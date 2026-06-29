"""``TrainingBridge`` — the remote-side glue that connects a live PyTorch loop to the control plane.

The bridge is what makes a run *interactive* for a local agent:

* it **pushes telemetry** (training state + optional **MLflow** linkage) to the broker;
* it **claims commands** the agent enqueued and runs them through a :class:`HandlerRegistry` at **safe sync
  points** (epoch/batch boundaries), then **posts results** back;
* built-in handlers cover hyperparameters, pause/resume, augmentation, sample-flagging and evaluation, while a
  registry lets users expose **any** loop influence point (preprocessing, filtering, balancing, interrogations,
  custom knobs) — the agent reaches them generically via ``invoke`` / ``interrogate`` / ``set_knob``.

``torch`` is only needed for the optional grad-clip helper; the bridge itself imports it lazily.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from .contract import (
    Command,
    KnobSpec,
    MlflowInfo,
    ParamGroupState,
    PerSampleLoss,
    Telemetry,
    TrainingConfig,
    TrainingState,
)
from .controlplane import ControlPlaneClient
from .telemetry import gpu_telemetry

Handler = Callable[[dict[str, Any], "HandlerContext"], "dict[str, Any] | None"]
logger = logging.getLogger("agentic_optimizer.bridge")


@dataclass
class HandlerContext:
    """Passed to every handler so it can act on the live run."""

    bridge: "TrainingBridge"
    optimizer: Any
    model: Any
    args: dict[str, Any]
    command: Command


class HandlerRegistry:
    """Name → handler map. Built-ins are pre-registered; users add their own loop influence points."""

    def __init__(self) -> None:
        self._handlers: dict[str, Handler] = {}

    def register(self, name: str, fn: Handler) -> None:
        self._handlers[name] = fn

    def get(self, name: str) -> Handler | None:
        return self._handlers.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._handlers

    def names(self) -> list[str]:
        return list(self._handlers)


def _opt_float(v: Any) -> float | None:
    return None if v is None else float(v)


class TrainingBridge:
    """Bridge a PyTorch training loop to a :class:`ControlPlaneClient`.

    Call the lifecycle hooks from your own loop::

        bridge = TrainingBridge(optimizer, client, model=model, mlflow=True)
        bridge.register_knob("label_smoothing", set_label_smoothing, description="...")
        bridge.on_train_begin()
        for epoch in range(epochs):
            for x, y in loader:
                ...; loss.backward()
                bridge.clip_gradients(model)            # applies any agent-set grad_clip
                optimizer.step(); optimizer.zero_grad()
                bridge.on_batch_end(loss.item(), batch_size=len(x), grad_norm=gn)
            bridge.on_epoch_end(epoch, metrics={"val_acc": acc})   # pushes telemetry + applies commands
        bridge.on_train_end()
    """

    def __init__(
        self,
        optimizer: Any,
        client: ControlPlaneClient,
        model: Any = None,
        *,
        apply_every_batches: int = 0,
        history_len: int = 100,
        max_epochs: int | None = None,
        mlflow: bool = False,
        mlflow_info_provider: Callable[["TrainingBridge", dict[str, float]], MlflowInfo | None]
        | None = None,
        pause_poll_s: float = 0.5,
        run_id: str = "default",
        susp_topk: int = 256,
        on_training_config: Callable[[TrainingConfig], None] | None = None,
        on_flagged_samples: Callable[[list[int]], None] | None = None,
        poll_interval: float = 0.0,
        max_pause_s: float = 0.0,
    ) -> None:
        self.optimizer = optimizer
        self.client = client
        self.model = model
        self.apply_every_batches = apply_every_batches
        self.history_len = history_len
        self.max_epochs = max_epochs
        self.mlflow_enabled = mlflow
        self._mlflow_info_provider = mlflow_info_provider
        self.pause_poll_s = pause_poll_s
        self.run_id = run_id
        self.susp_topk = max(0, int(susp_topk))
        self.on_training_config = on_training_config
        self.on_flagged_samples = on_flagged_samples
        self.poll_interval = float(poll_interval)
        self.max_pause_s = float(max_pause_s)

        self.registry = HandlerRegistry()
        self._safe: dict[str, bool] = {}
        self.knobs: dict[str, KnobSpec] = {}
        self._knob_hooks: dict[str, Callable[[Any], None]] = {}

        # live state
        self.step = 0
        self.epoch = 0
        self.loss_history: list[float] = []
        self.grad_norm: float | None = None
        self.grad_clip: float | None = None
        self.paused = False
        self.augmentation_enabled: bool | None = None
        self.flagged_indices: set[int] = set()
        self.knob_values: dict[str, Any] = {}
        self.processed_commands: list[Command] = []
        self.last_error: str | None = None
        self.training_config = TrainingConfig()
        self.grad_accum_steps = 1
        self.amp_enabled = False

        self._win_t: float | None = None
        self._win_samples = 0
        self._susp: dict[int, float] = {}
        self._deferred: list[Command] = []
        self._dlock = threading.Lock()
        self._poll_stop = threading.Event()
        self._poll_thread: threading.Thread | None = None
        self._error_serial = 0

        self._register_builtins()

    # ------------------------------------------------------------- handlers
    def _register_builtins(self) -> None:
        self.registry.register("set_hyperparameters", self._h_set_hyperparameters)
        self.registry.register("pause", self._h_pause)
        self.registry.register("resume", self._h_resume)
        self.registry.register("set_augmentation", self._h_set_augmentation)
        self.registry.register("flag_samples", self._h_flag_samples)
        self.registry.register("run_evaluation", self._h_run_evaluation)
        self.registry.register("set_knob", self._h_set_knob)
        self.registry.register("set_training_config", self._h_set_training_config)

    def register(self, name: str, fn: Handler, safe_async: bool = False) -> None:
        """Register an action; ``safe_async`` handlers must be read-only when polled in-thread."""
        self.registry.register(name, fn)
        self._safe[name] = bool(safe_async)

    def register_knob(
        self, name: str, hook: Callable[[Any], None], description: str = "", value: Any = None
    ) -> None:
        """Advertise a named knob; ``hook(value)`` is invoked when the agent sets it."""
        self._knob_hooks[name] = hook
        self.knobs[name] = KnobSpec(name=name, description=description, value=value)
        self.knob_values[name] = value

    # built-in handler implementations
    def _h_set_hyperparameters(self, args: dict[str, Any], ctx: HandlerContext) -> dict[str, Any]:
        return {"applied": self.apply_hyperparameters(**_filter_hp_args(args))}

    def _h_pause(self, args: dict[str, Any], ctx: HandlerContext) -> dict[str, Any]:
        self.paused = True
        return {"paused": True}

    def _h_resume(self, args: dict[str, Any], ctx: HandlerContext) -> dict[str, Any]:
        self.paused = False
        return {"paused": False}

    def _h_set_augmentation(self, args: dict[str, Any], ctx: HandlerContext) -> dict[str, Any]:
        enabled = bool(args.get("enabled", True))
        self.augmentation_enabled = enabled
        hook = self._knob_hooks.get("augmentation")
        if hook is not None:
            hook(enabled)
        return {"augmentation": enabled}

    def _h_flag_samples(self, args: dict[str, Any], ctx: HandlerContext) -> dict[str, Any]:
        indices = [int(i) for i in args.get("indices", [])]
        before = set(self.flagged_indices)
        self.flagged_indices.update(indices)
        new_indices = sorted(self.flagged_indices - before)
        if new_indices and self.on_flagged_samples is not None:
            self.on_flagged_samples(new_indices)
        return {"flagged_now": len(indices), "flagged_total": len(self.flagged_indices)}

    def _h_run_evaluation(self, args: dict[str, Any], ctx: HandlerContext) -> dict[str, Any]:
        fn = self.registry.get("evaluate")
        if fn is None:
            raise RuntimeError("no 'evaluate' handler registered; call bridge.register('evaluate', fn)")
        return fn(args, ctx) or {}

    def _h_set_knob(self, args: dict[str, Any], ctx: HandlerContext) -> dict[str, Any]:
        name = args["name"]
        value = args.get("value")
        self.knob_values[name] = value
        if name in self.knobs:
            self.knobs[name].value = value
        hook = self._knob_hooks.get(name)
        if hook is not None:
            hook(value)
        return {"knob": name, "value": value}

    def _h_set_training_config(self, args: dict[str, Any], ctx: HandlerContext) -> dict[str, Any]:
        cfg = TrainingConfig.model_validate({k: v for k, v in args.items() if v is not None})
        applied = cfg.model_dump(exclude_none=True)
        if not applied:
            return {"applied": {}}

        current = self.training_config.model_dump()
        current.update(applied)
        self.training_config = TrainingConfig.model_validate(current)
        if cfg.grad_accum_steps is not None:
            self.grad_accum_steps = int(cfg.grad_accum_steps)
        if cfg.amp is not None:
            self.amp_enabled = bool(cfg.amp)
        if self.on_training_config is not None:
            self.on_training_config(cfg)
        return {"applied": applied}

    # ------------------------------------------------------------- mutation
    def apply_hyperparameters(
        self,
        lr: float | None = None,
        weight_decay: float | None = None,
        momentum: float | None = None,
        grad_clip: float | None = None,
        extra: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        applied: dict[str, Any] = {}
        for pg in self.optimizer.param_groups:
            if lr is not None:
                pg["lr"] = lr
                applied["lr"] = lr
            if weight_decay is not None:
                pg["weight_decay"] = weight_decay
                applied["weight_decay"] = weight_decay
            if momentum is not None and "momentum" in pg:
                pg["momentum"] = momentum
                applied["momentum"] = momentum
        if grad_clip is not None:
            self.grad_clip = grad_clip
            applied["grad_clip"] = grad_clip
        if extra:
            applied["extra"] = dict(extra)
        return applied

    def clip_gradients(self, model: Any = None) -> float | None:
        if self.grad_clip is None:
            return None
        target = model or self.model
        if target is None:
            return None
        import torch

        return float(torch.nn.utils.clip_grad_norm_(target.parameters(), self.grad_clip))

    # --------------------------------------------------------------- hooks
    def on_train_begin(self) -> None:
        self._win_t = time.time()
        self._win_samples = 0
        if self.knobs:
            try:
                self.client.register_knobs(list(self.knobs.values()), run_id=self.run_id)
            except Exception as e:
                self._set_last_error(e)
                logger.warning("failed to register knobs", exc_info=True)
        if self.poll_interval > 0 and self._poll_thread is None:
            self._poll_stop.clear()
            self._poll_thread = threading.Thread(
                target=self._poll_loop, name="TrainingBridgePoller", daemon=True
            )
            self._poll_thread.start()

    def on_batch_end(
        self,
        loss: float,
        batch_size: int = 0,
        grad_norm: float | None = None,
        sample_indices: Any = None,
        sample_losses: Any = None,
    ) -> None:
        self.step += 1
        self._win_samples += int(batch_size)
        try:
            self.loss_history.append(float(loss))
        except (TypeError, ValueError):
            pass
        if len(self.loss_history) > self.history_len:
            self.loss_history = self.loss_history[-self.history_len :]
        if grad_norm is not None:
            self.grad_norm = float(grad_norm)
        self._track_suspicious_losses(sample_indices, sample_losses)
        if self.apply_every_batches and (self.step % self.apply_every_batches == 0):
            self.push_telemetry({})
            self.drain_commands()

    def on_epoch_end(self, epoch: int, metrics: dict[str, float] | None = None) -> list[Command]:
        self.epoch = epoch
        self.push_telemetry(metrics or {})
        processed = self.drain_commands()
        # Honour a live pause: keep serving commands until the agent resumes.
        pause_started = time.monotonic()
        broker_errors = 0
        while self.paused:
            before_error_serial = self._error_serial
            self.push_telemetry({})
            processed += self.drain_commands()
            if self._error_serial > before_error_serial:
                broker_errors += self._error_serial - before_error_serial
                if broker_errors >= 3:
                    logger.error("broker errors during pause; waiting for resume may be blocked")
            else:
                broker_errors = 0
            if self.max_pause_s > 0 and time.monotonic() - pause_started > self.max_pause_s:
                logger.warning("auto-resume after max_pause_s=%s", self.max_pause_s)
                self.paused = False
                break
            if self.paused:
                time.sleep(self.pause_poll_s)
        return processed

    def on_train_end(self) -> None:
        if self._poll_thread is not None:
            self._poll_stop.set()
            self._poll_thread.join(timeout=max(self.poll_interval, 0.1) + 1.0)
            self._poll_thread = None
        self.push_telemetry({})
        self.drain_commands()

    # --------------------------------------------------------- telemetry/io
    def build_state(self, metrics: dict[str, float]) -> TrainingState:
        param_groups = [
            ParamGroupState(
                lr=float(pg.get("lr", 0.0)),
                weight_decay=_opt_float(pg.get("weight_decay")),
                momentum=_opt_float(pg.get("momentum")),
            )
            for pg in self.optimizer.param_groups
        ]
        now = time.time()
        throughput = None
        if self._win_t is not None and now > self._win_t and self._win_samples > 0:
            throughput = self._win_samples / (now - self._win_t)
        self._win_t = now
        self._win_samples = 0
        per_sample_losses = [
            PerSampleLoss(index=index, loss=sample_loss)
            for index, sample_loss in sorted(
                self._susp.items(), key=lambda item: item[1], reverse=True
            )
        ]
        return TrainingState(
            step=self.step,
            epoch=self.epoch,
            max_epochs=self.max_epochs,
            timestamp=now,
            metrics={k: float(v) for k, v in metrics.items()},
            loss_history=list(self.loss_history),
            param_groups=param_groups,
            grad_norm=self.grad_norm,
            throughput_samples_per_s=throughput,
            gpu=gpu_telemetry(),
            per_sample_losses=per_sample_losses,
        )

    def push_telemetry(self, metrics: dict[str, float]) -> None:
        state = self.build_state(metrics)
        telem = Telemetry(
            run_id=self.run_id,
            state=state,
            mlflow=self._mlflow_info(metrics),
            paused=self.paused,
            knobs=list(self.knobs.values()),
            last_error=self.last_error,
        )
        try:
            self.client.push_telemetry(telem)
            self._susp = {}
        except Exception as e:
            # Telemetry is best-effort: never let a transient broker hiccup crash training.
            self._set_last_error(e)
            logger.warning("failed to push telemetry", exc_info=True)

    def drain_commands(self, max_commands: int = 100) -> list[Command]:
        processed: list[Command] = []
        if self._poll_thread is not None or self._deferred:
            with self._dlock:
                deferred = self._deferred
                self._deferred = []
            for cmd in deferred:
                self._execute(cmd)
                processed.append(cmd)
                self.processed_commands.append(cmd)
        for _ in range(max_commands):
            try:
                cmd = self.client.next_command(wait=0.0, run_id=self.run_id)
            except Exception as e:
                self._set_last_error(e)
                logger.warning("failed to drain commands", exc_info=True)
                break
            if cmd is None:
                break
            self._execute(cmd)
            processed.append(cmd)
            self.processed_commands.append(cmd)
        return processed

    def _resolve(self, cmd: Command) -> tuple[Handler, dict[str, Any]]:
        t = cmd.type
        handler = self.registry.get(t)
        if handler is not None:
            return handler, cmd.args
        if t in ("invoke", "interrogate"):
            name = cmd.args.get("action") or cmd.args.get("name")
            target = self.registry.get(name) if name else None
            if target is not None:
                return target, cmd.args.get("args", cmd.args)
            raise KeyError(f"no handler registered for action {name!r}")
        raise KeyError(f"no handler for command type {t!r}")

    def _execute(self, cmd: Command) -> None:
        try:
            handler, hargs = self._resolve(cmd)
            ctx = HandlerContext(
                bridge=self, optimizer=self.optimizer, model=self.model, args=hargs, command=cmd
            )
            data = handler(hargs, ctx) or {}
            self.client.complete_command(cmd.id, ok=True, data=data)
        except Exception as e:  # noqa: BLE001 - report any handler failure back to the agent
            self._set_last_error(e)
            logger.warning("command execution failed", exc_info=True)
            try:
                self.client.complete_command(
                    cmd.id, ok=False, error=f"{type(e).__name__}: {e}"
                )
            except Exception:
                self._set_last_error(e)
                logger.warning("failed to report command failure", exc_info=True)

    def _set_last_error(self, e: Exception) -> None:
        self.last_error = f"{type(e).__name__}: {e}"
        self._error_serial += 1

    def _track_suspicious_losses(self, sample_indices: Any, sample_losses: Any) -> None:
        if sample_indices is None or sample_losses is None or self.susp_topk <= 0:
            return
        for index, loss in zip(sample_indices, sample_losses, strict=False):
            i = int(index)
            sample_loss = float(loss)
            if i not in self._susp or sample_loss > self._susp[i]:
                self._susp[i] = sample_loss
        if len(self._susp) > self.susp_topk:
            self._susp = dict(
                sorted(self._susp.items(), key=lambda item: item[1], reverse=True)[
                    : self.susp_topk
                ]
            )

    def _is_safe_async(self, cmd: Command) -> bool:
        if cmd.type in {"pause", "resume", "flag_samples"}:
            return True
        if cmd.type in {"invoke", "interrogate"}:
            name = cmd.args.get("action") or cmd.args.get("name")
            return bool(name and self.registry.get(name) is not None and self._safe.get(name, False))
        return False

    def _poll_once(self) -> int:
        handled = 0
        for _ in range(10):
            cmd = self.client.next_command(wait=0.0, run_id=self.run_id)
            if cmd is None:
                break
            if self._is_safe_async(cmd):
                self._execute(cmd)
                self.processed_commands.append(cmd)
            else:
                with self._dlock:
                    self._deferred.append(cmd)
            handled += 1
        return handled

    def _poll_loop(self) -> None:
        while not self._poll_stop.is_set():
            try:
                self._poll_once()
                self._poll_stop.wait(self.poll_interval)
            except Exception as e:  # noqa: BLE001 - poller must never silently die
                self._set_last_error(e)
                logger.warning("command poller iteration failed", exc_info=True)
                self._poll_stop.wait(min(self.poll_interval * 2, 5.0))

    # --------------------------------------------------------------- mlflow
    def _mlflow_info(self, metrics: dict[str, float]) -> MlflowInfo | None:
        if self._mlflow_info_provider is not None:
            return self._mlflow_info_provider(self, metrics)
        if not self.mlflow_enabled:
            return None
        return _default_mlflow_info(self, metrics)


def _filter_hp_args(args: dict[str, Any]) -> dict[str, Any]:
    """Keep only recognised hyperparameter keys from a command's args."""
    allowed = {"lr", "weight_decay", "momentum", "grad_clip", "extra"}
    return {k: v for k, v in args.items() if k in allowed}


def _default_mlflow_info(bridge: TrainingBridge, metrics: dict[str, float]) -> MlflowInfo | None:
    """Read the active MLflow run and log the just-pushed metrics to it (best-effort)."""
    try:
        import mlflow

        run = mlflow.active_run()
        if run is None:
            return None
        if metrics:
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=bridge.step)
        info = run.info
        return MlflowInfo(
            run_id=info.run_id,
            tracking_uri=mlflow.get_tracking_uri(),
            experiment=getattr(info, "experiment_id", None),
            run_name=getattr(info, "run_name", None),
            metrics={k: float(v) for k, v in metrics.items()},
        )
    except Exception:
        return None
