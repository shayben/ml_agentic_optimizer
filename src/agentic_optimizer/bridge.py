"""``TrainingBridge`` — the remote-side glue that connects a live PyTorch loop to the control plane.

The bridge is what makes a run *interactive* for a local agent — **without ever pausing or idling it**:

* it **pushes telemetry** (training state + optional **MLflow** linkage) to the broker;
* it **claims commands** the agent enqueued and runs them through a :class:`HandlerRegistry` at **safe sync
  points** (epoch/batch boundaries), then **posts results** back;
* built-in handlers cover hyperparameters, augmentation, sample-flagging and evaluation, while a
  registry lets users expose **any** loop influence point (preprocessing, filtering, balancing, interrogations,
  custom knobs) — the agent reaches them generically via ``invoke`` / ``interrogate`` / ``set_knob``.

The training loop is **never blocked**: telemetry is fire-and-forget and command draining is non-blocking, so the
agent always observes *slightly stale* state and its influence lands **asynchronously** at the next sync point
(never a barrier that idles the GPU). A graceful ``stop_training`` sets a flag the loop polls; it does not idle.

``torch`` is only needed for the optional grad-clip helper; the bridge itself imports it lazily.
"""
from __future__ import annotations

import contextlib
import copy
import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from . import distributed as dist
from .contract import (
    AnomalyEvent as ContractAnomalyEvent,
    CheckpointInfo,
    Command,
    DistributedInfo,
    GuardrailBound,
    GuardrailConfig,
    KnobSpec,
    MlflowInfo,
    ParamGroupState,
    PerSampleLoss,
    ProfileSection,
    ProfileSummary,
    SchedulerState,
    Telemetry,
    TrainingConfig,
    TrainingState,
)
from .controlplane import ControlPlaneClient
from .profiling import StepProfiler
from .safety import AnomalyDetector, AnomalyEvent, Guardrails
from .telemetry import compute_grad_norm, gpu_telemetry

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


def _finite_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Drop non-finite metric values so telemetry stays JSON-compliant during divergence.

    ``inf``/``nan`` are not JSON-serializable; leaking them makes the whole telemetry
    push fail (taking anomalies riding the same payload down with it) exactly when the
    run is diverging and the agent most needs to see it."""
    out: dict[str, float] = {}
    for k, v in metrics.items():
        fv = _as_float(v)
        if math.isfinite(fv):
            out[k] = fv
    return out


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
        run_id: str = "default",
        susp_topk: int = 256,
        on_training_config: Callable[[TrainingConfig], None] | None = None,
        on_flagged_samples: Callable[[list[int]], None] | None = None,
        poll_interval: float = 0.0,
        scheduler: Any = None,
        scaler: Any = None,
        on_scheduler_reconfig: Callable[[dict[str, Any]], Any] | None = None,
        anomaly_detector: AnomalyDetector | None = None,
        guardrails: GuardrailConfig | dict[str, Any] | None = None,
        auto_grad_norm: bool = True,
        profiler: StepProfiler | None = None,
        checkpoint_dir: str | None = None,
        max_checkpoints: int = 5,
    ) -> None:
        self.optimizer = optimizer
        self.client = client
        self.model = model
        self.apply_every_batches = apply_every_batches
        self.history_len = history_len
        self.max_epochs = max_epochs
        self.mlflow_enabled = mlflow
        self._mlflow_info_provider = mlflow_info_provider
        self.run_id = run_id
        self.susp_topk = max(0, int(susp_topk))
        self.on_training_config = on_training_config
        self.on_flagged_samples = on_flagged_samples
        self.poll_interval = float(poll_interval)
        self.scheduler = scheduler
        self.scaler = scaler
        self.on_scheduler_reconfig = on_scheduler_reconfig
        self.auto_grad_norm = bool(auto_grad_norm)
        self._anomaly = anomaly_detector or AnomalyDetector()
        self._guardrails = _as_guardrails(guardrails)
        self.profiler = profiler if profiler is not None else StepProfiler()
        self._checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max(0, int(max_checkpoints))

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
        self.augmentation_enabled: bool | None = None
        self.flagged_indices: set[int] = set()
        self.knob_values: dict[str, Any] = {}
        self.processed_commands: list[Command] = []
        self.last_error: str | None = None
        self.training_config = TrainingConfig()
        self.grad_accum_steps = 1
        self.amp_enabled = False
        self._stop_requested = False
        self._checkpoints: dict[str, CheckpointInfo] = {}
        self._ckpt_blobs: dict[str, Any] = {}
        self._anomalies: list[ContractAnomalyEvent] = []

        self._win_t: float | None = None
        self._win_samples = 0
        self._susp: dict[int, float] = {}
        self._deferred: list[Command] = []
        self._pending_flag_indices: list[list[int]] = []
        self._dlock = threading.Lock()
        self._poll_stop = threading.Event()
        self._poll_thread: threading.Thread | None = None
        self._error_serial = 0

        self._register_builtins()

    # ------------------------------------------------------------- handlers
    def _register_builtins(self) -> None:
        self.registry.register("set_hyperparameters", self._h_set_hyperparameters)
        self.registry.register("set_augmentation", self._h_set_augmentation)
        self.registry.register("flag_samples", self._h_flag_samples)
        self.registry.register("run_evaluation", self._h_run_evaluation)
        self.registry.register("set_knob", self._h_set_knob)
        self.registry.register("set_training_config", self._h_set_training_config)
        self.registry.register("save_checkpoint", self._h_save_checkpoint)
        self.registry.register("restore_checkpoint", self._h_restore_checkpoint)
        self.registry.register("set_guardrails", self._h_set_guardrails)
        self.registry.register("set_scheduler", self._h_set_scheduler)
        self.registry.register("stop_training", self._h_stop_training)
        self.registry.register("extend_training", self._h_extend_training)

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
        hp = _filter_hp_args(args)
        clamped, notes = self._apply_guardrails(hp)
        applied = self.apply_hyperparameters(**clamped)
        result: dict[str, Any] = {"applied": applied}
        if notes:
            result["guardrails"] = notes
        return result

    def _apply_guardrails(
        self, hp: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Clamp requested hyperparameters to configured bounds/relative-change limits."""
        current: dict[str, Any] = {"grad_clip": self.grad_clip}
        if self.optimizer is not None and self.optimizer.param_groups:
            pg0 = self.optimizer.param_groups[0]
            current["lr"] = pg0.get("lr")
            current["weight_decay"] = pg0.get("weight_decay")
            current["momentum"] = pg0.get("momentum")
        out = dict(hp)
        notes: dict[str, Any] = {}
        for name in ("lr", "weight_decay", "momentum", "grad_clip"):
            if out.get(name) is None:
                continue
            res = self._guardrails.validate(name, float(out[name]), current.get(name))
            if res.changed:
                out[name] = res.value
                notes[name] = {
                    "requested": float(hp[name]),
                    "applied": res.value,
                    "reason": res.reason,
                }
        return out, notes

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
            # flag_samples runs on the poller thread (it is safe_async so the flag registers
            # immediately), but the user callback typically mutates shared training tensors
            # (e.g. zeroing sample weights). Defer it to the next sync point so it runs on the
            # training thread instead of racing it. drain_commands flushes the queue.
            with self._dlock:
                self._pending_flag_indices.append(new_indices)
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

    def _h_stop_training(self, args: dict[str, Any], ctx: HandlerContext) -> dict[str, Any]:
        self._stop_requested = True
        return {"stopping": True, "epoch": self.epoch, "step": self.step}

    def _h_extend_training(self, args: dict[str, Any], ctx: HandlerContext) -> dict[str, Any]:
        if "max_epochs" not in args:
            raise ValueError("extend_training requires 'max_epochs'")
        self.max_epochs = int(args["max_epochs"])
        return {"max_epochs": self.max_epochs}

    def _h_set_guardrails(self, args: dict[str, Any], ctx: HandlerContext) -> dict[str, Any]:
        self._guardrails.configure(args)
        return {"guardrails": self._guardrails.to_dict()}

    def _h_set_scheduler(self, args: dict[str, Any], ctx: HandlerContext) -> dict[str, Any]:
        if self.on_scheduler_reconfig is None:
            raise RuntimeError(
                "no scheduler reconfigure hook; pass on_scheduler_reconfig=... to TrainingBridge"
            )
        new = self.on_scheduler_reconfig(dict(args))
        if new is not None:
            self.scheduler = new
        state = self._scheduler_state()
        return {"scheduler": state.model_dump() if state is not None else None}

    def _h_save_checkpoint(self, args: dict[str, Any], ctx: HandlerContext) -> dict[str, Any]:
        return self.save_checkpoint(
            checkpoint_id=args.get("id"), note=args.get("note"), metrics=args.get("metrics")
        )

    def _h_restore_checkpoint(self, args: dict[str, Any], ctx: HandlerContext) -> dict[str, Any]:
        checkpoint_id = args.get("id")
        if not checkpoint_id:
            if not self._checkpoints:
                raise RuntimeError("no checkpoints have been saved")
            checkpoint_id = next(reversed(self._checkpoints))  # most recently saved
        return self.restore_checkpoint(checkpoint_id)

    # ------------------------------------------------------- checkpoint/rollback
    def save_checkpoint(
        self,
        checkpoint_id: str | None = None,
        note: str | None = None,
        metrics: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Snapshot model/optimizer/scheduler/scaler/RNG so the agent can roll back later."""
        import uuid

        cid = checkpoint_id or uuid.uuid4().hex[:12]
        # Re-saving an existing id must move it to the most-recently-saved position: a plain
        # dict update keeps the original insertion order, which would make restore-latest and
        # eviction target the wrong checkpoint. Pop first so the re-insert appends at the end.
        self._checkpoints.pop(cid, None)
        self._ckpt_blobs.pop(cid, None)
        blob: dict[str, Any] = {"step": self.step, "epoch": self.epoch}
        for name, obj in (
            ("model", self.model),
            ("optimizer", self.optimizer),
            ("scheduler", self.scheduler),
            ("scaler", self.scaler),
        ):
            if obj is not None and hasattr(obj, "state_dict"):
                blob[name] = obj.state_dict()
        blob["rng"] = _capture_rng()
        path: str | None = None
        if self._checkpoint_dir:
            try:
                import torch

                os.makedirs(self._checkpoint_dir, exist_ok=True)
                path = os.path.join(self._checkpoint_dir, f"{cid}.pt")
                torch.save(blob, path)
            except Exception:
                path = None
                self._ckpt_blobs[cid] = _clone_state(blob)
        else:
            self._ckpt_blobs[cid] = _clone_state(blob)
        info = CheckpointInfo(
            id=cid,
            step=self.step,
            epoch=self.epoch,
            metrics={k: float(v) for k, v in (metrics or {}).items()},
            path=path,
            note=note,
        )
        self._checkpoints[cid] = info
        self._evict_checkpoints()
        return {"id": cid, "step": self.step, "epoch": self.epoch, "path": path}

    def restore_checkpoint(self, checkpoint_id: str) -> dict[str, Any]:
        """Roll the live run back to a saved checkpoint (weights/optimizer/scheduler/RNG)."""
        blob = self._load_blob(checkpoint_id)
        if blob is None:
            raise RuntimeError(f"unknown checkpoint {checkpoint_id!r}")
        for name, obj in (
            ("model", self.model),
            ("optimizer", self.optimizer),
            ("scheduler", self.scheduler),
            ("scaler", self.scaler),
        ):
            if obj is not None and name in blob and hasattr(obj, "load_state_dict"):
                obj.load_state_dict(blob[name])
        _restore_rng(blob.get("rng"))
        self.step = int(blob.get("step", self.step))
        self.epoch = int(blob.get("epoch", self.epoch))
        self._anomaly.reset()
        return {"restored": checkpoint_id, "step": self.step, "epoch": self.epoch}

    def _load_blob(self, checkpoint_id: str) -> dict[str, Any] | None:
        if checkpoint_id in self._ckpt_blobs:
            return self._ckpt_blobs[checkpoint_id]
        info = self._checkpoints.get(checkpoint_id)
        if info is not None and info.path:
            import torch

            return torch.load(info.path, map_location="cpu", weights_only=False)
        return None

    def _evict_checkpoints(self) -> None:
        if self.max_checkpoints <= 0:
            return
        while len(self._checkpoints) > self.max_checkpoints:
            oldest_id = next(iter(self._checkpoints))  # least recently saved
            oldest = self._checkpoints.pop(oldest_id, None)
            self._ckpt_blobs.pop(oldest_id, None)
            if oldest is not None and oldest.path:
                with contextlib.suppress(OSError):
                    os.remove(oldest.path)

    # ------------------------------------------------------------- scheduler
    def scheduler_step(self, *args: Any, **kwargs: Any) -> None:
        """Step the attached LR scheduler (call where you'd normally call ``scheduler.step()``)."""
        if self.scheduler is not None and hasattr(self.scheduler, "step"):
            self.scheduler.step(*args, **kwargs)

    def _scheduler_state(self) -> SchedulerState | None:
        scheduler = self.scheduler
        if scheduler is None:
            return None
        last_lr: list[float] = []
        try:
            if hasattr(scheduler, "get_last_lr"):
                last_lr = [float(x) for x in scheduler.get_last_lr()]
            elif self.optimizer is not None:
                last_lr = [float(pg.get("lr", 0.0)) for pg in self.optimizer.param_groups]
        except Exception:
            last_lr = []
        return SchedulerState(
            name=type(scheduler).__name__,
            last_lr=last_lr,
            last_epoch=getattr(scheduler, "last_epoch", None),
            config=_scheduler_config(scheduler),
        )

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
        loss_val = _as_float(loss)
        if math.isfinite(loss_val):
            self.loss_history.append(loss_val)
            if len(self.loss_history) > self.history_len:
                self.loss_history = self.loss_history[-self.history_len :]
        if grad_norm is None and self.auto_grad_norm and self.model is not None:
            grad_norm = self._auto_grad_norm()
        if grad_norm is not None:
            gn = float(grad_norm)
            self.grad_norm = gn if math.isfinite(gn) else None
        self._track_suspicious_losses(sample_indices, sample_losses)
        event = self._anomaly.update(loss=loss_val, grad_norm=grad_norm, step=self.step)
        if event is not None:
            self._record_anomaly(event)
        self.profiler.mark_step()
        if self.apply_every_batches and (self.step % self.apply_every_batches == 0):
            self.push_telemetry({})
            self.drain_commands()

    def on_epoch_end(self, epoch: int, metrics: dict[str, float] | None = None) -> list[Command]:
        self.epoch = epoch
        self.push_telemetry(metrics or {})
        # Non-blocking by design: push the latest telemetry and apply whatever commands the agent
        # has already enqueued, then return immediately so the loop keeps training. The agent's
        # influence is therefore always *asynchronous* — it acts on slightly stale telemetry and
        # its next command lands here one or more epochs later. The loop never idles waiting for it.
        return self.drain_commands()

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
        gpu = gpu_telemetry()
        dinfo = dist.info()
        return TrainingState(
            step=self.step,
            epoch=self.epoch,
            max_epochs=self.max_epochs,
            timestamp=now,
            metrics=_finite_metrics(metrics),
            loss_history=list(self.loss_history),
            param_groups=param_groups,
            grad_norm=self.grad_norm,
            throughput_samples_per_s=throughput,
            gpu=gpu,
            per_sample_losses=per_sample_losses,
            scheduler=self._scheduler_state(),
            profile=self._profile_summary(gpu),
            distributed=DistributedInfo(**dinfo) if dinfo.get("enabled") else None,
            stop_requested=self._stop_requested,
        )

    def push_telemetry(self, metrics: dict[str, float]) -> None:
        if dist.is_available() and not dist.is_main_process():
            return
        state = self.build_state(metrics)
        telem = Telemetry(
            run_id=self.run_id,
            state=state,
            mlflow=self._mlflow_info(metrics),
            knobs=list(self.knobs.values()),
            last_error=self.last_error,
            checkpoints=list(self._checkpoints.values()),
            anomalies=list(self._anomalies),
            guardrails=self._guardrails_config(),
            stopping=self._stop_requested,
        )
        try:
            self.client.push_telemetry(telem)
            self._susp = {}
        except Exception as e:
            # Telemetry is best-effort: never let a transient broker hiccup crash training.
            self._set_last_error(e)
            logger.warning("failed to push telemetry", exc_info=True)

    def drain_commands(self, max_commands: int = 100) -> list[Command]:
        # In distributed mode, only rank 0 talks to the broker; mutations are then
        # broadcast to the other ranks so every replica applies the same change.
        if dist.is_available() and not dist.is_main_process():
            applied = self._apply_replicated_commands()
            self._flush_flag_callbacks()
            return applied
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
        self._broadcast_processed(processed)
        self._flush_flag_callbacks()
        return processed

    def _flush_flag_callbacks(self) -> None:
        """Run any deferred ``on_flagged_samples`` callbacks on the calling (training) thread.

        ``flag_samples`` registers the flag immediately on the poller thread, but its callback
        usually mutates shared tensors, so it is queued here and applied at a training-thread
        sync point (drain) to avoid racing the loop."""
        if self.on_flagged_samples is None:
            return
        with self._dlock:
            pending = self._pending_flag_indices
            self._pending_flag_indices = []
        for indices in pending:
            self.on_flagged_samples(indices)

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
            if not math.isfinite(sample_loss):
                continue
            if i not in self._susp or sample_loss > self._susp[i]:
                self._susp[i] = sample_loss
        if len(self._susp) > self.susp_topk:
            self._susp = dict(
                sorted(self._susp.items(), key=lambda item: item[1], reverse=True)[
                    : self.susp_topk
                ]
            )

    def _is_safe_async(self, cmd: Command) -> bool:
        if cmd.type in {
            "flag_samples",
            "stop_training",
            "extend_training",
            "set_guardrails",
        }:
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

    # ----------------------------------------------------------- ergonomics
    def train_step(
        self,
        loss: Any,
        optimizer: Any = None,
        *,
        backward: bool = True,
        batch_size: int = 0,
        grad_norm: float | None = None,
        sample_indices: Any = None,
        sample_losses: Any = None,
        zero_grad: bool = True,
    ) -> None:
        """One-call training step: backward → clip → opt.step → zero_grad → on_batch_end.

        ``loss`` may be a tensor (``backward=True`` calls ``loss.backward()``) or a float.
        Grad-norm is captured *before* ``zero_grad`` so anomaly detection sees real gradients.
        The bridge is also callable, so ``bridge(loss, batch_size=n)`` is shorthand for this.
        """
        opt = optimizer or self.optimizer
        loss_val = _as_float(loss)
        if backward and hasattr(loss, "backward"):
            loss.backward()
        captured = grad_norm
        if captured is None and self.grad_clip is not None:
            captured = self.clip_gradients()
        if captured is None and self.auto_grad_norm and self.model is not None:
            captured = self._auto_grad_norm()
        if opt is not None and hasattr(opt, "step"):
            opt.step()
        if zero_grad and opt is not None and hasattr(opt, "zero_grad"):
            opt.zero_grad()
        self.on_batch_end(
            loss_val,
            batch_size=batch_size,
            grad_norm=captured,
            sample_indices=sample_indices,
            sample_losses=sample_losses,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        """Shorthand for :meth:`train_step` so ``bridge(loss, batch_size=n)`` just works."""
        return self.train_step(*args, **kwargs)

    def epoch_end(self, epoch: int | None = None, **metrics: float) -> list[Command]:
        """Convenience wrapper around :meth:`on_epoch_end` (auto-increments epoch)."""
        ep = epoch if epoch is not None else self.epoch + 1
        return self.on_epoch_end(ep, {k: float(v) for k, v in metrics.items()})

    def should_stop(self) -> bool:
        """True once the agent has requested a graceful stop (poll this in your loop)."""
        return self._stop_requested

    def section(self, name: str) -> Any:
        """Profiler timing section context manager (e.g. ``with bridge.section("forward"):``)."""
        return self.profiler.section(name)

    def __enter__(self) -> TrainingBridge:
        self.on_train_begin()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.on_train_end()

    @classmethod
    def from_env(
        cls, optimizer: Any = None, model: Any = None, **kwargs: Any
    ) -> TrainingBridge | NoOpBridge:
        """Build a bridge from ``CONTROL_PLANE_*`` env vars.

        Returns a :class:`NoOpBridge` when ``CONTROL_PLANE_URL`` is unset so the *same*
        training script runs unchanged with or without the control plane configured.
        """
        client = ControlPlaneClient.from_env()
        if client is None:
            return NoOpBridge(optimizer, model, **kwargs)
        run_id = kwargs.pop("run_id", None) or os.environ.get("CONTROL_PLANE_RUN_ID") or "default"
        return cls(optimizer, client, model=model, run_id=run_id, **kwargs)

    # --------------------------------------------------- anomaly / profiler
    def _auto_grad_norm(self) -> float | None:
        if self.model is None:
            return None
        try:
            return compute_grad_norm(self.model.parameters())
        except Exception:
            return None

    def _record_anomaly(self, event: AnomalyEvent) -> None:
        value = event.value if event.value is not None and math.isfinite(event.value) else None
        contract_event = ContractAnomalyEvent(
            kind=event.kind,
            message=event.message,
            value=value,
            step=event.step if event.step is not None else self.step,
        )
        self._anomalies.append(contract_event)
        if len(self._anomalies) > 50:
            self._anomalies = self._anomalies[-50:]
        self.last_error = f"anomaly: {event.message}"
        self._error_serial += 1
        logger.warning("training anomaly detected: %s", event.message)

    def _profile_summary(self, gpu: Any) -> ProfileSummary | None:
        summary = self.profiler.summary()
        if not summary.get("steps"):
            return None
        gpu_util = getattr(gpu, "util_pct", None) if gpu is not None else None
        sections = [
            ProfileSection(name=s["name"], ms_avg=s["ms_avg"], pct=s["pct"])
            for s in summary.get("sections", [])
        ]
        return ProfileSummary(
            steps=summary["steps"],
            step_ms_avg=summary["step_ms_avg"],
            sections=sections,
            suggestions=self.profiler.suggest(gpu_util_pct=gpu_util),
        )

    def _guardrails_config(self) -> GuardrailConfig | None:
        cfg = self._guardrails.to_dict()
        if not cfg.get("bounds") and cfg.get("max_rel_change") is None:
            return None
        bounds = {
            name: GuardrailBound(min=b.get("min"), max=b.get("max"))
            for name, b in cfg.get("bounds", {}).items()
        }
        return GuardrailConfig(bounds=bounds, max_rel_change=cfg.get("max_rel_change"))

    # ------------------------------------------------------- distributed io
    def _broadcast_processed(self, processed: list[Command]) -> None:
        if not dist.is_available():
            return
        payload = [
            {"type": c.type, "args": c.args}
            for c in processed
            if c.type not in _NON_REPLICATED
        ]
        dist.broadcast_object(payload, src=0)

    def _apply_replicated_commands(self) -> list[Command]:
        payload = dist.broadcast_object(None, src=0)
        applied: list[Command] = []
        for item in payload or []:
            cmd = Command(type=item["type"], args=item.get("args", {}))
            try:
                handler, hargs = self._resolve(cmd)
                ctx = HandlerContext(
                    bridge=self, optimizer=self.optimizer, model=self.model, args=hargs, command=cmd
                )
                handler(hargs, ctx)
                applied.append(cmd)
            except Exception:
                logger.warning("replica failed to apply %s", cmd.type, exc_info=True)
        return applied

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
            mlflow.log_metrics(_finite_metrics(metrics), step=bridge.step)
        info = run.info
        return MlflowInfo(
            run_id=info.run_id,
            tracking_uri=mlflow.get_tracking_uri(),
            experiment=getattr(info, "experiment_id", None),
            run_name=getattr(info, "run_name", None),
            metrics=_finite_metrics(metrics),
        )
    except Exception:
        return None


# Commands that must NOT be replayed on non-zero ranks (rank-0-only side effects).
_NON_REPLICATED = {
    "interrogate",
    "invoke",
    "run_evaluation",
    "flag_samples",
    "save_checkpoint",
    "stop_training",
    "extend_training",
}


def _as_guardrails(g: Guardrails | GuardrailConfig | dict[str, Any] | None) -> Guardrails:
    """Coerce assorted guardrail inputs into a :class:`Guardrails` instance."""
    if isinstance(g, Guardrails):
        return g
    if g is None:
        return Guardrails()
    if isinstance(g, GuardrailConfig):
        return Guardrails(g.model_dump())
    return Guardrails(dict(g))


def _as_float(v: Any) -> float:
    """Best-effort float conversion (tensor ``.item()`` aware); NaN on failure."""
    if v is None:
        return math.nan
    item = getattr(v, "item", None)
    if callable(item):
        with contextlib.suppress(Exception):
            return float(item())
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def _clone_state(value: Any) -> Any:
    """Deep-copy a (possibly nested) ``state_dict`` so an in-memory checkpoint does not
    alias live tensors.

    Real ``torch`` ``state_dict()`` tensors share storage with live parameters; the next
    ``optimizer.step()`` mutates that storage in place, so an un-cloned snapshot silently
    drifts and ``restore_checkpoint`` becomes a no-op. Tensors are detached, moved to CPU,
    and cloned; dict/list/tuple containers recurse; anything else is deep-copied."""
    try:
        import torch

        if torch.is_tensor(value):
            return value.detach().cpu().clone()
    except Exception:
        pass
    if isinstance(value, dict):
        return {k: _clone_state(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone_state(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_state(v) for v in value)
    with contextlib.suppress(Exception):
        return copy.deepcopy(value)
    return value


def _capture_rng() -> dict[str, Any]:
    """Snapshot Python/torch/CUDA RNG state for reproducible checkpoint restore."""
    import random

    state: dict[str, Any] = {"python": random.getstate()}
    try:
        import torch

        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
    except Exception:
        pass
    return state


def _restore_rng(state: dict[str, Any] | None) -> None:
    if not state:
        return
    import random

    with contextlib.suppress(Exception):
        random.setstate(state["python"])
    try:
        import torch

        if "torch" in state:
            torch.set_rng_state(state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception:
        pass


def _scheduler_config(scheduler: Any) -> dict[str, Any]:
    """Extract JSON-serialisable scheduler attributes for telemetry."""
    keys = (
        "base_lrs",
        "gamma",
        "step_size",
        "T_max",
        "eta_min",
        "factor",
        "patience",
        "milestones",
        "total_steps",
        "max_lr",
        "min_lr",
    )
    cfg: dict[str, Any] = {}
    for key in keys:
        if not hasattr(scheduler, key):
            continue
        val = getattr(scheduler, key)
        if not _json_safe(val):
            with contextlib.suppress(TypeError, ValueError):
                val = list(val)
        if _json_safe(val):
            cfg[key] = val
    return cfg


def _json_safe(val: Any) -> bool:
    try:
        json.dumps(val)
        return True
    except (TypeError, ValueError):
        return False


class NoOpBridge:
    """A control-plane-free stand-in returned by :func:`attach` when no broker is configured.

    Training still runs normally: :meth:`train_step` / :meth:`__call__` perform the real
    ``loss.backward()`` → grad-clip → ``optimizer.step()`` → ``zero_grad()``, and
    :meth:`scheduler_step` advances the scheduler. Only the *control-plane* surface (telemetry
    push, command draining, checkpoints, interrogations) is inert. This way the *same* training
    script runs unchanged — and actually trains — whether or not ``CONTROL_PLANE_URL`` is set,
    becoming agent-steerable only when a broker is present. ``should_stop()`` is always ``False``
    and :meth:`section` yields a null context.
    """

    def __init__(
        self,
        optimizer: Any = None,
        model: Any = None,
        *,
        scheduler: Any = None,
        grad_clip: float | None = None,
        **_: Any,
    ) -> None:
        self.optimizer = optimizer
        self.model = model
        self.scheduler = scheduler
        self.grad_clip = grad_clip

    def __enter__(self) -> NoOpBridge:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def should_stop(self) -> bool:
        return False

    def section(self, name: str) -> Any:
        return contextlib.nullcontext()

    def train_step(
        self,
        loss: Any = None,
        optimizer: Any = None,
        *,
        backward: bool = True,
        zero_grad: bool = True,
        **_: Any,
    ) -> None:
        """Drive the real optimization step (telemetry is dropped, training is not)."""
        opt = optimizer or self.optimizer
        if backward and hasattr(loss, "backward"):
            loss.backward()
        self.clip_gradients()
        if opt is not None and hasattr(opt, "step"):
            opt.step()
        if zero_grad and opt is not None and hasattr(opt, "zero_grad"):
            opt.zero_grad()

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        return self.train_step(*args, **kwargs)

    def clip_gradients(self, model: Any = None) -> float | None:
        target = model or self.model
        if self.grad_clip is None or target is None or not hasattr(target, "parameters"):
            return None
        try:
            import torch

            return float(
                torch.nn.utils.clip_grad_norm_(target.parameters(), float(self.grad_clip))
            )
        except Exception:
            return None

    def scheduler_step(self, *args: Any, **kwargs: Any) -> None:
        if self.scheduler is not None and hasattr(self.scheduler, "step"):
            self.scheduler.step(*args, **kwargs)

    def register(self, *args: Any, **kwargs: Any) -> None:
        return None

    def register_knob(self, *args: Any, **kwargs: Any) -> None:
        return None

    def __getattr__(self, name: str) -> Callable[..., None]:
        def _noop(*args: Any, **kwargs: Any) -> None:
            return None

        return _noop


def attach(
    optimizer: Any = None, model: Any = None, **kwargs: Any
) -> TrainingBridge | NoOpBridge:
    """One-call entry point: build a bridge from the environment.

    Equivalent to :meth:`TrainingBridge.from_env`. Returns a :class:`NoOpBridge` when
    ``CONTROL_PLANE_URL`` is unset, so you can sprinkle ``bridge = attach(optimizer, model)``
    into any script and it stays inert until a control plane is configured::

        with attach(optimizer, model) as bridge:
            for epoch in range(epochs):
                for x, y in loader:
                    loss = loss_fn(model(x), y)
                    bridge.train_step(loss, batch_size=len(x))  # or: bridge(loss, batch_size=len(x))
                bridge.epoch_end(epoch, val_acc=acc)
                if bridge.should_stop():
                    break
    """
    return TrainingBridge.from_env(optimizer, model=model, **kwargs)
