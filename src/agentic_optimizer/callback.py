"""``AgenticCallback`` — the minimal glue between a PyTorch training loop and the Copilot CLI optimizer.

The callback is framework-agnostic about *how* you train: you call its hooks from your own loop. At a
configurable cadence it builds a :class:`~agentic_optimizer.contract.TrainingState`, asks the driver for a
:class:`~agentic_optimizer.contract.ControlSignal`, and applies it to the optimizer at the sync point.

``torch`` is imported lazily (only for optional GPU telemetry / grad-norm / grad-clip helpers) so the
callback can be unit-tested without a GPU and the package can be imported without PyTorch.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from .contract import ControlSignal, GpuTelemetry, ParamGroupState, TrainingState
from .driver import CopilotOptimizerDriver, OptimizerDriver


def _opt_float(value: Any) -> float | None:
    return None if value is None else float(value)


class AgenticCallback:
    """Bridge a training loop to an :class:`OptimizerDriver`.

    Typical use::

        cb = AgenticCallback(optimizer, optimize_every=1)
        cb.on_train_begin()
        for epoch in range(epochs):
            for x, y in loader:
                ...
                loss.backward()
                gn = cb.compute_grad_norm(model)
                cb.clip_gradients(model)          # applies any agent-set grad_clip
                optimizer.step(); optimizer.zero_grad()
                cb.on_batch_end(loss.item(), batch_size=len(x), grad_norm=gn)
            cb.on_epoch_end(epoch, metrics={"val_acc": acc})
        cb.on_train_end()
    """

    def __init__(
        self,
        optimizer: Any,
        driver: OptimizerDriver | None = None,
        optimize_every: int = 1,
        state_path: str | Path = "state.json",
        control_path: str | Path = "control.json",
        history_len: int = 100,
        async_mode: bool = False,
        max_epochs: int | None = None,
        model: Any = None,
    ) -> None:
        self.optimizer = optimizer
        self.state_path = Path(state_path)
        self.control_path = Path(control_path)
        if driver is None:
            driver = CopilotOptimizerDriver(
                workdir=self.state_path.resolve().parent,
                state_filename=self.state_path.name,
                control_filename=self.control_path.name,
            )
        self.driver = driver
        self.optimize_every = optimize_every
        self.history_len = history_len
        self.async_mode = async_mode
        self.max_epochs = max_epochs
        self.model = model

        # runtime state
        self.loss_history: list[float] = []
        self.step = 0
        self.epoch = 0
        self.grad_norm: float | None = None
        self.grad_clip: float | None = None
        self.pending_batch_size: int | None = None
        self.augmentation_enabled: bool | None = None
        self.flagged_indices: set[int] = set()
        self.applied_controls: list[ControlSignal] = []
        self.last_control: ControlSignal | None = None

        self._t0: float | None = None
        self._win_t: float | None = None
        self._win_samples = 0
        self._lock = threading.Lock()
        self._pending_result: ControlSignal | None = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ hooks
    def on_train_begin(self) -> None:
        self._t0 = time.time()
        self._win_t = self._t0
        self._win_samples = 0

    def on_batch_end(
        self, loss: float, batch_size: int = 0, grad_norm: float | None = None
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

    def on_epoch_end(self, epoch: int, metrics: dict[str, float] | None = None) -> ControlSignal:
        """Consult the agent (per ``optimize_every``) and apply its decision.

        In async mode the decision computed during the *previous* epoch is applied first (non-blocking),
        then a new consultation is launched in the background.
        """
        self.epoch = epoch
        if self.async_mode:
            self._apply_pending_async()

        if self.optimize_every <= 0 or (epoch % self.optimize_every) != 0:
            return ControlSignal.empty()

        state = self.build_state(metrics or {})
        if self.async_mode:
            self._launch_async(state)
            return ControlSignal.empty()

        control = self.driver.optimize(state)
        return self.apply_control(control)

    def on_train_end(self) -> None:
        if self.async_mode:
            self._join_async()
            self._apply_pending_async()

    # ------------------------------------------------------------------ core
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
            gpu=self._gpu_telemetry(),
        )

    def apply_control(self, control: ControlSignal | None) -> ControlSignal:
        if control is None:
            return ControlSignal.empty()
        self.last_control = control
        if control.is_empty():
            return control
        for pg in self.optimizer.param_groups:
            if control.set_lr is not None:
                pg["lr"] = control.set_lr
            if control.set_weight_decay is not None:
                pg["weight_decay"] = control.set_weight_decay
            if control.set_momentum is not None and "momentum" in pg:
                pg["momentum"] = control.set_momentum
        if control.grad_clip is not None:
            self.grad_clip = control.grad_clip
        if control.batch_size is not None:
            self.pending_batch_size = control.batch_size
        if control.enable_augmentation is not None:
            self.augmentation_enabled = control.enable_augmentation
        if control.flag_noisy_indices:
            self.flagged_indices.update(control.flag_noisy_indices)
        self.applied_controls.append(control)
        return control

    # -------------------------------------------------------------- helpers
    def clip_gradients(self, model: Any = None) -> float | None:
        """Apply the agent-set ``grad_clip`` (if any) to ``model``'s gradients."""
        if self.grad_clip is None:
            return None
        target = model or self.model
        if target is None:
            return None
        import torch

        return float(torch.nn.utils.clip_grad_norm_(target.parameters(), self.grad_clip))

    @staticmethod
    def compute_grad_norm(model: Any, norm_type: float = 2.0) -> float:
        import torch

        grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
        if not grads:
            return 0.0
        stacked = torch.stack([torch.norm(g, norm_type) for g in grads])
        return float(torch.norm(stacked, norm_type))

    def _gpu_telemetry(self) -> GpuTelemetry | None:
        try:
            import torch

            if not torch.cuda.is_available():
                return None
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            return GpuTelemetry(
                device=props.name,
                mem_used_mb=torch.cuda.memory_allocated(idx) / 1e6,
                mem_total_mb=props.total_memory / 1e6,
                util_pct=None,
            )
        except Exception:
            return None

    # ---------------------------------------------------------- async plumbing
    def _launch_async(self, state: TrainingState) -> None:
        def run() -> None:
            control = self.driver.optimize(state)
            with self._lock:
                self._pending_result = control

        self._thread = threading.Thread(target=run, name="agentic-optimize", daemon=True)
        self._thread.start()

    def _apply_pending_async(self) -> None:
        with self._lock:
            control = self._pending_result
            self._pending_result = None
        if control is not None:
            self.apply_control(control)

    def _join_async(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None
