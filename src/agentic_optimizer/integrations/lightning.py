"""PyTorch Lightning integration for :class:`agentic_optimizer.bridge.TrainingBridge`.

Copy-paste usage::

    import pytorch_lightning as pl
    from agentic_optimizer.integrations.lightning import BridgeCallback

    trainer = pl.Trainer(callbacks=[BridgeCallback.from_env()])
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

The callback wires the bridge to Lightning's optimizer, module, and scheduler at
train start, reports loss after each training batch, reports callback metrics at
epoch boundaries, and propagates agent-requested graceful stops through
``trainer.should_stop``.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

try:
    from pytorch_lightning.callbacks import Callback as _PLCallback
except Exception:  # pragma: no cover
    try:
        from lightning.pytorch.callbacks import Callback as _PLCallback
    except Exception:
        _PLCallback = object

__all__ = ["BridgeCallback"]


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_loss(outputs: Any) -> float | None:
    """Extract a scalar loss from Lightning batch outputs."""
    if isinstance(outputs, Mapping):
        outputs = outputs.get("loss")
    return _as_float(outputs)


def _infer_batch_size(batch: Any) -> int:
    """Best-effort batch-size inference from common Lightning batch shapes."""
    try:
        first = batch[0] if isinstance(batch, Sequence) and not isinstance(batch, str) else batch
    except Exception:
        return 0
    if first is None:
        return 0
    shape = getattr(first, "shape", None)
    if shape:
        try:
            return int(shape[0])
        except Exception:
            return 0
    try:
        return len(first)
    except Exception:
        return 0


def _metrics_to_float(metrics: Mapping[str, Any] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in (metrics or {}).items():
        converted = _as_float(value)
        if converted is not None:
            out[str(key)] = converted
    return out


class BridgeCallback(_PLCallback):
    """Lightning callback that forwards training lifecycle events to a bridge."""

    def __init__(
        self, bridge: Any = None, *, optimizer: Any = None, model: Any = None, **bridge_kw: Any
    ) -> None:
        if bridge is None:
            from agentic_optimizer.bridge import attach

            bridge = attach(optimizer, model, **bridge_kw)
        self.bridge = bridge
        self._started = False

    @classmethod
    def from_env(cls, **bridge_kw: Any) -> "BridgeCallback":
        return cls(**bridge_kw)

    def _wire_bridge(self, trainer: Any, pl_module: Any) -> None:
        optimizers = getattr(trainer, "optimizers", None) or []
        if optimizers:
            self.bridge.optimizer = optimizers[0]
        self.bridge.model = pl_module
        scheduler_configs = getattr(trainer, "lr_scheduler_configs", None) or []
        if scheduler_configs:
            scheduler = getattr(scheduler_configs[0], "scheduler", None)
            if scheduler is not None:
                self.bridge.scheduler = scheduler

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        self._start(trainer, pl_module)

    def on_train_start(self, trainer: Any, pl_module: Any) -> None:
        self._start(trainer, pl_module)

    def _start(self, trainer: Any, pl_module: Any) -> None:
        self._wire_bridge(trainer, pl_module)
        if not self._started:
            self.bridge.on_train_begin()
            self._started = True

    def on_train_batch_end(
        self, trainer: Any, pl_module: Any, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        del pl_module, batch_idx
        loss = _extract_loss(outputs)
        if loss is not None:
            self.bridge.on_batch_end(loss, batch_size=_infer_batch_size(batch))
        if self.bridge.should_stop():
            trainer.should_stop = True

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        del pl_module
        metrics = _metrics_to_float(getattr(trainer, "callback_metrics", None))
        self.bridge.on_epoch_end(int(getattr(trainer, "current_epoch", 0) or 0), metrics)
        if self.bridge.should_stop():
            trainer.should_stop = True

    def on_train_end(self, trainer: Any, pl_module: Any) -> None:
        del trainer, pl_module
        self.bridge.on_train_end()
        self._started = False
