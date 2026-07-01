"""HuggingFace Transformers ``Trainer`` integration for ``TrainingBridge``.

Copy-paste usage::

    from transformers import Trainer
    from agentic_optimizer.integrations.hf import HFBridgeCallback

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[HFBridgeCallback.from_env()],
    )
    trainer.train()

The callback wires the bridge to the Trainer model, optimizer, and LR scheduler,
reports logged training losses, reports epoch metrics from Trainer state, and
propagates agent-requested graceful stops through ``control.should_training_stop``.
"""
from __future__ import annotations

from numbers import Number
from typing import Any

try:
    from transformers import TrainerCallback as _HFCallback
except Exception:  # pragma: no cover
    _HFCallback = object

__all__ = ["BridgeCallback", "HFBridgeCallback"]


def _numeric_metrics(values: dict[str, Any] | None) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in (values or {}).items():
        if isinstance(value, Number):
            metrics[str(key)] = float(value)
    return metrics


class HFBridgeCallback(_HFCallback):
    """Transformers Trainer callback that forwards events to a bridge."""

    def __init__(
        self, bridge: Any = None, *, optimizer: Any = None, model: Any = None, **bridge_kw: Any
    ) -> None:
        if bridge is None:
            from agentic_optimizer.bridge import attach

            bridge = attach(optimizer, model, **bridge_kw)
        self.bridge = bridge

    @classmethod
    def from_env(cls, **bridge_kw: Any) -> "HFBridgeCallback":
        return cls(**bridge_kw)

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        del args, state
        self.bridge.optimizer = kwargs.get("optimizer") or self.bridge.optimizer
        self.bridge.model = kwargs.get("model") or self.bridge.model
        self.bridge.scheduler = kwargs.get("lr_scheduler") or self.bridge.scheduler
        self.bridge.on_train_begin()
        return control

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        del state, kwargs
        loss = (logs or {}).get("loss")
        if isinstance(loss, Number):
            batch_size = int(getattr(args, "per_device_train_batch_size", 0) or 0)
            self.bridge.on_batch_end(float(loss), batch_size=batch_size)
        if self.bridge.should_stop():
            control.should_training_stop = True
        return control

    def on_epoch_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        del args, kwargs
        epoch = getattr(state, "epoch", None)
        metrics = {"epoch": float(epoch)} if isinstance(epoch, Number) else {"epoch": 0.0}
        log_history = getattr(state, "log_history", None) or []
        if log_history:
            metrics.update(_numeric_metrics(log_history[-1]))
        self.bridge.on_epoch_end(int(epoch or 0), metrics)
        if self.bridge.should_stop():
            control.should_training_stop = True
        return control

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        del args, state, kwargs
        self.bridge.on_train_end()
        return control


BridgeCallback = HFBridgeCallback
