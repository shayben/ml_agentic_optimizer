from __future__ import annotations

from types import SimpleNamespace

import torch

from agentic_optimizer.integrations.hf import HFBridgeCallback
from agentic_optimizer.integrations.lightning import (
    BridgeCallback,
    _extract_loss,
    _infer_batch_size,
)


class FakeBridge:
    def __init__(self) -> None:
        self.optimizer = None
        self.model = None
        self.scheduler = None
        self.batch_end: list[dict] = []
        self.epoch_end: list[dict] = []
        self.train_begin = False
        self.train_end = False
        self.paused = False
        self._stop = False

    def on_train_begin(self) -> None:
        self.train_begin = True

    def on_train_end(self) -> None:
        self.train_end = True

    def on_batch_end(self, loss: float, batch_size: int = 0, **kwargs) -> None:
        self.batch_end.append({"loss": loss, "batch_size": batch_size, **kwargs})

    def on_epoch_end(self, epoch: int, metrics: dict[str, float] | None = None) -> list:
        self.epoch_end.append({"epoch": epoch, "metrics": metrics or {}})
        return []

    def should_stop(self) -> bool:
        return self._stop


def test_lightning_callback_wires_and_reports_hooks() -> None:
    fake = FakeBridge()
    cb = BridgeCallback(bridge=fake)
    opt = SimpleNamespace(param_groups=[{"lr": 0.1}])
    trainer = SimpleNamespace(
        optimizers=[opt],
        lr_scheduler_configs=[],
        current_epoch=0,
        should_stop=False,
        callback_metrics={"val_acc": torch.tensor(0.5)},
    )
    pl_module = object()

    cb.on_train_start(trainer, pl_module)
    assert fake.optimizer is opt
    assert fake.model is pl_module
    assert fake.train_begin is True

    cb.on_train_batch_end(
        trainer,
        pl_module,
        {"loss": torch.tensor(1.23)},
        (torch.zeros(8, 3), torch.zeros(8)),
        0,
    )
    assert fake.batch_end[-1]["loss"] == torch.tensor(1.23).item()
    assert fake.batch_end[-1]["batch_size"] == 8

    fake._stop = True
    cb.on_train_batch_end(trainer, pl_module, torch.tensor(0.9), torch.zeros(4, 3), 1)
    assert trainer.should_stop is True

    cb.on_train_epoch_end(trainer, pl_module)
    assert fake.epoch_end[-1] == {"epoch": 0, "metrics": {"val_acc": 0.5}}

    cb.on_train_end(trainer, pl_module)
    assert fake.train_end is True


def test_lightning_helpers_extract_loss() -> None:
    assert _extract_loss({"loss": torch.tensor(1.5)}) == 1.5
    assert _extract_loss(torch.tensor(2.5)) == 2.5
    assert _extract_loss(3.5) == 3.5
    assert _extract_loss(None) is None
    assert _extract_loss({"not_loss": torch.tensor(1.0)}) is None


def test_lightning_helpers_infer_batch_size() -> None:
    assert _infer_batch_size((torch.zeros(8, 3), torch.zeros(8))) == 8
    assert _infer_batch_size(torch.zeros(4, 3)) == 4
    assert _infer_batch_size(None) == 0


def test_hf_callback_wires_and_reports_hooks() -> None:
    fake = FakeBridge()
    cb = HFBridgeCallback(bridge=fake)
    args = SimpleNamespace(per_device_train_batch_size=16)
    state = SimpleNamespace(epoch=1.0, log_history=[])
    control = SimpleNamespace(should_training_stop=False)
    model = object()
    opt = SimpleNamespace(param_groups=[{"lr": 0.1}])
    scheduler = object()

    returned = cb.on_train_begin(
        args, state, control, model=model, optimizer=opt, lr_scheduler=scheduler
    )
    assert returned is control
    assert fake.model is model
    assert fake.optimizer is opt
    assert fake.scheduler is scheduler
    assert fake.train_begin is True

    cb.on_log(args, state, control, logs={"loss": 0.7})
    assert fake.batch_end[-1] == {"loss": 0.7, "batch_size": 16}

    fake._stop = True
    cb.on_log(args, state, control, logs={"loss": 0.6})
    assert control.should_training_stop is True

    cb.on_epoch_end(args, state, control)
    assert fake.epoch_end[-1] == {"epoch": 1, "metrics": {"epoch": 1.0}}

    cb.on_train_end(args, state, control)
    assert fake.train_end is True


def test_lightning_adapter_applies_real_bridge_lr_change() -> None:
    from agentic_optimizer.bridge import TrainingBridge
    from agentic_optimizer.controlplane import ControlPlaneClient, ControlPlaneStore, create_app

    client = ControlPlaneClient.from_app(create_app(ControlPlaneStore()))
    opt = SimpleNamespace(param_groups=[{"lr": 0.1}])
    bridge = TrainingBridge(opt, client)
    cb = BridgeCallback(bridge=bridge)
    trainer = SimpleNamespace(
        optimizers=[opt],
        lr_scheduler_configs=[],
        current_epoch=0,
        should_stop=False,
        callback_metrics={},
    )

    cb.on_train_start(trainer, object())
    client.enqueue_command("set_hyperparameters", {"lr": 0.02})
    cb.on_train_epoch_end(trainer, object())

    assert opt.param_groups[0]["lr"] == 0.02
