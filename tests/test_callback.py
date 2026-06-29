from types import SimpleNamespace

from agentic_optimizer.callback import AgenticCallback
from agentic_optimizer.contract import ControlSignal
from agentic_optimizer.driver import FunctionDriver


def make_opt(lr=0.1):
    return SimpleNamespace(param_groups=[{"lr": lr, "weight_decay": 0.0, "momentum": 0.9}])


def test_applies_control_at_epoch_end():
    opt = make_opt(0.1)
    driver = FunctionDriver(
        lambda s: ControlSignal(
            set_lr=0.05, grad_clip=1.0, enable_augmentation=True, flag_noisy_indices=[2]
        )
    )
    cb = AgenticCallback(opt, driver=driver, optimize_every=1)
    cb.on_train_begin()
    cb.on_batch_end(1.0, batch_size=8)
    ctrl = cb.on_epoch_end(0, {"val_acc": 0.3})
    assert opt.param_groups[0]["lr"] == 0.05
    assert cb.grad_clip == 1.0
    assert cb.augmentation_enabled is True
    assert 2 in cb.flagged_indices
    assert not ctrl.is_empty()


def test_optimize_every_cadence():
    opt = make_opt(0.1)
    seen = []
    driver = FunctionDriver(lambda s: (seen.append(s.epoch), ControlSignal())[1])
    cb = AgenticCallback(opt, driver=driver, optimize_every=2)
    cb.on_train_begin()
    for e in range(4):
        cb.on_epoch_end(e)
    assert seen == [0, 2]


def test_build_state_reflects_optimizer_and_history():
    opt = make_opt(0.2)
    cb = AgenticCallback(opt, driver=FunctionDriver(lambda s: ControlSignal()), optimize_every=1)
    cb.on_train_begin()
    cb.on_batch_end(0.5, batch_size=4, grad_norm=3.0)
    st = cb.build_state({"val_acc": 0.4})
    assert st.param_groups[0].lr == 0.2
    assert st.loss_history[-1] == 0.5
    assert st.grad_norm == 3.0
    assert st.metrics["val_acc"] == 0.4


def test_empty_control_does_not_change_lr():
    opt = make_opt(0.1)
    cb = AgenticCallback(opt, driver=FunctionDriver(lambda s: ControlSignal()), optimize_every=1)
    cb.on_train_begin()
    cb.on_epoch_end(0)
    assert opt.param_groups[0]["lr"] == 0.1


def test_async_applies_next_epoch():
    opt = make_opt(0.1)
    driver = FunctionDriver(lambda s: ControlSignal(set_lr=0.01))
    cb = AgenticCallback(opt, driver=driver, optimize_every=1, async_mode=True)
    cb.on_train_begin()
    cb.on_epoch_end(0)  # launches async; not applied yet
    cb._join_async()  # make the result deterministic
    assert opt.param_groups[0]["lr"] == 0.1
    cb.on_epoch_end(1)  # applies the decision computed during epoch 0
    assert opt.param_groups[0]["lr"] == 0.01
