from agentic_optimizer.contract import (
    Command,
    CommandRequest,
    ControlSignal,
    ParamGroupState,
    Telemetry,
    TrainingConfig,
    TrainingState,
)


def test_training_state_roundtrip(tmp_path):
    s = TrainingState(
        step=10,
        epoch=2,
        metrics={"val_acc": 0.5},
        loss_history=[1.0, 0.9],
        param_groups=[ParamGroupState(lr=0.1, weight_decay=1e-4, momentum=0.9)],
        grad_norm=2.5,
    )
    p = s.write_json(tmp_path / "state.json")
    s2 = TrainingState.read_json(p)
    assert s2.step == 10 and s2.epoch == 2
    assert s2.param_groups[0].lr == 0.1
    assert s2.metrics["val_acc"] == 0.5
    assert s2.grad_norm == 2.5


def test_control_is_empty():
    assert ControlSignal().is_empty()
    assert ControlSignal.empty().is_empty()
    assert not ControlSignal(set_lr=0.01).is_empty()
    assert not ControlSignal(flag_noisy_indices=[1]).is_empty()
    assert not ControlSignal(enable_augmentation=True).is_empty()


def test_control_from_text_tolerant():
    assert ControlSignal.from_text('{"set_lr": 0.01}').set_lr == 0.01
    c = ControlSignal.from_text("Here is my decision:\n{\"grad_clip\": 1.0}\nDone")
    assert c.grad_clip == 1.0
    assert ControlSignal.from_text("").is_empty()
    assert ControlSignal.from_text("no json here").is_empty()


def test_control_roundtrip(tmp_path):
    c = ControlSignal(set_lr=0.02, grad_clip=2.0, flag_noisy_indices=[3, 7], notes="reduce lr")
    p = c.write_json(tmp_path / "control.json")
    c2 = ControlSignal.read_json(p)
    assert c2.set_lr == 0.02 and c2.grad_clip == 2.0 and c2.flag_noisy_indices == [3, 7]
    assert c2.notes == "reduce lr"


def test_training_config_is_empty():
    assert TrainingConfig().is_empty()
    assert not TrainingConfig(batch_size=64).is_empty()
    assert not TrainingConfig(amp=True).is_empty()
    assert not TrainingConfig(grad_accum_steps=2).is_empty()
    assert not TrainingConfig(num_workers=4).is_empty()


def test_v2_runid_and_lease_defaults():
    assert Telemetry().run_id == "default"
    assert Telemetry().last_error is None
    assert CommandRequest(type="set_knob").run_id == "default"
    cmd = Command(type="set_hyperparameters")
    assert cmd.run_id == "default"
    assert cmd.attempts == 0
    assert cmd.lease_expires_at is None
    # explicit run namespacing round-trips through JSON
    t = Telemetry(run_id="exp-7", last_error="boom")
    assert Telemetry.model_validate_json(t.model_dump_json()).run_id == "exp-7"
