"""Integration tests for the live-control features: checkpoints, anomaly detection, guardrails,
profiler/scheduler telemetry, run lifecycle (stop/extend), distributed gating, and the
``step``/``epoch_end``/``attach`` ergonomics layer."""
from __future__ import annotations

import math

import pytest

from agentic_optimizer import bridge as bridge_module
from agentic_optimizer.bridge import NoOpBridge, TrainingBridge, attach
from agentic_optimizer.safety import AnomalyDetector
from agentic_optimizer.controlplane import (
    ControlPlaneClient,
    ControlPlaneStore,
    create_app,
)


class FakeModule:
    def __init__(self, val: float = 1.0) -> None:
        self.val = val

    def state_dict(self) -> dict:
        return {"val": self.val}

    def load_state_dict(self, sd: dict) -> None:
        self.val = sd["val"]


class FakeOpt:
    def __init__(self) -> None:
        self.param_groups = [{"lr": 0.1, "weight_decay": 0.0, "momentum": 0.9}]
        self.steps = 0
        self.zeroed = 0

    def state_dict(self) -> dict:
        return {"pg": [dict(pg) for pg in self.param_groups]}

    def load_state_dict(self, sd: dict) -> None:
        self.param_groups = [dict(pg) for pg in sd["pg"]]

    def step(self) -> None:
        self.steps += 1

    def zero_grad(self) -> None:
        self.zeroed += 1


class FakeLoss:
    def __init__(self, v: float) -> None:
        self.v = v
        self.backwarded = False

    def backward(self) -> None:
        self.backwarded = True

    def item(self) -> float:
        return self.v


class FakeScheduler:
    def __init__(self) -> None:
        self.last_epoch = 2
        self.base_lrs = [0.1]
        self.gamma = 0.9
        self.stepped = 0

    def get_last_lr(self) -> list[float]:
        return [0.05]

    def step(self) -> None:
        self.stepped += 1
        self.last_epoch += 1


def make_bridge(optimizer=None, **kwargs):
    opt = optimizer or FakeOpt()
    client = ControlPlaneClient.from_app(create_app(ControlPlaneStore()))
    return TrainingBridge(opt, client, **kwargs), opt, client


# --------------------------------------------------------------- checkpoints
def test_checkpoint_save_restore_roundtrip_in_memory():
    model = FakeModule(val=1.0)
    bridge, opt, client = make_bridge(model=model)
    bridge.step = 5
    bridge.epoch = 2
    saved = bridge.save_checkpoint(note="before risky change")
    cid = saved["id"]

    model.val = 99.0
    opt.param_groups[0]["lr"] = 5.0
    bridge.step = 50
    bridge.epoch = 9

    restored = bridge.restore_checkpoint(cid)
    assert restored["restored"] == cid
    assert model.val == 1.0
    assert opt.param_groups[0]["lr"] == 0.1
    assert bridge.step == 5 and bridge.epoch == 2


def test_checkpoint_save_restore_via_commands_and_listing():
    model = FakeModule(val=2.0)
    bridge, opt, client = make_bridge(model=model)
    save = client.enqueue_command("save_checkpoint", {"note": "ckpt-a"})
    bridge.drain_commands()
    cid = client.get_command(save.id).result.data["id"]

    bridge.push_telemetry({})
    listed = client.get_telemetry().checkpoints
    assert any(c.id == cid and c.note == "ckpt-a" for c in listed)

    model.val = -1.0
    client.enqueue_command("restore_checkpoint", {"id": cid})
    bridge.drain_commands()
    assert model.val == 2.0


def test_restore_latest_checkpoint_when_id_omitted():
    model = FakeModule(val=1.0)
    bridge, opt, client = make_bridge(model=model)
    bridge.save_checkpoint()
    model.val = 7.0
    bridge.save_checkpoint()  # latest snapshot captures val=7.0
    model.val = 123.0

    client.enqueue_command("restore_checkpoint", {})
    bridge.drain_commands()
    assert model.val == 7.0


def test_checkpoint_eviction_respects_max_checkpoints():
    bridge, opt, client = make_bridge(model=FakeModule(), max_checkpoints=2)
    bridge.save_checkpoint(checkpoint_id="a")
    bridge.save_checkpoint(checkpoint_id="b")
    bridge.save_checkpoint(checkpoint_id="c")
    assert set(bridge._checkpoints) == {"b", "c"}


def test_resaving_existing_id_moves_it_to_latest():
    """Re-saving an id must move it to the most-recent position so restore-latest picks it up,
    rather than the entry that merely has the newest content but the oldest insertion slot."""
    model = FakeModule(val=1.0)
    bridge, opt, client = make_bridge(model=model)
    bridge.save_checkpoint(checkpoint_id="a")  # val=1.0
    model.val = 2.0
    bridge.save_checkpoint(checkpoint_id="b")  # val=2.0
    model.val = 3.0
    bridge.save_checkpoint(checkpoint_id="a")  # re-save "a"; it becomes the latest
    model.val = 99.0

    client.enqueue_command("restore_checkpoint", {})  # no id -> latest
    bridge.drain_commands()
    assert model.val == 3.0  # the re-saved "a", not "b"
    assert list(bridge._checkpoints) == ["b", "a"]


def test_resaving_existing_id_updates_eviction_order():
    """A re-save also refreshes eviction order so the just-touched id is not the next evicted."""
    bridge, opt, client = make_bridge(model=FakeModule(), max_checkpoints=2)
    bridge.save_checkpoint(checkpoint_id="a")
    bridge.save_checkpoint(checkpoint_id="b")
    bridge.save_checkpoint(checkpoint_id="a")  # refresh "a" -> "b" is now oldest
    bridge.save_checkpoint(checkpoint_id="c")  # evicts the oldest ("b")
    assert set(bridge._checkpoints) == {"a", "c"}


def test_checkpoint_disk_roundtrip(tmp_path):
    pytest.importorskip("torch")  # the disk path uses torch.save/torch.load
    model = FakeModule(val=3.0)
    bridge, opt, client = make_bridge(model=model, checkpoint_dir=str(tmp_path))
    saved = bridge.save_checkpoint(checkpoint_id="disk")
    assert saved["path"] is not None
    assert (tmp_path / "disk.pt").exists()

    model.val = 0.0
    bridge.restore_checkpoint("disk")
    assert model.val == 3.0


# --------------------------------------------------------------- anomalies
def test_anomaly_recorded_on_nan_loss():
    bridge, opt, client = make_bridge()
    bridge.on_batch_end(float("nan"), batch_size=4)
    # Anomalies are recorded and surfaced to the agent, but NEVER pause the loop.
    assert len(bridge._anomalies) == 1
    assert bridge._anomalies[0].kind == "nan_loss"
    assert bridge.last_error and "anomaly" in bridge.last_error


def test_anomaly_recorded_in_telemetry():
    bridge, opt, client = make_bridge()
    bridge.on_batch_end(float("inf"), batch_size=4)
    bridge.push_telemetry({})
    anomalies = client.get_telemetry().anomalies
    assert anomalies and anomalies[0].kind == "inf_loss"


def test_anomaly_grad_explosion_detected_after_warmup():
    detector = AnomalyDetector(warmup=2, grad_explosion_factor=2.0)
    bridge, opt, client = make_bridge(anomaly_detector=detector, auto_grad_norm=False)
    for _ in range(3):
        bridge.on_batch_end(0.5, grad_norm=1.0)
    bridge.on_batch_end(0.5, grad_norm=20.0)
    assert any(a.kind == "grad_explosion" for a in bridge._anomalies)


# --------------------------------------------------------------- guardrails
def test_guardrails_clamp_hyperparameters_on_set():
    bridge, opt, client = make_bridge(guardrails={"bounds": {"lr": {"min": 1e-3, "max": 0.1}}})
    cmd = client.enqueue_command("set_hyperparameters", {"lr": 1.0})
    bridge.drain_commands()
    assert opt.param_groups[0]["lr"] == 0.1
    data = client.get_command(cmd.id).result.data
    assert data["applied"]["lr"] == 0.1
    assert data["guardrails"]["lr"]["requested"] == 1.0


def test_set_guardrails_command_then_clamps():
    bridge, opt, client = make_bridge()
    client.enqueue_command("set_guardrails", {"bounds": {"lr": {"max": 0.05}}})
    bridge.drain_commands()
    client.enqueue_command("set_hyperparameters", {"lr": 0.9})
    bridge.drain_commands()
    assert opt.param_groups[0]["lr"] == 0.05


def test_guardrails_max_rel_change_limits_jump():
    bridge, opt, client = make_bridge(guardrails={"max_rel_change": 2.0})
    client.enqueue_command("set_hyperparameters", {"lr": 100.0})
    bridge.drain_commands()
    assert opt.param_groups[0]["lr"] == 0.2  # 0.1 current * 2.0 cap


# --------------------------------------------------------------- profiler
def test_profiler_sections_surface_in_telemetry():
    bridge, opt, client = make_bridge()
    for _ in range(3):
        with bridge.section("data"):
            pass
        with bridge.section("forward"):
            pass
        bridge.on_batch_end(0.5, batch_size=8)
    bridge.push_telemetry({})
    profile = client.get_telemetry().state.profile
    assert profile is not None
    assert profile.steps >= 3
    assert {s.name for s in profile.sections} == {"data", "forward"}


# --------------------------------------------------------------- scheduler
def test_scheduler_state_in_telemetry_and_step():
    sched = FakeScheduler()
    bridge, opt, client = make_bridge(scheduler=sched)
    bridge.push_telemetry({})
    state = client.get_telemetry().state.scheduler
    assert state is not None
    assert state.name == "FakeScheduler"
    assert state.last_lr == [0.05]
    assert state.config.get("gamma") == 0.9

    bridge.scheduler_step()
    assert sched.stepped == 1


def test_set_scheduler_command_invokes_hook():
    sched = FakeScheduler()
    seen = {}

    def reconfig(args):
        seen.update(args)
        return FakeScheduler()

    bridge, opt, client = make_bridge(scheduler=sched, on_scheduler_reconfig=reconfig)
    cmd = client.enqueue_command("set_scheduler", {"gamma": 0.5})
    bridge.drain_commands()
    assert seen == {"gamma": 0.5}
    assert bridge.scheduler is not sched
    assert client.get_command(cmd.id).result.data["scheduler"]["name"] == "FakeScheduler"


def test_set_scheduler_without_hook_fails():
    bridge, opt, client = make_bridge(scheduler=FakeScheduler())
    cmd = client.enqueue_command("set_scheduler", {"gamma": 0.5})
    bridge.drain_commands()
    done = client.get_command(cmd.id)
    assert done.status.value == "failed"


# --------------------------------------------------------------- run lifecycle
def test_stop_training_sets_should_stop_and_telemetry():
    bridge, opt, client = make_bridge()
    client.enqueue_command("stop_training", {})
    bridge.drain_commands()
    assert bridge.should_stop() is True
    bridge.push_telemetry({})
    t = client.get_telemetry()
    assert t.stopping is True
    assert t.state.stop_requested is True


def test_extend_training_raises_max_epochs():
    bridge, opt, client = make_bridge(max_epochs=5)
    client.enqueue_command("extend_training", {"max_epochs": 12})
    bridge.drain_commands()
    assert bridge.max_epochs == 12


def test_extend_training_requires_max_epochs():
    bridge, opt, client = make_bridge(max_epochs=5)
    cmd = client.enqueue_command("extend_training", {})
    bridge.drain_commands()
    assert client.get_command(cmd.id).status.value == "failed"


# --------------------------------------------------------------- distributed
class FakeDist:
    def __init__(self, *, available=True, main=True, info=None, recv=None) -> None:
        self.available = available
        self.main = main
        self._info = info or {"enabled": True, "rank": 0, "world_size": 2, "backend": "gloo"}
        self._recv = recv
        self.broadcasts: list = []

    def is_available(self) -> bool:
        return self.available

    def is_main_process(self) -> bool:
        return self.main

    def info(self) -> dict:
        return self._info

    def broadcast_object(self, obj, src: int = 0):
        self.broadcasts.append(obj)
        return self._recv if obj is None else obj


def test_distributed_non_main_skips_telemetry(monkeypatch):
    bridge, opt, client = make_bridge()
    monkeypatch.setattr(bridge_module, "dist", FakeDist(available=True, main=False))
    bridge.push_telemetry({"val_acc": 0.5})
    assert client.get_telemetry() is None


def test_distributed_non_main_applies_replicated_commands(monkeypatch):
    bridge, opt, client = make_bridge()
    fake = FakeDist(
        available=True,
        main=False,
        recv=[{"type": "set_hyperparameters", "args": {"lr": 0.02}}],
    )
    monkeypatch.setattr(bridge_module, "dist", fake)
    bridge.drain_commands()
    assert opt.param_groups[0]["lr"] == 0.02


def test_distributed_main_broadcasts_processed(monkeypatch):
    bridge, opt, client = make_bridge()
    fake = FakeDist(available=True, main=True)
    monkeypatch.setattr(bridge_module, "dist", fake)
    client.enqueue_command("set_hyperparameters", {"lr": 0.03})
    client.enqueue_command("flag_samples", {"indices": [1]})  # non-replicated
    bridge.drain_commands()
    payload = fake.broadcasts[-1]
    assert payload == [{"type": "set_hyperparameters", "args": {"lr": 0.03}}]


def test_distributed_info_in_telemetry_on_main(monkeypatch):
    bridge, opt, client = make_bridge()
    monkeypatch.setattr(bridge_module, "dist", FakeDist(available=True, main=True))
    bridge.push_telemetry({})
    dinfo = client.get_telemetry().state.distributed
    assert dinfo is not None and dinfo.world_size == 2


# --------------------------------------------------------------- ergonomics
def test_step_runs_full_dance():
    model = FakeModule()
    bridge, opt, client = make_bridge(model=model)
    loss = FakeLoss(0.42)
    bridge.train_step(loss, batch_size=8)
    assert loss.backwarded is True
    assert opt.steps == 1 and opt.zeroed == 1
    assert bridge.step == 1
    assert bridge.loss_history[-1] == 0.42


def test_step_without_backward_skips_backward():
    bridge, opt, client = make_bridge(model=FakeModule())
    loss = FakeLoss(0.3)
    bridge.train_step(loss, backward=False, batch_size=4)
    assert loss.backwarded is False
    assert opt.steps == 1


def test_callable_shorthand_delegates_to_train_step():
    bridge, opt, client = make_bridge(model=FakeModule())
    bridge(FakeLoss(0.25), batch_size=2)
    assert opt.steps == 1
    assert bridge.loss_history[-1] == 0.25


def test_epoch_end_auto_increments_and_pushes():
    bridge, opt, client = make_bridge()
    bridge.epoch_end(val_acc=0.7)
    assert bridge.epoch == 1
    assert client.get_telemetry().state.metrics["val_acc"] == 0.7


def test_context_manager_runs_begin_and_end():
    bridge, opt, client = make_bridge()
    with bridge as b:
        assert b is bridge
    # on_train_end pushes a final telemetry frame
    assert client.get_telemetry() is not None


# --------------------------------------------------------------- attach / NoOp
def test_attach_returns_noop_when_unconfigured(monkeypatch):
    monkeypatch.delenv("CONTROL_PLANE_URL", raising=False)
    b = attach(FakeOpt(), model=FakeModule())
    assert isinstance(b, NoOpBridge)


def test_noop_bridge_control_surface_is_inert():
    nb = NoOpBridge()
    with nb as b:
        assert b is nb
        assert b.should_stop() is False
        with b.section("anything"):
            pass
        b.epoch_end(val_acc=0.1)
        b.on_batch_end(0.5)
        assert b.clip_gradients() is None
        b.register("x", lambda a, c: None)
        b.register_knob("k", lambda v: None)
        b.scheduler_step()
    assert nb.should_stop() is False


def test_noop_bridge_still_drives_training():
    """The control plane is inert, but the optimizer/scheduler must still be driven so the
    same script actually trains when no broker is configured."""
    opt, model, sched = FakeOpt(), FakeModule(), FakeScheduler()
    nb = NoOpBridge(opt, model, scheduler=sched, grad_clip=1.0)

    loss = FakeLoss(0.5)
    nb.train_step(loss, batch_size=4)
    assert loss.backwarded is True
    assert opt.steps == 1 and opt.zeroed == 1

    nb(FakeLoss(0.4), batch_size=4)  # __call__ delegates to train_step
    assert opt.steps == 2 and opt.zeroed == 2

    nb.scheduler_step()
    assert sched.stepped == 1


# ------------------------------------------------- code-review regressions
def test_in_memory_checkpoint_survives_optimizer_step():
    """Real-tensor regression for the aliasing bug: an in-memory snapshot must be a deep
    clone, so a subsequent optimizer.step() cannot silently drift it (which would turn
    restore_checkpoint into a no-op). FakeModule missed this because its state_dict is a
    primitive; only real torch tensors share storage with live params."""
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    client = ControlPlaneClient.from_app(create_app(ControlPlaneStore()))
    bridge = TrainingBridge(opt, client, model=model)

    original = model.weight.detach().clone()
    bridge.save_checkpoint(checkpoint_id="pre")  # default in-memory path

    # A real training step mutates parameter storage in place.
    loss = model(torch.randn(8, 4)).pow(2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert not torch.allclose(model.weight, original)  # sanity: the weights moved

    bridge.restore_checkpoint("pre")
    assert torch.allclose(model.weight, original)  # snapshot was a clone, so restore works


def test_non_finite_values_do_not_break_telemetry():
    """inf/nan in grad_norm, metrics, or per-sample losses must not blow up the JSON
    telemetry push (which would blind the agent during divergence, dropping the anomalies
    that ride the same payload)."""
    bridge, opt, client = make_bridge()
    bridge.on_batch_end(
        0.5,
        batch_size=4,
        grad_norm=float("inf"),
        sample_indices=[1, 2],
        sample_losses=[float("nan"), 3.0],
    )
    bridge.push_telemetry({"train_loss": float("inf"), "val_acc": 0.9})

    telem = client.get_telemetry()
    assert telem is not None  # the push actually succeeded (payload was JSON-compliant)
    assert telem.state.grad_norm is None  # non-finite grad norm coerced away
    assert "train_loss" not in telem.state.metrics  # non-finite metric dropped
    assert telem.state.metrics["val_acc"] == 0.9
    assert all(math.isfinite(p.loss) for p in telem.state.per_sample_losses)
    assert {p.index for p in telem.state.per_sample_losses} == {2}
    # The grad-explosion anomaly still reaches the agent on the same payload, and is
    # itself JSON-safe (its non-finite value is nulled rather than serialized).
    assert telem.anomalies
    assert telem.anomalies[0].value is None


def test_from_env_builds_bridge_when_configured(monkeypatch):
    monkeypatch.setenv("CONTROL_PLANE_URL", "http://127.0.0.1:65500")
    monkeypatch.setenv("CONTROL_PLANE_RUN_ID", "run-xyz")
    b = TrainingBridge.from_env(FakeOpt())
    assert isinstance(b, TrainingBridge)
    assert b.run_id == "run-xyz"
