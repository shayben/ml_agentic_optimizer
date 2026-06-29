"""Offline integration test: agent (MCP tools) <-> broker <-> TrainingBridge, in-process (no sockets)."""
from types import SimpleNamespace

from agentic_optimizer import mcp_server as tools
from agentic_optimizer.bridge import TrainingBridge
from agentic_optimizer.controlplane import (
    ControlPlaneClient,
    ControlPlaneStore,
    create_app,
)


def test_full_agent_bridge_flow():
    client = ControlPlaneClient.from_app(create_app(ControlPlaneStore()))
    opt = SimpleNamespace(param_groups=[{"lr": 0.1, "weight_decay": 0.0, "momentum": 0.9}])
    bridge = TrainingBridge(opt, client, max_epochs=2)
    bridge.register("per_class_loss", lambda args, ctx: {"0": 0.05, "1": 0.4})
    bridge.register_knob("label_smoothing", lambda v: None, value=0.0)

    # training emits telemetry the agent can read
    bridge.on_train_begin()
    bridge.on_batch_end(0.7, batch_size=16, grad_norm=1.2)
    bridge.on_epoch_end(0, metrics={"val_acc": 0.5})

    state = tools.get_training_state_impl(client)
    assert state["available"] and state["param_groups"][0]["lr"] == 0.1

    # agent injects a live LR change and an interrogation
    c1 = tools.set_hyperparameters_impl(client, lr=0.01)
    c2 = tools.interrogate_impl(client, "per_class_loss")
    bridge.drain_commands()

    r1 = tools.wait_for_result_impl(client, c1["command_id"], timeout=2.0)
    r2 = tools.wait_for_result_impl(client, c2["command_id"], timeout=2.0)
    assert r1["ok"] and opt.param_groups[0]["lr"] == 0.01
    assert r2["ok"] and r2["data"]["1"] == 0.4
    assert "label_smoothing" in {k["name"] for k in tools.list_knobs_impl(client)}


def test_set_training_config_applies_via_bridge():
    client = ControlPlaneClient.from_app(create_app(ControlPlaneStore()))
    opt = SimpleNamespace(param_groups=[{"lr": 0.1}])
    seen: dict = {}
    bridge = TrainingBridge(
        opt, client, on_training_config=lambda cfg: seen.update(cfg.model_dump(exclude_none=True))
    )

    cmd = tools.set_training_config_impl(client, batch_size=64, amp=True)
    bridge.drain_commands()
    r = tools.wait_for_result_impl(client, cmd["command_id"], timeout=2.0)

    assert r["ok"]
    assert bridge.training_config.batch_size == 64
    assert bridge.amp_enabled is True
    assert seen == {"batch_size": 64, "amp": True}


def test_suspicious_samples_surface_highest_losses():
    client = ControlPlaneClient.from_app(create_app(ControlPlaneStore()))
    opt = SimpleNamespace(param_groups=[{"lr": 0.1}])
    bridge = TrainingBridge(opt, client)

    bridge.on_train_begin()
    bridge.on_batch_end(
        0.5, batch_size=4, sample_indices=[0, 1, 2, 3], sample_losses=[0.1, 9.0, 0.2, 5.0]
    )
    bridge.on_epoch_end(0, metrics={"val_acc": 0.5})

    susp = tools.get_suspicious_samples_impl(client, limit=2)
    assert susp["available"]
    assert [s["index"] for s in susp["samples"]] == [1, 3]  # sorted by loss, descending


def test_run_id_isolation():
    client = ControlPlaneClient.from_app(create_app(ControlPlaneStore()))
    opt_a = SimpleNamespace(param_groups=[{"lr": 0.1}])
    opt_b = SimpleNamespace(param_groups=[{"lr": 0.2}])
    bridge_a = TrainingBridge(opt_a, client, run_id="a")
    bridge_b = TrainingBridge(opt_b, client, run_id="b")

    for b, acc in ((bridge_a, 0.7), (bridge_b, 0.6)):
        b.on_train_begin()
        b.on_batch_end(0.3, batch_size=4)
        b.on_epoch_end(0, metrics={"val_acc": acc})

    assert tools.get_training_state_impl(client, run_id="a")["param_groups"][0]["lr"] == 0.1
    assert tools.get_training_state_impl(client, run_id="b")["param_groups"][0]["lr"] == 0.2

    # a command addressed to run "a" must not be delivered to run "b"
    tools.set_hyperparameters_impl(client, lr=0.05, run_id="a")
    bridge_b.drain_commands()
    assert opt_b.param_groups[0]["lr"] == 0.2
    bridge_a.drain_commands()
    assert opt_a.param_groups[0]["lr"] == 0.05

    assert {"a", "b"} <= {r["run_id"] for r in tools.list_runs_impl(client)}


def test_poll_once_runs_safe_commands_and_defers_unsafe():
    client = ControlPlaneClient.from_app(create_app(ControlPlaneStore()))
    opt = SimpleNamespace(param_groups=[{"lr": 0.1}])
    bridge = TrainingBridge(opt, client)

    tools.flag_samples_impl(client, [7, 8])  # safe: runs in the poller thread immediately
    cmd = tools.set_hyperparameters_impl(client, lr=0.02)  # unsafe: deferred to a sync point

    handled = bridge._poll_once()
    assert handled == 2
    assert bridge.flagged_indices == {7, 8}
    assert opt.param_groups[0]["lr"] == 0.1  # deferred command not applied yet

    bridge.drain_commands()  # sync point drains deferred commands
    assert opt.param_groups[0]["lr"] == 0.02
    assert tools.wait_for_result_impl(client, cmd["command_id"], timeout=2.0)["ok"]
