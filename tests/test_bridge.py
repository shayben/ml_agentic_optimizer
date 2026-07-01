from types import SimpleNamespace
import time

from agentic_optimizer.bridge import TrainingBridge
from agentic_optimizer.contract import MlflowInfo, TrainingConfig
from agentic_optimizer.controlplane import (
    ControlPlaneClient,
    ControlPlaneStore,
    create_app,
)


def make_bridge(**kwargs):
    opt = SimpleNamespace(param_groups=[{"lr": 0.1, "weight_decay": 0.0, "momentum": 0.9}])
    client = ControlPlaneClient.from_app(create_app(ControlPlaneStore()))
    return TrainingBridge(opt, client, **kwargs), opt, client


def test_set_hyperparameters_applied():
    bridge, opt, client = make_bridge()
    cmd = client.enqueue_command("set_hyperparameters", {"lr": 0.01, "grad_clip": 1.5})
    bridge.drain_commands()
    assert opt.param_groups[0]["lr"] == 0.01
    assert bridge.grad_clip == 1.5
    done = client.get_command(cmd.id)
    assert done.status.value == "done"
    assert done.result.data["applied"]["lr"] == 0.01


def test_custom_invoke_routes_to_registered_handler():
    bridge, opt, client = make_bridge()
    seen = {}

    def rebalance(args, ctx):
        seen["ratio"] = args.get("ratio")
        return {"rebalanced": True}

    bridge.register("rebalance", rebalance)
    cmd = client.enqueue_command("invoke", {"action": "rebalance", "args": {"ratio": 0.5}})
    bridge.drain_commands()
    assert seen["ratio"] == 0.5
    assert client.get_command(cmd.id).result.data == {"rebalanced": True}


def test_interrogate_returns_data():
    bridge, opt, client = make_bridge()
    bridge.register("per_class_loss", lambda args, ctx: {"0": 0.1, "1": 0.2})
    cmd = client.enqueue_command("interrogate", {"name": "per_class_loss"})
    bridge.drain_commands()
    assert client.get_command(cmd.id).result.data["1"] == 0.2


def test_pause_then_resume_drained_together():
    bridge, opt, client = make_bridge()
    client.enqueue_command("pause", {})
    client.enqueue_command("resume", {})
    processed = bridge.on_epoch_end(0, {"val_acc": 0.5})  # applies pause then resume -> no block
    assert bridge.paused is False
    assert len(processed) == 2


def test_flag_samples_accumulates():
    bridge, opt, client = make_bridge()
    client.enqueue_command("flag_samples", {"indices": [1, 2, 2, 3]})
    bridge.drain_commands()
    assert bridge.flagged_indices == {1, 2, 3}


def test_run_evaluation_requires_handler_then_succeeds():
    bridge, opt, client = make_bridge()
    c1 = client.enqueue_command("run_evaluation", {})
    bridge.drain_commands()
    failed = client.get_command(c1.id)
    assert failed.status.value == "failed" and "evaluate" in (failed.result.error or "")

    bridge.register("evaluate", lambda args, ctx: {"val_acc": 0.71})
    c2 = client.enqueue_command("run_evaluation", {})
    bridge.drain_commands()
    assert client.get_command(c2.id).result.data == {"val_acc": 0.71}


def test_unknown_command_type_fails_gracefully():
    bridge, opt, client = make_bridge()
    cmd = client.enqueue_command("does_not_exist", {})
    bridge.drain_commands()
    done = client.get_command(cmd.id)
    assert done.status.value == "failed" and done.result.ok is False


def test_register_knob_advertised_and_set():
    bridge, opt, client = make_bridge()
    received = {}
    bridge.register_knob(
        "label_smoothing", lambda v: received.update(v=v), description="LS factor", value=0.0
    )
    bridge.on_train_begin()  # advertises knobs to the broker
    assert "label_smoothing" in [k.name for k in client.get_knobs()]
    client.enqueue_command("set_knob", {"name": "label_smoothing", "value": 0.1})
    bridge.drain_commands()
    assert received["v"] == 0.1
    assert bridge.knob_values["label_smoothing"] == 0.1


def test_telemetry_pushed_with_state():
    bridge, opt, client = make_bridge()
    bridge.on_train_begin()
    bridge.on_batch_end(0.5, batch_size=8, grad_norm=2.0)
    bridge.push_telemetry({"val_acc": 0.3})
    t = client.get_telemetry()
    assert t.state.loss_history[-1] == 0.5
    assert t.state.grad_norm == 2.0
    assert t.state.metrics["val_acc"] == 0.3


def test_mlflow_info_provider_injected():
    def provider(bridge, metrics):
        return MlflowInfo(run_id="abc123", tracking_uri="file:./mlruns", metrics=metrics)

    bridge, opt, client = make_bridge(mlflow_info_provider=provider)
    bridge.push_telemetry({"loss": 0.9})
    t = client.get_telemetry()
    assert t.mlflow is not None and t.mlflow.run_id == "abc123"
    assert t.mlflow.metrics["loss"] == 0.9


def test_per_sample_loss_surfacing_topk_sorted_and_reset():
    bridge, opt, client = make_bridge(susp_topk=3)
    bridge.on_batch_end(0.5, sample_indices=[1, 2, 3], sample_losses=[0.2, 0.9, 0.1])
    bridge.on_batch_end(0.4, sample_indices=[4, 5], sample_losses=[1.4, 0.8])
    bridge.push_telemetry({})

    losses = client.get_telemetry().state.per_sample_losses
    assert [(item.index, item.loss) for item in losses] == [(4, 1.4), (2, 0.9), (5, 0.8)]

    bridge.push_telemetry({})
    assert client.get_telemetry().state.per_sample_losses == []


def test_set_training_config_applies_and_calls_callback():
    seen = []
    bridge, opt, client = make_bridge(on_training_config=seen.append)
    client.enqueue_command("set_training_config", {"batch_size": 64, "amp": True})

    bridge.drain_commands()

    assert seen == [TrainingConfig(batch_size=64, amp=True)]
    assert bridge.training_config == TrainingConfig(batch_size=64, amp=True)
    assert bridge.amp_enabled is True


def test_flag_samples_hook_receives_new_indices():
    seen = []
    bridge, opt, client = make_bridge(on_flagged_samples=seen.append)
    client.enqueue_command("flag_samples", {"indices": [3, 1, 3, 2]})

    bridge.drain_commands()

    assert seen == [[1, 2, 3]]
    assert bridge.flagged_indices == {1, 2, 3}


def test_flag_samples_callback_deferred_from_poller_to_drain():
    """flag_samples registers the indices immediately on the poller thread, but the
    user callback (which typically mutates shared training tensors) must be deferred to the
    training thread's next drain so it cannot race the loop."""
    seen = []
    bridge, opt, client = make_bridge(on_flagged_samples=seen.append)
    client.enqueue_command("flag_samples", {"indices": [5, 6]})

    assert bridge._poll_once() == 1  # poller thread applies the safe command
    assert bridge.flagged_indices == {5, 6}  # metadata registered immediately
    assert seen == []  # ...but the tensor-mutating callback has NOT run yet

    bridge.drain_commands()  # training-thread sync point
    assert seen == [[5, 6]]  # callback runs here, on the training thread


def test_poll_once_executes_safe_and_defers_mutation_until_drain():
    bridge, opt, client = make_bridge()
    bridge.register("read_lr", lambda args, ctx: {"lr": ctx.optimizer.param_groups[0]["lr"]}, safe_async=True)
    safe_cmd = client.enqueue_command("interrogate", {"name": "read_lr"})

    assert bridge._poll_once() == 1
    assert client.get_command(safe_cmd.id).status.value == "done"
    assert client.get_command(safe_cmd.id).result.data == {"lr": 0.1}

    mutation = client.enqueue_command("set_hyperparameters", {"lr": 0.01})
    assert bridge._poll_once() == 1
    assert opt.param_groups[0]["lr"] == 0.1
    assert client.get_command(mutation.id).status.value == "in_progress"

    bridge.drain_commands()
    assert opt.param_groups[0]["lr"] == 0.01
    assert client.get_command(mutation.id).status.value == "done"


def test_push_telemetry_failure_sets_last_error_without_crashing():
    bridge, opt, client = make_bridge()
    client.close()

    bridge.push_telemetry({})

    assert bridge.last_error is not None


def test_max_pause_auto_resumes_within_bound():
    bridge, opt, client = make_bridge(max_pause_s=0.01, pause_poll_s=0.001)
    bridge.paused = True

    started = time.monotonic()
    bridge.on_epoch_end(0, {})

    assert time.monotonic() - started < 1.0
    assert bridge.paused is False
