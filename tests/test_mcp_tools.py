from types import SimpleNamespace

import asyncio
import pytest

from agentic_optimizer import mcp_server as ms
from agentic_optimizer.bridge import TrainingBridge
from agentic_optimizer.contract import PerSampleLoss, Telemetry, TrainingState
from agentic_optimizer.controlplane import (
    ControlPlaneClient,
    ControlPlaneStore,
    create_app,
)


def _client():
    return ControlPlaneClient.from_app(create_app(ControlPlaneStore()))


def test_get_training_state_impl_before_and_after():
    c = _client()
    assert ms.get_training_state_impl(c)["available"] is False
    c.push_telemetry(Telemetry(state=TrainingState(step=7, metrics={"val_acc": 0.6})))
    state = ms.get_training_state_impl(c)
    assert state["available"] is True and state["step"] == 7


def test_set_hyperparameters_impl_enqueues_filtered():
    c = _client()
    out = ms.set_hyperparameters_impl(c, lr=0.02, momentum=None, grad_clip=1.0)
    cmd = c.get_command(out["command_id"])
    assert cmd.type == "set_hyperparameters"
    assert cmd.args == {"lr": 0.02, "grad_clip": 1.0}  # None fields dropped


def test_interrogate_then_wait_for_result_via_bridge():
    c = _client()
    opt = SimpleNamespace(param_groups=[{"lr": 0.1}])
    bridge = TrainingBridge(opt, c)
    bridge.register("confusion_matrix", lambda args, ctx: {"tp": 10, "fp": 2})

    out = ms.interrogate_impl(c, "confusion_matrix")
    bridge.drain_commands()
    res = ms.wait_for_result_impl(c, out["command_id"], timeout=2.0)
    assert res["ready"] is True and res["ok"] is True and res["data"]["tp"] == 10


def test_wait_for_result_times_out_when_unhandled():
    c = _client()
    out = ms.set_knob_impl(c, "x", 1)
    res = ms.wait_for_result_impl(c, out["command_id"], timeout=0.3)
    assert res["ready"] is False


def test_invoke_and_flag_augment_impls():
    c = _client()
    assert ms.invoke_impl(c, "rebalance", {"ratio": 0.5})["type"] == "invoke"
    assert ms.flag_samples_impl(c, [1, 2])["type"] == "flag_samples"
    assert ms.set_augmentation_impl(c, True)["type"] == "set_augmentation"


def test_list_knobs_impl():
    c = _client()
    opt = SimpleNamespace(param_groups=[{"lr": 0.1}])
    bridge = TrainingBridge(opt, c)
    bridge.register_knob("mixup_alpha", lambda v: None, description="mixup", value=0.2)
    bridge.on_train_begin()
    knobs = ms.list_knobs_impl(c)
    assert {"mixup_alpha"} == {k["name"] for k in knobs}


def test_build_server_constructs():
    c = _client()
    server = ms.build_server(c)
    assert server is not None


def test_new_tools_callable_through_fastmcp_server():
    c = _client()
    server = ms.build_server(c)

    async def call_tools():
        config = await server.call_tool("set_training_config", {"batch_size": 64})
        samples = await server.call_tool("get_suspicious_samples", {"limit": 5})
        return config, samples

    config, samples = asyncio.run(call_tools())
    assert config is not None
    assert samples is not None


def test_impls_return_structured_error_when_broker_down():
    c = ControlPlaneClient.from_url("http://127.0.0.1:1", timeout=0.2)
    out = ms.get_training_state_impl(c)
    assert out["available"] is False
    assert "error" in out


def test_get_suspicious_samples_impl_sorts_and_caps():
    c = _client()
    c.push_telemetry(
        Telemetry(
            state=TrainingState(
                per_sample_losses=[
                    PerSampleLoss(index=1, loss=0.5),
                    PerSampleLoss(index=2, loss=1.5),
                    PerSampleLoss(index=3, loss=1.0),
                ]
            )
        )
    )

    out = ms.get_suspicious_samples_impl(c, limit=2)

    assert out == {
        "available": True,
        "count": 2,
        "samples": [{"index": 2, "loss": 1.5}, {"index": 3, "loss": 1.0}],
    }


def test_set_training_config_impl_enqueues_filtered_args():
    c = _client()
    out = ms.set_training_config_impl(c, batch_size=64, num_workers=None, amp=True)
    cmd = c.get_command(out["command_id"])
    assert cmd.type == "set_training_config"
    assert cmd.args == {"batch_size": 64, "amp": True}


def test_invoke_and_wait_impl_times_out_when_unhandled():
    c = _client()
    res = ms.invoke_and_wait_impl(c, "rebalance", {"ratio": 0.5}, timeout=0.01)
    assert res["ready"] is False


def test_hpo_impls_happy_path():
    pytest.importorskip("optuna")
    configured = ms.hpo_configure_impl(
        {"lr": {"type": "float", "low": 0.001, "high": 0.01}},
        direction="minimize",
    )
    assert configured["ok"] is True
    suggestion = ms.hpo_suggest_impl()
    assert "trial_id" in suggestion and "params" in suggestion
    reported = ms.hpo_report_impl(suggestion["trial_id"], 0.25)
    assert reported["state"] == "complete"
    best = ms.hpo_best_impl()
    assert best["value"] == 0.25
