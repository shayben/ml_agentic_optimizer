import threading
import time

import httpx
import pytest

from agentic_optimizer.contract import CommandResult, CommandStatus, KnobSpec, Telemetry, TrainingState
from agentic_optimizer.controlplane import (
    ControlPlaneClient,
    ControlPlaneStore,
    _check_exposure,
    create_app,
)


# ------------------------------------------------------------------ store unit
def test_store_command_lifecycle():
    store = ControlPlaneStore()
    cmd = store.enqueue_command("set_hyperparameters", {"lr": 0.01})
    assert cmd.status.value == "pending"
    claimed = store.claim_next_command()
    assert claimed is not None and claimed.id == cmd.id
    assert claimed.status.value == "in_progress"
    assert claimed.lease_expires_at is not None
    assert claimed.attempts == 1
    assert store.claim_next_command() is None  # queue drained
    done = store.complete_command(CommandResult(command_id=cmd.id, ok=True, data={"applied": True}))
    assert done is not None and done.status.value == "done"
    assert store.get_command(cmd.id).result.data == {"applied": True}


def test_store_telemetry_and_metrics():
    store = ControlPlaneStore()
    assert store.get_telemetry() is None
    store.push_telemetry(Telemetry(state=TrainingState(step=5, epoch=1, metrics={"val_acc": 0.4})))
    t = store.get_telemetry()
    assert t is not None and t.state.step == 5
    hist = store.get_metrics()
    assert hist[-1]["metrics"]["val_acc"] == 0.4


def test_store_knobs():
    store = ControlPlaneStore()
    store.register_knobs([KnobSpec(name="mixup", description="alpha")])
    assert [k.name for k in store.get_knobs()] == ["mixup"]


def test_run_id_isolation_for_telemetry_and_commands():
    store = ControlPlaneStore()
    store.push_telemetry(Telemetry(run_id="run-a", state=TrainingState(step=1)))
    store.push_telemetry(Telemetry(run_id="run-b", state=TrainingState(step=2)))
    assert store.get_telemetry("run-a").state.step == 1
    assert store.get_telemetry("run-b").state.step == 2
    assert store.get_telemetry().state.step == 0 if store.get_telemetry() else True

    cmd_a = store.enqueue_command("do-a", run_id="run-a")
    cmd_b = store.enqueue_command("do-b", run_id="run-b")
    assert store.claim_next_command(run_id="run-b").id == cmd_b.id
    assert store.claim_next_command(run_id="run-a").id == cmd_a.id
    assert {cmd.id for cmd in store.list_commands(run_id="run-a")} == {cmd_a.id}
    assert {cmd.id for cmd in store.list_commands(run_id="run-b")} == {cmd_b.id}


def test_lease_reclaim_and_max_attempts_failure():
    store = ControlPlaneStore()
    cmd = store.enqueue_command("retry-me")
    claimed = store.claim_next_command(lease_s=0.01)
    assert claimed.id == cmd.id
    time.sleep(0.02)
    changed = store.reclaim_expired(max_attempts=5)
    assert [item.id for item in changed] == [cmd.id]
    assert store.get_command(cmd.id).status == CommandStatus.pending
    assert store.claim_next_command().id == cmd.id

    failing = store.enqueue_command("fail-me")
    store.claim_next_command(lease_s=0.01)
    time.sleep(0.02)
    store.reclaim_expired(max_attempts=1)
    failed = store.get_command(failing.id)
    assert failed.status == CommandStatus.failed
    assert failed.result is not None
    assert failed.result.ok is False
    assert failed.result.error == "lease expired after 1 attempts"


def test_sqlite_persistence_round_trip(tmp_path):
    persist_path = tmp_path / "controlplane.sqlite"
    store = ControlPlaneStore(persist_path=str(persist_path))
    store.push_telemetry(
        Telemetry(
            run_id="persisted",
            state=TrainingState(step=7, metrics={"loss": 0.5}),
            knobs=[KnobSpec(name="temperature", value=0.2)],
        )
    )
    cmd = store.enqueue_command("persist-command", {"x": 1}, run_id="persisted")
    recovered = ControlPlaneStore(persist_path=str(persist_path))

    assert recovered.get_telemetry("persisted").state.step == 7
    assert recovered.get_command(cmd.id).type == "persist-command"
    assert recovered.claim_next_command(run_id="persisted").id == cmd.id
    assert [knob.name for knob in recovered.get_knobs("persisted")] == ["temperature"]


# ------------------------------------------------------------------ app + client
def _client(token=None, store=None, max_body_bytes=16 * 1024 * 1024):
    app = create_app(store or ControlPlaneStore(), token=token, max_body_bytes=max_body_bytes)
    return ControlPlaneClient.from_app(app, token=token)


def test_app_telemetry_roundtrip():
    c = _client()
    assert c.get_telemetry() is None
    c.push_telemetry(Telemetry(state=TrainingState(step=3, metrics={"loss": 1.0})))
    t = c.get_telemetry()
    assert t.state.step == 3
    assert c.get_metrics(limit=10)[-1]["metrics"]["loss"] == 1.0
    assert c.health() is True


def test_app_command_flow_and_wait():
    c = _client()
    cmd = c.enqueue_command("interrogate", {"name": "per_class_loss"})
    claimed = c.next_command()
    assert claimed.id == cmd.id
    c.complete_command(cmd.id, ok=True, data={"0": 0.1, "1": 0.3})
    result = c.wait_for_result(cmd.id, timeout=2.0)
    assert result is not None and result.ok and result.data["1"] == 0.3


def test_app_next_command_empty_returns_none():
    c = _client()
    assert c.next_command(wait=0.0) is None


def test_app_unknown_command_result_404():
    c = _client()
    r = c._client.post("/commands/nope/result", json={"command_id": "nope", "ok": True})
    assert r.status_code == 404


def test_app_auth_enforced():
    app = create_app(ControlPlaneStore(), token="secret")
    good = ControlPlaneClient.from_app(app, token="secret")
    bad = ControlPlaneClient.from_app(app, token="wrong")
    good.push_telemetry(Telemetry(state=TrainingState(step=1)))
    assert good.get_telemetry() is not None
    with pytest.raises(httpx.HTTPStatusError):
        bad.push_telemetry(Telemetry(state=TrainingState(step=1)))
    assert bad.health() is True


def test_constant_time_auth_correct_wrong_and_missing():
    app = create_app(ControlPlaneStore(), token="secret")
    good = ControlPlaneClient.from_app(app, token="secret")
    wrong = ControlPlaneClient.from_app(app, token="wrong")
    missing = ControlPlaneClient.from_app(app)

    good.push_telemetry(Telemetry(state=TrainingState(step=1)))
    assert good.get_telemetry() is not None
    assert wrong._client.get("/telemetry/latest").status_code == 401
    assert missing._client.get("/telemetry/latest").status_code == 401


# ------------------------------------------------------------------ exposure guard
def test_check_exposure_refuses_unauthenticated_tunnel():
    with pytest.raises(SystemExit):
        _check_exposure(token=None, tunnel=True, host="127.0.0.1", insecure=False)


def test_check_exposure_refuses_unauthenticated_non_loopback():
    with pytest.raises(SystemExit):
        _check_exposure(token=None, tunnel=False, host="0.0.0.0", insecure=False)


def test_check_exposure_allows_loopback_without_token():
    for host in ("127.0.0.1", "localhost", "::1"):
        _check_exposure(token=None, tunnel=False, host=host, insecure=False)


def test_check_exposure_token_permits_tunnel_and_non_loopback():
    _check_exposure(token="secret", tunnel=True, host="0.0.0.0", insecure=False)


def test_check_exposure_insecure_optout_permits_exposure():
    _check_exposure(token=None, tunnel=True, host="0.0.0.0", insecure=True)


def test_max_body_413():
    c = _client(token="secret", max_body_bytes=10)
    response = c._client.post(
        "/telemetry",
        content=b"x" * 11,
        headers={"Authorization": "Bearer secret", "content-type": "application/json"},
    )
    assert response.status_code == 413


def test_long_poll_returns_command_enqueued_mid_wait():
    store = ControlPlaneStore()
    c = _client(store=store)

    def enqueue_later():
        time.sleep(0.05)
        store.enqueue_command("late", run_id="run-late")

    thread = threading.Thread(target=enqueue_later)
    thread.start()
    try:
        cmd = c.next_command(wait=1.0, run_id="run-late")
    finally:
        thread.join(timeout=1.0)
    assert cmd is not None
    assert cmd.type == "late"
    assert cmd.run_id == "run-late"


def test_runs_endpoint():
    store = ControlPlaneStore()
    c = _client(token="secret", store=store)
    c.push_telemetry(Telemetry(run_id="run-a", state=TrainingState(step=1)))
    c.enqueue_command("pending", run_id="run-a")
    c.enqueue_command("other", run_id="run-b")

    runs = {item["run_id"]: item for item in c.list_runs()}
    assert runs["run-a"]["has_telemetry"] is True
    assert runs["run-a"]["pending"] == 1
    assert runs["run-b"]["has_telemetry"] is False
    assert runs["run-b"]["pending"] == 1
