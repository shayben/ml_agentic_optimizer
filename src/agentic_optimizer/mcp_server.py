"""MCP server that exposes the live training run to a local GitHub Copilot CLI session.

It runs as the CLI's **stdio** subprocess (see ``agent/mcp-config.json``) and is a thin client of the
control-plane broker. Each tool maps to a broker call: read telemetry/metrics, or enqueue a command the
remote :class:`~agentic_optimizer.bridge.TrainingBridge` will apply at the next safe sync point.

Run directly with ``python -m agentic_optimizer.mcp_server`` (configured via ``CONTROL_PLANE_URL`` /
``CONTROL_PLANE_TOKEN``). The tool *implementations* are standalone functions taking a
:class:`ControlPlaneClient`, so they can be unit-tested in-process without stdio.
"""
from __future__ import annotations

import os
from functools import wraps
from typing import Any, Callable, TypeVar

from .controlplane import ControlPlaneClient

DEFAULT_URL = "http://127.0.0.1:8765"
_CURRENT_RUN_ID = os.environ.get("CONTROL_PLANE_RUN_ID", "default")
_HPO_ADVISOR: Any | None = None

F = TypeVar("F", bound=Callable[..., Any])


def _active_run_id(run_id: str | None = None) -> str:
    return run_id or _CURRENT_RUN_ID


def _error_response(exc: Exception) -> dict[str, Any]:
    return {"error": f"{type(exc).__name__}: {exc}", "available": False}


def _safe_impl(func: F) -> F:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            return _error_response(exc)

    return wrapper  # type: ignore[return-value]


# --------------------------------------------------------------------------- impls
@_safe_impl
def get_training_state_impl(
    client: ControlPlaneClient, run_id: str | None = None
) -> dict[str, Any]:
    t = client.get_telemetry(_active_run_id(run_id))
    if t is None:
        return {"available": False, "note": "no telemetry pushed yet"}
    return {"available": True, **t.state.model_dump()}


@_safe_impl
def get_metrics_impl(
    client: ControlPlaneClient, limit: int = 50, run_id: str | None = None
) -> list[dict[str, Any]] | dict[str, Any]:
    return client.get_metrics(limit, _active_run_id(run_id))


@_safe_impl
def get_mlflow_info_impl(
    client: ControlPlaneClient, run_id: str | None = None
) -> dict[str, Any]:
    t = client.get_telemetry(_active_run_id(run_id))
    if t is None or t.mlflow is None:
        return {"available": False}
    return {"available": True, **t.mlflow.model_dump()}


@_safe_impl
def list_runs_impl(client: ControlPlaneClient) -> list[dict[str, Any]] | dict[str, Any]:
    return client.list_runs()


@_safe_impl
def select_run_impl(run_id: str) -> dict[str, str]:
    global _CURRENT_RUN_ID
    _CURRENT_RUN_ID = run_id
    return {"run_id": _CURRENT_RUN_ID}


def _enqueue(
    client: ControlPlaneClient, type: str, args: dict[str, Any], run_id: str | None = None
) -> dict[str, Any]:
    cmd = client.enqueue_command(type, args, _active_run_id(run_id))
    return {"command_id": cmd.id, "type": cmd.type, "status": cmd.status.value}


@_safe_impl
def set_hyperparameters_impl(
    client: ControlPlaneClient,
    lr: float | None = None,
    weight_decay: float | None = None,
    momentum: float | None = None,
    grad_clip: float | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    args = {
        k: v
        for k, v in {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "grad_clip": grad_clip,
        }.items()
        if v is not None
    }
    return _enqueue(client, "set_hyperparameters", args, run_id)


@_safe_impl
def set_training_config_impl(
    client: ControlPlaneClient,
    batch_size: int | None = None,
    num_workers: int | None = None,
    grad_accum_steps: int | None = None,
    amp: bool | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    args = {
        k: v
        for k, v in {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "grad_accum_steps": grad_accum_steps,
            "amp": amp,
        }.items()
        if v is not None
    }
    return _enqueue(client, "set_training_config", args, run_id)


@_safe_impl
def set_knob_impl(
    client: ControlPlaneClient, name: str, value: Any, run_id: str | None = None
) -> dict[str, Any]:
    return _enqueue(client, "set_knob", {"name": name, "value": value}, run_id)


@_safe_impl
def invoke_impl(
    client: ControlPlaneClient,
    action: str,
    args: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    return _enqueue(client, "invoke", {"action": action, "args": args or {}}, run_id)


@_safe_impl
def interrogate_impl(
    client: ControlPlaneClient,
    name: str,
    args: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    return _enqueue(client, "interrogate", {"name": name, "args": args or {}}, run_id)


@_safe_impl
def set_augmentation_impl(
    client: ControlPlaneClient, enabled: bool = True, run_id: str | None = None
) -> dict[str, Any]:
    return _enqueue(client, "set_augmentation", {"enabled": enabled}, run_id)


@_safe_impl
def flag_samples_impl(
    client: ControlPlaneClient, indices: list[int], run_id: str | None = None
) -> dict[str, Any]:
    return _enqueue(client, "flag_samples", {"indices": [int(i) for i in indices]}, run_id)


@_safe_impl
def run_evaluation_impl(
    client: ControlPlaneClient, args: dict[str, Any] | None = None, run_id: str | None = None
) -> dict[str, Any]:
    return _enqueue(client, "run_evaluation", args or {}, run_id)


@_safe_impl
def wait_for_result_impl(
    client: ControlPlaneClient,
    command_id: str,
    timeout: float = 60.0,
    run_id: str | None = None,
) -> dict[str, Any]:
    del run_id
    result = client.wait_for_result(command_id, timeout=timeout)
    if result is None:
        return {"ready": False, "note": "timed out waiting for the training job to apply the command"}
    return {"ready": True, "ok": result.ok, "data": result.data, "error": result.error}


@_safe_impl
def list_knobs_impl(
    client: ControlPlaneClient, run_id: str | None = None
) -> list[dict[str, Any]] | dict[str, Any]:
    return [k.model_dump() for k in client.get_knobs(_active_run_id(run_id))]


@_safe_impl
def get_suspicious_samples_impl(
    client: ControlPlaneClient, limit: int = 20, run_id: str | None = None
) -> dict[str, Any]:
    t = client.get_telemetry(_active_run_id(run_id))
    if t is None or not t.state.per_sample_losses:
        return {"available": False}
    samples = sorted(t.state.per_sample_losses, key=lambda item: item.loss, reverse=True)[:limit]
    return {
        "available": True,
        "count": len(samples),
        "samples": [{"index": sample.index, "loss": sample.loss} for sample in samples],
    }


# ---- checkpoint / rollback
@_safe_impl
def save_checkpoint_impl(
    client: ControlPlaneClient,
    note: str | None = None,
    metrics: dict[str, float] | None = None,
    checkpoint_id: str | None = None,
    timeout: float = 60.0,
    run_id: str | None = None,
) -> dict[str, Any]:
    args = {
        k: v
        for k, v in {"note": note, "metrics": metrics, "id": checkpoint_id}.items()
        if v is not None
    }
    queued = _enqueue(client, "save_checkpoint", args, run_id)
    if "error" in queued:
        return queued
    return wait_for_result_impl(client, queued["command_id"], timeout, run_id)


@_safe_impl
def restore_checkpoint_impl(
    client: ControlPlaneClient,
    checkpoint_id: str | None = None,
    timeout: float = 60.0,
    run_id: str | None = None,
) -> dict[str, Any]:
    args = {"id": checkpoint_id} if checkpoint_id else {}
    queued = _enqueue(client, "restore_checkpoint", args, run_id)
    if "error" in queued:
        return queued
    return wait_for_result_impl(client, queued["command_id"], timeout, run_id)


@_safe_impl
def list_checkpoints_impl(
    client: ControlPlaneClient, run_id: str | None = None
) -> dict[str, Any]:
    t = client.get_telemetry(_active_run_id(run_id))
    if t is None:
        return {"available": False}
    return {"available": True, "checkpoints": [c.model_dump() for c in t.checkpoints]}


# ---- guardrails
@_safe_impl
def set_guardrails_impl(
    client: ControlPlaneClient,
    bounds: dict[str, dict[str, float]] | None = None,
    max_rel_change: float | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    args: dict[str, Any] = {}
    if bounds is not None:
        args["bounds"] = bounds
    if max_rel_change is not None:
        args["max_rel_change"] = max_rel_change
    return _enqueue(client, "set_guardrails", args, run_id)


# ---- profiler / scheduler
@_safe_impl
def get_profile_impl(client: ControlPlaneClient, run_id: str | None = None) -> dict[str, Any]:
    t = client.get_telemetry(_active_run_id(run_id))
    if t is None or t.state.profile is None:
        return {"available": False}
    return {"available": True, **t.state.profile.model_dump()}


@_safe_impl
def get_scheduler_impl(client: ControlPlaneClient, run_id: str | None = None) -> dict[str, Any]:
    t = client.get_telemetry(_active_run_id(run_id))
    if t is None or t.state.scheduler is None:
        return {"available": False}
    return {"available": True, **t.state.scheduler.model_dump()}


@_safe_impl
def set_scheduler_impl(
    client: ControlPlaneClient,
    config: dict[str, Any] | None = None,
    timeout: float = 60.0,
    run_id: str | None = None,
) -> dict[str, Any]:
    queued = _enqueue(client, "set_scheduler", config or {}, run_id)
    if "error" in queued:
        return queued
    return wait_for_result_impl(client, queued["command_id"], timeout, run_id)


# ---- run lifecycle
@_safe_impl
def stop_training_impl(client: ControlPlaneClient, run_id: str | None = None) -> dict[str, Any]:
    return _enqueue(client, "stop_training", {}, run_id)


@_safe_impl
def extend_training_impl(
    client: ControlPlaneClient, max_epochs: int, run_id: str | None = None
) -> dict[str, Any]:
    return _enqueue(client, "extend_training", {"max_epochs": int(max_epochs)}, run_id)


# ---- anomalies
@_safe_impl
def get_anomalies_impl(
    client: ControlPlaneClient, limit: int = 20, run_id: str | None = None
) -> dict[str, Any]:
    t = client.get_telemetry(_active_run_id(run_id))
    if t is None or not t.anomalies:
        return {"available": False}
    items = t.anomalies[-limit:]
    return {
        "available": True,
        "count": len(items),
        "anomalies": [a.model_dump() for a in items],
    }


@_safe_impl
def invoke_and_wait_impl(
    client: ControlPlaneClient,
    action: str,
    args: dict[str, Any] | None = None,
    timeout: float = 60.0,
    run_id: str | None = None,
) -> dict[str, Any]:
    queued = invoke_impl(client, action, args, run_id)
    if "error" in queued:
        return queued
    return wait_for_result_impl(client, queued["command_id"], timeout, run_id)


@_safe_impl
def interrogate_and_wait_impl(
    client: ControlPlaneClient,
    name: str,
    args: dict[str, Any] | None = None,
    timeout: float = 60.0,
    run_id: str | None = None,
) -> dict[str, Any]:
    queued = interrogate_impl(client, name, args, run_id)
    if "error" in queued:
        return queued
    return wait_for_result_impl(client, queued["command_id"], timeout, run_id)


def _hpo_unavailable() -> dict[str, Any]:
    return {"available": False, "note": "pip install agentic-optimizer[hpo]"}


@_safe_impl
def hpo_configure_impl(
    param_space: dict[str, dict],
    direction: str = "minimize",
    storage: str | None = None,
) -> dict[str, Any]:
    global _HPO_ADVISOR
    try:
        from .optuna_advisor import OptunaAdvisor, optuna_available
    except ImportError:
        return _hpo_unavailable()
    if not optuna_available():
        return _hpo_unavailable()
    _HPO_ADVISOR = OptunaAdvisor(param_space, direction=direction, storage=storage)
    return {"ok": True, "direction": direction, "params": list(param_space)}


@_safe_impl
def hpo_suggest_impl() -> dict[str, Any]:
    if _HPO_ADVISOR is None:
        return {"available": False}
    return _HPO_ADVISOR.suggest()


@_safe_impl
def hpo_report_impl(trial_id: int, value: float, step: int | None = None) -> dict[str, Any]:
    if _HPO_ADVISOR is None:
        return {"available": False}
    return _HPO_ADVISOR.report(trial_id, value, step)


@_safe_impl
def hpo_report_intermediate_impl(trial_id: int, value: float, step: int) -> dict[str, Any]:
    if _HPO_ADVISOR is None:
        return {"available": False}
    return _HPO_ADVISOR.report_intermediate(trial_id, value, step)


@_safe_impl
def hpo_best_impl() -> dict[str, Any] | None:
    if _HPO_ADVISOR is None:
        return {"available": False}
    return _HPO_ADVISOR.best() or {"available": False}


# --------------------------------------------------------------------------- server
def build_server(client: ControlPlaneClient):
    """Build a FastMCP server whose tools are bound to ``client``."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("agentic-optimizer-control")

    @mcp.tool()
    def get_training_state() -> dict[str, Any]:
        """Latest training telemetry: step/epoch, loss history, optimizer state, GPU utilization."""
        return get_training_state_impl(client)

    @mcp.tool()
    def get_metrics(limit: int = 50) -> list[dict[str, Any]] | dict[str, Any]:
        """Recent metric history (most recent last): step, epoch, metrics, grad_norm, throughput."""
        return get_metrics_impl(client, limit)

    @mcp.tool()
    def get_mlflow_info() -> dict[str, Any]:
        """The training run's MLflow linkage (run_id, tracking_uri, experiment, latest metrics) if any."""
        return get_mlflow_info_impl(client)

    @mcp.tool()
    def list_runs() -> list[dict[str, Any]] | dict[str, Any]:
        """List broker-known training runs and their telemetry/pending-command status."""
        return list_runs_impl(client)

    @mcp.tool()
    def select_run(run_id: str) -> dict[str, str]:
        """Select the active run_id used by subsequent MCP tools in this server process."""
        return select_run_impl(run_id)

    @mcp.tool()
    def set_hyperparameters(
        lr: float | None = None,
        weight_decay: float | None = None,
        momentum: float | None = None,
        grad_clip: float | None = None,
    ) -> dict[str, Any]:
        """Change optimizer hyperparameters live; use wait_for_result to confirm application."""
        return set_hyperparameters_impl(client, lr, weight_decay, momentum, grad_clip)

    @mcp.tool()
    def set_training_config(
        batch_size: int | None = None,
        num_workers: int | None = None,
        grad_accum_steps: int | None = None,
        amp: bool | None = None,
    ) -> dict[str, Any]:
        """Change live training-loop config; only non-null fields are included in the command."""
        return set_training_config_impl(client, batch_size, num_workers, grad_accum_steps, amp)

    @mcp.tool()
    def set_knob(name: str, value: Any) -> dict[str, Any]:
        """Set a named, training-job-registered knob (e.g. label_smoothing, mixup_alpha) to a value."""
        return set_knob_impl(client, name, value)

    @mcp.tool()
    def invoke(action: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        """Invoke a custom action the training job registered (preprocessing, filtering, balancing, ...)."""
        return invoke_impl(client, action, args)

    @mcp.tool()
    def invoke_and_wait(
        action: str, args: dict[str, Any] | None = None, timeout: float = 60.0
    ) -> dict[str, Any]:
        """Invoke a custom action and wait for its command result in one call."""
        return invoke_and_wait_impl(client, action, args, timeout)

    @mcp.tool()
    def interrogate(name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        """Queue a model/data interrogation; fetch the answer with wait_for_result."""
        return interrogate_impl(client, name, args)

    @mcp.tool()
    def interrogate_and_wait(
        name: str, args: dict[str, Any] | None = None, timeout: float = 60.0
    ) -> dict[str, Any]:
        """Queue a model/data interrogation and wait for its result in one call."""
        return interrogate_and_wait_impl(client, name, args, timeout)

    @mcp.tool()
    def get_suspicious_samples(limit: int = 20) -> dict[str, Any]:
        """Return top per-sample losses sorted descending for label-noise triage."""
        return get_suspicious_samples_impl(client, limit)

    @mcp.tool()
    def set_augmentation(enabled: bool = True) -> dict[str, Any]:
        """Enable or disable data augmentation in the live run."""
        return set_augmentation_impl(client, enabled)

    @mcp.tool()
    def flag_samples(indices: list[int]) -> dict[str, Any]:
        """Flag dataset sample indices as suspected label noise (the job decides how to use them)."""
        return flag_samples_impl(client, indices)

    @mcp.tool()
    def run_evaluation(args: dict[str, Any] | None = None) -> dict[str, Any]:
        """Trigger an evaluation pass in the live run; fetch results with wait_for_result."""
        return run_evaluation_impl(client, args)

    @mcp.tool()
    def wait_for_result(command_id: str, timeout: float = 60.0) -> dict[str, Any]:
        """Block until the training job applies a command (or times out) and return its result/data."""
        return wait_for_result_impl(client, command_id, timeout)

    @mcp.tool()
    def list_knobs() -> list[dict[str, Any]] | dict[str, Any]:
        """List the custom knobs the training job has advertised as agent-controllable."""
        return list_knobs_impl(client)

    @mcp.tool()
    def hpo_configure(
        param_space: dict[str, dict],
        direction: str = "minimize",
        storage: str | None = None,
    ) -> dict[str, Any]:
        """Configure Optuna HPO. Loop: hpo_suggest -> set_hyperparameters -> get_metrics ->
        hpo_report; repeat, then hpo_best to finalize."""
        return hpo_configure_impl(param_space, direction, storage)

    @mcp.tool()
    def hpo_suggest() -> dict[str, Any]:
        """Ask Optuna for a trial and parameters to apply with set_hyperparameters."""
        return hpo_suggest_impl()

    @mcp.tool()
    def hpo_report(trial_id: int, value: float, step: int | None = None) -> dict[str, Any]:
        """Report a completed trial objective after reading metrics from get_metrics."""
        return hpo_report_impl(trial_id, value, step)

    @mcp.tool()
    def hpo_report_intermediate(trial_id: int, value: float, step: int) -> dict[str, Any]:
        """Report an intermediate metric and receive an Optuna pruning recommendation."""
        return hpo_report_intermediate_impl(trial_id, value, step)

    @mcp.tool()
    def hpo_best() -> dict[str, Any] | None:
        """Return the best completed Optuna trial, used after the HPO loop finalizes."""
        return hpo_best_impl()

    @mcp.tool()
    def save_checkpoint(
        note: str | None = None,
        metrics: dict[str, float] | None = None,
        checkpoint_id: str | None = None,
        timeout: float = 60.0,
    ) -> dict[str, Any]:
        """Snapshot the live run (weights/optimizer/scheduler/RNG) so you can roll back later.

        Waits for the job to apply it and returns the checkpoint id."""
        return save_checkpoint_impl(client, note, metrics, checkpoint_id, timeout)

    @mcp.tool()
    def restore_checkpoint(
        checkpoint_id: str | None = None, timeout: float = 60.0
    ) -> dict[str, Any]:
        """Roll the run back to a saved checkpoint (latest if checkpoint_id omitted); waits for apply."""
        return restore_checkpoint_impl(client, checkpoint_id, timeout)

    @mcp.tool()
    def list_checkpoints() -> dict[str, Any]:
        """List checkpoints saved this run (id, step, epoch, metrics, note) for rollback selection."""
        return list_checkpoints_impl(client)

    @mcp.tool()
    def set_guardrails(
        bounds: dict[str, dict[str, float]] | None = None,
        max_rel_change: float | None = None,
    ) -> dict[str, Any]:
        """Set safety bounds for live hyperparameter changes.

        ``bounds`` maps a name (lr/weight_decay/momentum/grad_clip) to ``{min, max}``;
        ``max_rel_change`` caps the per-change fractional delta. Out-of-range values are clamped."""
        return set_guardrails_impl(client, bounds, max_rel_change)

    @mcp.tool()
    def get_profile() -> dict[str, Any]:
        """Step-time breakdown (dataloader vs fwd/bwd vs H2D) with throughput-tuning suggestions."""
        return get_profile_impl(client)

    @mcp.tool()
    def get_scheduler() -> dict[str, Any]:
        """The live LR scheduler's state (name, last_lr, last_epoch, config) if one is attached."""
        return get_scheduler_impl(client)

    @mcp.tool()
    def set_scheduler(
        config: dict[str, Any] | None = None, timeout: float = 60.0
    ) -> dict[str, Any]:
        """Reconfigure/replace the LR scheduler (job-defined hook); waits for apply and returns state."""
        return set_scheduler_impl(client, config, timeout)

    @mcp.tool()
    def stop_training() -> dict[str, Any]:
        """Request a graceful stop; the loop exits at the next epoch boundary and frees the GPU."""
        return stop_training_impl(client)

    @mcp.tool()
    def extend_training(max_epochs: int) -> dict[str, Any]:
        """Raise the run's max_epochs so a promising run can keep training past its original budget."""
        return extend_training_impl(client, max_epochs)

    @mcp.tool()
    def get_anomalies(limit: int = 20) -> dict[str, Any]:
        """Recent training anomalies (NaN/Inf loss, grad explosion, loss divergence) the job detected."""
        return get_anomalies_impl(client, limit)

    return mcp


def client_from_env() -> ControlPlaneClient:
    url = os.environ.get("CONTROL_PLANE_URL", DEFAULT_URL)
    token = os.environ.get("CONTROL_PLANE_TOKEN")
    tunnel_access_token = os.environ.get("CONTROL_PLANE_TUNNEL_ACCESS_TOKEN")
    return ControlPlaneClient.from_url(url, token, tunnel_access_token=tunnel_access_token)


def main() -> None:
    build_server(client_from_env()).run()


if __name__ == "__main__":
    main()
