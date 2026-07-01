"""A scripted stand-in for the local agent (what the GitHub Copilot CLI does via the MCP tools).

It connects to the broker and calls the very same tool implementations the MCP server exposes
(:mod:`agentic_optimizer.mcp_server`), so this exercises the real agent→broker→bridge path end-to-end —
just without a human/LLM in the loop. Use it to validate or demo live interjection.

    python examples/agent_sim.py --broker http://127.0.0.1:8765
"""
from __future__ import annotations

import argparse
import os
import time

from agentic_optimizer import mcp_server as tools
from agentic_optimizer.controlplane import ControlPlaneClient


def _wait_for_telemetry(client: ControlPlaneClient, timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if tools.get_training_state_impl(client).get("available"):
            return True
        time.sleep(0.2)
    return False


def run_agent(client: ControlPlaneClient, new_lr: float = 0.02, verbose: bool = True) -> dict:
    """Drive the live run end to end and return observations.

    Steps: read telemetry, change LR, interrogate per-class loss, set a knob, flag suspected label
    noise from per-sample losses, raise the batch size live, and (if optuna is installed) run a short
    HPO loop.
    """
    obs: dict = {"ok": True}
    if not _wait_for_telemetry(client):
        return {"ok": False, "error": "no telemetry appeared"}

    state = tools.get_training_state_impl(client)
    obs["initial_lr"] = state["param_groups"][0]["lr"]
    obs["initial_step"] = state["step"]
    if verbose:
        print(f"[agent] saw step={state['step']} lr={obs['initial_lr']} "
              f"loss_hist[-1]={state['loss_history'][-1] if state['loss_history'] else None}")

    # 1) live hyperparameter change
    cmd = tools.set_hyperparameters_impl(client, lr=new_lr, grad_clip=5.0)
    res = tools.wait_for_result_impl(client, cmd["command_id"], timeout=10.0)
    obs["set_lr_result"] = res
    if verbose:
        print(f"[agent] set_hyperparameters -> {res}")

    # 2) mid-run interrogation (returns data)
    cmd = tools.interrogate_impl(client, "per_class_loss")
    res = tools.wait_for_result_impl(client, cmd["command_id"], timeout=10.0)
    obs["per_class_loss"] = res
    if verbose:
        print(f"[agent] interrogate per_class_loss -> {res}")

    # 3) set a custom knob the job advertised
    obs["knobs"] = [k["name"] for k in tools.list_knobs_impl(client)]
    cmd = tools.set_knob_impl(client, "label_smoothing", 0.1)
    obs["set_knob_result"] = tools.wait_for_result_impl(client, cmd["command_id"], timeout=10.0)

    # 4) flag suspected-noisy samples + trigger an eval
    tools.flag_samples_impl(client, [3, 17, 42])
    cmd = tools.run_evaluation_impl(client)
    obs["evaluation"] = tools.wait_for_result_impl(client, cmd["command_id"], timeout=10.0)

    # 5) confirm the change landed in live telemetry
    time.sleep(0.5)
    after = tools.get_training_state_impl(client)
    obs["after_lr"] = after["param_groups"][0]["lr"]
    if verbose:
        print(f"[agent] confirmed live lr now = {obs['after_lr']}")

    # 6) label-noise triage: pull the worst per-sample losses and flag the top few as suspected noise
    susp = tools.get_suspicious_samples_impl(client, limit=5)
    obs["suspicious"] = susp
    if susp.get("available") and susp.get("samples"):
        worst = [s["index"] for s in susp["samples"][:3]]
        tools.flag_samples_impl(client, worst)
        obs["flagged"] = worst
        # read-only RCA interrogation answered by the bridge's background poller (safe_async)
        obs["noise_report"] = tools.interrogate_and_wait_impl(client, "noise_report", timeout=10.0)
        if verbose:
            print(f"[agent] flagged suspected-noisy {worst} -> {obs['noise_report']}")

    # 7) throughput / hardware-utilization lever: raise the batch size live
    cmd = tools.set_training_config_impl(client, batch_size=256)
    obs["set_training_config"] = tools.wait_for_result_impl(client, cmd["command_id"], timeout=10.0)
    if verbose:
        print(f"[agent] set_training_config(batch_size=256) -> {obs['set_training_config']}")

    # 8) one-call convenience: invoke a custom action and wait for its result
    obs["evaluation2"] = tools.invoke_and_wait_impl(client, "evaluate", timeout=10.0)

    # 9) safe-live-control surfaces: profiler RCA, scheduler awareness, checkpoint + rollback,
    #    guardrail clamp, and run-lifecycle extension.
    obs["profile"] = tools.get_profile_impl(client)
    obs["scheduler"] = tools.get_scheduler_impl(client)

    ckpt = tools.save_checkpoint_impl(client, note="pre-clamp", timeout=10.0)
    obs["checkpoint"] = ckpt
    if verbose:
        print(f"[agent] saved checkpoint -> {ckpt}")

    # request an out-of-range LR; guardrails should clamp it to the configured max
    clamp = tools.set_hyperparameters_impl(client, lr=10.0)
    obs["guardrail"] = tools.wait_for_result_impl(client, clamp["command_id"], timeout=10.0)
    if verbose:
        print(f"[agent] guardrail-clamped lr=10.0 -> {obs['guardrail']}")

    # roll back to the checkpoint we just saved
    cid = (ckpt.get("data") or {}).get("id") if ckpt.get("ready") else None
    if cid:
        obs["restore"] = tools.restore_checkpoint_impl(client, cid, timeout=10.0)
    obs["checkpoints"] = tools.list_checkpoints_impl(client)

    # extend the training budget (the agent decided the run is promising)
    ext = tools.extend_training_impl(client, max_epochs=999)
    obs["extend"] = tools.wait_for_result_impl(client, ext["command_id"], timeout=10.0)
    obs["anomalies"] = tools.get_anomalies_impl(client)

    # 10) optional principled HPO loop (only runs if the optuna extra is installed)
    obs["hpo"] = _run_hpo(client, verbose=verbose)
    return obs


def _latest_metric(metrics, key: str, default: float = 1.0) -> float:
    if isinstance(metrics, list) and metrics:
        return float(metrics[-1].get("metrics", {}).get(key, default))
    return default


def _run_hpo(client: ControlPlaneClient, trials: int = 3, verbose: bool = True) -> dict:
    """Drive a short Optuna ask -> apply -> read-metric -> tell loop over the live run.

    This demonstrates the HPO wiring; the per-trial objective (live ``train_loss``) is intentionally
    lightweight rather than a full per-config retrain.
    """
    cfg = tools.hpo_configure_impl(
        {"lr": {"type": "float", "low": 1e-3, "high": 0.3, "log": True}}, direction="minimize"
    )
    if not cfg.get("ok"):
        return {"available": False, "note": cfg.get("note")}
    results = []
    for _ in range(trials):
        suggestion = tools.hpo_suggest_impl()
        params = suggestion.get("params")
        if not params:
            break
        lr = params["lr"]
        applied = tools.set_hyperparameters_impl(client, lr=lr)
        tools.wait_for_result_impl(client, applied["command_id"], timeout=10.0)
        time.sleep(0.6)  # let the new lr drive at least one epoch before reading the objective
        value = _latest_metric(tools.get_metrics_impl(client, limit=1), "train_loss")
        tools.hpo_report_impl(suggestion["trial_id"], value)
        results.append({"trial_id": suggestion["trial_id"], "lr": round(lr, 5), "train_loss": value})
    best = tools.hpo_best_impl()
    if verbose:
        print(f"[agent] HPO ({len(results)} trials) best -> {best}")
    return {"available": True, "trials": results, "best": best}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--broker", default=os.environ.get("CONTROL_PLANE_URL", "http://127.0.0.1:8765"))
    ap.add_argument("--token", default=os.environ.get("CONTROL_PLANE_TOKEN"))
    ap.add_argument("--lr", type=float, default=0.02)
    args = ap.parse_args()
    client = ControlPlaneClient.from_url(args.broker, args.token)
    obs = run_agent(client, new_lr=args.lr)
    print(f"[agent] observations: {obs}")


if __name__ == "__main__":
    main()
