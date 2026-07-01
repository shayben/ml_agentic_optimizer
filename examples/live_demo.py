"""One-command end-to-end demo: broker + remote-style training + scripted agent, over real HTTP.

Starts the control-plane broker (uvicorn) in-process, launches the :mod:`train_with_bridge` loop in a
background thread, and runs the :mod:`agent_sim` agent against it — proving live telemetry + interjection
(an LR change and a mid-run interrogation) without needing the real Copilot CLI.

    python examples/live_demo.py
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import uvicorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agentic_optimizer.controlplane import ControlPlaneClient, ControlPlaneStore, create_app  # noqa: E402

import agent_sim  # noqa: E402
import train_with_bridge  # noqa: E402


class _Server:
    def __init__(self, app, host: str, port: int) -> None:
        self._server = uvicorn.Server(
            uvicorn.Config(app, host=host, port=port, log_level="warning")
        )
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def wait_ready(self, client: ControlPlaneClient, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if client.health():
                return True
            time.sleep(0.1)
        return False

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=5.0)


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=0, help="0 = auto-pick a free port")
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--token", default=os.environ.get("CONTROL_PLANE_TOKEN"))
    args = ap.parse_args()
    port = args.port or _free_port()
    base_url = f"http://127.0.0.1:{port}"

    server = _Server(create_app(ControlPlaneStore(), token=args.token), "127.0.0.1", port)
    server.start()
    ctrl = ControlPlaneClient.from_url(base_url, args.token)
    if not server.wait_ready(ctrl):
        print("broker failed to start", file=sys.stderr)
        return 1
    print(f"broker up at {base_url}")

    # remote-style training job in the background (poll_interval>0 exercises the low-latency poller)
    train_client = ControlPlaneClient.from_url(base_url, args.token)
    trained: dict = {}

    def _train() -> None:
        trained["bridge"] = train_with_bridge.run_training(
            client=train_client, epochs=args.epochs, epoch_pause_s=0.2, poll_interval=0.05
        )

    train_thread = threading.Thread(target=_train, daemon=True)
    train_thread.start()

    # local agent drives the live run
    control = ControlPlaneClient.from_url(base_url, args.token)
    obs = agent_sim.run_agent(control, new_lr=0.02)
    train_thread.join(timeout=60.0)
    control.close()
    bridge = trained.get("bridge")

    guardrail = obs.get("guardrail", {}) or {}
    guardrail_data = guardrail.get("data") or {}
    extend = obs.get("extend", {}) or {}
    extend_data = extend.get("data") or {}

    print("\n=== RESULT ===")
    checks = {
        "agent_ok": bool(obs.get("ok")),
        "lr_applied": bool(obs.get("set_lr_result", {}).get("ok"))
        and abs(obs.get("after_lr", -1) - 0.02) < 1e-9,
        "interrogation": bool(obs.get("per_class_loss", {}).get("ready")),
        "evaluation": bool(obs.get("evaluation", {}).get("ready")),
        "suspicious_seen": bool(obs.get("suspicious", {}).get("available")),
        "config_applied": bool(obs.get("set_training_config", {}).get("ready")),
        "flagged_applied": bridge is not None and len(bridge.flagged_indices) > 0,
        "batch_size_applied": bridge is not None and bridge.training_config.batch_size == 256,
        "profile_seen": bool(obs.get("profile", {}).get("available")),
        "scheduler_seen": bool(obs.get("scheduler", {}).get("available")),
        "checkpoint_saved": bool(obs.get("checkpoint", {}).get("ok")),
        "guardrail_clamped": bool(guardrail.get("ok"))
        and "guardrails" in guardrail_data
        and guardrail_data.get("applied", {}).get("lr", 99.0) <= 0.5 + 1e-9,
        "checkpoint_restored": bool(obs.get("restore", {}).get("ok")),
        "extended": bool(extend.get("ok")) and extend_data.get("max_epochs") == 999,
    }
    ok = all(checks.values())
    print(f"agent observations: {obs}")
    print(f"checks: {checks}")
    print(f"\nlive interjection {'SUCCEEDED' if ok else 'FAILED'}: "
          f"lr {obs.get('initial_lr')} -> {obs.get('after_lr')}, "
          f"flagged={sorted(bridge.flagged_indices)[:5] if bridge else None}, "
          f"batch_size={bridge.training_config.batch_size if bridge else None}")
    server.stop()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
