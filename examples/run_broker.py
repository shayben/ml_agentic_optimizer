"""Start the control-plane broker (the only network-reachable component).

    python examples/run_broker.py --host 0.0.0.0 --port 8765
    CONTROL_PLANE_TOKEN=secret python examples/run_broker.py   # enable bearer auth
    python examples/run_broker.py --persist control-plane.db    # survive restarts (SQLite)

Binding a non-loopback ``--host`` without a token is refused unless ``--insecure`` is passed.

The local MCP server (``CONTROL_PLANE_URL``) and the remote ``TrainingBridge`` both connect here.
"""
from __future__ import annotations

import argparse
import os

import uvicorn

from agentic_optimizer.controlplane import ControlPlaneStore, create_app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--token", default=os.environ.get("CONTROL_PLANE_TOKEN"))
    ap.add_argument("--persist", default=os.environ.get("CONTROL_PLANE_PERSIST"),
                    help="SQLite path to persist runs/commands/knobs across restarts")
    ap.add_argument("--max-body-bytes", type=int,
                    default=int(os.environ.get("CONTROL_PLANE_MAX_BODY_BYTES", str(16 * 1024 * 1024))),
                    help="reject request bodies larger than this many bytes")
    ap.add_argument("--insecure", action="store_true",
                    help="allow binding a non-loopback host without a token (UNSAFE)")
    args = ap.parse_args()

    loopback = args.host in {"127.0.0.1", "localhost", "::1"}
    if not args.token and not loopback and not args.insecure:
        raise SystemExit(
            "refusing to bind a non-loopback host without a token; "
            "set CONTROL_PLANE_TOKEN / --token, or pass --insecure to override"
        )

    store = ControlPlaneStore(persist_path=args.persist)
    app = create_app(store, token=args.token, max_body_bytes=args.max_body_bytes)
    auth = "bearer-token" if args.token else "disabled (dev)"
    persist = args.persist or "in-memory"
    print(f"control plane on http://{args.host}:{args.port}  (auth: {auth}, store: {persist})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
