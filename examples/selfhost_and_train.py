"""Node-hosted "submit-then-attach" entrypoint: run the broker + Dev Tunnel ON the training node.

This is **Topology B** (see the README "Node-hosted broker" section). Instead of the agent's box
hosting the broker and the job dialing out to it, everything runs on the node:

* the control-plane broker binds to ``localhost`` and is published through a Dev Tunnel;
* the :class:`TrainingBridge` talks to the broker over **loopback** (no internet hop for telemetry);
* the public tunnel URL is printed and written to ``--url-file`` so a local agent can **opt in**
  whenever it likes.

Because the job self-publishes its control plane, you submit it *first* and attach *later* — there is
no "bring the broker up before submitting" ordering constraint. Pair it with a persistent
``--tunnel-id`` for a stable URL so the agent's ``CONTROL_PLANE_URL`` / MCP config stays static.

    python examples/selfhost_and_train.py --tunnel-id my-stable-id --epochs 30

Hosting a tunnel from a headless node requires *non-interactive* Dev Tunnels authentication (the
``--allow-anonymous`` flag only grants clients access; the host must be logged in). Provide it via
``--tunnel-login`` / ``$CONTROL_PLANE_TUNNEL_LOGIN`` (for example an access-token wrapper).

By default the tunnel allows anonymous clients (the broker's bearer token is the only gate). Pass
``--no-tunnel-anonymous`` to make the relay itself reject unauthenticated clients; the node then mints
a **connect token** (``--token-file``) that the agent supplies via ``CONTROL_PLANE_TUNNEL_ACCESS_TOKEN``.
Connect tokens expire after ~24h — see the README "non-anonymous" section for the trade-offs.
"""
from __future__ import annotations

import argparse
import os
import shlex
import threading
import time
from pathlib import Path

from agentic_optimizer.controlplane import ControlPlaneClient, ControlPlaneStore, create_app
from agentic_optimizer.tunnel import TunnelError, issue_connect_token, serve_with_tunnel

# ``run_training`` lives alongside this file in examples/; import it directly.
from train_with_bridge import run_training


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=int(os.environ.get("CONTROL_PLANE_PORT", "8765")))
    ap.add_argument("--token", default=os.environ.get("CONTROL_PLANE_TOKEN"))
    ap.add_argument("--tunnel-id", default=os.environ.get("CONTROL_PLANE_TUNNEL_ID"),
                    help="persistent named tunnel id for a stable public URL (recommended)")
    ap.add_argument("--tunnel-login", default=os.environ.get("CONTROL_PLANE_TUNNEL_LOGIN"),
                    help="non-interactive Dev Tunnels host login command (headless auth)")
    ap.add_argument("--url-file", default=os.environ.get("CONTROL_PLANE_TUNNEL_URL_FILE"),
                    help="write the discovered public URL here (e.g. an AML outputs folder)")
    ap.add_argument("--tunnel-anonymous", action=argparse.BooleanOptionalAction,
                    default=os.environ.get("CONTROL_PLANE_TUNNEL_ANONYMOUS", "1") != "0",
                    help="allow anonymous tunnel clients (default on). --no-tunnel-anonymous makes "
                         "the tunnel non-anonymous; the relay then requires a connect token.")
    ap.add_argument("--token-file", default=os.environ.get("CONTROL_PLANE_TUNNEL_TOKEN_FILE"),
                    help="for a non-anonymous named tunnel: mint a connect token and write it here "
                         "so the agent can set CONTROL_PLANE_TUNNEL_ACCESS_TOKEN (expires ~24h)")
    ap.add_argument("--devtunnel-cmd",
                    default=os.environ.get("CONTROL_PLANE_DEVTUNNEL_CMD", "devtunnel"))
    ap.add_argument("--persist", default=os.environ.get("CONTROL_PLANE_PERSIST"),
                    help="SQLite path so telemetry/commands survive a broker restart")
    ap.add_argument("--run-id", default=os.environ.get("CONTROL_PLANE_RUN_ID", "default"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--noise-frac", type=float, default=0.08)
    ap.add_argument("--poll-interval", type=float, default=0.0)
    ap.add_argument("--mlflow", action="store_true")
    ap.add_argument("--startup-timeout", type=float, default=90.0,
                    help="seconds to wait for the Dev Tunnel to publish a URL")
    args = ap.parse_args()

    if not args.token and args.tunnel_anonymous:
        raise SystemExit(
            "refusing to publish an unauthenticated broker over an anonymous public Dev Tunnel; "
            "set CONTROL_PLANE_TOKEN / --token, or host a non-anonymous tunnel "
            "(--no-tunnel-anonymous)."
        )

    mint_token = bool(args.token_file and not args.tunnel_anonymous and args.tunnel_id)
    if args.token_file and not mint_token:
        print("[selfhost] WARN: --token-file requires --no-tunnel-anonymous and --tunnel-id; "
              "no connect token will be minted.")

    store = ControlPlaneStore(persist_path=args.persist)
    app = create_app(store, token=args.token)
    login_cmd = shlex.split(args.tunnel_login) if args.tunnel_login else None

    ready = threading.Event()
    holder: dict[str, object] = {}

    def on_url(url: str) -> None:
        holder["url"] = url
        if args.url_file:
            try:
                path = Path(args.url_file)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(url + "\n", encoding="utf-8")
            except OSError as exc:
                print(f"[selfhost] WARN: could not write url-file {args.url_file}: {exc}")
        if mint_token:
            try:
                connect_token = issue_connect_token(args.tunnel_id, cmd=args.devtunnel_cmd)
                path = Path(args.token_file)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(connect_token + "\n", encoding="utf-8")
                print(f"[selfhost] wrote connect token to {args.token_file} (expires ~24h); "
                      "set the agent's CONTROL_PLANE_TUNNEL_ACCESS_TOKEN to its contents.")
            except (TunnelError, OSError) as exc:
                print(f"[selfhost] WARN: could not mint/write connect token: {exc}")
        print(f"[selfhost] public Dev Tunnel URL (set the agent's CONTROL_PLANE_URL to this): {url}")
        ready.set()

    def serve() -> None:
        try:
            serve_with_tunnel(
                app,
                "127.0.0.1",
                args.port,
                cmd=args.devtunnel_cmd,
                tunnel_id=args.tunnel_id,
                allow_anonymous=args.tunnel_anonymous,
                login_cmd=login_cmd,
                on_url=on_url,
            )
        except Exception as exc:  # surface tunnel/broker startup failures to the main thread
            holder["error"] = exc
            ready.set()

    threading.Thread(target=serve, name="selfhost-broker", daemon=True).start()

    if not ready.wait(timeout=args.startup_timeout):
        raise SystemExit("timed out waiting for the Dev Tunnel to publish a public URL")
    if "error" in holder:
        raise SystemExit(f"broker/tunnel failed to start: {holder['error']}")

    client = ControlPlaneClient.from_url(f"http://127.0.0.1:{args.port}", args.token)
    deadline = time.monotonic() + 30.0
    while not client.health():
        if time.monotonic() > deadline:
            raise SystemExit("broker did not become healthy on loopback")
        time.sleep(0.2)

    print(f"[selfhost] broker healthy on loopback; starting training (run_id={args.run_id!r}). "
          "An agent can attach at any time.")
    run_training(
        client,
        epochs=args.epochs,
        lr=args.lr,
        mlflow=args.mlflow,
        run_id=args.run_id,
        poll_interval=args.poll_interval,
        noise_frac=args.noise_frac,
    )
    print("[selfhost] training complete; the broker/tunnel shut down with this process.")


if __name__ == "__main__":
    main()
