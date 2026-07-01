"""The control-plane broker: the only network-reachable component.

It mediates between the **local** agent (GitHub Copilot CLI, via the MCP server) and the **remote** PyTorch
training job (via :class:`~agentic_optimizer.bridge.TrainingBridge`):

* the bridge **pushes telemetry** and **claims commands** (long-poll) and **posts results**;
* the agent **reads telemetry/metrics** and **enqueues commands** and **waits for results**.

Three pieces live here:

* :class:`ControlPlaneStore` — a thread-safe state machine with optional SQLite mirroring.
* :func:`create_app` — a thin FastAPI wrapper exposing the store over REST with bearer-token auth.
* :class:`ControlPlaneClient` — an httpx client used by both the MCP server and the bridge. It can wrap an
  in-process ASGI transport for tests (``from_app``) or a real URL (``from_url``).
"""
from __future__ import annotations

import asyncio
import hmac
import logging
import os
import shlex
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .contract import (
    Command,
    CommandRequest,
    CommandResult,
    CommandStatus,
    KnobSpec,
    Telemetry,
)

logger = logging.getLogger("agentic_optimizer.controlplane")


@dataclass
class _RunState:
    metric_history: deque[dict[str, Any]]
    telemetry: Telemetry | None = None
    pending: deque[Command] = field(default_factory=deque)
    knobs: dict[str, KnobSpec] = field(default_factory=dict)
    updated_at: float = field(default_factory=time.time)


class ControlPlaneStore:
    """Thread-safe control-plane state, namespaced by run_id, with optional SQLite mirroring."""

    def __init__(self, history_limit: int = 1000, persist_path: str | None = None) -> None:
        self._history_limit = history_limit
        self._lock = threading.RLock()
        self._runs: dict[str, _RunState] = {}
        self._commands: dict[str, Command] = {}
        self._persist_path = persist_path
        self._db: sqlite3.Connection | None = None
        if persist_path is not None:
            Path(persist_path).parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(persist_path, check_same_thread=False)
            with self._lock:
                self._init_db()
                self._load_db()

    def _run(self, run_id: str = "default") -> _RunState:
        state = self._runs.get(run_id)
        if state is None:
            state = _RunState(metric_history=deque(maxlen=self._history_limit))
            self._runs[run_id] = state
        return state

    def _init_db(self) -> None:
        if self._db is None:
            return
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS telemetry "
            "(run_id TEXT PRIMARY KEY, json TEXT NOT NULL, updated_at REAL NOT NULL)"
        )
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS commands "
            "(id TEXT PRIMARY KEY, run_id TEXT NOT NULL, json TEXT NOT NULL)"
        )
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS knobs "
            "(run_id TEXT NOT NULL, name TEXT NOT NULL, json TEXT NOT NULL, "
            "PRIMARY KEY(run_id, name))"
        )
        self._db.commit()

    def _load_db(self) -> None:
        if self._db is None:
            return
        for run_id, payload, updated_at in self._db.execute(
            "SELECT run_id, json, updated_at FROM telemetry"
        ):
            run = self._run(run_id)
            run.telemetry = Telemetry.model_validate_json(payload)
            run.updated_at = max(run.updated_at, float(updated_at))
        for command_id, run_id, payload in self._db.execute("SELECT id, run_id, json FROM commands"):
            cmd = Command.model_validate_json(payload)
            self._commands[command_id] = cmd
            run = self._run(run_id)
            if cmd.status == CommandStatus.pending:
                run.pending.append(cmd)
            run.updated_at = max(run.updated_at, cmd.completed_at or cmd.claimed_at or cmd.created_at)
        for run_id in self._runs:
            self._runs[run_id].pending = deque(
                sorted(self._runs[run_id].pending, key=lambda cmd: cmd.created_at)
            )
        for run_id, _name, payload in self._db.execute("SELECT run_id, name, json FROM knobs"):
            knob = KnobSpec.model_validate_json(payload)
            self._run(run_id).knobs[knob.name] = knob

    def _persist_telemetry(self, telemetry: Telemetry, updated_at: float) -> None:
        if self._db is None:
            return
        self._db.execute(
            "INSERT OR REPLACE INTO telemetry(run_id, json, updated_at) VALUES (?, ?, ?)",
            (telemetry.run_id, telemetry.model_dump_json(), updated_at),
        )
        self._db.commit()

    def _persist_command(self, cmd: Command) -> None:
        if self._db is None:
            return
        self._db.execute(
            "INSERT OR REPLACE INTO commands(id, run_id, json) VALUES (?, ?, ?)",
            (cmd.id, cmd.run_id, cmd.model_dump_json()),
        )
        self._db.commit()

    def _persist_knob(self, run_id: str, knob: KnobSpec) -> None:
        if self._db is None:
            return
        self._db.execute(
            "INSERT OR REPLACE INTO knobs(run_id, name, json) VALUES (?, ?, ?)",
            (run_id, knob.name, knob.model_dump_json()),
        )
        self._db.commit()

    # ------------------------------------------------------------- telemetry
    def push_telemetry(self, telemetry: Telemetry) -> None:
        with self._lock:
            now = time.time()
            run = self._run(telemetry.run_id)
            run.telemetry = telemetry
            run.updated_at = now
            st = telemetry.state
            if st.metrics or st.loss_history:
                run.metric_history.append(
                    {
                        "step": st.step,
                        "epoch": st.epoch,
                        "timestamp": st.timestamp or now,
                        "metrics": dict(st.metrics),
                        "grad_norm": st.grad_norm,
                        "throughput_samples_per_s": st.throughput_samples_per_s,
                    }
                )
            for knob in telemetry.knobs:
                run.knobs[knob.name] = knob
                self._persist_knob(telemetry.run_id, knob)
            self._persist_telemetry(telemetry, now)

    def get_telemetry(self, run_id: str = "default") -> Telemetry | None:
        with self._lock:
            run = self._runs.get(run_id)
            return run.telemetry if run is not None else None

    def get_metrics(self, limit: int = 100, run_id: str = "default") -> list[dict[str, Any]]:
        with self._lock:
            run = self._runs.get(run_id)
            items = list(run.metric_history) if run is not None else []
        return items[-limit:] if limit and limit > 0 else items

    # -------------------------------------------------------------- commands
    def enqueue_command(
        self, type: str, args: dict[str, Any] | None = None, run_id: str = "default"
    ) -> Command:
        cmd = Command(type=type, args=args or {}, run_id=run_id)
        with self._lock:
            run = self._run(run_id)
            self._commands[cmd.id] = cmd
            run.pending.append(cmd)
            run.updated_at = time.time()
            self._persist_command(cmd)
        return cmd

    def reclaim_expired(self, now: float | None = None, max_attempts: int = 5) -> list[Command]:
        with self._lock:
            return self._reclaim_expired_locked(now=now, max_attempts=max_attempts)

    def _reclaim_expired_locked(
        self, now: float | None = None, max_attempts: int = 5
    ) -> list[Command]:
        now = time.time() if now is None else now
        changed: list[Command] = []
        expired = sorted(
            (
                cmd
                for cmd in self._commands.values()
                if cmd.status == CommandStatus.in_progress
                and cmd.lease_expires_at is not None
                and cmd.lease_expires_at < now
            ),
            key=lambda cmd: cmd.created_at,
            reverse=True,
        )
        for cmd in expired:
            run = self._run(cmd.run_id)
            run.updated_at = now
            if cmd.attempts >= max_attempts:
                cmd.status = CommandStatus.failed
                cmd.completed_at = now
                cmd.result = CommandResult(
                    command_id=cmd.id,
                    ok=False,
                    error=f"lease expired after {cmd.attempts} attempts",
                    applied_at=now,
                )
                logger.info("command lease expired after max attempts", extra={"command_id": cmd.id})
            else:
                cmd.status = CommandStatus.pending
                cmd.claimed_at = None
                cmd.lease_expires_at = None
                run.pending.appendleft(cmd)
                logger.debug("reclaimed expired command lease", extra={"command_id": cmd.id})
            self._persist_command(cmd)
            changed.append(cmd)
        return changed

    def claim_next_command(self, run_id: str = "default", lease_s: float = 30.0) -> Command | None:
        with self._lock:
            self._reclaim_expired_locked()
            run = self._runs.get(run_id)
            if run is None:
                return None
            while run.pending:
                cmd = run.pending.popleft()
                if cmd.status == CommandStatus.pending and cmd.run_id == run_id:
                    now = time.time()
                    cmd.status = CommandStatus.in_progress
                    cmd.claimed_at = now
                    cmd.lease_expires_at = now + lease_s
                    cmd.attempts += 1
                    run.updated_at = now
                    self._persist_command(cmd)
                    return cmd
        return None

    def complete_command(self, result: CommandResult) -> Command | None:
        with self._lock:
            cmd = self._commands.get(result.command_id)
            if cmd is None:
                return None
            cmd.status = CommandStatus.done if result.ok else CommandStatus.failed
            cmd.completed_at = time.time()
            if result.applied_at is None:
                result.applied_at = cmd.completed_at
            cmd.result = result
            cmd.lease_expires_at = None
            self._run(cmd.run_id).updated_at = cmd.completed_at
            self._persist_command(cmd)
            return cmd

    def get_command(self, command_id: str) -> Command | None:
        with self._lock:
            self._reclaim_expired_locked()
            return self._commands.get(command_id)

    def list_commands(self, run_id: str | None = None) -> list[Command]:
        with self._lock:
            self._reclaim_expired_locked()
            commands = list(self._commands.values())
        if run_id is None:
            return commands
        return [cmd for cmd in commands if cmd.run_id == run_id]

    # ----------------------------------------------------------------- knobs
    def register_knobs(self, knobs: list[KnobSpec], run_id: str = "default") -> None:
        with self._lock:
            run = self._run(run_id)
            run.updated_at = time.time()
            for knob in knobs:
                run.knobs[knob.name] = knob
                self._persist_knob(run_id, knob)

    def get_knobs(self, run_id: str = "default") -> list[KnobSpec]:
        with self._lock:
            run = self._runs.get(run_id)
            return list(run.knobs.values()) if run is not None else []

    def list_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            self._reclaim_expired_locked()
            return [
                {
                    "run_id": run_id,
                    "has_telemetry": run.telemetry is not None,
                    "pending": sum(1 for cmd in run.pending if cmd.status == CommandStatus.pending),
                    "updated_at": run.updated_at,
                }
                for run_id, run in sorted(self._runs.items())
            ]


def create_app(
    store: ControlPlaneStore | None = None,
    token: str | None = None,
    max_body_bytes: int = 16 * 1024 * 1024,
) -> Any:
    """Wrap a :class:`ControlPlaneStore` in a FastAPI app. ``token`` enables bearer auth when set."""
    from fastapi import Depends, FastAPI, Header, HTTPException, Query
    from fastapi.responses import JSONResponse, Response

    store = store or ControlPlaneStore()
    app = FastAPI(title="agentic-optimizer control plane", version="0.1.0")
    app.state.store = store

    @app.middleware("http")
    async def limit_request_body(request: Any, call_next: Any) -> Any:
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > max_body_bytes:
                    logger.warning("request rejected because content-length exceeds limit")
                    return Response(status_code=413)
            except ValueError:
                logger.warning("request rejected because content-length is invalid")
                return Response(status_code=413)
        body = await request.body()
        if len(body) > max_body_bytes:
            logger.warning("request rejected because body exceeds limit")
            return Response(status_code=413)
        return await call_next(request)

    def auth_dep() -> Any:
        async def check(authorization: str | None = Header(default=None)) -> None:
            if not token:
                return
            expected = f"Bearer {token}"
            if not hmac.compare_digest(authorization or "", expected):
                logger.warning("invalid or missing bearer token")
                raise HTTPException(status_code=401, detail="invalid or missing bearer token")

        return check

    auth = Depends(auth_dep())

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"ok": True, "runs": len(store.list_runs())}

    @app.post("/telemetry", dependencies=[auth])
    async def post_telemetry(telemetry: Telemetry) -> dict[str, bool]:
        store.push_telemetry(telemetry)
        return {"ok": True}

    @app.get("/telemetry/latest", dependencies=[auth])
    async def get_latest_telemetry(run_id: str = "default") -> Response:
        t = store.get_telemetry(run_id=run_id)
        if t is None:
            return Response(status_code=204)
        return JSONResponse(content=t.model_dump(mode="json"))

    @app.get("/metrics", dependencies=[auth])
    async def get_metrics(
        run_id: str = "default", limit: int = Query(default=100, ge=1, le=10000)
    ) -> list[dict[str, Any]]:
        return store.get_metrics(limit=limit, run_id=run_id)

    @app.post("/commands", dependencies=[auth])
    async def post_command(req: CommandRequest) -> Command:
        return store.enqueue_command(req.type, req.args, run_id=req.run_id)

    @app.get("/commands/next", dependencies=[auth])
    async def next_command(
        run_id: str = "default", wait: float = Query(default=0.0, ge=0.0, le=60.0)
    ) -> Response:
        deadline = time.monotonic() + wait
        while True:
            cmd = store.claim_next_command(run_id=run_id)
            if cmd is not None:
                return JSONResponse(content=cmd.model_dump(mode="json"))
            if time.monotonic() >= deadline:
                return Response(status_code=204)
            await asyncio.sleep(0.1)

    @app.post("/commands/{command_id}/result", dependencies=[auth])
    async def post_result(command_id: str, result: CommandResult) -> Command:
        result.command_id = command_id
        cmd = store.complete_command(result)
        if cmd is None:
            raise HTTPException(status_code=404, detail="unknown command_id")
        return cmd

    @app.get("/commands/{command_id}", dependencies=[auth])
    async def get_command(command_id: str) -> Command:
        cmd = store.get_command(command_id)
        if cmd is None:
            raise HTTPException(status_code=404, detail="unknown command_id")
        return cmd

    @app.get("/commands", dependencies=[auth])
    async def list_commands(run_id: str | None = Query(default=None)) -> list[Command]:
        return store.list_commands(run_id=run_id or None)

    @app.post("/knobs", dependencies=[auth])
    async def post_knobs(knobs: list[KnobSpec], run_id: str = "default") -> dict[str, bool]:
        store.register_knobs(knobs, run_id=run_id)
        return {"ok": True}

    @app.get("/knobs", dependencies=[auth])
    async def get_knobs(run_id: str = "default") -> list[KnobSpec]:
        return store.get_knobs(run_id=run_id)

    @app.get("/runs", dependencies=[auth])
    async def list_runs() -> list[dict[str, Any]]:
        return store.list_runs()

    return app


def _client_headers(
    token: str | None, tunnel_access_token: str | None = None
) -> dict[str, str]:
    """Assemble request headers for the two independent auth layers.

    ``Authorization: Bearer`` is our application token (validated by the broker).
    ``X-Tunnel-Authorization: tunnel`` is a Dev Tunnels *connect* token (validated by the relay,
    only needed for a non-anonymous tunnel). Distinct headers means neither shadows the other.
    """
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if tunnel_access_token:
        headers["X-Tunnel-Authorization"] = f"tunnel {tunnel_access_token}"
    return headers


class ControlPlaneClient:
    """httpx-based client for the broker, shared by the MCP server and the training bridge."""

    def __init__(self, client: httpx.Client) -> None:
        self._client = client

    # -- constructors -------------------------------------------------------
    @classmethod
    def from_url(
        cls,
        base_url: str,
        token: str | None = None,
        timeout: float = 120.0,
        *,
        tunnel_access_token: str | None = None,
    ) -> "ControlPlaneClient":
        """Build a real networked client.

        ``token`` is our application bearer token (``Authorization: Bearer``), validated by the
        broker. ``tunnel_access_token`` is an independent Dev Tunnels *connect* token, sent as
        ``X-Tunnel-Authorization: tunnel <token>`` so a **non-anonymous** tunnel relay admits the
        request before it ever reaches the broker. The two headers are deliberately distinct so
        they never clash — see :func:`agentic_optimizer.tunnel.issue_connect_token`.
        """
        headers = _client_headers(token, tunnel_access_token)
        return cls(httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout))

    @classmethod
    def from_app(cls, app: Any, token: str | None = None) -> "ControlPlaneClient":
        """In-process client driving the ASGI app synchronously — used by tests (no sockets)."""
        from starlette.testclient import TestClient

        headers = {"Authorization": f"Bearer {token}"} if token else {}
        return cls(TestClient(app, headers=headers))

    @classmethod
    def from_env(cls, timeout: float = 120.0) -> "ControlPlaneClient | None":
        """Build a client from ``CONTROL_PLANE_URL`` / ``CONTROL_PLANE_TOKEN``.

        Returns ``None`` when ``CONTROL_PLANE_URL`` is unset, so callers (e.g. :func:`attach`) can
        run the *same* training script transparently with or without a control plane attached.

        When the broker is fronted by a **non-anonymous** Dev Tunnel, also set
        ``CONTROL_PLANE_TUNNEL_ACCESS_TOKEN`` to a connect token — it is forwarded as the
        ``X-Tunnel-Authorization`` header so the relay admits the client.
        """
        url = os.environ.get("CONTROL_PLANE_URL")
        if not url:
            return None
        return cls.from_url(
            url,
            token=os.environ.get("CONTROL_PLANE_TOKEN"),
            timeout=timeout,
            tunnel_access_token=os.environ.get("CONTROL_PLANE_TUNNEL_ACCESS_TOKEN"),
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ControlPlaneClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- telemetry ----------------------------------------------------------
    def push_telemetry(self, telemetry: Telemetry) -> None:
        r = self._client.post("/telemetry", json=telemetry.model_dump(mode="json"))
        r.raise_for_status()

    def get_telemetry(self, run_id: str | None = None) -> Telemetry | None:
        r = self._client.get("/telemetry/latest", params={"run_id": run_id or "default"})
        if r.status_code == 204:
            return None
        r.raise_for_status()
        return Telemetry.model_validate(r.json())

    def get_metrics(self, limit: int = 100, run_id: str | None = None) -> list[dict[str, Any]]:
        r = self._client.get("/metrics", params={"limit": limit, "run_id": run_id or "default"})
        r.raise_for_status()
        return r.json()

    # -- commands -----------------------------------------------------------
    def enqueue_command(
        self, type: str, args: dict[str, Any] | None = None, run_id: str | None = None
    ) -> Command:
        r = self._client.post(
            "/commands", json={"type": type, "args": args or {}, "run_id": run_id or "default"}
        )
        r.raise_for_status()
        return Command.model_validate(r.json())

    def next_command(self, wait: float = 0.0, run_id: str | None = None) -> Command | None:
        r = self._client.get(
            "/commands/next", params={"wait": wait, "run_id": run_id or "default"}
        )
        if r.status_code == 204:
            return None
        r.raise_for_status()
        return Command.model_validate(r.json())

    def complete_command(
        self,
        command_id: str,
        ok: bool = True,
        data: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> Command:
        result = CommandResult(command_id=command_id, ok=ok, data=data or {}, error=error)
        r = self._client.post(f"/commands/{command_id}/result", json=result.model_dump(mode="json"))
        r.raise_for_status()
        return Command.model_validate(r.json())

    def get_command(self, command_id: str) -> Command | None:
        r = self._client.get(f"/commands/{command_id}")
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return Command.model_validate(r.json())

    def wait_for_result(
        self, command_id: str, timeout: float = 60.0, poll: float = 0.25
    ) -> CommandResult | None:
        deadline = time.monotonic() + timeout
        while True:
            cmd = self.get_command(command_id)
            if cmd is not None and cmd.status in (CommandStatus.done, CommandStatus.failed):
                return cmd.result
            if time.monotonic() >= deadline:
                return None
            time.sleep(poll)

    # -- knobs --------------------------------------------------------------
    def register_knobs(self, knobs: list[KnobSpec], run_id: str | None = None) -> None:
        r = self._client.post(
            "/knobs", params={"run_id": run_id or "default"}, json=[k.model_dump() for k in knobs]
        )
        r.raise_for_status()

    def get_knobs(self, run_id: str | None = None) -> list[KnobSpec]:
        r = self._client.get("/knobs", params={"run_id": run_id or "default"})
        r.raise_for_status()
        return [KnobSpec.model_validate(k) for k in r.json()]

    def list_runs(self) -> list[dict[str, Any]]:
        r = self._client.get("/runs")
        r.raise_for_status()
        return r.json()

    def health(self) -> bool:
        try:
            r = self._client.get("/healthz")
            return r.status_code == 200
        except httpx.HTTPError:
            return False


def _check_exposure(
    *, token: str | None, tunnel: bool, host: str, insecure: bool, tunnel_anonymous: bool = True
) -> None:
    """Refuse to expose an unauthenticated control plane.

    A Dev Tunnel is always a public endpoint and a non-loopback bind is reachable off-box;
    either one without a bearer token lets anyone drive the training run. Require
    ``CONTROL_PLANE_TOKEN`` in those cases unless the operator explicitly opts out with
    ``CONTROL_PLANE_INSECURE=1``. Raises :class:`SystemExit` to abort startup.

    A **non-anonymous** tunnel (``tunnel_anonymous=False``) is itself an authentication layer —
    the Dev Tunnels relay rejects clients that lack a connect token before requests reach us — so
    it satisfies the check even without an application bearer token (defence in depth still
    recommends setting both)."""
    if token or insecure:
        return
    if tunnel and tunnel_anonymous:
        raise SystemExit(
            "Refusing to expose an unauthenticated control plane over an anonymous public Dev "
            "Tunnel. Set CONTROL_PLANE_TOKEN (recommended), host a non-anonymous tunnel "
            "(--no-tunnel-anonymous), or set CONTROL_PLANE_INSECURE=1 to override."
        )
    if not tunnel and host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit(
            "Refusing to bind an unauthenticated control plane to a non-loopback host. "
            "Set CONTROL_PLANE_TOKEN or CONTROL_PLANE_INSECURE=1 to override."
        )


def _run_cli() -> None:  # pragma: no cover - thin console entry point
    """``agentic-optimizer-broker`` console script: serve the broker from env config."""
    import argparse
    import os

    import uvicorn

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Serve the agentic optimizer control-plane broker.")
    parser.add_argument("--tunnel", action="store_true", help="Expose the broker with Microsoft Dev Tunnels.")
    parser.add_argument("--host", default=os.environ.get("CONTROL_PLANE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("CONTROL_PLANE_PORT", "8765")))
    parser.add_argument("--devtunnel-cmd", default="devtunnel")
    parser.add_argument(
        "--tunnel-id",
        default=os.environ.get("CONTROL_PLANE_TUNNEL_ID"),
        help="Host a persistent (named) Dev Tunnel with a STABLE public URL so the node's "
        "CONTROL_PLANE_URL stays constant across broker restarts (no config churn). "
        "Defaults to $CONTROL_PLANE_TUNNEL_ID. Requires --tunnel.",
    )
    parser.add_argument(
        "--tunnel-login",
        default=os.environ.get("CONTROL_PLANE_TUNNEL_LOGIN"),
        help="Non-interactive Dev Tunnels login command run before hosting (node-hosted mode: a "
        "headless node must authenticate to HOST a tunnel; --allow-anonymous only grants clients). "
        "E.g. an access-token wrapper or 'devtunnel user login -g -d'. Defaults to "
        "$CONTROL_PLANE_TUNNEL_LOGIN. Requires --tunnel.",
    )
    parser.add_argument(
        "--tunnel-url-file",
        default=os.environ.get("CONTROL_PLANE_TUNNEL_URL_FILE"),
        help="Write the discovered public Dev Tunnel URL to this path so a remote agent can "
        "discover it (e.g. an AML outputs folder). Defaults to $CONTROL_PLANE_TUNNEL_URL_FILE. "
        "Requires --tunnel.",
    )
    parser.add_argument(
        "--tunnel-anonymous",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("CONTROL_PLANE_TUNNEL_ANONYMOUS", "1") != "0",
        help="Whether the Dev Tunnel allows anonymous client access (default: on, for backward "
        "compatibility). Pass --no-tunnel-anonymous (or CONTROL_PLANE_TUNNEL_ANONYMOUS=0) to make "
        "the tunnel NON-ANONYMOUS: the Dev Tunnels relay then rejects clients that do not present a "
        "connect token, before requests ever reach the broker. Clients supply that token via "
        "CONTROL_PLANE_TUNNEL_ACCESS_TOKEN (sent as the X-Tunnel-Authorization header).",
    )
    parser.add_argument(
        "--tunnel-token-file",
        default=os.environ.get("CONTROL_PLANE_TUNNEL_TOKEN_FILE"),
        help="For a NON-ANONYMOUS named tunnel: mint a Dev Tunnels connect token and write it here "
        "so a remote agent can pick it up (set its CONTROL_PLANE_TUNNEL_ACCESS_TOKEN). Requires "
        "--no-tunnel-anonymous and --tunnel-id. NOTE: connect tokens expire after ~24h. Defaults to "
        "$CONTROL_PLANE_TUNNEL_TOKEN_FILE.",
    )
    args = parser.parse_args()

    host = args.host
    port = args.port
    token = os.environ.get("CONTROL_PLANE_TOKEN")
    _check_exposure(
        token=token,
        tunnel=args.tunnel,
        host=host,
        insecure=os.environ.get("CONTROL_PLANE_INSECURE") == "1",
        tunnel_anonymous=args.tunnel_anonymous,
    )
    if args.tunnel_id and not args.tunnel:
        logger.warning(
            "--tunnel-id/$CONTROL_PLANE_TUNNEL_ID is ignored without --tunnel; "
            "add --tunnel to host the persistent Dev Tunnel."
        )
    if (args.tunnel_login or args.tunnel_url_file) and not args.tunnel:
        logger.warning(
            "--tunnel-login/--tunnel-url-file are ignored without --tunnel."
        )
    if args.tunnel_token_file and (args.tunnel_anonymous or not args.tunnel_id or not args.tunnel):
        logger.warning(
            "--tunnel-token-file requires --tunnel, --no-tunnel-anonymous and --tunnel-id; "
            "no connect token will be minted."
        )
    persist_path = os.environ.get("CONTROL_PLANE_PERSIST")
    max_body_bytes = int(os.environ.get("CONTROL_PLANE_MAX_BODY_BYTES", str(16 * 1024 * 1024)))
    app = create_app(
        ControlPlaneStore(persist_path=persist_path), token=token, max_body_bytes=max_body_bytes
    )
    if args.tunnel:
        from .tunnel import serve_with_tunnel

        tunnel_host = "127.0.0.1"
        logger.info(
            "control plane on http://%s:%s via Dev Tunnel (bearer auth: %s, relay access: %s)",
            tunnel_host,
            port,
            "on" if token else "off",
            "anonymous" if args.tunnel_anonymous else "non-anonymous (connect token required)",
        )
        logger.info(
            "Use the Dev Tunnel public URL as CONTROL_PLANE_URL on the remote node; "
            "the bearer token is still required when configured."
        )
        if not args.tunnel_anonymous:
            logger.info(
                "Non-anonymous tunnel: clients must present a connect token via "
                "CONTROL_PLANE_TUNNEL_ACCESS_TOKEN (X-Tunnel-Authorization header). Mint one with "
                "`devtunnel token %s --scopes connect` (expires ~24h).",
                args.tunnel_id or "<tunnel-id>",
            )
        if args.tunnel_id:
            logger.info(
                "Persistent Dev Tunnel '%s': its public URL is stable across restarts, so the "
                "node's CONTROL_PLANE_URL can be set once and left unchanged.",
                args.tunnel_id,
            )
        else:
            logger.info(
                "Temporary Dev Tunnel: the public URL changes each run. Pass --tunnel-id "
                "(or $CONTROL_PLANE_TUNNEL_ID) for a stable URL."
            )

        login_cmd = shlex.split(args.tunnel_login) if args.tunnel_login else None
        url_file = args.tunnel_url_file
        mint_token = bool(
            args.tunnel_token_file and not args.tunnel_anonymous and args.tunnel_id
        )
        token_file = args.tunnel_token_file if mint_token else None

        def _on_url(url: str) -> None:
            if url_file:
                try:
                    path = Path(url_file)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(url + "\n", encoding="utf-8")
                    logger.info("Wrote Dev Tunnel public URL to %s", url_file)
                except OSError:
                    logger.warning("Failed to write Dev Tunnel URL to %s", url_file, exc_info=True)
            if token_file:
                from .tunnel import TunnelError, issue_connect_token

                try:
                    connect_token = issue_connect_token(args.tunnel_id, cmd=args.devtunnel_cmd)
                    path = Path(token_file)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(connect_token + "\n", encoding="utf-8")
                    logger.info("Wrote Dev Tunnel connect token to %s (expires ~24h)", token_file)
                except (TunnelError, OSError):
                    logger.warning(
                        "Failed to mint/write Dev Tunnel connect token to %s", token_file,
                        exc_info=True,
                    )

        serve_with_tunnel(
            app,
            tunnel_host,
            port,
            cmd=args.devtunnel_cmd,
            tunnel_id=args.tunnel_id,
            allow_anonymous=args.tunnel_anonymous,
            login_cmd=login_cmd,
            on_url=_on_url,
        )
        return

    logger.info("control plane on http://%s:%s (auth: %s)", host, port, "on" if token else "off")
    uvicorn.run(app, host=host, port=port, log_level="warning")
