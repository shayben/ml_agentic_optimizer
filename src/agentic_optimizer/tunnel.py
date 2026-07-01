"""Microsoft Dev Tunnels integration for the control-plane broker."""
from __future__ import annotations

import logging
import queue
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from typing import IO, Any

logger = logging.getLogger("agentic_optimizer.tunnel")

_DEV_TUNNEL_URL_RE = re.compile(
    r"https://[A-Za-z0-9.-]+\.devtunnels\.ms(?::\d+)?(?:/[^\s\"'<>]*)?"
)


class TunnelError(RuntimeError):
    """Raised when a Dev Tunnel cannot be launched or its public URL is unavailable."""


def devtunnel_available(cmd: str = "devtunnel") -> bool:
    """Return whether the Dev Tunnels CLI executable is available on ``PATH``."""
    return shutil.which(cmd) is not None


def build_host_command(
    port: int, *, allow_anonymous: bool = True, cmd: str = "devtunnel"
) -> list[str]:
    """Build ``devtunnel host -p <PORT>`` argv for forwarding to local ``localhost:<PORT>``."""
    argv = [cmd, "host", "-p", str(port)]
    if allow_anonymous:
        argv.append("--allow-anonymous")
    return argv


def build_named_host_command(
    tunnel_id: str, *, allow_anonymous: bool = True, cmd: str = "devtunnel"
) -> list[str]:
    """Build ``devtunnel host <TUNNEL_ID>`` argv for an existing named tunnel."""
    argv = [cmd, "host", tunnel_id]
    if allow_anonymous:
        argv.append("--allow-anonymous")
    return argv


def ensure_named_tunnel(
    tunnel_id: str,
    port: int,
    *,
    allow_anonymous: bool = True,
    cmd: str = "devtunnel",
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    """Ensure a persistent Dev Tunnel and forwarded port exist."""
    create_argv = [cmd, "create", tunnel_id]
    if allow_anonymous:
        create_argv.append("--allow-anonymous")
    port_argv = [cmd, "port", "create", tunnel_id, "-p", str(port)]

    for argv in (create_argv, port_argv):
        result = run(argv, capture_output=True, text=True)
        if result.returncode == 0 or _already_exists(result):
            continue
        stderr = (result.stderr or result.stdout or "").strip()
        detail = f": {stderr}" if stderr else ""
        raise TunnelError(f"Dev Tunnel command failed ({' '.join(argv)}){detail}")


def _already_exists(result: subprocess.CompletedProcess[str]) -> bool:
    output = f"{result.stderr or ''}\n{result.stdout or ''}".lower()
    if result.returncode == 0:
        return False
    # The Dev Tunnels service reports a pre-existing named tunnel/port either as "... already exists"
    # or as "Conflict with existing entity" depending on the operation; both mean the entity is there,
    # so ensure_named_tunnel can treat them as success (it is meant to be idempotent).
    return "already exists" in output or "conflict with existing entity" in output


def run_login(
    login_cmd: list[str],
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    """Run a non-interactive Dev Tunnels *host* login (for headless hosts such as an AML node).

    Hosting a tunnel — even an anonymous-access one — requires the host to be authenticated to the
    Dev Tunnels service (``--allow-anonymous`` only grants *client* access). On your local box you are
    already logged in, but a remote training node is not. ``login_cmd`` is the command that
    authenticates it, e.g. an access-token wrapper or ``devtunnel user login -g -d``; it must be
    non-interactive (return promptly) to be usable in an automated job.
    """
    if not login_cmd:
        return
    logger.info("Authenticating Dev Tunnels host: %s", " ".join(login_cmd))
    try:
        result = run(login_cmd, capture_output=True, text=True)
    except OSError as exc:
        raise TunnelError(f"Dev Tunnels login command failed to start: {exc}") from exc
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "").strip()
        detail = f": {stderr}" if stderr else ""
        raise TunnelError(f"Dev Tunnels login failed ({' '.join(login_cmd)}){detail}")


_ACCESS_TOKEN_RE = re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+")


def parse_connect_token(output: str) -> str | None:
    """Extract a Dev Tunnels access token (a JWT) from ``devtunnel token`` output.

    The preview CLI prints the token with a surrounding label (e.g. ``Tunnel access token: eyJ...``)
    whose exact wording changes between versions, so we match the JWT itself rather than the label.
    """
    match = _ACCESS_TOKEN_RE.search(output or "")
    return match.group(0) if match else None


def issue_connect_token(
    tunnel_id: str,
    *,
    scopes: str = "connect",
    cmd: str = "devtunnel",
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    """Mint a Dev Tunnels **connect** token for ``tunnel_id`` so clients can reach a non-anonymous
    tunnel without their own interactive login.

    Runs ``devtunnel token <id> --scopes connect`` (the host must already be authenticated with a
    *manage* scope over the tunnel — true on your local box, or on a node after :func:`run_login`).
    The returned token is handed to clients via ``CONTROL_PLANE_TUNNEL_ACCESS_TOKEN`` and forwarded
    as the ``X-Tunnel-Authorization`` header.

    .. warning::
       Connect tokens **expire after ~24 hours** and can only be refreshed by a real user identity
       holding *manage* scope. For runs longer than a day, re-issue periodically, or grant access to
       an Entra tenant (``--tenant``) / GitHub org (``--organization``) instead of distributing a
       token — see the non-anonymous docs.
    """
    argv = [cmd, "token", tunnel_id, "--scopes", scopes]
    try:
        result = run(argv, capture_output=True, text=True)
    except OSError as exc:
        raise TunnelError(f"Dev Tunnels token command failed to start: {exc}") from exc
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "").strip()
        detail = f": {stderr}" if stderr else ""
        raise TunnelError(f"Dev Tunnels token issuance failed ({' '.join(argv)}){detail}")
    token = parse_connect_token(result.stdout or "")
    if not token:
        raise TunnelError(
            f"could not parse a connect token from `{' '.join(argv)}` output"
        )
    return token


def parse_tunnel_url(line: str, port: int | None = None) -> str | None:
    """Extract a public ``https://*.devtunnels.ms`` URL from a devtunnel line.

    When ``port`` is given, only return a URL whose host label encodes that forwarded port
    (``https://<id>-<port>.<cluster>.devtunnels.ms``). A named tunnel can forward several ports, and
    ``devtunnel host`` then prints one connect URL per port; without this filter the first-listed
    (often a stale, lower-numbered) port would be returned instead of the port we actually host. The
    per-port *inspect* URL (``<id>-<port>-inspect...``) never matches ``-<port>`` and is rejected.
    """
    match = _DEV_TUNNEL_URL_RE.search(line)
    if match is None:
        return None
    url = match.group(0).rstrip(" \t\r\n.,);]")
    if port is not None:
        host_label = url.split("://", 1)[-1].split("/", 1)[0].split(".", 1)[0]
        if host_label.rsplit("-", 1)[-1] != str(port):
            return None
    return url


class DevTunnel:
    """Manage a ``devtunnel host`` subprocess for the broker lifetime."""

    def __init__(
        self,
        port: int,
        *,
        allow_anonymous: bool = True,
        cmd: str = "devtunnel",
        tunnel_id: str | None = None,
        login_cmd: list[str] | None = None,
        on_url: Callable[[str], None] | None = None,
        popen: Callable[..., Any] = subprocess.Popen,
        run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        timeout: float = 30.0,
    ) -> None:
        self.port = port
        self.allow_anonymous = allow_anonymous
        self.cmd = cmd
        self.tunnel_id = tunnel_id
        self.login_cmd = login_cmd
        self.on_url = on_url
        self.popen = popen
        self.run = run
        self.timeout = timeout
        self.url: str | None = None
        self.proc: Any | None = None
        self._startup_done = threading.Event()

    def start(self) -> str:
        """Start the Dev Tunnel process and return its public HTTPS URL."""
        if self.popen is subprocess.Popen and not devtunnel_available(self.cmd):
            raise TunnelError(
                f"Microsoft Dev Tunnels CLI '{self.cmd}' was not found on PATH. "
                "Install it from https://learn.microsoft.com/azure/developer/dev-tunnels/ "
                "or pass --devtunnel-cmd with the CLI path."
            )

        if self.login_cmd:
            run_login(self.login_cmd, run=self.run)

        if self.tunnel_id is not None:
            ensure_named_tunnel(
                self.tunnel_id,
                self.port,
                allow_anonymous=self.allow_anonymous,
                cmd=self.cmd,
                run=self.run,
            )
            argv = build_named_host_command(
                self.tunnel_id, allow_anonymous=self.allow_anonymous, cmd=self.cmd
            )
        else:
            argv = build_host_command(
                self.port, allow_anonymous=self.allow_anonymous, cmd=self.cmd
            )
        try:
            self.proc = self.popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise TunnelError(f"Failed to start Dev Tunnels CLI: {exc}") from exc

        stdout = getattr(self.proc, "stdout", None)
        if stdout is None:
            self.stop()
            raise TunnelError("Dev Tunnels CLI did not provide stdout for URL discovery.")

        lines: queue.Queue[str | None] = queue.Queue()
        reader = threading.Thread(
            target=self._read_stdout,
            args=(stdout, lines),
            name="agentic-optimizer-devtunnel-output",
            daemon=True,
        )
        reader.start()

        captured: list[str] = []
        deadline = time.monotonic() + self.timeout
        # A named tunnel may forward several ports (e.g. a stale one left over from a prior run);
        # devtunnel host then prints a connect URL per port, so require the URL for the port we host
        # rather than blindly taking the first. Keep first-match behaviour for ad-hoc tunnels.
        want_port = self.port if self.tunnel_id is not None else None
        seen: list[str] = []  # distinct non-inspect URLs, for an unambiguous single-URL fallback

        def _accept(u: str) -> str:
            self.url = u
            self._startup_done.set()
            self._emit_url(u)
            return u

        def _fallback() -> str | None:
            # Fall back only when exactly one URL was advertised (e.g. a named tunnel that does not
            # port-tag its connect URL). With several ports advertised and none matching the port we
            # host, refuse to guess -- returning the wrong port would send the node to a dead relay.
            uniq = list(dict.fromkeys(seen))
            return uniq[0] if len(uniq) == 1 else None

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                fb = _fallback()
                if fb is not None:
                    return _accept(fb)
                self.stop()
                raise TunnelError(self._format_start_failure("Timed out", captured))
            try:
                line = lines.get(timeout=min(0.1, remaining))
            except queue.Empty:
                if self.proc is not None and self.proc.poll() is not None:
                    fb = _fallback()
                    if fb is not None:
                        return _accept(fb)
                    raise TunnelError(self._format_start_failure("Process exited", captured))
                continue

            if line is None:
                fb = _fallback()
                if fb is not None:
                    return _accept(fb)
                raise TunnelError(self._format_start_failure("Process exited", captured))
            captured.append(line.rstrip())
            url = parse_tunnel_url(line, port=want_port)
            if url is not None:
                return _accept(url)
            # Track any non-inspect URL for the unambiguous fallback, and take it as soon as the host
            # signals readiness (so we don't wait out the whole timeout on a portless single-port URL).
            any_url = parse_tunnel_url(line)
            if any_url is not None and "-inspect." not in any_url:
                seen.append(any_url)
            if "ready to accept connections" in line.lower():
                fb = _fallback()
                if fb is not None:
                    return _accept(fb)

    def stop(self) -> None:
        """Terminate the tunnel subprocess if it is still running."""
        proc = self.proc
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
                    try:
                        proc.wait(timeout=5)
                    except Exception:
                        pass
        except Exception:
            logger.debug("error while stopping Dev Tunnel", exc_info=True)

    def __enter__(self) -> str:
        """Start the tunnel and return its public URL."""
        return self.start()

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Stop the tunnel when leaving a context manager."""
        self.stop()

    def _emit_url(self, url: str) -> None:
        if self.on_url is None:
            return
        try:
            self.on_url(url)
        except Exception:
            logger.warning("Dev Tunnel on_url callback raised", exc_info=True)

    def _read_stdout(self, stdout: IO[str], lines: queue.Queue[str | None]) -> None:
        try:
            for line in stdout:
                if self._startup_done.is_set():
                    logger.info("devtunnel: %s", line.rstrip())
                else:
                    lines.put(line)
        finally:
            if not self._startup_done.is_set():
                lines.put(None)

    @staticmethod
    def _format_start_failure(reason: str, captured: list[str]) -> str:
        output = "\n".join(captured).strip()
        if output:
            return f"{reason} before Dev Tunnel public URL was available. Output:\n{output}"
        return f"{reason} before Dev Tunnel public URL was available; no output captured."


def serve_with_tunnel(
    app: object,
    host: str,
    port: int,
    *,
    cmd: str = "devtunnel",
    tunnel_id: str | None = None,
    allow_anonymous: bool = True,
    login_cmd: list[str] | None = None,
    on_url: Callable[[str], None] | None = None,
) -> None:  # pragma: no cover - thin server wrapper
    """Serve the broker with a Microsoft Dev Tunnel forwarding public HTTPS traffic.

    ``login_cmd`` authenticates a headless host (node-hosted mode); ``on_url`` receives the public URL
    once discovered (e.g. to write it to a file for cross-machine discovery).
    """
    tunnel = DevTunnel(
        port,
        cmd=cmd,
        tunnel_id=tunnel_id,
        allow_anonymous=allow_anonymous,
        login_cmd=login_cmd,
        on_url=on_url,
    )
    try:
        url = tunnel.start()
        logger.info("Dev Tunnel public URL: %s", url)
        logger.info("Set the remote node CONTROL_PLANE_URL to %s", url)
        logger.info("CONTROL_PLANE_TOKEN bearer authentication is still required when configured.")
        import uvicorn

        uvicorn.run(app, host=host, port=port, log_level="warning")
    finally:
        tunnel.stop()
