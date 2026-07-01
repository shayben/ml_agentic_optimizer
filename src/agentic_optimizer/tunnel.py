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
    output = f"{result.stderr or ''}\n{result.stdout or ''}"
    return result.returncode != 0 and "already exists" in output.lower()


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


def parse_tunnel_url(line: str) -> str | None:
    """Extract the first public ``https://*.devtunnels.ms`` URL from a devtunnel line."""
    match = _DEV_TUNNEL_URL_RE.search(line)
    if match is None:
        return None
    return match.group(0).rstrip(" \t\r\n.,);]")


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
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.stop()
                raise TunnelError(self._format_start_failure("Timed out", captured))
            try:
                line = lines.get(timeout=min(0.1, remaining))
            except queue.Empty:
                if self.proc is not None and self.proc.poll() is not None:
                    raise TunnelError(self._format_start_failure("Process exited", captured))
                continue

            if line is None:
                raise TunnelError(self._format_start_failure("Process exited", captured))
            captured.append(line.rstrip())
            url = parse_tunnel_url(line)
            if url is not None:
                self.url = url
                self._startup_done.set()
                self._emit_url(url)
                return url

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
