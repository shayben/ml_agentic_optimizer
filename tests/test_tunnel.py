from __future__ import annotations

import subprocess
import threading

import pytest

from agentic_optimizer import tunnel
from agentic_optimizer.tunnel import (
    DevTunnel,
    TunnelError,
    build_host_command,
    build_named_host_command,
    devtunnel_available,
    ensure_named_tunnel,
    issue_connect_token,
    parse_connect_token,
    parse_tunnel_url,
    run_login,
)


class FakeStdout:
    def __init__(self, lines: list[str]) -> None:
        self._lines = iter(lines)

    def __iter__(self) -> FakeStdout:
        return self

    def __next__(self) -> str:
        return next(self._lines)

    def readline(self) -> str:
        try:
            return next(self)
        except StopIteration:
            return ""


class BlockingStdout:
    def __init__(self) -> None:
        self._closed = threading.Event()

    def __iter__(self) -> BlockingStdout:
        return self

    def __next__(self) -> str:
        if self._closed.wait(timeout=5):
            raise StopIteration
        return "still starting\n"

    def readline(self) -> str:
        try:
            return next(self)
        except StopIteration:
            return ""

    def close(self) -> None:
        self._closed.set()


class FakeProcess:
    def __init__(self, stdout: FakeStdout | BlockingStdout) -> None:
        self.stdout = stdout
        self.terminated = False
        self.killed = False
        self._running = True

    def poll(self) -> int | None:
        return None if self._running else 0

    def terminate(self) -> None:
        self.terminated = True
        self._running = False
        close = getattr(self.stdout, "close", None)
        if close is not None:
            close()

    def kill(self) -> None:
        self.killed = True
        self._running = False
        close = getattr(self.stdout, "close", None)
        if close is not None:
            close()

    def wait(self, timeout: float | None = None) -> int:
        self._running = False
        return 0


def test_build_host_command_defaults() -> None:
    assert build_host_command(8765) == [
        "devtunnel",
        "host",
        "-p",
        "8765",
        "--allow-anonymous",
    ]


def test_build_host_command_without_anonymous_and_custom_cmd() -> None:
    assert build_host_command(9000, allow_anonymous=False, cmd="dt") == [
        "dt",
        "host",
        "-p",
        "9000",
    ]


def test_build_named_host_command_defaults() -> None:
    assert build_named_host_command("stable-id") == [
        "devtunnel",
        "host",
        "stable-id",
        "--allow-anonymous",
    ]


def test_ensure_named_tunnel_creates_tunnel_and_port_tolerating_existing() -> None:
    calls: list[list[str]] = []

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="already exists")

    ensure_named_tunnel("stable-id", 8765, run=run)

    assert calls == [
        ["devtunnel", "create", "stable-id", "--allow-anonymous"],
        ["devtunnel", "port", "create", "stable-id", "-p", "8765"],
    ]


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        (
            "Connect via browser: https://abc123-8765.usw2.devtunnels.ms",
            "https://abc123-8765.usw2.devtunnels.ms",
        ),
        (
            "Inspect at https://abc123.usw2.devtunnels.ms:8765/",
            "https://abc123.usw2.devtunnels.ms:8765/",
        ),
        (
            "Ready: https://abc123-8765.usw2.devtunnels.ms/path, copy this",
            "https://abc123-8765.usw2.devtunnels.ms/path",
        ),
        ("no tunnel url here", None),
    ],
)
def test_parse_tunnel_url(line: str, expected: str | None) -> None:
    assert parse_tunnel_url(line) == expected


def test_devtunnel_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tunnel.shutil, "which", lambda cmd: "C:\\tools\\devtunnel.exe")
    assert devtunnel_available("devtunnel") is True

    monkeypatch.setattr(tunnel.shutil, "which", lambda cmd: None)
    assert devtunnel_available("missing") is False


def test_dev_tunnel_start_happy_path() -> None:
    spawned: list[list[str]] = []
    fake_proc = FakeProcess(
        FakeStdout(
            [
                "Starting tunnel\n",
                "Connect via browser: https://abc123-8765.usw2.devtunnels.ms\n",
            ]
        )
    )

    def popen(argv: list[str], **kwargs: object) -> FakeProcess:
        spawned.append(argv)
        return fake_proc

    dev_tunnel = DevTunnel(8765, cmd="dt", popen=popen, timeout=1)

    assert dev_tunnel.start() == "https://abc123-8765.usw2.devtunnels.ms"
    assert dev_tunnel.url == "https://abc123-8765.usw2.devtunnels.ms"
    assert spawned == [["dt", "host", "-p", "8765", "--allow-anonymous"]]


def test_dev_tunnel_named_start_ensures_then_hosts_stable_url() -> None:
    events: list[tuple[str, list[str]]] = []
    fake_proc = FakeProcess(
        FakeStdout(["Connect via browser: https://stable-id.devtunnels.ms\n"])
    )

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        events.append(("run", argv))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    def popen(argv: list[str], **kwargs: object) -> FakeProcess:
        events.append(("popen", argv))
        return fake_proc

    dev_tunnel = DevTunnel(8765, tunnel_id="stable-id", popen=popen, run=run, timeout=1)

    assert dev_tunnel.start() == "https://stable-id.devtunnels.ms"
    assert events == [
        ("run", ["devtunnel", "create", "stable-id", "--allow-anonymous"]),
        ("run", ["devtunnel", "port", "create", "stable-id", "-p", "8765"]),
        ("popen", ["devtunnel", "host", "stable-id", "--allow-anonymous"]),
    ]


def test_run_login_runs_command() -> None:
    calls: list[list[str]] = []

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    run_login(["devtunnel", "user", "login", "-g", "-d"], run=run)

    assert calls == [["devtunnel", "user", "login", "-g", "-d"]]


def test_run_login_noop_when_empty() -> None:
    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("run should not be called for an empty login command")

    run_login([], run=run)  # no exception, no call


def test_run_login_raises_on_failure() -> None:
    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="auth boom")

    with pytest.raises(TunnelError, match="auth boom"):
        run_login(["devtunnel", "login"], run=run)


def test_dev_tunnel_runs_login_before_temp_host() -> None:
    events: list[tuple[str, list[str]]] = []
    fake_proc = FakeProcess(
        FakeStdout(["Connect via browser: https://abc-8765.usw2.devtunnels.ms\n"])
    )

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        events.append(("run", argv))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    def popen(argv: list[str], **kwargs: object) -> FakeProcess:
        events.append(("popen", argv))
        return fake_proc

    dev_tunnel = DevTunnel(
        8765, login_cmd=["devtunnel", "login", "-g", "-d"], popen=popen, run=run, timeout=1
    )

    assert dev_tunnel.start() == "https://abc-8765.usw2.devtunnels.ms"
    assert events == [
        ("run", ["devtunnel", "login", "-g", "-d"]),
        ("popen", ["devtunnel", "host", "-p", "8765", "--allow-anonymous"]),
    ]


def test_dev_tunnel_named_start_logs_in_then_ensures_then_hosts() -> None:
    events: list[tuple[str, list[str]]] = []
    fake_proc = FakeProcess(
        FakeStdout(["Connect via browser: https://stable-id.devtunnels.ms\n"])
    )

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        events.append(("run", argv))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    def popen(argv: list[str], **kwargs: object) -> FakeProcess:
        events.append(("popen", argv))
        return fake_proc

    dev_tunnel = DevTunnel(
        8765, tunnel_id="stable-id", login_cmd=["devtunnel", "login"], popen=popen, run=run, timeout=1
    )

    assert dev_tunnel.start() == "https://stable-id.devtunnels.ms"
    assert events == [
        ("run", ["devtunnel", "login"]),
        ("run", ["devtunnel", "create", "stable-id", "--allow-anonymous"]),
        ("run", ["devtunnel", "port", "create", "stable-id", "-p", "8765"]),
        ("popen", ["devtunnel", "host", "stable-id", "--allow-anonymous"]),
    ]


def test_dev_tunnel_invokes_on_url_with_public_url() -> None:
    seen: list[str] = []
    fake_proc = FakeProcess(
        FakeStdout(["Connect via browser: https://abc-8765.usw2.devtunnels.ms\n"])
    )

    def popen(argv: list[str], **kwargs: object) -> FakeProcess:
        return fake_proc

    dev_tunnel = DevTunnel(8765, popen=popen, on_url=seen.append, timeout=1)

    assert dev_tunnel.start() == "https://abc-8765.usw2.devtunnels.ms"
    assert seen == ["https://abc-8765.usw2.devtunnels.ms"]


def test_dev_tunnel_on_url_exception_does_not_break_start() -> None:
    fake_proc = FakeProcess(
        FakeStdout(["Connect via browser: https://abc-8765.usw2.devtunnels.ms\n"])
    )

    def popen(argv: list[str], **kwargs: object) -> FakeProcess:
        return fake_proc

    def boom(url: str) -> None:
        raise RuntimeError("callback fail")

    dev_tunnel = DevTunnel(8765, popen=popen, on_url=boom, timeout=1)

    assert dev_tunnel.start() == "https://abc-8765.usw2.devtunnels.ms"


def test_dev_tunnel_start_raises_when_process_exits_without_url() -> None:
    def popen(argv: list[str], **kwargs: object) -> FakeProcess:
        return FakeProcess(FakeStdout(["Starting tunnel\n", "No URL here\n"]))

    dev_tunnel = DevTunnel(8765, popen=popen, timeout=1)

    with pytest.raises(TunnelError, match="before Dev Tunnel public URL"):
        dev_tunnel.start()


def test_dev_tunnel_start_raises_on_deadline() -> None:
    fake_proc = FakeProcess(BlockingStdout())

    def popen(argv: list[str], **kwargs: object) -> FakeProcess:
        return fake_proc

    dev_tunnel = DevTunnel(8765, popen=popen, timeout=0.01)

    with pytest.raises(TunnelError, match="Timed out"):
        dev_tunnel.start()
    assert fake_proc.terminated is True


def test_dev_tunnel_context_manager_stops_process() -> None:
    fake_proc = FakeProcess(
        FakeStdout(["Connect via browser: https://abc123-8765.usw2.devtunnels.ms\n"])
    )

    def popen(argv: list[str], **kwargs: object) -> FakeProcess:
        return fake_proc

    with DevTunnel(8765, popen=popen, timeout=1) as url:
        assert url == "https://abc123-8765.usw2.devtunnels.ms"

    assert fake_proc.terminated is True


def test_start_checks_real_popen_availability_before_spawning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tunnel, "devtunnel_available", lambda cmd: False)

    with pytest.raises(TunnelError, match="Install it from"):
        DevTunnel(8765).start()


# ------------------------------------------------------------------ non-anonymous / connect token


def test_ensure_named_tunnel_non_anonymous_omits_flag() -> None:
    calls: list[list[str]] = []

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    ensure_named_tunnel("stable-id", 8765, allow_anonymous=False, run=run)

    assert calls == [
        ["devtunnel", "create", "stable-id"],
        ["devtunnel", "port", "create", "stable-id", "-p", "8765"],
    ]


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("Tunnel access token: eyJhbGci.eyJzdWIi.abc-DEF_1", "eyJhbGci.eyJzdWIi.abc-DEF_1"),
        ("eyJa.eyJb.c1\n", "eyJa.eyJb.c1"),
        ("no token here", None),
        ("", None),
    ],
)
def test_parse_connect_token(output: str, expected: str | None) -> None:
    assert parse_connect_token(output) == expected


def test_issue_connect_token_success() -> None:
    calls: list[list[str]] = []

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(
            argv, 0, stdout="Tunnel access token: eyJa.eyJb.sig_9", stderr=""
        )

    token = issue_connect_token("my-tid", run=run)

    assert token == "eyJa.eyJb.sig_9"
    assert calls == [["devtunnel", "token", "my-tid", "--scopes", "connect"]]


def test_issue_connect_token_raises_on_nonzero_return() -> None:
    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="not authorized")

    with pytest.raises(TunnelError, match="not authorized"):
        issue_connect_token("my-tid", run=run)


def test_issue_connect_token_raises_when_unparseable() -> None:
    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 0, stdout="issued, but no jwt printed", stderr="")

    with pytest.raises(TunnelError, match="could not parse a connect token"):
        issue_connect_token("my-tid", run=run)


def test_issue_connect_token_raises_on_oserror() -> None:
    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise OSError("devtunnel missing")

    with pytest.raises(TunnelError, match="failed to start"):
        issue_connect_token("my-tid", run=run)
