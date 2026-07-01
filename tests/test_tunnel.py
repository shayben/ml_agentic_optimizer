from __future__ import annotations

import threading

import pytest

from agentic_optimizer import tunnel
from agentic_optimizer.tunnel import (
    DevTunnel,
    TunnelError,
    build_host_command,
    devtunnel_available,
    parse_tunnel_url,
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
