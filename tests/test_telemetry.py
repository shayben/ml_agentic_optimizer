import builtins
import sys
from types import SimpleNamespace

import pytest

from agentic_optimizer.contract import GpuTelemetry
from agentic_optimizer.telemetry import compute_grad_norm, gpu_telemetry


def test_gpu_telemetry_returns_none_when_torch_and_nvml_unavailable(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in {"torch", "pynvml"}:
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert gpu_telemetry() is None


def test_gpu_telemetry_populates_util_pct_from_pynvml(monkeypatch):
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    fake_nvml = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda idx: f"handle-{idx}",
        nvmlDeviceGetName=lambda handle: b"Fake GPU",
        nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(used=123_000_000, total=456_000_000),
        nvmlDeviceGetUtilizationRates=lambda handle: SimpleNamespace(gpu=77),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml)

    telemetry = gpu_telemetry()

    assert isinstance(telemetry, GpuTelemetry)
    assert telemetry.device == "Fake GPU"
    assert telemetry.mem_used_mb == 123.0
    assert telemetry.mem_total_mb == 456.0
    assert telemetry.util_pct == 77.0


def test_compute_grad_norm_with_torch():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 1)
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()

    assert compute_grad_norm(model) > 0
