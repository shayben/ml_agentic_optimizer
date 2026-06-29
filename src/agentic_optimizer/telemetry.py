"""Standalone telemetry helpers shared by the training bridge (and usable elsewhere).

``torch`` is imported lazily so this module can be imported without PyTorch installed.
"""
from __future__ import annotations

import logging
from typing import Any

from .contract import GpuTelemetry

logger = logging.getLogger("agentic_optimizer.telemetry")


def compute_grad_norm(model: Any, norm_type: float = 2.0) -> float:
    """L2 (by default) norm of all gradients currently on ``model``'s parameters."""
    import torch

    grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    stacked = torch.stack([torch.norm(g, norm_type) for g in grads])
    return float(torch.norm(stacked, norm_type))


def gpu_telemetry() -> GpuTelemetry | None:
    """Best-effort CUDA/NVML snapshot; ``None`` when no GPU telemetry is available."""
    idx: int | None = None
    device: str | None = None
    mem_used_mb: float | None = None
    mem_total_mb: float | None = None
    util_pct: float | None = None
    torch_cuda_available = False

    try:
        import torch

        torch_cuda_available = bool(torch.cuda.is_available())
        if torch_cuda_available:
            idx = int(torch.cuda.current_device())
            props = torch.cuda.get_device_properties(idx)
            device = props.name
            mem_used_mb = float(torch.cuda.memory_allocated(idx) / 1e6)
            mem_total_mb = float(props.total_memory / 1e6)
    except Exception as exc:
        logger.debug("torch CUDA telemetry unavailable", exc_info=exc)

    try:
        import pynvml

        pynvml.nvmlInit()
        if idx is None:
            idx = 0
        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
        try:
            raw_name = pynvml.nvmlDeviceGetName(handle)
            device = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
        except Exception as exc:
            logger.debug("NVML device name unavailable", exc_info=exc)
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used_mb = float(mem.used / 1e6)
            mem_total_mb = float(mem.total / 1e6)
        except Exception as exc:
            logger.debug("NVML memory telemetry unavailable", exc_info=exc)
        util_pct = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
    except Exception as exc:
        util_pct = None
        logger.debug("NVML GPU utilization unavailable", exc_info=exc)

    if not torch_cuda_available and device is None and mem_used_mb is None and mem_total_mb is None:
        return None
    return GpuTelemetry(
        device=device,
        mem_used_mb=mem_used_mb,
        mem_total_mb=mem_total_mb,
        util_pct=util_pct,
    )
