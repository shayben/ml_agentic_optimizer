from __future__ import annotations

from typing import Any


def _dist() -> Any | None:
    """Return initialized torch.distributed, or None if unavailable."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist
    except Exception:
        return None
    return None


def is_available() -> bool:
    """Return True when torch.distributed is initialized and usable."""
    return _dist() is not None


def rank() -> int:
    """Return the global rank, or 0 in single-process mode."""
    try:
        dist = _dist()
        if dist is not None:
            return int(dist.get_rank())
    except Exception:
        return 0
    return 0


def world_size() -> int:
    """Return the world size, or 1 in single-process mode."""
    try:
        dist = _dist()
        if dist is not None:
            return int(dist.get_world_size())
    except Exception:
        return 1
    return 1


def is_main_process() -> bool:
    """Return True when this process is rank 0."""
    return rank() == 0


def backend() -> str | None:
    """Return the active distributed backend name, if initialized."""
    try:
        dist = _dist()
        if dist is not None:
            return str(dist.get_backend())
    except Exception:
        return None
    return None


def barrier() -> None:
    """Synchronize ranks when initialized; otherwise do nothing."""
    try:
        dist = _dist()
        if dist is not None:
            dist.barrier()
    except Exception:
        return


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """Broadcast a picklable object from src to all ranks."""
    try:
        dist = _dist()
        if dist is None:
            return obj
        objects = [obj]
        dist.broadcast_object_list(objects, src=src)
        return objects[0]
    except Exception:
        return obj


def all_reduce_mean(value: float) -> float:
    """Return the mean of value across ranks, or value in single-process mode."""
    try:
        dist = _dist()
        if dist is None:
            return float(value)

        import torch

        tensor_kwargs: dict[str, Any] = {"dtype": torch.float64}
        if str(dist.get_backend()).lower() == "nccl" and torch.cuda.is_available():
            tensor_kwargs["device"] = torch.device("cuda", torch.cuda.current_device())
        tensor = torch.tensor(float(value), **tensor_kwargs)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return float((tensor / dist.get_world_size()).item())
    except Exception:
        return float(value)


def info() -> dict:
    """Return distributed status information."""
    return {
        "enabled": is_available(),
        "rank": rank(),
        "world_size": world_size(),
        "backend": backend(),
    }
