from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_TOTAL_KEY = "__total_ms__"


class StepProfiler:
    def __init__(self, *, window: int = 50, sync_cuda: bool = False) -> None:
        """Track per-section wall time and full-step time over a rolling window."""
        if window < 1:
            raise ValueError("window must be >= 1")
        self._steps: deque[dict[str, float]] = deque(maxlen=window)
        self._current: dict[str, float] = {}
        self._last_mark = time.perf_counter()
        self._sync_cuda = sync_cuda

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        """Time a named region of the current step."""
        self._cuda_synchronize()
        started = time.perf_counter()
        try:
            yield
        finally:
            self._cuda_synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self._current[name] = self._current.get(name, 0.0) + elapsed_ms

    def mark_step(self) -> None:
        """Close the current step and add it to the rolling window."""
        now = time.perf_counter()
        step = dict(self._current)
        if step:
            total_ms = sum(step.values())
        else:
            total_ms = (now - self._last_mark) * 1000.0
        step[_TOTAL_KEY] = total_ms
        self._steps.append(step)
        self._current = {}
        self._last_mark = now

    def section_names(self) -> list[str]:
        """All section names seen in the current window, sorted."""
        names = {
            name
            for step in self._steps
            for name in step
            if name != _TOTAL_KEY
        }
        return sorted(names)

    def summary(self) -> dict[str, Any]:
        """Summarize average step time and per-section average time in milliseconds."""
        if not self._steps:
            return {"steps": 0, "step_ms_avg": None, "sections": []}

        step_count = len(self._steps)
        step_ms_avg = sum(step[_TOTAL_KEY] for step in self._steps) / step_count
        sections = []
        for name in self.section_names():
            values = [step[name] for step in self._steps if name in step]
            ms_avg = sum(values) / len(values)
            pct = (ms_avg / step_ms_avg * 100.0) if step_ms_avg else 0.0
            sections.append({"name": name, "ms_avg": ms_avg, "pct": pct})

        sections.sort(key=lambda section: section["ms_avg"], reverse=True)
        return {"steps": step_count, "step_ms_avg": step_ms_avg, "sections": sections}

    def suggest(self, gpu_util_pct: float | None = None) -> list[str]:
        """Return short recommendations based on the current timing summary."""
        summary = self.summary()
        if summary["steps"] == 0 or summary["step_ms_avg"] is None:
            return []

        suggestions: list[str] = []
        sections = summary["sections"]
        data_section = next(
            (
                section
                for section in sections
                if section["name"].lower() in {"data", "dataloader"}
            ),
            None,
        )
        if data_section is not None and data_section["pct"] > 30.0:
            pct = round(data_section["pct"])
            suggestions.append(
                f"Dataloader is {pct}% of step time; raise num_workers or prefetch / "
                "use faster storage."
            )

        if gpu_util_pct is not None and gpu_util_pct < 50.0:
            pct = round(gpu_util_pct)
            suggestions.append(
                f"GPU utilization is {pct}%; increase batch_size or enable AMP to better "
                "saturate the device."
            )

        compute_section = next(
            (
                section
                for section in sections
                if section["name"].lower() in {"backward", "forward"} and section["pct"] > 70.0
            ),
            None,
        )
        if compute_section is not None and gpu_util_pct is not None and gpu_util_pct >= 70.0:
            suggestions.append(
                "Compute-bound; consider AMP (amp=true), a larger batch, or gradient accumulation."
            )

        return suggestions

    def reset(self) -> None:
        """Clear all recorded timing state."""
        self._steps.clear()
        self._current = {}
        self._last_mark = time.perf_counter()

    def _cuda_synchronize(self) -> None:
        if not self._sync_cuda:
            return
        try:
            import torch

            torch.cuda.synchronize()
        except Exception:
            return
