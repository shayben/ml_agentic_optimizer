from __future__ import annotations

import pytest

import agentic_optimizer.profiling as profiling
from agentic_optimizer.profiling import StepProfiler


def set_times(monkeypatch: pytest.MonkeyPatch, values: list[float]) -> None:
    iterator = iter(values)
    monkeypatch.setattr(profiling.time, "perf_counter", lambda: next(iterator))


def test_section_accumulates_elapsed_ms(monkeypatch: pytest.MonkeyPatch) -> None:
    set_times(monkeypatch, [0.0, 1.0, 2.0, 2.1, 2.4, 2.5])
    profiler = StepProfiler()

    with profiler.section("forward"):
        pass
    with profiler.section("forward"):
        pass
    profiler.mark_step()

    summary = profiler.summary()
    assert summary["steps"] == 1
    assert summary["step_ms_avg"] == pytest.approx(1300.0)
    assert summary["sections"] == [
        {"name": "forward", "ms_avg": pytest.approx(1300.0), "pct": pytest.approx(100.0)}
    ]


def test_nested_sections_are_timed_independently(monkeypatch: pytest.MonkeyPatch) -> None:
    set_times(monkeypatch, [0.0, 1.0, 1.2, 1.5, 2.0, 2.1])
    profiler = StepProfiler()

    with profiler.section("outer"):
        with profiler.section("inner"):
            pass
    profiler.mark_step()

    sections = {section["name"]: section for section in profiler.summary()["sections"]}
    assert sections["outer"]["ms_avg"] == pytest.approx(1000.0)
    assert sections["inner"]["ms_avg"] == pytest.approx(300.0)
    assert profiler.summary()["step_ms_avg"] == pytest.approx(1300.0)


def test_mark_step_uses_wall_time_without_sections_and_evicts_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_times(monkeypatch, [0.0, 0.1, 0.3, 0.6])
    profiler = StepProfiler(window=2)

    profiler.mark_step()
    profiler.mark_step()
    profiler.mark_step()

    summary = profiler.summary()
    assert summary["steps"] == 2
    assert summary["step_ms_avg"] == pytest.approx(250.0)
    assert summary["sections"] == []


def test_summary_shape_pct_math_and_empty_state(monkeypatch: pytest.MonkeyPatch) -> None:
    set_times(monkeypatch, [0.0, 0.1, 0.2, 0.3, 0.5, 0.6])
    profiler = StepProfiler()

    assert profiler.summary() == {"steps": 0, "step_ms_avg": None, "sections": []}

    with profiler.section("dataloader"):
        pass
    with profiler.section("forward"):
        pass
    profiler.mark_step()

    summary = profiler.summary()
    assert summary["steps"] == 1
    assert summary["step_ms_avg"] == pytest.approx(300.0)
    assert summary["sections"] == [
        {"name": "forward", "ms_avg": pytest.approx(200.0), "pct": pytest.approx(66.6666667)},
        {"name": "dataloader", "ms_avg": pytest.approx(100.0), "pct": pytest.approx(33.3333333)},
    ]


def test_section_names_are_sorted_and_unique(monkeypatch: pytest.MonkeyPatch) -> None:
    set_times(monkeypatch, [0.0, 1.0, 1.1, 1.2, 1.3, 1.4])
    profiler = StepProfiler()

    with profiler.section("optimizer"):
        pass
    with profiler.section("backward"):
        pass
    profiler.mark_step()

    assert profiler.section_names() == ["backward", "optimizer"]


def test_summary_averages_sections_only_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    set_times(monkeypatch, [0.0, 0.1, 0.2, 0.3, 0.6])
    profiler = StepProfiler()

    with profiler.section("data"):
        pass
    profiler.mark_step()
    profiler.mark_step()

    summary = profiler.summary()
    assert summary["steps"] == 2
    assert summary["step_ms_avg"] == pytest.approx(200.0)
    assert summary["sections"] == [
        {"name": "data", "ms_avg": pytest.approx(100.0), "pct": pytest.approx(50.0)}
    ]


def test_suggest_returns_empty_without_data() -> None:
    assert StepProfiler().suggest() == []


def test_suggest_fires_dataloader_and_low_gpu_util_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_times(monkeypatch, [0.0, 0.1, 0.5, 0.6, 1.2, 1.3])
    profiler = StepProfiler()

    with profiler.section("dataloader"):
        pass
    with profiler.section("forward"):
        pass
    profiler.mark_step()

    assert profiler.suggest(gpu_util_pct=42.3) == [
        "Dataloader is 40% of step time; raise num_workers or prefetch / use faster storage.",
        "GPU utilization is 42%; increase batch_size or enable AMP to better saturate the device.",
    ]


def test_suggest_fires_compute_bound_message(monkeypatch: pytest.MonkeyPatch) -> None:
    set_times(monkeypatch, [0.0, 0.1, 0.9, 1.0])
    profiler = StepProfiler()

    with profiler.section("backward"):
        pass
    profiler.mark_step()

    assert profiler.suggest(gpu_util_pct=90.0) == [
        "Compute-bound; consider AMP (amp=true), a larger batch, or gradient accumulation."
    ]


def test_reset_clears_window_and_current_step(monkeypatch: pytest.MonkeyPatch) -> None:
    set_times(monkeypatch, [0.0, 0.1, 0.2, 0.3, 0.4])
    profiler = StepProfiler()

    with profiler.section("forward"):
        pass
    profiler.reset()
    profiler.mark_step()

    assert profiler.section_names() == []
    assert profiler.summary() == {"steps": 1, "step_ms_avg": pytest.approx(100.0), "sections": []}
