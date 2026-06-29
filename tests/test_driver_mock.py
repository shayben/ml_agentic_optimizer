import subprocess
from pathlib import Path

import agentic_optimizer.driver as drv
from agentic_optimizer.contract import ControlSignal, TrainingState
from agentic_optimizer.driver import CopilotOptimizerDriver, FunctionDriver


def test_function_driver():
    d = FunctionDriver(lambda s: ControlSignal(set_lr=0.01))
    assert d.optimize(TrainingState()).set_lr == 0.01


def test_function_driver_non_control_returns_empty():
    d = FunctionDriver(lambda s: "oops")  # type: ignore[arg-type]
    assert d.optimize(TrainingState()).is_empty()


def test_copilot_driver_parses_written_control(tmp_path, monkeypatch):
    d = CopilotOptimizerDriver(workdir=tmp_path)

    def fake_run(cmd, cwd, env, capture_output, text, timeout, check):
        (Path(cwd) / "control.json").write_text(
            '{"set_lr": 0.03, "grad_clip": 1.5}', encoding="utf-8"
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(drv.subprocess, "run", fake_run)
    ctrl = d.optimize(TrainingState(step=1))
    assert ctrl.set_lr == 0.03 and ctrl.grad_clip == 1.5
    assert (tmp_path / "state.json").exists()


def test_copilot_driver_empty_when_no_output(tmp_path, monkeypatch):
    d = CopilotOptimizerDriver(workdir=tmp_path)

    def fake_run(cmd, cwd, env, capture_output, text, timeout, check):
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(drv.subprocess, "run", fake_run)
    assert d.optimize(TrainingState()).is_empty()


def test_copilot_driver_stdout_fallback(tmp_path, monkeypatch):
    d = CopilotOptimizerDriver(workdir=tmp_path)

    def fake_run(cmd, cwd, env, capture_output, text, timeout, check):
        return subprocess.CompletedProcess(cmd, 0, stdout='{"set_momentum": 0.8}', stderr="")

    monkeypatch.setattr(drv.subprocess, "run", fake_run)
    assert d.optimize(TrainingState()).set_momentum == 0.8


def test_copilot_driver_missing_binary(tmp_path):
    d = CopilotOptimizerDriver(workdir=tmp_path, copilot_bin="copilot-does-not-exist-xyz")
    assert d.optimize(TrainingState()).is_empty()


def test_copilot_driver_uses_prompt_path(tmp_path):
    prompt_file = tmp_path / "p.md"
    prompt_file.write_text("CUSTOM PROMPT", encoding="utf-8")
    d = CopilotOptimizerDriver(workdir=tmp_path, prompt_path=prompt_file)
    assert d.prompt == "CUSTOM PROMPT"
    assert "CUSTOM PROMPT" in d._build_cmd()
