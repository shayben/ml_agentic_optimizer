"""Drivers that turn a :class:`~agentic_optimizer.contract.TrainingState` into a ``ControlSignal``.

:class:`CopilotOptimizerDriver` shells out to the GitHub Copilot CLI (``copilot -p``) running in a working
directory that contains ``state.json``; the agent writes ``control.json`` which is parsed back. Failures
(missing CLI, auth error, timeout, malformed output) degrade gracefully to an empty (no-op) control so a
flaky agent never crashes or corrupts training.

:class:`FunctionDriver` wraps a plain Python callable and is used for tests, offline runs, and the demo's
fallback heuristic when the CLI is not configured.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from .contract import ControlSignal, TrainingState

DEFAULT_PROMPT = """\
You are an autonomous in-the-loop optimizer for a neural-network training run.

Read the training snapshot in ./state.json (current epoch/step, recent loss_history, per param-group
lr/weight_decay/momentum, grad_norm, throughput_samples_per_s, gpu telemetry, and optional
per_sample_losses).

Decide whether to adjust optimization, then write ./control.json using EXACTLY this schema and ONLY include
fields you want to change (an empty object {} means "training is healthy, change nothing"):

{
  "set_lr": <float|null>,
  "set_weight_decay": <float|null>,
  "set_momentum": <float|null>,
  "grad_clip": <float|null>,
  "batch_size": <int|null>,
  "enable_augmentation": <bool|null>,
  "flag_noisy_indices": <int[]>,
  "notes": <string|null>
}

Guidance: if loss diverges or grad_norm explodes, reduce lr and/or set grad_clip; on a long plateau, nudge
lr; use per_sample_losses to flag_noisy_indices for suspected mislabeled samples. Be conservative — small,
justified changes only. Write the file with your file-write tool; do not print explanations.
"""


class OptimizerDriver(Protocol):
    """Anything that maps a training snapshot to a control decision."""

    def optimize(self, state: TrainingState) -> ControlSignal:  # pragma: no cover - protocol
        ...


@dataclass
class FunctionDriver:
    """Adapter that turns a ``Callable[[TrainingState], ControlSignal]`` into an :class:`OptimizerDriver`."""

    fn: Callable[[TrainingState], ControlSignal]

    def optimize(self, state: TrainingState) -> ControlSignal:
        result = self.fn(state)
        return result if isinstance(result, ControlSignal) else ControlSignal.empty()


@dataclass
class CopilotOptimizerDriver:
    """Invoke the GitHub Copilot CLI to produce a :class:`ControlSignal`.

    Parameters
    ----------
    copilot_bin:
        Path/name of the ``copilot`` executable.
    model:
        Optional model override passed via ``--model``.
    workdir:
        Directory the CLI runs in (where ``state.json`` is written and ``control.json`` is read). Defaults
        to a fresh temp dir.
    prompt / prompt_path:
        The instruction passed via ``-p``. ``prompt_path`` (e.g. ``agent/optimizer_prompt.md``) overrides
        ``prompt`` when provided.
    timeout_s:
        Hard timeout for each agent consultation.
    extra_env:
        Extra environment variables (e.g. ``COPILOT_GITHUB_TOKEN`` or the ``COPILOT_PROVIDER_*`` BYOK vars).
    """

    copilot_bin: str = "copilot"
    model: str | None = None
    workdir: str | Path | None = None
    prompt: str = DEFAULT_PROMPT
    prompt_path: str | Path | None = None
    timeout_s: float = 180.0
    extra_env: dict[str, str] = field(default_factory=dict)
    state_filename: str = "state.json"
    control_filename: str = "control.json"
    allow_all_paths: bool = True
    last_returncode: int | None = field(default=None, init=False)
    last_stderr: str = field(default="", init=False)

    def __post_init__(self) -> None:
        if self.workdir is None:
            self.workdir = Path(tempfile.mkdtemp(prefix="agentic-opt-"))
        else:
            self.workdir = Path(self.workdir)
            self.workdir.mkdir(parents=True, exist_ok=True)
        if self.prompt_path is not None:
            self.prompt = Path(self.prompt_path).read_text(encoding="utf-8")

    def _build_cmd(self) -> list[str]:
        cmd = [
            self.copilot_bin,
            "-p",
            self.prompt,
            "--allow-all-tools",
            "--no-ask-user",
            "-s",
            "--no-color",
            "--no-auto-update",
        ]
        if self.allow_all_paths:
            cmd.append("--allow-all-paths")
        if self.model:
            cmd += ["--model", self.model]
        return cmd

    def optimize(self, state: TrainingState) -> ControlSignal:
        workdir = Path(self.workdir)
        state_path = workdir / self.state_filename
        control_path = workdir / self.control_filename
        state.write_json(state_path)
        if control_path.exists():
            control_path.unlink()

        env = {**os.environ, **self.extra_env}
        try:
            proc = subprocess.run(
                self._build_cmd(),
                cwd=str(workdir),
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
            self.last_returncode = proc.returncode
            self.last_stderr = proc.stderr or ""
        except FileNotFoundError:
            # copilot binary not on PATH
            self.last_returncode = None
            self.last_stderr = f"copilot binary not found: {self.copilot_bin!r}"
            return ControlSignal.empty()
        except subprocess.TimeoutExpired:
            self.last_returncode = None
            self.last_stderr = "copilot timed out"
            return ControlSignal.empty()

        return self._read_control(control_path, fallback_stdout=proc.stdout)

    def _read_control(self, control_path: Path, fallback_stdout: str = "") -> ControlSignal:
        if control_path.exists():
            try:
                return ControlSignal.read_json(control_path)
            except Exception:
                try:
                    return ControlSignal.from_text(control_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
        # The agent may have printed the JSON instead of writing the file.
        if fallback_stdout:
            return ControlSignal.from_text(fallback_stdout)
        return ControlSignal.empty()
