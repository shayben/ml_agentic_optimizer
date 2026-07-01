"""The state/control contract exchanged with the Copilot CLI optimizer agent.

``TrainingState`` is written by the training loop (via :class:`~agentic_optimizer.callback.AgenticCallback`)
to ``state.json``. The agent reads it, decides, and writes ``control.json`` which is parsed back into a
:class:`ControlSignal` and applied to the optimizer.

Both models are plain pydantic v2 models with convenience JSON read/write helpers so the contract has a
single source of truth shared by the Python glue and the agent's prompt/instructions.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ParamGroupState(BaseModel):
    """Per optimizer param-group view of the tunable hyperparameters."""

    lr: float
    weight_decay: float | None = None
    momentum: float | None = None


class GpuTelemetry(BaseModel):
    """Best-effort GPU utilization snapshot (populated when CUDA is available)."""

    device: str | None = None
    mem_used_mb: float | None = None
    mem_total_mb: float | None = None
    util_pct: float | None = None


class PerSampleLoss(BaseModel):
    """A single (sample-index, loss) pair, used as a label-noise signal."""

    index: int
    loss: float


class SchedulerState(BaseModel):
    """Snapshot of the LR scheduler so the agent can see (and replace) the schedule."""

    name: str | None = None
    last_lr: list[float] = Field(default_factory=list)
    last_epoch: int | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class ProfileSection(BaseModel):
    """One timed region of the training step (e.g. dataloader/forward/backward/optimizer)."""

    name: str
    ms_avg: float
    pct: float


class ProfileSummary(BaseModel):
    """Per-region step-time breakdown for hardware/throughput root-cause analysis."""

    steps: int = 0
    step_ms_avg: float | None = None
    sections: list[ProfileSection] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class DistributedInfo(BaseModel):
    """DDP/multi-rank context for the run (single-process: enabled=False, world_size=1)."""

    enabled: bool = False
    rank: int = 0
    world_size: int = 1
    backend: str | None = None


class TrainingState(BaseModel):
    """Snapshot of training health written to ``state.json`` for the agent to read."""

    step: int = 0
    epoch: int = 0
    max_epochs: int | None = None
    timestamp: float | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    loss_history: list[float] = Field(default_factory=list, max_length=200_000)
    param_groups: list[ParamGroupState] = Field(default_factory=list)
    grad_norm: float | None = None
    throughput_samples_per_s: float | None = None
    gpu: GpuTelemetry | None = None
    per_sample_losses: list[PerSampleLoss] = Field(default_factory=list, max_length=200_000)
    scheduler: SchedulerState | None = None
    profile: ProfileSummary | None = None
    distributed: DistributedInfo | None = None
    stop_requested: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    def write_json(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")
        return p

    @classmethod
    def read_json(cls, path: str | Path) -> "TrainingState":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


class ControlSignal(BaseModel):
    """Hyperparameter overrides / flags the agent writes to ``control.json``.

    Every field is optional; ``None`` (or an empty list) means "leave unchanged". An all-empty signal is a
    no-op, which is the correct output when training is healthy.
    """

    set_lr: float | None = None
    set_weight_decay: float | None = None
    set_momentum: float | None = None
    grad_clip: float | None = None
    batch_size: int | None = None
    enable_augmentation: bool | None = None
    flag_noisy_indices: list[int] = Field(default_factory=list)
    notes: str | None = None

    def is_empty(self) -> bool:
        return (
            self.set_lr is None
            and self.set_weight_decay is None
            and self.set_momentum is None
            and self.grad_clip is None
            and self.batch_size is None
            and self.enable_augmentation is None
            and not self.flag_noisy_indices
        )

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    def write_json(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")
        return p

    @classmethod
    def read_json(cls, path: str | Path) -> "ControlSignal":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    @classmethod
    def from_text(cls, text: str) -> "ControlSignal":
        """Parse a control signal from arbitrary agent output.

        Tolerant of extra prose around the JSON object: extracts the first balanced ``{...}`` block.
        """
        text = text.strip()
        if not text:
            return cls()
        try:
            return cls.model_validate(json.loads(text))
        except Exception:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return cls.model_validate(json.loads(text[start : end + 1]))
            except Exception:
                pass
        return cls()

    @classmethod
    def empty(cls) -> "ControlSignal":
        return cls()


# ---------------------------------------------------------------------------
# MCP control-plane contract (v2): live agent <-> remote training-job messaging
# ---------------------------------------------------------------------------
import time  # noqa: E402
import uuid  # noqa: E402
from enum import Enum  # noqa: E402


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


class CommandStatus(str, Enum):
    """Lifecycle of a command flowing agent -> broker -> training bridge -> back."""

    pending = "pending"
    in_progress = "in_progress"
    done = "done"
    failed = "failed"


class CommandResult(BaseModel):
    """Result the training bridge posts back for a command (incl. interrogations)."""

    command_id: str
    ok: bool = True
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    applied_at: float | None = None


class Command(BaseModel):
    """An action the agent asks the live training run to perform.

    ``type`` is a free-form action name. Built-ins handled specially by the bridge:
    ``set_hyperparameters``, ``pause``, ``resume``, ``set_augmentation``, ``flag_samples``,
    ``run_evaluation``. Any other ``type`` (or ``set_knob`` / ``invoke`` / ``interrogate``) is routed to a
    user-registered handler — this is how the agent can influence *anything* in the loop.
    """

    id: str = Field(default_factory=_new_id)
    type: str
    args: dict[str, Any] = Field(default_factory=dict)
    status: CommandStatus = CommandStatus.pending
    run_id: str = "default"
    created_at: float = Field(default_factory=time.time)
    claimed_at: float | None = None
    lease_expires_at: float | None = None
    attempts: int = 0
    completed_at: float | None = None
    result: CommandResult | None = None


class CommandRequest(BaseModel):
    """Request body the agent/MCP server posts to enqueue a command."""

    type: str
    args: dict[str, Any] = Field(default_factory=dict)
    run_id: str = "default"


class Hyperparameters(BaseModel):
    """Hyperparameter overrides applied to every optimizer param group."""

    lr: float | None = None
    weight_decay: float | None = None
    momentum: float | None = None
    extra: dict[str, float] = Field(default_factory=dict)

    def is_empty(self) -> bool:
        return (
            self.lr is None
            and self.weight_decay is None
            and self.momentum is None
            and not self.extra
        )


class TrainingConfig(BaseModel):
    """Throughput / hardware-utilization levers, applied via the ``set_training_config`` command.

    The bridge records these and invokes a user-supplied callback that actually rebuilds the
    ``DataLoader`` (``batch_size``/``num_workers``), toggles AMP, or changes gradient accumulation. Only
    non-null fields are applied; the rest are left unchanged.
    """

    batch_size: int | None = None
    num_workers: int | None = None
    grad_accum_steps: int | None = None
    amp: bool | None = None

    def is_empty(self) -> bool:
        return (
            self.batch_size is None
            and self.num_workers is None
            and self.grad_accum_steps is None
            and self.amp is None
        )


class KnobSpec(BaseModel):
    """A named, agent-controllable knob the bridge advertises (custom loop influence point)."""

    name: str
    description: str = ""
    value: Any = None


class MlflowInfo(BaseModel):
    """Linkage to the training run's MLflow run, surfaced to the agent."""

    run_id: str | None = None
    tracking_uri: str | None = None
    experiment: str | None = None
    run_name: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)


class CheckpointInfo(BaseModel):
    """A checkpoint the bridge saved and can roll the live run back to."""

    id: str
    step: int = 0
    epoch: int = 0
    created_at: float = Field(default_factory=time.time)
    metrics: dict[str, float] = Field(default_factory=dict)
    path: str | None = None
    note: str | None = None


class AnomalyEvent(BaseModel):
    """A training-health anomaly the bridge detected (NaN/Inf, grad explosion, divergence)."""

    kind: str
    message: str
    value: float | None = None
    step: int | None = None
    epoch: int | None = None
    at: float = Field(default_factory=time.time)


class GuardrailBound(BaseModel):
    """Inclusive [min, max] bound for a single guardrailed knob (either may be null)."""

    min: float | None = None
    max: float | None = None


class GuardrailConfig(BaseModel):
    """Bounds + max relative-change-per-call the bridge enforces on hyperparameter mutations."""

    bounds: dict[str, GuardrailBound] = Field(default_factory=dict)
    max_rel_change: float | None = None


class Telemetry(BaseModel):
    """A telemetry snapshot pushed by the bridge: training state + MLflow linkage + flags."""

    run_id: str = "default"
    state: TrainingState = Field(default_factory=TrainingState)
    mlflow: MlflowInfo | None = None
    paused: bool = False
    knobs: list[KnobSpec] = Field(default_factory=list)
    last_error: str | None = None
    checkpoints: list[CheckpointInfo] = Field(default_factory=list)
    anomalies: list[AnomalyEvent] = Field(default_factory=list)
    guardrails: GuardrailConfig | None = None
    stopping: bool = False

