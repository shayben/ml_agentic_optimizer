"""agentic_optimizer.

Drive an existing PyTorch training loop with the GitHub Copilot CLI acting as the optimizer agent.

The top-level import is intentionally lightweight (no ``torch`` import) so the schemas and driver can be
used in environments without PyTorch. ``AgenticCallback`` is imported lazily on first access.
"""
from .contract import (
    AnomalyEvent,
    CheckpointInfo,
    Command,
    CommandResult,
    CommandStatus,
    ControlSignal,
    DistributedInfo,
    GpuTelemetry,
    GuardrailBound,
    GuardrailConfig,
    Hyperparameters,
    KnobSpec,
    MlflowInfo,
    ParamGroupState,
    PerSampleLoss,
    ProfileSection,
    ProfileSummary,
    SchedulerState,
    Telemetry,
    TrainingConfig,
    TrainingState,
)
from .controlplane import ControlPlaneClient, ControlPlaneStore, create_app
from .driver import (
    DEFAULT_PROMPT,
    CopilotOptimizerDriver,
    FunctionDriver,
    OptimizerDriver,
)

__all__ = [
    "TrainingState",
    "ControlSignal",
    "ParamGroupState",
    "GpuTelemetry",
    "PerSampleLoss",
    "Command",
    "CommandResult",
    "CommandStatus",
    "Hyperparameters",
    "KnobSpec",
    "MlflowInfo",
    "Telemetry",
    "TrainingConfig",
    "SchedulerState",
    "ProfileSection",
    "ProfileSummary",
    "DistributedInfo",
    "CheckpointInfo",
    "AnomalyEvent",
    "GuardrailBound",
    "GuardrailConfig",
    "ControlPlaneStore",
    "ControlPlaneClient",
    "create_app",
    "CopilotOptimizerDriver",
    "FunctionDriver",
    "OptimizerDriver",
    "DEFAULT_PROMPT",
    "AgenticCallback",
    "TrainingBridge",
    "NoOpBridge",
    "HandlerRegistry",
    "attach",
    "OptunaAdvisor",
    "optuna_available",
]


def __getattr__(name: str):  # pragma: no cover - thin lazy import shim
    if name == "AgenticCallback":
        from .callback import AgenticCallback

        return AgenticCallback
    if name in ("TrainingBridge", "HandlerRegistry", "NoOpBridge", "attach"):
        from . import bridge

        return getattr(bridge, name)
    if name in ("OptunaAdvisor", "optuna_available"):
        from . import optuna_advisor

        return getattr(optuna_advisor, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
