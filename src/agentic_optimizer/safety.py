from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import statistics
from typing import Any


@dataclass
class AnomalyEvent:
    kind: str
    message: str
    value: float | None = None
    step: int | None = None


class AnomalyDetector:
    def __init__(
        self,
        *,
        grad_explosion_factor: float = 8.0,
        loss_divergence_factor: float = 4.0,
        window: int = 50,
        warmup: int = 10,
    ) -> None:
        self.grad_explosion_factor = grad_explosion_factor
        self.loss_divergence_factor = loss_divergence_factor
        self.window = window
        self.warmup = warmup
        self.reset()

    def update(
        self,
        loss: float | None = None,
        grad_norm: float | None = None,
        step: int | None = None,
    ) -> AnomalyEvent | None:
        """Feed a batch loss and/or grad_norm and return the first detected anomaly."""
        if loss is not None:
            if math.isnan(loss):
                return AnomalyEvent("nan_loss", "loss was NaN", loss, step)
            if math.isinf(loss):
                return AnomalyEvent("inf_loss", "loss was infinite", loss, step)

        if grad_norm is not None:
            if math.isnan(grad_norm):
                return AnomalyEvent("nan_grad", "grad_norm was NaN", grad_norm, step)
            if math.isinf(grad_norm):
                return AnomalyEvent("inf_grad", "grad_norm was infinite", grad_norm, step)

        if (
            grad_norm is not None
            and self._grad_history
            and len(self._grad_history) >= self.warmup
        ):
            median = statistics.median(self._grad_history)
            threshold = self.grad_explosion_factor * median
            if grad_norm > threshold:
                message = (
                    f"grad_norm {grad_norm} exceeded {self.grad_explosion_factor:g}x "
                    f"rolling median {median}"
                )
                return AnomalyEvent("grad_explosion", message, grad_norm, step)

        if (
            loss is not None
            and self._best_loss is not None
            and len(self._loss_history) >= self.warmup
        ):
            threshold = self.loss_divergence_factor * self._best_loss
            if loss > threshold:
                message = (
                    f"loss {loss} exceeded {self.loss_divergence_factor:g}x "
                    f"best loss {self._best_loss}"
                )
                return AnomalyEvent("loss_divergence", message, loss, step)

        if grad_norm is not None:
            self._grad_history.append(grad_norm)
        if loss is not None:
            self._loss_history.append(loss)
            self._best_loss = loss if self._best_loss is None else min(self._best_loss, loss)

        return None

    def reset(self) -> None:
        self._grad_history: deque[float] = deque(maxlen=self.window)
        self._loss_history: deque[float] = deque(maxlen=self.window)
        self._best_loss: float | None = None


@dataclass
class ClampResult:
    value: float
    changed: bool
    reason: str | None = None


class Guardrails:
    def __init__(self, config: dict | None = None) -> None:
        """config shape: {"bounds": {"lr": {"min": 1e-6, "max": 1.0}, ...}, ...}."""
        self._bounds: dict[str, dict[str, float]] = {}
        self._max_rel_change: float | None = None
        if config is not None:
            self.configure(config)

    def configure(self, config: dict) -> None:
        """Merge new bounds / max_rel_change into the existing config (partial update)."""
        bounds = config.get("bounds")
        if isinstance(bounds, dict):
            for name, limits in bounds.items():
                if not isinstance(limits, dict):
                    continue
                existing = self._bounds.setdefault(str(name), {})
                for key in ("min", "max"):
                    if key in limits:
                        existing[key] = limits[key]

        if "max_rel_change" in config:
            self._max_rel_change = config["max_rel_change"]

    def validate(self, name: str, value: float, current: float | None = None) -> ClampResult:
        """Clamp `value` for knob `name` to configured bounds and relative-change limits."""
        if not math.isfinite(value):
            return ClampResult(value, False)

        original = value
        clamped = value
        reason = None
        bounds = self._bounds.get(name, {})

        minimum = bounds.get("min")
        maximum = bounds.get("max")
        if minimum is not None and clamped < minimum:
            clamped = minimum
            reason = f"{name} min"
        if maximum is not None and clamped > maximum:
            clamped = maximum
            reason = f"{name} max"

        if (
            self._max_rel_change is not None
            and current is not None
            and math.isfinite(current)
            and current > 0
            and self._max_rel_change > 0
        ):
            lower = current / self._max_rel_change
            upper = current * self._max_rel_change
            if clamped < lower:
                clamped = lower
                reason = f"{name} max_rel_change down"
            if clamped > upper:
                clamped = upper
                reason = f"{name} max_rel_change up"

        return ClampResult(clamped, clamped != original, reason if clamped != original else None)

    def to_dict(self) -> dict:
        """Return the current config as a plain dict (bounds + max_rel_change)."""
        config: dict[str, Any] = {"bounds": {name: limits.copy() for name, limits in self._bounds.items()}}
        if self._max_rel_change is not None:
            config["max_rel_change"] = self._max_rel_change
        return config
