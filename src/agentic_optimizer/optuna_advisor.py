"""Optional Optuna-backed hyperparameter advisor for live agent optimization.

The GitHub Copilot CLI agent can use this module as a principled HPO backend while it
controls a remote training run through MCP tools: call ``suggest()`` to ask Optuna for a
trial, apply the returned parameters to the live run with ``set_hyperparameters``, call
``report()`` with the observed objective metric, and query ``best()`` for the best
completed trial so far. During long trials, the agent may call ``report_intermediate()``
with step metrics to receive pruning guidance.

The advisor is optional and imports Optuna lazily, so base package imports still work
without the HPO extra installed. Install it with ``pip install agentic-optimizer[hpo]``.

``param_space`` is a mapping from parameter name to one of:

* ``{"type": "float", "low": x, "high": y, "log": bool(optional), "step": optional}``
* ``{"type": "int", "low": a, "high": b, "step": optional(default 1), "log": optional}``
* ``{"type": "categorical", "choices": [...]}``
"""

from __future__ import annotations

import importlib
from typing import Any


def optuna_available() -> bool:
    """Return True if Optuna can be imported in the current environment."""
    try:
        importlib.import_module("optuna")
    except ImportError:
        return False
    return True


class OptunaAdvisor:
    """Small wrapper around Optuna's ask/tell API for live training control."""

    def __init__(
        self,
        param_space: dict[str, dict],
        direction: str = "minimize",
        study_name: str | None = None,
        storage: str | None = None,
        sampler: str | None = None,
        pruner: str | None = None,
    ):
        optuna = importlib.import_module("optuna")
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        self._optuna = optuna
        self.param_space = param_space
        self._live_trials: dict[int, Any] = {}
        self.study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            sampler=self._resolve_sampler(sampler),
            pruner=self._resolve_pruner(pruner),
        )

    def suggest(self) -> dict:
        """Ask Optuna for a trial and return the suggested parameter values."""
        trial = self.study.ask()
        params = {
            name: self._suggest_param(trial, name, spec) for name, spec in self.param_space.items()
        }
        self._live_trials[trial.number] = trial
        return {"trial_id": trial.number, "params": params}

    def report(self, trial_id: int, value: float, step: int | None = None) -> dict:
        """Complete a live trial with its final objective value."""
        del step
        trial = self._live_trials.pop(trial_id, None)
        if trial is None:
            return {"error": "unknown trial_id"}
        self.study.tell(trial, value)
        return {"trial_id": trial_id, "state": "complete"}

    def report_intermediate(self, trial_id: int, value: float, step: int) -> dict:
        """Report an intermediate metric and return Optuna's pruning recommendation."""
        trial = self._live_trials.get(trial_id)
        if trial is None:
            return {"error": "unknown trial_id"}

        trial.report(value, step)
        should_prune = trial.should_prune()
        if should_prune:
            self.study.tell(trial, state=self._optuna.trial.TrialState.PRUNED)
            self._live_trials.pop(trial_id, None)
        return {"should_prune": bool(should_prune)}

    def best(self) -> dict | None:
        """Return the best completed trial so far, or None if none is complete."""
        completed = self._optuna.trial.TrialState.COMPLETE
        if not any(trial.state == completed for trial in self.study.trials):
            return None
        return {
            "trial_id": self.study.best_trial.number,
            "value": self.study.best_value,
            "params": dict(self.study.best_params),
        }

    def trials(self) -> list[dict]:
        """Return a compact serializable view of all Optuna trials."""
        return [
            {
                "trial_id": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "params": dict(trial.params),
            }
            for trial in self.study.trials
        ]

    def _resolve_sampler(self, sampler: str | None) -> Any:
        if sampler is None:
            return None

        sampler_name = sampler.lower()
        if sampler_name == "tpe":
            return self._optuna.samplers.TPESampler()
        if sampler_name == "random":
            return self._optuna.samplers.RandomSampler()
        if sampler_name == "cmaes":
            return self._optuna.samplers.CmaEsSampler()
        raise ValueError(f"unknown sampler: {sampler}")

    def _resolve_pruner(self, pruner: str | None) -> Any:
        if pruner is None:
            return None

        pruner_name = pruner.lower()
        if pruner_name == "median":
            return self._optuna.pruners.MedianPruner()
        if pruner_name in {"asha", "successivehalving"}:
            return self._optuna.pruners.SuccessiveHalvingPruner()
        if pruner_name == "hyperband":
            return self._optuna.pruners.HyperbandPruner()
        if pruner_name == "none":
            return self._optuna.pruners.NopPruner()
        raise ValueError(f"unknown pruner: {pruner}")

    @staticmethod
    def _suggest_param(trial: Any, name: str, spec: dict) -> Any:
        param_type = spec.get("type")
        if param_type == "float":
            return trial.suggest_float(
                name,
                spec["low"],
                spec["high"],
                log=spec.get("log", False),
                step=spec.get("step"),
            )
        if param_type == "int":
            return trial.suggest_int(
                name,
                spec["low"],
                spec["high"],
                step=spec.get("step", 1),
                log=spec.get("log", False),
            )
        if param_type == "categorical":
            return trial.suggest_categorical(name, spec["choices"])
        raise ValueError(f"unknown parameter type for {name!r}: {param_type!r}")
