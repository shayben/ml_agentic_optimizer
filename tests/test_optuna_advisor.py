import pytest

pytest.importorskip("optuna")

from agentic_optimizer.optuna_advisor import OptunaAdvisor, optuna_available


PARAM_SPACE = {
    "lr": {"type": "float", "low": 1e-4, "high": 1e-1, "log": True},
    "wd": {"type": "float", "low": 0.0, "high": 1e-3},
    "layers": {"type": "int", "low": 1, "high": 4},
    "activation": {"type": "categorical", "choices": ["relu", "gelu"]},
}


def _objective(params: dict) -> float:
    activation_penalty = 0.0 if params["activation"] == "gelu" else 0.01
    return (
        (params["lr"] - 0.01) ** 2
        + abs(params["wd"] - 1e-4)
        + ((params["layers"] - 2) ** 2) * 0.001
        + activation_penalty
    )


def test_optuna_available_is_true_when_test_runs():
    assert optuna_available() is True


def test_suggest_report_and_best_trial():
    advisor = OptunaAdvisor(PARAM_SPACE, sampler="random", pruner="none")

    for _ in range(12):
        suggestion = advisor.suggest()
        params = suggestion["params"]
        advisor.report(suggestion["trial_id"], _objective(params))

    best = advisor.best()

    assert best is not None
    assert {"trial_id", "value", "params"} <= set(best)
    assert set(best["params"]) == set(PARAM_SPACE)
    assert isinstance(best["value"], float)
    assert 1e-4 <= best["params"]["lr"] <= 1e-1
    assert 0.0 <= best["params"]["wd"] <= 1e-3
    assert 1 <= best["params"]["layers"] <= 4
    assert best["params"]["activation"] in {"relu", "gelu"}
    assert len(advisor.trials()) == 12


def test_report_intermediate_returns_pruning_decision():
    advisor = OptunaAdvisor(PARAM_SPACE, pruner="none")
    suggestion = advisor.suggest()

    result = advisor.report_intermediate(suggestion["trial_id"], value=1.0, step=1)

    assert isinstance(result["should_prune"], bool)
    advisor.report(suggestion["trial_id"], 1.0)


def test_unknown_trial_id_error_path():
    advisor = OptunaAdvisor(PARAM_SPACE)

    assert advisor.report(12345, 1.0) == {"error": "unknown trial_id"}
