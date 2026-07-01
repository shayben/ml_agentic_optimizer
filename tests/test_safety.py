import math

import pytest

from agentic_optimizer.safety import AnomalyDetector, ClampResult, Guardrails


@pytest.mark.parametrize(
    ("kwargs", "kind"),
    [
        ({"loss": math.nan}, "nan_loss"),
        ({"loss": math.inf}, "inf_loss"),
        ({"grad_norm": math.nan}, "nan_grad"),
        ({"grad_norm": math.inf}, "inf_grad"),
    ],
)
def test_non_finite_anomalies_fire(kwargs, kind):
    event = AnomalyDetector().update(step=7, **kwargs)

    assert event is not None
    assert event.kind == kind
    assert event.step == 7


def test_grad_explosion_fires_after_warmup():
    detector = AnomalyDetector(grad_explosion_factor=8.0, warmup=3)
    for _ in range(3):
        assert detector.update(grad_norm=2.0) is None

    event = detector.update(grad_norm=17.0, step=4)

    assert event is not None
    assert event.kind == "grad_explosion"
    assert event.value == 17.0
    assert "rolling median" in event.message


def test_loss_divergence_fires_after_warmup():
    detector = AnomalyDetector(loss_divergence_factor=4.0, warmup=3)
    for loss in (10.0, 8.0, 5.0):
        assert detector.update(loss=loss) is None

    event = detector.update(loss=21.0, step=4)

    assert event is not None
    assert event.kind == "loss_divergence"
    assert event.value == 21.0
    assert "best loss 5.0" in event.message


def test_warmup_suppresses_early_rolling_anomalies():
    detector = AnomalyDetector(grad_explosion_factor=2.0, loss_divergence_factor=2.0, warmup=3)

    assert detector.update(loss=10.0, grad_norm=1.0) is None
    assert detector.update(loss=100.0, grad_norm=100.0) is None


def test_finite_only_history_skips_non_finite_values():
    detector = AnomalyDetector(grad_explosion_factor=2.0, loss_divergence_factor=2.0, warmup=2)

    assert detector.update(loss=10.0, grad_norm=10.0) is None
    assert detector.update(loss=math.inf) is not None
    assert detector.update(grad_norm=math.nan) is not None
    assert detector.update(loss=8.0, grad_norm=10.0) is None

    grad_event = detector.update(grad_norm=21.0)
    loss_event = detector.update(loss=17.0)

    assert grad_event is not None
    assert grad_event.kind == "grad_explosion"
    assert loss_event is not None
    assert loss_event.kind == "loss_divergence"


def test_reset_clears_histories_and_best_loss():
    detector = AnomalyDetector(grad_explosion_factor=2.0, loss_divergence_factor=2.0, warmup=2)
    assert detector.update(loss=10.0, grad_norm=10.0) is None
    assert detector.update(loss=8.0, grad_norm=10.0) is None

    detector.reset()

    assert detector.update(loss=17.0, grad_norm=21.0) is None


def test_guardrails_min_max_clamp():
    guardrails = Guardrails({"bounds": {"lr": {"min": 0.001, "max": 1.0}}})

    assert guardrails.validate("lr", 0.0001) == ClampResult(0.001, True, "lr min")
    assert guardrails.validate("lr", 2.0) == ClampResult(1.0, True, "lr max")
    assert guardrails.validate("momentum", 2.0) == ClampResult(2.0, False)


def test_guardrails_max_rel_change_up_and_down():
    guardrails = Guardrails({"max_rel_change": 10.0})

    up = guardrails.validate("lr", 20.0, current=1.0)
    down = guardrails.validate("lr", 0.01, current=1.0)

    assert up == ClampResult(10.0, True, "lr max_rel_change up")
    assert down == ClampResult(0.1, True, "lr max_rel_change down")


def test_guardrails_configure_partially_merges_config():
    guardrails = Guardrails({"bounds": {"lr": {"min": 0.001, "max": 1.0}}})

    guardrails.configure({"bounds": {"lr": {"max": 0.5}, "wd": {"min": 0.0}}})
    guardrails.configure({"max_rel_change": 2.0})

    assert guardrails.to_dict() == {
        "bounds": {"lr": {"min": 0.001, "max": 0.5}, "wd": {"min": 0.0}},
        "max_rel_change": 2.0,
    }


def test_guardrails_to_dict_round_trips():
    config = {"bounds": {"lr": {"min": 1e-6, "max": 1.0}}, "max_rel_change": 10.0}

    assert Guardrails(Guardrails(config).to_dict()).to_dict() == config


def test_guardrails_value_in_range_and_non_finite_are_unchanged():
    guardrails = Guardrails({"bounds": {"lr": {"min": 0.001, "max": 1.0}}})

    assert guardrails.validate("lr", 0.1) == ClampResult(0.1, False)
    assert guardrails.validate("lr", math.inf) == ClampResult(math.inf, False)
