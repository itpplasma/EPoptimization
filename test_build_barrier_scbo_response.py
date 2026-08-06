from __future__ import annotations

import json

import pytest

from build_barrier_scbo_response import base_response, successful_response

REQUEST = {"candidate_id": 7, "unit_x": [0.5, 0.25]}

GEOMETRY = {
    "schema_name": "alpha-loss.direct-geometry",
    "constraint_names": ["mirror_ratio", "max_elongation"],
    "constraint_limits": [0.2, 6.0],
    "constraints": [-0.5, -0.25],
    "feasible": True,
    "metrics": {"aspect": 4.5},
}


def barrier_result(**overrides):
    barrier = {
        "continuous": {
            "jpar": {
                "birth_mean": 0.10,
                "barrier_defect": 0.0875,
                "resolved_fraction_by_surface": [0.9, 0.8],
            },
            "rotation": {
                "birth_mean": 0.03,
                "barrier_defect": 0.04,
                "resolved_fraction_by_surface": [0.85, 0.75],
            },
        },
        "prompt_loss_birth": 0.02,
        "trapped_count_by_surface": [300, 295, 290, 280],
        "particles_per_surface": 1024,
        "s_inner": 0.25,
        "s_outer": 0.7,
    }
    barrier.update(overrides)
    return {
        "schema_name": "alpha-loss.continuous-fast-classifier-result",
        "barrier": barrier,
        "wout_sha256": "abc",
    }


def build(result=None, geometry=None, **kwargs):
    options = {
        "objective": "barrier-jpar",
        "inner_surface": 0.25,
        "outer_surface": 0.7,
        "particles_per_surface": 1024,
        "prompt_limit": 0.05,
    }
    options.update(kwargs)
    return successful_response(
        REQUEST, result or barrier_result(), geometry or GEOMETRY, **options
    )


def test_objective_is_the_continuous_barrier_defect() -> None:
    response = build()
    assert response["observation"]["value"] == pytest.approx(0.0875)
    assert response["status"] == "ok"


def test_prompt_loss_enters_as_a_third_constraint() -> None:
    response = build()
    constraints = response["observation"]["constraints"]
    assert len(constraints) == 3
    # 0.02 / 0.05 - 1 = -0.6, satisfied
    assert constraints[2] == pytest.approx(-0.6)
    assert response["metrics"]["constraint_names"][2] == "prompt_loss"


def test_prompt_loss_above_the_limit_is_a_violation() -> None:
    response = build(barrier_result(prompt_loss_birth=0.10))
    assert response["observation"]["constraints"][2] == pytest.approx(1.0)


def test_surface_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError):
        build(barrier_result(s_outer=0.5))


def test_particle_count_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError):
        build(barrier_result(particles_per_surface=512))


def test_empty_trapped_population_is_rejected() -> None:
    with pytest.raises(ValueError):
        build(barrier_result(trapped_count_by_surface=[300, 0]))


def test_non_finite_overlap_is_rejected() -> None:
    result = barrier_result()
    result["barrier"]["continuous"]["jpar"]["barrier_defect"] = float("nan")
    with pytest.raises(ValueError):
        build(result)


def test_wrong_schema_is_rejected() -> None:
    result = barrier_result()
    result["schema_name"] = "something-else"
    with pytest.raises(ValueError):
        build(result)


def test_failure_response_carries_no_observation() -> None:
    response = base_response(REQUEST, "failed", "equilibrium_failure")
    assert response["observation"] is None
    assert response["failure_kind"] == "equilibrium_failure"
    assert json.dumps(response)


# --- objective selection --------------------------------------------------


def test_raw_and_barrier_objectives_are_selectable() -> None:
    assert build(objective="jpar")["observation"]["value"] == pytest.approx(0.10)
    assert build(objective="rotation")["observation"]["value"] == pytest.approx(0.03)
    assert build(objective="barrier-rotation")["observation"]["value"] == pytest.approx(0.04)


def test_both_continuous_classifiers_are_recorded() -> None:
    response = build(objective="barrier-jpar")
    recorded = response["metrics"]["continuous_fast_classifier"]
    assert set(recorded) == {"jpar", "rotation"}
    assert response["metrics"]["objective_kind"] == "barrier-jpar"


def test_an_objective_without_scores_is_rejected() -> None:
    result = barrier_result()
    result["barrier"]["continuous"] = {}
    with pytest.raises(ValueError, match="no value"):
        build(result, objective="barrier-jpar")


def test_unknown_objective_is_rejected() -> None:
    with pytest.raises(ValueError):
        build(objective="fractal")


def test_a_null_continuous_value_reports_resolution() -> None:
    result = barrier_result()
    result["barrier"]["continuous"]["rotation"]["barrier_defect"] = None
    with pytest.raises(ValueError, match="resolved fractions"):
        build(result, objective="barrier-rotation")
