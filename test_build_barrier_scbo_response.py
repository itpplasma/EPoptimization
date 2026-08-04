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
        "barrier_overlap": 0.125,
        "classifier": "topology",
        "prompt_loss_inner": 0.02,
        "trapped_inner": 300,
        "trapped_outer": 280,
        "particles_per_surface": 1024,
        "s_inner": 0.25,
        "s_outer": 0.6,
    }
    barrier.update(overrides)
    return {
        "schema_name": "alpha-loss.barrier-overlap-result",
        "barrier": barrier,
        "wout_sha256": "abc",
    }


def build(result=None, geometry=None, **kwargs):
    options = {
        "inner_surface": 0.25,
        "outer_surface": 0.6,
        "particles_per_surface": 1024,
        "prompt_limit": 0.05,
    }
    options.update(kwargs)
    return successful_response(
        REQUEST, result or barrier_result(), geometry or GEOMETRY, **options
    )


def test_objective_is_the_barrier_overlap() -> None:
    response = build()
    assert response["observation"]["value"] == pytest.approx(0.125)
    assert response["status"] == "ok"


def test_prompt_loss_enters_as_a_third_constraint() -> None:
    response = build()
    constraints = response["observation"]["constraints"]
    assert len(constraints) == 3
    # 0.02 / 0.05 - 1 = -0.6, satisfied
    assert constraints[2] == pytest.approx(-0.6)
    assert response["metrics"]["constraint_names"][2] == "prompt_loss"


def test_prompt_loss_above_the_limit_is_a_violation() -> None:
    response = build(barrier_result(prompt_loss_inner=0.10))
    assert response["observation"]["constraints"][2] == pytest.approx(1.0)


def test_surface_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError):
        build(barrier_result(s_outer=0.5))


def test_particle_count_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError):
        build(barrier_result(particles_per_surface=512))


def test_empty_trapped_population_is_rejected() -> None:
    with pytest.raises(ValueError):
        build(barrier_result(trapped_outer=0))


def test_non_finite_overlap_is_rejected() -> None:
    with pytest.raises(ValueError):
        build(barrier_result(barrier_overlap=float("nan")))


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
