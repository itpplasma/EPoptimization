import numpy as np

from build_direct_scbo_response import jeffreys_variance, successful_response


def test_direct_response_uses_only_measured_total_loss_and_geometry_constraints():
    result = {
        "particles": 256,
        "seed": 12345,
        "birth_surface": 0.25,
        "trace_time": 0.1,
        "wout_sha256": "a" * 64,
        "direct": {
            "prompt_count": 32,
            "late_count": 48,
            "total_count": 80,
            "prompt_loss": 0.125,
            "late_loss": 0.1875,
            "total_loss": 0.3125,
        },
    }
    geometry = {
        "schema_name": "alpha-loss.direct-geometry",
        "constraint_names": ["mirror_ratio", "max_elongation"],
        "constraint_limits": [0.2, 6.0],
        "constraints": [-0.5, -0.25],
        "feasible": True,
        "metrics": {"mirror_ratio": 0.1, "max_elongation": 4.5},
    }
    response = successful_response(
        {"candidate_id": 9, "unit_x": [0.2, 0.8]},
        result,
        geometry,
        particles=256,
        seed=12345,
        birth_surface=0.25,
        trace_time=0.1,
    )
    assert response["observation"]["value"] == 0.3125
    assert response["observation"]["constraints"] == [-0.5, -0.25]
    assert response["metrics"]["objective_name"] == (
        "direct_total_loss_fraction_100ms_s025"
    )


def test_jeffreys_variance_remains_positive_at_zero_and_complete_loss():
    assert jeffreys_variance(0, 256) > 0.0
    assert jeffreys_variance(256, 256) > 0.0
    assert np.isclose(jeffreys_variance(0, 256), jeffreys_variance(256, 256))
