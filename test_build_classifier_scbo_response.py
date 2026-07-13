import numpy as np

from build_classifier_scbo_response import successful_response


def test_classifier_response_keeps_late_objective_and_prompt_constraints_separate() -> None:
    request = {"candidate_id": 7, "unit_x": [0.2, 0.8]}
    common = {
        "surfaces": [0.3, 0.425, 0.4875, 0.55, 0.8],
        "simple_sha256": "simple-hash",
        "trace_time": 0.02,
    }
    topology = {
        **common,
        "classifier": "topology",
        "shift_escape_volumes": [0.08, 0.07],
        "shift_nonideal_volumes": [0.18, 0.17],
        "wout_sha256": "candidate-hash",
    }
    reference_topology = {
        **common,
        "classifier": "topology",
        "shift_escape_volumes": [0.10, 0.09],
        "shift_nonideal_volumes": [0.20, 0.19],
        "wout_sha256": "reference-hash",
    }
    jpar = {
        **common,
        "classifier": "jpar",
        "shift_escape_volumes": [0.11, 0.08],
        "shift_nonideal_volumes": [0.21, 0.18],
        "wout_sha256": "candidate-hash",
    }
    reference_jpar = {
        **common,
        "classifier": "jpar",
        "shift_escape_volumes": [0.10, 0.09],
        "shift_nonideal_volumes": [0.20, 0.19],
        "wout_sha256": "reference-hash",
    }
    prompt = {
        "shift_unclassified_fractions": [0.11, 0.10],
        "shift_nonideal_fractions": [0.12, 0.09],
        "shift_jpar_nonideal_fractions": [0.12, 0.09],
        "wout_sha256": "candidate-hash",
        "surface": 0.25,
        "simple_sha256": "simple-hash",
        "trace_time": 0.02,
    }
    reference_prompt = {
        "shift_unclassified_fractions": [0.10, 0.10],
        "shift_nonideal_fractions": [0.10, 0.10],
        "shift_jpar_nonideal_fractions": [0.10, 0.10],
        "wout_sha256": "reference-hash",
        "surface": 0.25,
        "simple_sha256": "simple-hash",
        "trace_time": 0.02,
    }

    response = successful_response(
        request,
        topology,
        reference_topology,
        jpar,
        reference_jpar,
        prompt,
        reference_prompt,
        {
            "angular_status": "passed",
            "fractal_features": [],
            "heldout_status": "passed",
            "horizon_status": "passed",
            "late": {"feature": "late_topology_escape", "slope": 2.0},
            "prompt": {"feature": "prompt_topology_nonideal", "slope": 3.0},
            "radial_status": "passed",
        },
        0.004,
        0.002,
        0.005,
    )

    observation = response["observation"]
    assert np.isclose(observation["value"], -0.04)
    np.testing.assert_allclose(
        observation["constraints"],
        [-0.04, -0.04, 0.055, -0.035, 0.0],
    )
    assert len(response["metrics"]["constraint_names"]) == 5
    assert response["metrics"]["late_feature"] == "late_topology_escape"
    assert response["metrics"]["prompt_feature"] == "prompt_topology_nonideal"
    assert response["metrics"]["wout_sha256"] == "candidate-hash"
