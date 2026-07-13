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
        "wout_sha256": "candidate-hash",
    }
    reference_topology = {
        **common,
        "classifier": "topology",
        "shift_escape_volumes": [0.10, 0.09],
        "wout_sha256": "reference-hash",
    }
    jpar = {
        **common,
        "classifier": "jpar",
        "shift_escape_volumes": [0.11, 0.08],
        "wout_sha256": "candidate-hash",
    }
    reference_jpar = {
        **common,
        "classifier": "jpar",
        "shift_escape_volumes": [0.10, 0.09],
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
        0.004,
        0.002,
        0.005,
    )

    observation = response["observation"]
    assert np.isclose(observation["value"], -0.02)
    np.testing.assert_allclose(
        observation["constraints"],
        [0.01, -0.01, 0.005, -0.005, 0.015, -0.015, 0.015, -0.015, 0.0],
    )
    assert len(response["metrics"]["constraint_names"]) == 9
    assert response["metrics"]["wout_sha256"] == "candidate-hash"
