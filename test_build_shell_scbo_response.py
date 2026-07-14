import numpy as np

from build_shell_scbo_response import successful_response


def test_shell_response_targets_prompt_and_predicted_late_loss() -> None:
    request = {"candidate_id": 7, "unit_x": [0.2, 0.8]}
    common = {
        "classifier": "topology",
        "shell_inner": 0.675,
        "shell_outer": 0.8,
        "simple_sha256": "simple-hash",
        "surfaces": [0.25, 0.675, 0.8],
        "trace_time": 0.02,
    }
    shell = {
        **common,
        "shift_shell_nonideal_volumes": [0.08, 0.07],
        "wout_sha256": "candidate-hash",
    }
    reference = {
        **common,
        "shift_shell_nonideal_volumes": [0.10, 0.09],
        "wout_sha256": "reference-hash",
    }

    response = successful_response(
        request,
        shell,
        reference,
        {
            "feature": "topology_fixed_outer_shell_volume",
            "slope": 2.0,
            "status": "passed",
        },
        {"change": 0.01, "paired_se": 0.02},
        0.004,
        0.002,
        0.005,
        0.0,
    )

    observation = response["observation"]
    assert np.isclose(observation["value"], -0.03)
    np.testing.assert_allclose(
        observation["constraints"], [-0.04, -0.04, 0.005, 0.0]
    )
    np.testing.assert_allclose(
        observation["constraint_variances"], [0.0, 0.0, 0.0004, 0.0]
    )
    assert response["metrics"]["predicted_total_change"] == observation["value"]
    assert response["metrics"]["late_feature"] == "topology_fixed_outer_shell_volume"


def test_shell_response_rejects_escape_volume_head() -> None:
    with np.testing.assert_raises_regex(ValueError, "not the fixed outer shell"):
        successful_response(
            {"candidate_id": 1, "unit_x": [0.5]},
            {},
            {},
            {"feature": "late_topology_escape", "slope": 1.0, "status": "passed"},
            {"change": 0.0, "paired_se": 0.0},
            0.0,
            0.0,
            0.0,
            0.0,
        )
