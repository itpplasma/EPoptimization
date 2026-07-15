from pathlib import Path

import numpy as np

from build_shell_scbo_response import successful_response, taxonomy_indicators


def direct(prompt=0.01, early=-0.02):
    return {
        "prompt": {"change": prompt, "paired_se": 0.02},
        "early": {"change": early, "paired_se": 0.03},
        "short": {"change": prompt + early, "paired_se": 0.025},
    }


def shells():
    common = {
        "classifier": "topology",
        "shell_inner": 0.675,
        "shell_outer": 0.8,
        "simple_sha256": "simple-hash",
        "surfaces": [0.25, 0.675, 0.8],
        "trace_time": 0.02,
    }
    candidate = {
        **common,
        "shift_shell_nonideal_volumes": [0.08, 0.07],
        "wout_sha256": "candidate-hash",
    }
    reference = {
        **common,
        "shift_shell_nonideal_volumes": [0.10, 0.09],
        "wout_sha256": "reference-hash",
    }
    return candidate, reference


def head():
    return {
        "feature": "topology_fixed_outer_shell_volume",
        "slope": 2.0,
        "proxy_reference": 0.095,
        "proxy_floor": 0.05,
        "late_reference": 0.1,
        "status": "passed",
        "target": "loss_1_100ms",
    }


def test_taxonomy_assigns_exact_boundaries_to_later_windows(tmp_path: Path) -> None:
    times = np.array(
        [
            [1, 0.0],
            [2, 0.000099],
            [3, 0.0001],
            [4, 0.000999],
            [5, 0.001],
        ]
    )
    path = tmp_path / "times_lost.dat"
    np.savetxt(path, times)
    windows = taxonomy_indicators(path)
    np.testing.assert_array_equal(windows["prompt"], [False, True, False, False, False])
    np.testing.assert_array_equal(windows["early"], [False, False, True, True, False])
    np.testing.assert_array_equal(
        windows["short"], windows["prompt"] | windows["early"]
    )


def test_shell_response_preserves_unit_weighted_total_and_three_objectives() -> None:
    shell, reference = shells()
    response = successful_response(
        {"candidate_id": 7, "unit_x": [0.2, 0.8], "generation": 2},
        shell,
        reference,
        head(),
        direct(),
        0.004,
        0.002,
        0.005,
        0.005,
        0.0,
    )

    observation = response["observation"]
    assert np.isclose(observation["value"], -0.05)
    assert np.isclose(observation["variance"], 0.025**2)
    np.testing.assert_allclose(observation["constraints"], [-1.0])
    assert response["metrics"]["scalar_constraint_names"] == ["always_feasible"]
    assert not response["metrics"]["loss_component_limits_enforced"]
    np.testing.assert_allclose(
        response["pareto_observation"]["values"], [0.01, -0.02, -0.04]
    )
    assert response["generation"] == 2
    assert response["metrics"]["predicted_total_change"] == observation["value"]
    assert not response["metrics"]["late_floor_active"]


def test_shell_response_caps_late_loss_at_calibration_floor() -> None:
    shell, reference = shells()
    shell["shift_shell_nonideal_volumes"] = [0.04, 0.04]
    response = successful_response(
        {"candidate_id": 9, "unit_x": [0.2, 0.8]},
        shell,
        reference,
        head(),
        direct(prompt=0.0, early=0.0),
        0.004,
        0.002,
        0.005,
        0.005,
        0.0,
    )

    assert response["metrics"]["late_floor_active"]
    np.testing.assert_allclose(response["metrics"]["late_predictions"], [-0.1, -0.1])
    assert response["observation"]["value"] == -0.1
    assert (
        response["observation"]["variance"]
        == direct(0.0, 0.0)["short"]["paired_se"] ** 2
    )


def test_shell_response_allows_loss_window_tradeoffs_in_scalar_search() -> None:
    shell, reference = shells()
    response = successful_response(
        {"candidate_id": 8, "unit_x": [0.4, 0.6]},
        shell,
        reference,
        head(),
        direct(prompt=0.03, early=0.02),
        0.003,
        0.002,
        0.0,
        0.0,
        0.0,
    )

    assert np.isclose(response["observation"]["value"], 0.01)
    assert response["observation"]["constraints"] == [-1.0]
    assert response["metrics"]["gamma_c_s03"] == 0.003


def test_shell_response_uses_short_covariance_for_scalar_variance() -> None:
    shell, reference = shells()
    response = successful_response(
        {"candidate_id": 1, "unit_x": [0.5]},
        shell,
        reference,
        head(),
        direct(),
        0.002,
        0.002,
        0.1,
        0.1,
        0.1,
    )
    assert np.isclose(response["observation"]["variance"], 0.025**2)
    assert not np.isclose(response["observation"]["variance"], 0.02**2 + 0.03**2)


def test_shell_response_rejects_superseded_head() -> None:
    shell, reference = shells()
    with np.testing.assert_raises_regex(ValueError, "another loss window"):
        successful_response(
            {"candidate_id": 1, "unit_x": [0.5]},
            shell,
            reference,
            {**head(), "target": "late_1_10ms_loss"},
            direct(),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
