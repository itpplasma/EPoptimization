from pathlib import Path

import numpy as np

import calibrate_classifier_proxy
from calibrate_classifier_proxy import grouped_scalar_fit, predict_case, radial_convergence


def test_grouped_scalar_fit_requires_rank_sign_and_shift_agreement() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = 2.0 * x
    se = np.full(4, 0.1)
    shifts = np.column_stack((0.9 * x, 1.1 * x))
    result = grouped_scalar_fit(x, shifts, y, se)
    assert result["passes"]
    assert result["spearman"] == 1.0
    assert result["sign_correct_fraction"] == 1.0
    assert result["resolved_ranking_correct_fraction"] == 1.0

    shifts[2, 0] *= -1.0
    rejected = grouped_scalar_fit(x, shifts, y, se)
    assert not rejected["passes"]
    assert rejected["shift_concordant"] == [True, True, False, True]


def test_grouped_scalar_fit_rejects_false_safe_predictions() -> None:
    x = np.array([-1.0, -2.0, -3.0, 4.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    se = np.full(4, 0.1)
    shifts = np.column_stack((x, x))
    result = grouped_scalar_fit(x, shifts, y, se)
    assert not result["passes"]
    assert result["sign_correct_fraction"] < 1.0


def test_grouped_scalar_fit_does_not_rank_unresolved_ties() -> None:
    x = np.array([1.0, 4.0, 2.0, 3.0])
    y = np.array([1.00, 1.01, 0.99, 1.02])
    se = np.full(4, 0.1)
    shifts = np.column_stack((x, x))
    result = grouped_scalar_fit(x, shifts, y, se)
    assert result["passes"]
    assert result["resolved_ranking_pairs"] == []
    assert result["resolved_ranking_correct_fraction"] is None


def test_radial_convergence_requires_value_sign_and_order_stability() -> None:
    fine = np.array(
        [[1.0, 1.1], [0.9, 1.0], [0.8, 0.9], [0.7, 0.8], [0.6, 0.7]]
    )
    result = radial_convergence(
        {"coarse": 1.02 * fine, "medium": 1.01 * fine, "fine": fine}
    )
    assert result["passes"]
    bad = fine.copy()
    bad[2, 0] = 1.2
    rejected = radial_convergence(
        {"coarse": 1.02 * fine, "medium": bad, "fine": fine}
    )
    assert not rejected["passes"]


def test_radial_convergence_compares_successive_orderings() -> None:
    rankings = (
        np.array([4, 3, 1, 3, 3, 0, 1]),
        np.array([4, 1, 0, 3, 4, 0, 0]),
        np.array([3, 0, 0, 2, 4, 0, 0]),
    )
    levels = {}
    for name, ranking in zip(("coarse", "medium", "fine"), rankings, strict=True):
        values = np.concatenate(([100.0], 100.0 + 1.0e-3 * (ranking + 10)))
        levels[name] = np.column_stack((values, values))
    result = radial_convergence(levels)
    assert result["passes"]
    assert np.allclose(
        result["ordering_spearman"],
        [0.9019607843137254, 0.9019607843137254, 0.9290701563922508, 0.9290701563922508],
    )


def test_prediction_uses_the_frozen_radial_grid(monkeypatch) -> None:
    calls = []

    def fake_extract(path: Path, surfaces: tuple[str, ...]) -> dict:
        calls.append((path, surfaces))
        return {
            "prompt_topology_nonideal": np.array([0.2, 0.3]),
            "late_topology_escape": np.array([0.4, 0.5]),
            "wout_sha256": "candidate" if path.name == "candidate" else "reference",
        }

    monkeypatch.setattr(calibrate_classifier_proxy, "extract_features", fake_extract)
    frozen = {
        "fractal_features": [],
        "radial_surfaces": ["s0p25000", "s0p45625", "s0p80000"],
        "prompt": {"feature": "prompt_topology_nonideal", "slope": 2.0},
        "late": {"feature": "late_topology_escape", "slope": -3.0},
    }
    result = predict_case(frozen, Path("candidate"), Path("reference"))
    expected = ("s0p25000", "s0p45625", "s0p80000")
    assert calls == [(Path("candidate"), expected), (Path("reference"), expected)]
    assert result["fractal_features"] == []
