import numpy as np

from calibrate_classifier_proxy import grouped_scalar_fit, radial_convergence


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
