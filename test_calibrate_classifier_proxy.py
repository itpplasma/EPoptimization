import numpy as np

from calibrate_classifier_proxy import grouped_scalar_fit


def test_grouped_scalar_fit_requires_rank_sign_and_shift_agreement() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = 2.0 * x
    se = np.full(4, 0.1)
    shifts = np.column_stack((0.9 * x, 1.1 * x))
    result = grouped_scalar_fit(x, shifts, y, se)
    assert result["passes"]
    assert result["spearman"] == 1.0
    assert result["sign_correct_fraction"] == 1.0

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
