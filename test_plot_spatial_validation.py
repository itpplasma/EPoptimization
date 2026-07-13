import numpy as np

from plot_spatial_validation import validation_points


def test_validation_points_preserve_shift_and_direct_uncertainty() -> None:
    summary = {
        "configurations": {
            "alpes": {
                "direct_late_change": 0.0,
                "direct_late_paired_se": 0.0,
                "spatial_score_change": 0.0,
            },
            "candidate34": {
                "direct_late_change": -0.05,
                "direct_late_paired_se": 0.01,
                "spatial_score_change": -0.002,
            },
        },
        "shift_score_changes": {"candidate34": [-0.006, 0.002]},
    }
    points = validation_points(summary)
    np.testing.assert_array_equal(points["labels"], ["ALPES", "candidate34"])
    np.testing.assert_allclose(points["x"], [0.0, -0.002])
    np.testing.assert_allclose(points["xerr"], [[0.0, 0.004], [0.0, 0.004]])
    np.testing.assert_allclose(points["y"], [0.0, -0.05])
    np.testing.assert_allclose(points["yerr"], [0.0, 0.01])
