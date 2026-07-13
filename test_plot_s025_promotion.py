from __future__ import annotations

import numpy as np

from plot_s025_promotion import promotion_points


def test_promotion_points_follow_ranked_candidates_and_double_standard_errors() -> None:
    summary = {
        "ranked_candidates": ["11", "7"],
        "candidates": {
            "7": {
                "aggregate": {
                    "total": {"change": 0.02, "paired_se": 0.003},
                    "prompt": {"change": 0.05, "paired_se": 0.004},
                    "late": {"change": -0.03, "paired_se": 0.005},
                }
            },
            "11": {
                "aggregate": {
                    "total": {"change": 0.01, "paired_se": 0.006},
                    "prompt": {"change": 0.04, "paired_se": 0.007},
                    "late": {"change": -0.03, "paired_se": 0.008},
                }
            },
        },
    }

    points = promotion_points(summary)

    assert points["labels"].tolist() == ["11", "7"]
    np.testing.assert_allclose(points["total"], [0.01, 0.02])
    np.testing.assert_allclose(points["total_error"], [0.012, 0.006])
    np.testing.assert_allclose(points["prompt"], [0.04, 0.05])
    np.testing.assert_allclose(points["late"], [-0.03, -0.03])
