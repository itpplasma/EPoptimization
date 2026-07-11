#!/usr/bin/env python3
import numpy as np

from analyze_reactor_calibration import cross_validate, fit_ridge, paired_se, proposal


def main() -> None:
    names = ["base"] + [
        f"d{index:02d}_{sign}"
        for index in range(4)
        for sign in ("minus", "plus")
    ]
    x = np.array(
        [[0.0, 0.0]]
        + [
            [sign * (index + 1), sign * (4 - index)]
            for index in range(4)
            for sign in (-1.0, 1.0)
        ]
    )
    y = 0.4 + 0.03 * x[:, 0] + 0.01 * x[:, 1]
    model = fit_ridge(x, y, penalty=0.01)
    np.testing.assert_allclose(model["prediction"], y, atol=2e-4)
    agreements, count = cross_validate(names, x, y, penalty=0.01)
    assert (agreements, count) == (4, 4)

    results = {"base": {"candidate_metadata": {"perturbation": [0.0, 0.0]}}}
    for index in range(4):
        vector = np.zeros(2)
        vector[index % 2] = 0.005
        for sign, factor in (("minus", -1.0), ("plus", 1.0)):
            results[f"d{index:02d}_{sign}"] = {
                "candidate_metadata": {
                    "perturbation": (factor * vector).tolist(),
                    "amplitude_over_a": 0.005,
                }
            }
    step = proposal(names, results, model["prediction"])
    np.testing.assert_allclose(np.linalg.norm(step), 0.005)

    first = np.array([1e-3, 2e-3, 3e-1, -1.0])
    second = np.array([1e-3, -1.0, 3e-1, -1.0])
    assert paired_se(first, second, "late") > 0.0
    print("test_analyze_reactor_calibration: all checks passed")


if __name__ == "__main__":
    main()
