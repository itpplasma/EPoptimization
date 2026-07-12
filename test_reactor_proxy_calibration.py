#!/usr/bin/env python3
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from reactor_proxy_calibration import candidate_specs, geometry_constraints, parser, write_json


def main() -> None:
    specs = candidate_specs(dimension=5, directions=3, amplitude=0.02, seed=7)
    assert len(specs) == 7
    assert specs[0][0] == "base"
    np.testing.assert_array_equal(specs[0][1], np.zeros(5))
    for index in range(3):
        minus = specs[1 + 2 * index]
        plus = specs[2 + 2 * index]
        assert minus[2:] == (index, -1)
        assert plus[2:] == (index, 1)
        np.testing.assert_allclose(minus[1], -plus[1])
        np.testing.assert_allclose(np.linalg.norm(plus[1]), 0.02)

    base = {
        "aspect": 4.0,
        "mean_iota": -0.3,
        "vacuum_well": -0.2,
        "mirror_ratio": 0.04,
        "max_elongation": 2.5,
        "qa_residual": 0.1,
    }
    args = SimpleNamespace(
        max_aspect_relative=0.02,
        max_iota_change=0.02,
        max_qa_increase=0.01,
        max_mirror_increase=0.02,
        max_elongation_increase=0.2,
    )
    feasible = geometry_constraints(dict(base), base, args)
    assert feasible["feasible"]
    bad = dict(base)
    bad["aspect"] = 4.2
    rejected = geometry_constraints(bad, base, args)
    assert not rejected["feasible"]
    assert not rejected["checks"]["aspect_relative"]

    with tempfile.TemporaryDirectory() as tmp:
        try:
            write_json(Path(tmp) / "bad.json", {"value": float("nan")})
        except ValueError:
            pass
        else:
            raise AssertionError("strict JSON accepted NaN")
    args = parser().parse_args(
        [
            "evaluate",
            "--candidate",
            "/tmp/candidate",
            "--out",
            "/tmp/result",
            "--simple-executable",
            "/tmp/simple.x",
            "--simple-sha256",
            "0" * 64,
            "--desc-python",
            "/tmp/python",
            "--desc-version",
            "test",
            "--desc-source-sha256",
            "1" * 64,
            "--direct-trace-time",
            "0.02",
            "--seed",
            "22345",
        ]
    )
    assert args.class_particles == 3000
    assert args.direct_trace_time == 0.02
    assert args.seed == 22345
    print("test_reactor_proxy_calibration: all checks passed")


if __name__ == "__main__":
    main()
