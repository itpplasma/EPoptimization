#!/usr/bin/env python
"""Check barrier_overlap and chaotic_fraction against hand-computed values.

Protects the optimization objective: a wrong port of the sensopt metric would
silently drive the optimizer toward something other than the validated
barrier-breach fraction.
"""
import tempfile
from pathlib import Path

import numpy as np

from evaluate_barrier_proxy import parser as barrier_parser
from simple_barrier import (
    barrier_overlap,
    chaotic_fraction,
    classification_loss_metrics,
    composite_proxy,
    loss_windows,
    paired_barrier_bootstrap,
)


def write_class_parts(
    directory: Path,
    mu: list,
    topology: list,
    trap_par: list | None = None,
    jpar: list | None = None,
) -> Path:
    if jpar is None:
        jpar = topology
    rows = [
        [i + 1, 0.3, m, j, t, 0]
        for i, (m, j, t) in enumerate(zip(mu, jpar, topology))
    ]
    if trap_par is None:
        trap_par = [1.0] * len(rows)
    directory.mkdir(parents=True, exist_ok=True)
    np.savetxt(directory / "class_parts.dat", np.array(rows, dtype=float))
    particles = np.zeros((len(rows), 3))
    particles[:, 0] = np.arange(1, len(rows) + 1)
    particles[:, 1] = -1.0
    particles[:, 2] = trap_par
    np.savetxt(directory / "times_lost.dat", particles)
    return directory


def main():
    args = barrier_parser().parse_args(
        [
            "--wout",
            "/tmp/wout.nc",
            "--out",
            "/tmp/output",
            "--simple-executable",
            "/tmp/simple.x",
            "--simple-sha256",
            "0" * 64,
            "--wout-sha256",
            "1" * 64,
        ]
    )
    assert args.particles == 3000
    assert args.seed == 12345
    assert (args.inner_surface, args.outer_surface, args.bins) == (0.3, 0.6, 16)
    windows = loss_windows(
        np.array([-1.0, 5e-4, 1e-3, 2e-3, 0.3, 0.31]),
        prompt_time=1e-3,
        final_time=0.3,
    )
    assert windows["prompt_count"] == 2
    assert windows["late_count"] == 1
    assert windows["total_count"] == 3
    assert windows["total_loss"] == windows["prompt_loss"] + windows["late_loss"]
    rounded_endpoint = np.nextafter(0.3, 0.0)
    windows = loss_windows(
        np.array([2e-3, rounded_endpoint]), final_time=rounded_endpoint
    )
    assert windows["late_count"] == 1
    assert windows["total_count"] == 1
    assert composite_proxy(0.1, 0.4, short_weight=0.25) == 0.2
    assert composite_proxy(
        0.1,
        0.5,
        short_weight=0.25,
        short_limit=0.4,
        excess_penalty=100.0,
    ) > 1.2

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        # Inner surface: 8 trapped plus one forced-regular passing particle.
        inner = write_class_parts(
            base / "inner",
            mu=[0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.5],
            topology=[2, 1, 2, 1, 1, 2, 1, 1, 1],
            trap_par=[1, 1, 1, 1, 1, 1, 1, 1, -1],
            jpar=[1] * 9,
        )
        # Outer (barrier) surface: 6 trapped.
        outer = write_class_parts(
            base / "outer",
            mu=[0.1, 0.3, 0.45, 0.55, 0.7, 0.9],
            topology=[2, 1, 2, 1, 2, 1],
        )

        # nbins=2 on common mu range [0.1, 0.9]: edges 0.1 | 0.5 | 0.9.
        # bin 1: inner topo [2,1,2,1] -> p_birth_chaotic = 2/8;
        #        outer topo [2,1,2]   -> breach 2/3
        # bin 2: inner topo [1,2,1,1] -> p_birth_chaotic = 1/8;
        #        outer topo [1,2,1]   -> breach 1/3
        expected = (2 / 8) * (2 / 3) + (1 / 8) * (1 / 3)
        got = barrier_overlap(inner, outer, nbins=2)
        assert abs(got - expected) < 1e-12, f"barrier_overlap {got} != {expected}"
        assert barrier_overlap(inner, outer, nbins=2, col=3) == 0.0

        chaotic_trapped, chaotic_all, n_trapped = chaotic_fraction(inner)
        assert n_trapped == 8
        assert abs(chaotic_trapped - 3 / 8) < 1e-12
        assert abs(chaotic_all - 3 / 9) < 1e-12

        # No classified trapped particles on a surface -> NaN, not a crash.
        dead = write_class_parts(
            base / "dead", mu=[0.2, 0.4], topology=[1, 1], trap_par=[-1, -1]
        )
        assert np.isnan(barrier_overlap(inner, dead, nbins=2))

        candidate_inner = write_class_parts(
            base / "candidate_inner",
            mu=[0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.5],
            topology=[1, 1, 1, 1, 1, 2, 1, 1, 1],
            trap_par=[1, 1, 1, 1, 1, 1, 1, 1, -1],
            jpar=[1] * 9,
        )
        candidate_outer = write_class_parts(
            base / "candidate_outer",
            mu=[0.1, 0.3, 0.45, 0.55, 0.7, 0.9],
            topology=[2, 1, 2, 1, 2, 1],
        )
        comparison = paired_barrier_bootstrap(
            inner,
            outer,
            candidate_inner,
            candidate_outer,
            nbins=2,
            replicates=50,
            seed=7,
        )
        assert comparison["candidate"] < comparison["reference"]
        assert comparison["change"] < 0.0
        assert comparison["paired_standard_error"] > 0.0

        loss_run = base / "loss"
        loss_run.mkdir()
        endpoint = np.nextafter(0.02, 0.0)
        np.savetxt(
            loss_run / "times_lost.dat",
            np.array([[1, 5e-4], [2, 5e-3], [3, endpoint], [4, endpoint]]),
        )
        np.savetxt(
            loss_run / "confined_fraction.dat",
            np.array([[endpoint, 0.5, 0.0, 4]]),
        )
        loss = classification_loss_metrics(loss_run)
        assert loss["prompt_count"] == 1
        assert loss["total_count"] == 2
        assert loss["total_loss"] == 0.5

    print("test_simple_barrier: all checks passed")


if __name__ == "__main__":
    main()
