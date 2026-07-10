#!/usr/bin/env python
"""Check barrier_overlap and chaotic_fraction against hand-computed values.

Protects the optimization objective: a wrong port of the sensopt metric would
silently drive the optimizer toward something other than the validated
barrier-breach fraction.
"""
import tempfile
from pathlib import Path

import numpy as np

from simple_barrier import barrier_overlap, chaotic_fraction


def write_class_parts(
    directory: Path, mu: list, topology: list, trap_par: list | None = None
) -> Path:
    rows = [
        [i + 1, 0.3, m, t, t, 0]
        for i, (m, t) in enumerate(zip(mu, topology))
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
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        # Inner surface: 8 trapped plus one forced-regular passing particle.
        inner = write_class_parts(
            base / "inner",
            mu=[0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.5],
            topology=[2, 1, 2, 1, 1, 2, 1, 1, 1],
            trap_par=[1, 1, 1, 1, 1, 1, 1, 1, -1],
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

        chaotic_trapped, chaotic_all, n_trapped = chaotic_fraction(inner)
        assert n_trapped == 8
        assert abs(chaotic_trapped - 3 / 8) < 1e-12
        assert abs(chaotic_all - 3 / 9) < 1e-12

        # No classified trapped particles on a surface -> NaN, not a crash.
        dead = write_class_parts(
            base / "dead", mu=[0.2, 0.4], topology=[1, 1], trap_par=[-1, -1]
        )
        assert np.isnan(barrier_overlap(inner, dead, nbins=2))

    print("test_simple_barrier: all checks passed")


if __name__ == "__main__":
    main()
