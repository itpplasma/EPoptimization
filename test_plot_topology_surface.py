import numpy as np

from scripts.plot_topology_surface import periodic_critical_points


def test_periodic_critical_points_find_extrema_and_saddles() -> None:
    angles = 2.0 * np.pi * np.arange(16) / 16
    theta, zeta = np.meshgrid(angles, angles, indexing="ij")
    field = np.cos(theta) + np.cos(zeta)
    maxima, minima, saddles = periodic_critical_points(field)
    assert [0, 0] in maxima.tolist()
    assert [8, 8] in minima.tolist()
    assert [0, 8] in saddles.tolist()
    assert [8, 0] in saddles.tolist()
