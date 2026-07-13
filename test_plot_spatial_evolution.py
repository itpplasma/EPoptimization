import numpy as np

from plot_spatial_evolution import evolution_fields


def test_evolution_fields_track_disconnected_holes() -> None:
    topology = np.ones((3, 1, 2, 1, 8, 8), dtype=np.int8)
    topology[:, 0, 0, 0, 1:4, 2:5] = 2
    topology[0, 0, 1, 0, 1:4, 2:5] = 2
    topology[1, 0, 1, 0, 5:8, 2:5] = 2
    topology[2, 0, 1, 0, 1:4, 2:5] = 2
    weights = np.ones((3, 1, 8, 8)) / 64.0

    area, components, capacity = evolution_fields(topology, weights, shift=0)

    np.testing.assert_allclose(area[:, 0, :], 9.0 / 64.0)
    np.testing.assert_array_equal(components[:, 0, :], 1)
    assert np.all(capacity[:, 0, 0] > 0.0)
    assert capacity[0, 0, 1] > 0.0
    assert capacity[1, 0, 1] == 0.0
    assert capacity[2, 0, 1] == 0.0
