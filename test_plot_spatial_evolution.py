import numpy as np
from matplotlib.figure import Figure

from plot_spatial_evolution import evolution_fields, plot_evolution


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


def test_evolution_rotates_close_adaptive_surface_ticks(tmp_path, monkeypatch) -> None:
    topology_files = []
    for surface in (0.3, 0.425, 0.4875, 0.55, 0.8):
        path = tmp_path / f"surface-{surface}.npz"
        topology = np.ones((1, 2, 1, 3, 3), dtype=np.int8)
        np.savez_compressed(
            path,
            topology=topology,
            jpar=topology,
            particle_index=np.zeros_like(topology),
            weights=np.ones((1, 3, 3)) / 9.0,
            lambda_values=np.array([0.75]),
            shifts=np.array([0.0]),
            signs=np.array([1.0, -1.0]),
            surface=np.array(surface),
            simple_sha256=np.array("simple"),
            wout_sha256=np.array("wout"),
            trace_time=np.array(0.02),
        )
        topology_files.append(path)
    rotations = []
    original = Figure.savefig

    def record_ticks(figure, *args, **kwargs):
        for axis in figure.axes[:6]:
            rotations.extend(label.get_rotation() for label in axis.get_xticklabels())
        return original(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", record_ticks)
    plot_evolution(topology_files, tmp_path / "evolution.png", "test")
    assert 45.0 in rotations
