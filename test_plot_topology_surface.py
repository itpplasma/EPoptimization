import numpy as np
from matplotlib.figure import Figure

from scripts.plot_topology_surface import periodic_critical_points, plot_surface


def test_periodic_critical_points_find_extrema_and_saddles() -> None:
    angles = 2.0 * np.pi * np.arange(16) / 16
    theta, zeta = np.meshgrid(angles, angles, indexing="ij")
    field = np.cos(theta) + np.cos(zeta)
    maxima, minima, saddles = periodic_critical_points(field)
    assert [0, 0] in maxima.tolist()
    assert [8, 8] in minima.tolist()
    assert [0, 8] in saddles.tolist()
    assert [8, 0] in saddles.tolist()


def test_surface_title_preserves_adaptive_radius(tmp_path, monkeypatch) -> None:
    topology = np.ones((1, 2, 2, 3, 3), dtype=np.int8)
    topology_file = tmp_path / "topology.npz"
    np.savez_compressed(
        topology_file,
        topology=topology,
        particle_index=np.zeros_like(topology),
        lambda_values=np.array([0.75]),
        signs=np.array([1.0, -1.0]),
        surface=np.array(0.425),
        b=np.ones((2, 3, 3)),
    )
    titles = []
    original = Figure.suptitle

    def record_title(figure, title, *args, **kwargs):
        titles.append(title)
        return original(figure, title, *args, **kwargs)

    monkeypatch.setattr(Figure, "suptitle", record_title)
    plot_surface(topology_file, tmp_path / "surface.png", "test")
    assert "s=0.425" in titles[0]
