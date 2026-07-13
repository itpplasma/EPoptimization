from pathlib import Path

import numpy as np

from spatial_atlas import load_spatial_atlas, score_spatial_atlas


def _write_surface(path: Path, surface: float, hole: slice) -> Path:
    topology = np.ones((1, 2, 2, 8, 8), dtype=np.int8)
    topology[:, :, :, hole, 2:5] = 2
    particle_index = np.arange(topology.size).reshape(topology.shape)
    np.savez_compressed(
        path,
        topology=topology,
        particle_index=particle_index,
        weights=np.ones((2, 8, 8)) / 64.0,
        lambda_values=np.array([0.5]),
        shifts=np.array([0.0, 0.5]),
        signs=np.array([1.0, -1.0]),
        surface=np.array(surface),
    )
    return path


def test_atlas_sorts_surfaces_and_scores_each_shift(tmp_path: Path) -> None:
    outer = _write_surface(tmp_path / "outer.npz", 0.8, slice(1, 4))
    inner = _write_surface(tmp_path / "inner.npz", 0.3, slice(1, 4))
    atlas = load_spatial_atlas([outer, inner])
    result = score_spatial_atlas(atlas, sigma_cells=0.5)
    np.testing.assert_allclose(result.surfaces, [0.3, 0.8])
    assert result.score > 0.0
    assert result.shift_scores.shape == (2,)
    assert result.refinement_interval == (0.3, 0.8)


def test_misaligned_middle_surface_closes_channel(tmp_path: Path) -> None:
    paths = [
        _write_surface(tmp_path / "inner.npz", 0.3, slice(1, 4)),
        _write_surface(tmp_path / "middle.npz", 0.55, slice(5, 8)),
        _write_surface(tmp_path / "outer.npz", 0.8, slice(1, 4)),
    ]
    result = score_spatial_atlas(load_spatial_atlas(paths), sigma_cells=0.5)
    assert result.score == 0.0
