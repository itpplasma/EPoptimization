from pathlib import Path

import numpy as np

from spatial_atlas import (
    fixed_shell_nonideal_volumes,
    load_spatial_atlas,
    score_spatial_atlas,
)


def _write_surface(path: Path, surface: float, hole: slice) -> Path:
    topology = np.ones((1, 2, 2, 8, 8), dtype=np.int8)
    topology[:, :, :, hole, 2:5] = 2
    particle_index = np.arange(topology.size).reshape(topology.shape)
    np.savez_compressed(
        path,
        topology=topology,
        jpar=topology,
        particle_index=particle_index,
        weights=np.ones((2, 8, 8)) / 64.0,
        lambda_values=np.array([0.5]),
        shifts=np.array([0.0, 0.5]),
        signs=np.array([1.0, -1.0]),
        surface=np.array(surface),
        simple_sha256=np.array("simple-hash"),
        trace_time=np.array(0.02),
        wout_sha256=np.array("wout-hash"),
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
    assert result.shift_nonideal_volumes.shape == (2,)
    assert result.shift_escape_volumes.shape == (2,)
    assert result.shift_minimum_separator_widths.shape == (2,)
    assert result.shift_mean_separator_widths.shape == (2,)
    assert result.shift_open_channel_fractions.shape == (2,)
    assert result.refinement_interval == (0.3, 0.8)
    jpar_result = score_spatial_atlas(atlas, sigma_cells=0.5, classifier="jpar")
    assert jpar_result.score == result.score


def test_misaligned_middle_surface_closes_channel(tmp_path: Path) -> None:
    paths = [
        _write_surface(tmp_path / "inner.npz", 0.3, slice(1, 4)),
        _write_surface(tmp_path / "middle.npz", 0.55, slice(5, 8)),
        _write_surface(tmp_path / "outer.npz", 0.8, slice(1, 4)),
    ]
    result = score_spatial_atlas(load_spatial_atlas(paths), sigma_cells=0.5)
    assert result.score == 0.0


def test_fixed_shell_integrates_only_the_declared_outer_band(tmp_path: Path) -> None:
    birth = _write_surface(tmp_path / "birth.npz", 0.25, slice(0, 8))
    inner = _write_surface(tmp_path / "inner.npz", 0.675, slice(1, 3))
    outer = _write_surface(tmp_path / "outer.npz", 0.8, slice(1, 5))
    atlas = load_spatial_atlas([outer, birth, inner])

    result = fixed_shell_nonideal_volumes(atlas, "topology", 0.675, 0.8)

    np.testing.assert_allclose(result, 9.0 / 64.0)


def test_fixed_shell_requires_both_boundaries(tmp_path: Path) -> None:
    paths = [
        _write_surface(tmp_path / "birth.npz", 0.25, slice(0, 8)),
        _write_surface(tmp_path / "outer.npz", 0.8, slice(1, 5)),
    ]

    with np.testing.assert_raises_regex(ValueError, "both fixed shell boundaries"):
        fixed_shell_nonideal_volumes(
            load_spatial_atlas(paths), "topology", 0.675, 0.8
        )
