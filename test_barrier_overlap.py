from __future__ import annotations

import numpy as np
import pytest

import barrier_overlap as bo


def test_pitch_grid_is_symmetric_and_inside_the_unit_interval() -> None:
    pitch = bo.pitch_grid(8)
    assert pitch.size == 8
    assert np.allclose(np.sort(pitch), pitch)
    assert np.allclose(pitch, -pitch[::-1])
    assert np.all(np.abs(pitch) < 1.0)
    assert np.all(np.abs(pitch) > 0.0)


def test_pitch_grid_rejects_odd_counts() -> None:
    with pytest.raises(ValueError):
        bo.pitch_grid(7)


def test_starting_grid_is_the_full_product_of_its_axes() -> None:
    rows = bo.starting_grid(0.25, ntheta=8, nzeta=8, npitch=16, nfp=4)
    assert rows.shape == (1024, 5)
    assert np.allclose(rows[:, 0], 0.25)
    assert np.allclose(rows[:, 3], 1.0)
    assert len(np.unique(rows[:, 1].round(12))) == 8
    assert len(np.unique(rows[:, 2].round(12))) == 8
    assert len(np.unique(rows[:, 4].round(12))) == 16
    # every (theta, zeta, pitch) combination appears exactly once
    assert len(np.unique(rows[:, [1, 2, 4]].round(12), axis=0)) == 1024


def test_starting_grid_spans_one_field_period_in_zeta() -> None:
    rows = bo.starting_grid(0.3, ntheta=4, nzeta=4, npitch=2, nfp=5)
    assert rows[:, 2].max() < 2.0 * np.pi / 5.0
    assert rows[:, 1].max() < 2.0 * np.pi


def test_starting_grid_avoids_symmetry_planes() -> None:
    rows = bo.starting_grid(0.3, ntheta=4, nzeta=4, npitch=2, nfp=2)
    assert np.all(rows[:, 1] > 0.0)
    assert np.all(rows[:, 2] > 0.0)


def test_overlap_is_zero_when_the_barrier_surface_is_wholly_regular() -> None:
    mu = np.array([0.01, 0.02, 0.03, 0.04])
    trapped = np.ones(4, dtype=bool)
    edges = np.linspace(0.0, 0.05, 3)
    value = bo.barrier_overlap_samples(
        mu, np.array([2, 2, 2, 2]), trapped,
        mu, np.array([1, 1, 1, 1]), trapped,
        edges=edges,
    )
    assert value == 0.0


def test_overlap_is_one_when_both_surfaces_are_wholly_chaotic() -> None:
    mu = np.array([0.01, 0.02, 0.03, 0.04])
    trapped = np.ones(4, dtype=bool)
    edges = np.linspace(0.0, 0.05, 3)
    value = bo.barrier_overlap_samples(
        mu, np.array([2, 2, 2, 2]), trapped,
        mu, np.array([2, 2, 2, 2]), trapped,
        edges=edges,
    )
    assert value == pytest.approx(1.0)


def test_overlap_matches_a_hand_computed_two_bin_case() -> None:
    # bin [0, 0.02): inner 1 of 2 chaotic, outer 1 of 2 chaotic
    # bin [0.02, 0.04]: inner 2 of 2 chaotic, outer 0 of 2 chaotic
    # overlap = (1/4)*(1/2) + (2/4)*(0/2) = 0.125
    mu = np.array([0.005, 0.015, 0.025, 0.035])
    trapped = np.ones(4, dtype=bool)
    edges = np.array([0.0, 0.02, 0.04])
    value = bo.barrier_overlap_samples(
        mu, np.array([2, 1, 2, 2]), trapped,
        mu, np.array([2, 1, 1, 1]), trapped,
        edges=edges,
    )
    assert value == pytest.approx(0.125)


def test_overlap_ignores_passing_particles() -> None:
    mu = np.array([0.01, 0.01, 0.03, 0.03])
    edges = np.array([0.0, 0.02, 0.04])
    trapped_only_first = np.array([True, False, True, False])
    chaotic = np.array([2, 2, 2, 2])
    regular_where_passing = np.array([2, 1, 2, 1])
    assert bo.barrier_overlap_samples(
        mu, chaotic, trapped_only_first,
        mu, chaotic, trapped_only_first,
        edges=edges,
    ) == bo.barrier_overlap_samples(
        mu, regular_where_passing, trapped_only_first,
        mu, regular_where_passing, trapped_only_first,
        edges=edges,
    )


def test_overlap_is_nan_without_trapped_particles() -> None:
    mu = np.array([0.01, 0.02])
    none = np.zeros(2, dtype=bool)
    value = bo.barrier_overlap_samples(
        mu, np.array([2, 2]), none,
        mu, np.array([2, 2]), np.ones(2, dtype=bool),
        edges=np.array([0.0, 0.05]),
    )
    assert np.isnan(value)


def test_mu_edges_do_not_depend_on_the_sample() -> None:
    first = bo.fixed_mu_edges(1.0, nbins=16)
    second = bo.fixed_mu_edges(2.0, nbins=16)
    assert np.array_equal(first, second)
    assert first.size == 17
    assert first[0] == 0.0


def test_namelist_pins_the_fast_classifier_without_the_fractal_cut() -> None:
    text = bo.CLASSIFY_NAMELIST.format(
        n=1024, ttime="2d-2", sbeg="2.5d-1", field_type=2, integmode=3,
        nturns=8, rz="1d0", b="1d0", seed=12345,
    )
    assert "fast_class = .True." in text
    assert "class_plot = .False." in text
    assert "tcut = -1d0" in text
    assert "notrace_passing = 1" in text
    assert "multharm = 5" in text
    assert "startmode = 2" in text
    assert "integmode = 3" in text
    assert "isw_field_type = 2" in text


def test_loss_windows_partition_the_trace(tmp_path) -> None:
    rows = np.array(
        [
            [1, -1.0, 0.5],
            [2, 5.0e-4, 0.5],
            [3, 5.0e-3, 0.5],
            [4, 2.0e-2, 0.5],
        ]
    )
    np.savetxt(tmp_path / "times_lost.dat", rows)
    metrics = bo.classification_loss_metrics(
        tmp_path, prompt_time=1.0e-3, trace_time=2.0e-2
    )
    assert metrics["prompt_count"] == 1
    assert metrics["late_count"] == 1
    assert metrics["prompt_loss"] == pytest.approx(0.25)
    assert metrics["total_loss"] == pytest.approx(0.5)


def test_classification_reader_rejects_mismatched_particle_indices(tmp_path) -> None:
    np.savetxt(tmp_path / "class_parts.dat", np.array([[1, 0.25, 0.01, 1, 2, 0]]))
    np.savetxt(tmp_path / "times_lost.dat", np.array([[2, -1.0, 0.5]]))
    with pytest.raises(ValueError):
        bo.load_classification(tmp_path, 4)
