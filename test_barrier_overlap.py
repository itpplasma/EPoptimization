from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import barrier_overlap as bo


def test_pitch_grid_is_symmetric_and_inside_the_unit_interval() -> None:
    pitch = bo.pitch_grid(8, pitch_max=1.0)
    assert pitch.size == 8
    assert np.allclose(np.sort(pitch), pitch)
    assert np.allclose(pitch, -pitch[::-1])
    assert np.all(np.abs(pitch) < 1.0)
    assert np.all(np.abs(pitch) > 0.0)


def test_pitch_quadrature_integrates_isotropic_pitch_second_moment() -> None:
    pitch, weights = bo.pitch_quadrature(7)
    assert weights.sum() == pytest.approx(1.0)
    assert np.sum(weights * pitch**2) == pytest.approx(1.0 / 3.0)


def test_pitch_grid_respects_the_trapped_bound() -> None:
    pitch = bo.pitch_grid(16, pitch_max=0.6)
    assert np.max(np.abs(pitch)) <= 0.6
    assert np.max(np.abs(pitch)) > 0.5


def test_pitch_grid_rejects_bounds_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError):
        bo.pitch_grid(8, pitch_max=1.5)
    with pytest.raises(ValueError):
        bo.pitch_grid(8, pitch_max=0.0)


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


def test_starting_weights_match_grid_order_and_normalise() -> None:
    weights = bo.starting_weights(3, 2, 8)
    assert weights.shape == (48,)
    assert weights.sum() == pytest.approx(1.0)
    assert np.array_equal(weights[:8], weights[8:16])


def test_overlap_is_zero_when_the_barrier_surface_is_wholly_regular() -> None:
    mu = np.array([0.01, 0.02, 0.03, 0.04])
    trapped = np.ones(4, dtype=bool)
    edges = np.linspace(0.0, 0.05, 3)
    value = bo.barrier_overlap_samples(
        mu,
        np.array([2, 2, 2, 2]),
        trapped,
        mu,
        np.array([1, 1, 1, 1]),
        trapped,
        edges=edges,
    )
    assert value == 0.0


def test_overlap_is_one_when_both_surfaces_are_wholly_chaotic() -> None:
    mu = np.array([0.01, 0.02, 0.03, 0.04])
    trapped = np.ones(4, dtype=bool)
    edges = np.linspace(0.0, 0.05, 3)
    value = bo.barrier_overlap_samples(
        mu,
        np.array([2, 2, 2, 2]),
        trapped,
        mu,
        np.array([2, 2, 2, 2]),
        trapped,
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
        mu,
        np.array([2, 1, 2, 2]),
        trapped,
        mu,
        np.array([2, 1, 1, 1]),
        trapped,
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
        mu,
        chaotic,
        trapped_only_first,
        mu,
        chaotic,
        trapped_only_first,
        edges=edges,
    ) == bo.barrier_overlap_samples(
        mu,
        regular_where_passing,
        trapped_only_first,
        mu,
        regular_where_passing,
        trapped_only_first,
        edges=edges,
    )


def test_overlap_is_nan_without_trapped_particles() -> None:
    mu = np.array([0.01, 0.02])
    none = np.zeros(2, dtype=bool)
    value = bo.barrier_overlap_samples(
        mu,
        np.array([2, 2]),
        none,
        mu,
        np.array([2, 2]),
        np.ones(2, dtype=bool),
        edges=np.array([0.0, 0.05]),
    )
    assert np.isnan(value)


def test_mu_edges_do_not_depend_on_the_sample() -> None:
    first = bo.fixed_mu_edges(1.0, nbins=16)
    second = bo.fixed_mu_edges(2.0, nbins=16)
    assert np.array_equal(first, second)
    assert first.size == 17


def test_mu_edges_sit_on_the_gauss_scale_simple_actually_uses() -> None:
    """A measured trapped band from a reactor-scale run must land inside.

    SIMPLE carries the field in Gauss, so perp_inv is of order 1e-5, not 0.1.
    Edges on the Tesla scale put every trapped particle in the first bin and
    silently reduce a mu-resolved metric to a single-bin one.
    """
    edges = bo.fixed_mu_edges(1.0, nbins=16)
    observed_low, observed_high = 1.4324e-05, 1.7896e-05
    assert edges[0] < observed_low
    assert edges[-1] > observed_high
    occupied = np.unique(np.digitize([observed_low, observed_high], edges))
    assert occupied.size > 1, "the trapped band must span more than one bin"


def test_mu_band_is_centred_not_started_at_zero() -> None:
    edges = bo.fixed_mu_edges(1.0, nbins=8)
    centre = bo.reference_mu()
    assert edges[0] > 0.0
    assert edges[0] < centre < edges[-1]


def test_namelist_pins_the_fast_classifier_without_the_fractal_cut() -> None:
    text = bo.CLASSIFY_NAMELIST.format(
        n=1024,
        ttime="2d-2",
        num_surf=1,
        sbeg="2.5d-1",
        field_type=2,
        integmode=3,
        nturns=8,
        rz="1d0",
        b="1d0",
        seed=12345,
    )
    assert "fast_class = .True." in text
    assert "class_plot = .False." in text
    assert "tcut = -1d0" in text
    assert "notrace_passing = 1" in text
    assert "multharm = 5" in text
    assert "startmode = 2" in text
    assert "integmode = 3" in text
    assert "isw_field_type = 2" in text


def test_multi_surface_namelist_requests_radial_bminmax_cache(
    tmp_path, monkeypatch
) -> None:
    executable = tmp_path / "simple.x"
    executable.touch()
    wout = tmp_path / "wout.nc"
    wout.touch()
    captured = {}

    def fake_run(command, **kwargs):
        captured["namelist"] = (Path(kwargs["cwd"]) / "simple.in").read_text()
        (Path(kwargs["cwd"]) / "class_parts.dat").touch()
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(bo.subprocess, "run", fake_run)
    work = tmp_path / "run"
    bo.run_classification(
        wout,
        surface=0.25,
        surface_bounds=(0.25, 0.7),
        starts=np.zeros((2, 5)),
        rz_scale=1.0,
        b_scale=1.0,
        trace_time=0.02,
        nturns=8,
        seed=1,
        workdir=work,
        simple_executable=executable,
    )
    assert "num_surf = 2" in captured["namelist"]
    assert (
        "sbeg = 2.5000000000000000d-01, 6.9999999999999996d-01" in captured["namelist"]
    )


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


def test_coarse_equilibria_are_rejected_before_simple_runs(monkeypatch) -> None:
    monkeypatch.setattr(bo, "flux_surface_count", lambda path: 3)
    with pytest.raises(ValueError, match="flux surfaces"):
        bo.check_radial_resolution("wout_coarse.nc")


def test_the_threshold_is_the_spline_stencil_width(monkeypatch) -> None:
    monkeypatch.setattr(bo, "flux_surface_count", lambda path: bo.MINIMUM_FLUX_SURFACES)
    assert bo.check_radial_resolution("wout.nc") == bo.MINIMUM_FLUX_SURFACES
    monkeypatch.setattr(
        bo, "flux_surface_count", lambda path: bo.MINIMUM_FLUX_SURFACES - 1
    )
    with pytest.raises(ValueError):
        bo.check_radial_resolution("wout.nc")


def test_adequate_radial_resolution_passes(monkeypatch) -> None:
    monkeypatch.setattr(bo, "flux_surface_count", lambda path: 16)
    assert bo.check_radial_resolution("wout_fine.nc") == 16


def test_classifier_work_directory_uses_requested_storage(
    monkeypatch, tmp_path
) -> None:
    requested = tmp_path / "campaign-scratch"
    observed = {}

    def fake_mkdtemp(*, prefix, dir):
        observed.update(prefix=prefix, parent=dir)
        path = Path(dir) / f"{prefix}oracle"
        path.mkdir()
        return str(path)

    monkeypatch.setattr(bo.tempfile, "mkdtemp", fake_mkdtemp)
    monkeypatch.setattr(bo, "find_simple_x", lambda path: tmp_path / "simple.x")
    monkeypatch.setattr(bo, "file_sha256", lambda path: "expected")
    monkeypatch.setattr(bo, "check_radial_resolution", lambda path: 51)
    monkeypatch.setattr(bo, "reactor_scale", lambda path: (1.0, 1.0))
    monkeypatch.setattr(bo, "field_periods", lambda path: 4)
    monkeypatch.setattr(
        bo,
        "run_classification",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("oracle stop")),
    )

    with pytest.raises(RuntimeError, match="oracle stop"):
        bo.barrier_metrics(
            "wout.nc",
            expected_simple_sha256="expected",
            surfaces=[0.25, 0.7],
            ntheta=1,
            nzeta=1,
            npitch=2,
            pitch_max=1.0,
            nmu=3,
            trace_time=0.02,
            prompt_time=0.001,
            nturns=4,
            seed=1,
            simple_executable="simple.x",
            work_root=requested,
        )

    assert requested.is_dir()
    assert observed == {"prefix": "continuous_barrier_", "parent": requested}


def test_continuous_aggregator_reports_only_raw_fast_classifiers(tmp_path) -> None:
    n = 6
    runs = []
    for index, (jpar, rotation) in enumerate(((0.2, 0.03), (0.4, 0.05))):
        run = tmp_path / f"surface_{index}"
        run.mkdir()
        score_rows = np.column_stack(
            [
                np.arange(1, n + 1),
                np.full(n, jpar),
                np.full(n, rotation),
                np.ones(n),
                np.full(n, 4),
                np.full(n, 2),
                np.full(n, 6),
                np.zeros(n),
                np.ones(n),
                np.zeros(n),
                np.ones(n),
                np.ones(n),
            ]
        )
        class_rows = np.column_stack(
            [
                np.arange(1, n + 1),
                np.full(n, 0.25 + 0.25 * index),
                np.linspace(1.3e-5, 1.9e-5, n),
                np.zeros((n, 3)),
            ]
        )
        np.savetxt(run / "class_scores.dat", score_rows)
        np.savetxt(run / "class_parts.dat", class_rows)
        runs.append(run)

    nodes = bo.fixed_mu_nodes(8)
    result = bo._continuous_metrics(
        runs,
        np.full(n, 1.0 / n),
        nodes=nodes,
        radial_weights=np.array([0.5, 0.5]),
        settings=None,
    )
    expected_jpar = 0.2 - 0.1 * np.log((1.0 + np.exp(-2.0)) / 2.0)
    assert result["jpar"]["birth_mean"] == pytest.approx(0.2)
    assert result["jpar"]["barrier_defect"] == pytest.approx(expected_jpar)
    assert result["rotation"]["birth_mean"] == pytest.approx(0.03)
    assert "topology" not in result
    assert "radial" not in result


def test_batched_surface_output_preserves_the_same_softmin_oracle(tmp_path) -> None:
    n = 6
    score_rows = []
    class_rows = []
    for index, (jpar, rotation) in enumerate(((0.2, 0.03), (0.4, 0.05))):
        score_rows.append(
            np.column_stack(
                [
                    np.arange(index * n + 1, (index + 1) * n + 1),
                    np.full(n, jpar),
                    np.full(n, rotation),
                    np.ones(n),
                    np.full(n, 4),
                    np.full(n, 2),
                    np.full(n, 6),
                    np.zeros(n),
                    np.ones(n),
                    np.zeros(n),
                    np.ones(n),
                    np.ones(n),
                ]
            )
        )
        class_rows.append(
            np.column_stack(
                [
                    np.arange(index * n + 1, (index + 1) * n + 1),
                    np.full(n, 0.25 + 0.25 * index),
                    np.linspace(1.3e-5, 1.9e-5, n),
                    np.zeros((n, 3)),
                ]
            )
        )
    np.savetxt(tmp_path / "class_scores.dat", np.concatenate(score_rows))
    np.savetxt(tmp_path / "class_parts.dat", np.concatenate(class_rows))

    result = bo._continuous_metrics(
        [tmp_path],
        np.full(n, 1.0 / n),
        nodes=bo.fixed_mu_nodes(8),
        radial_weights=np.array([0.5, 0.5]),
        settings=None,
        surface_slices=[slice(0, n), slice(n, 2 * n)],
    )
    expected_jpar = 0.2 - 0.1 * np.log((1.0 + np.exp(-2.0)) / 2.0)
    assert result["jpar"]["birth_mean"] == pytest.approx(0.2)
    assert result["jpar"]["barrier_defect"] == pytest.approx(expected_jpar)


def test_radial_quadrature_matches_a_linear_integral() -> None:
    surfaces = np.array([0.2, 0.3, 0.7])
    weights = bo.radial_quadrature_weights(surfaces)
    average = np.sum(weights * surfaces)
    assert weights.sum() == pytest.approx(1.0)
    assert average == pytest.approx(0.45)
