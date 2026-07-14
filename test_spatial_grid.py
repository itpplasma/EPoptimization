import numpy as np
from pathlib import Path

import simple_barrier
from spatial_grid import (
    ChartMap,
    _periodic_sample,
    angular_grid,
    calibrate_fixed_invariant,
    fourier_field,
    pitch_quantile_lambdas,
    spatial_classify_namelist,
    write_compatible_chartmap,
)


def test_pitch_quantiles_represent_uniform_absolute_pitch() -> None:
    values = pitch_quantile_lambdas(9)
    np.testing.assert_allclose(np.sort(np.sqrt(1.0 - values)), (np.arange(9) + 0.5) / 9)
    assert np.all((0.0 < values) & (values < 1.0))


def test_angular_grid_is_endpoint_excluded_and_shifted() -> None:
    theta, zeta = angular_grid(4, 6, 2, 0.0)
    shifted_theta, shifted_zeta = angular_grid(4, 6, 2, 0.5)
    assert theta.shape == zeta.shape == (4, 6)
    assert theta.max() < 2.0 * np.pi
    assert zeta.max() < np.pi
    np.testing.assert_allclose(shifted_theta - theta, np.pi / 4)
    np.testing.assert_allclose(shifted_zeta - zeta, np.pi / 12)


def test_fourier_field_uses_boozer_phase_convention() -> None:
    theta, zeta = angular_grid(5, 7, 2, 0.0)
    value = fourier_field(
        np.array([2.0, 3.0]),
        np.array([0.0, 4.0]),
        np.array([0, 1]),
        np.array([0, 2]),
        theta,
        zeta,
    )
    expected = 2.0 + 3.0 * np.cos(theta - 2.0 * zeta) + 4.0 * np.sin(theta - 2.0 * zeta)
    np.testing.assert_allclose(value, expected)


def test_spatial_classifier_preserves_toroidal_launch_position(monkeypatch) -> None:
    monkeypatch.setattr(simple_barrier, "reactor_scale", lambda _: (2.0, 3.0))
    text = spatial_classify_namelist(
        particle_count=64,
        surface=0.55,
        wout=Path("wout.nc"),
        trace_time=0.02,
    )
    assert "startmode = 2" in text
    assert "class_plot = .False." in text
    assert "trace_time = 0.02" in text
    assert "tcut = -1d0" in text


def test_spatial_classifier_uses_explicit_chartmap_scales(monkeypatch) -> None:
    monkeypatch.setattr(
        simple_barrier,
        "reactor_scale",
        lambda _: (_ for _ in ()).throw(AssertionError("must not inspect chartmap")),
    )
    text = spatial_classify_namelist(
        particle_count=64,
        surface=0.55,
        wout=Path("chartmap.nc"),
        trace_time=0.02,
        rz_scale=10.0,
        b_scale=6.0,
    )
    assert "vmec_RZ_scale = 10" in text
    assert "vmec_B_scale = 6" in text


def test_chartmap_sampling_is_radial_linear_and_angular_periodic() -> None:
    rho = np.array([0.25, 0.75])
    theta = 2.0 * np.pi * np.arange(4) / 4
    zeta = np.pi * np.arange(6) / 6
    radial, poloidal, toroidal = np.meshgrid(rho, theta, zeta, indexing="ij")
    values = radial + np.cos(poloidal) + 2.0 * np.cos(2.0 * toroidal)
    stored = np.transpose(values, (2, 1, 0))
    chartmap = ChartMap(
        rho=rho,
        theta=theta,
        zeta=zeta,
        xyz=np.zeros((3, len(zeta), len(theta), len(rho))),
        bmod=stored,
        nfp=2,
    )
    query_theta = np.array([[0.0, 2.0 * np.pi]])
    query_zeta = np.array([[0.0, np.pi]])
    sampled = _periodic_sample(
        stored, chartmap, 0.25, query_theta, query_zeta
    )
    np.testing.assert_allclose(sampled, 3.5)


def test_modern_chartmap_copy_preserves_bytes(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.nc"
    output = tmp_path / "output.nc"
    source.write_bytes(b"exact modern chartmap")
    monkeypatch.setattr(
        "spatial_grid._reference_chartmap",
        lambda _: __import__("contextlib").nullcontext(source),
    )
    write_compatible_chartmap(source, output)
    assert output.read_bytes() == source.read_bytes()


def test_invariant_calibration_uses_pinned_executable_field() -> None:
    shape = (2, 2, 1, 1, 2)
    particle_index = np.arange(np.prod(shape)).reshape(shape)
    starts = np.zeros((particle_index.size, 5))
    starts[:, 3] = 1.0
    starts[:, 4] = np.tile([0.5, 0.5], particle_index.size // 2)
    field = np.broadcast_to(np.array([5.0e4, 7.0e4]), shape)
    classes = np.zeros((particle_index.size, 6))
    classes[:, 0] = np.arange(1, particle_index.size + 1)
    classes[:, 2] = 0.75 / field.ravel()
    design = {
        "lambda": np.array([0.5, 1.0]),
        "particle_index": particle_index,
        "signs": np.array([1.0, -1.0]),
        "start": starts,
    }
    calibrated = calibrate_fixed_invariant(design, classes)
    assert calibrated["start"].shape[0] == 6
    np.testing.assert_allclose(calibrated["b"], [[[5.0, 7.0]]])
    assert np.count_nonzero(calibrated["particle_index"] < 0) == 2
