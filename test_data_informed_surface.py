from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from data_informed_surface import (
    fit_weighted_surface_transform,
    load_weighted_surface_transform,
    surface_from_unit,
)


def _write_training_data(path: Path) -> None:
    rng = np.random.default_rng(9)
    data = rng.normal(size=(24, 8))
    weights = np.repeat([1.0, 2.0, 4.0], 8)
    with h5py.File(path, "w") as archive:
        archive["data"] = data
        archive["weights"] = weights
        archive["n_theta"] = 2
        archive["n_phi"] = 2


def test_weighted_surface_transform_roundtrips_and_freezes(tmp_path: Path) -> None:
    source = tmp_path / "training.h5"
    sidecar = tmp_path / "transform.npz"
    _write_training_data(source)
    transform = fit_weighted_surface_transform(source, dimension=8)
    repeated = fit_weighted_surface_transform(source, dimension=8)
    unit = np.linspace(0.15, 0.85, 8)

    decoded = transform.decode(unit)
    reconstructed = transform.decode(transform.encode(decoded))
    transform.save(sidecar)
    loaded = load_weighted_surface_transform(sidecar)

    np.testing.assert_allclose(reconstructed, decoded, rtol=0.0, atol=1.0e-12)
    np.testing.assert_array_equal(loaded.decode(unit), transform.decode(unit))
    np.testing.assert_array_equal(repeated.components, transform.components)


def test_weighted_surface_chart_preserves_geometry_contract(tmp_path: Path) -> None:
    source = tmp_path / "training.h5"
    _write_training_data(source)
    transform = fit_weighted_surface_transform(source, dimension=4)
    surface = surface_from_unit(
        transform,
        np.full(4, 0.5),
        nfp=2,
        major_radius=1.0,
        minor_radius=0.25,
        mpol=2,
        ntor=2,
    )

    assert surface.nfp == 2
    assert surface.stellsym
    np.testing.assert_allclose(surface.major_radius(), 1.0, rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(surface.minor_radius(), 0.25, rtol=0.0, atol=1.0e-10)


def test_anchored_chart_places_reference_at_box_center(tmp_path: Path) -> None:
    source = tmp_path / "training.h5"
    _write_training_data(source)
    transform = fit_weighted_surface_transform(source, dimension=4)
    anchor = np.linspace(-0.3, 0.4, 8)

    center = transform.decode_anchored(np.full(4, 0.5), anchor)

    np.testing.assert_allclose(center, anchor, rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(
        transform.encode_anchored(center, anchor),
        np.full(4, 0.5),
        rtol=0.0,
        atol=1.0e-15,
    )
