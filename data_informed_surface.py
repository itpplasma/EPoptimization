from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy import optimize
from weightedpca import WeightedPCA, WeightedQuantileTransformer


@dataclass(frozen=True, slots=True)
class WeightedSurfaceTransform:
    n_theta: int
    n_phi: int
    mean: np.ndarray
    components: np.ndarray
    references: np.ndarray
    quantiles: np.ndarray
    explained_variance: np.ndarray
    source_sha256: str

    @property
    def dimension(self) -> int:
        return len(self.components)

    def decode(self, unit: np.ndarray, margin: float = 0.0) -> np.ndarray:
        values = _unit_values(unit, self.dimension, margin)
        probabilities = margin + (1.0 - 2.0 * margin) * values
        scores = np.array(
            [
                np.interp(probabilities[index], self.references, self.quantiles[:, index])
                for index in range(self.dimension)
            ]
        )
        return self.mean + scores @ self.components

    def encode(self, real_space: np.ndarray, margin: float = 0.0) -> np.ndarray:
        values = np.asarray(real_space, dtype=float)
        if values.shape != self.mean.shape or not np.isfinite(values).all():
            raise ValueError("real-space surface vector has the wrong shape")
        scores = (values - self.mean) @ self.components.T
        probabilities = np.array(
            [
                np.interp(scores[index], self.quantiles[:, index], self.references)
                for index in range(self.dimension)
            ]
        )
        unit = (probabilities - margin) / (1.0 - 2.0 * margin)
        if np.any((unit < 0.0) | (unit > 1.0)):
            raise ValueError("surface lies outside the quantile margin")
        return unit

    def decode_anchored(
        self, unit: np.ndarray, anchor: np.ndarray, margin: float = 0.0
    ) -> np.ndarray:
        values = _unit_values(unit, self.dimension, margin)
        base = _real_space_values(anchor, self.mean.shape)
        probabilities = margin + (1.0 - 2.0 * margin) * values
        scores = _interpolate_columns(probabilities, self.references, self.quantiles)
        medians = _interpolate_columns(
            np.full(self.dimension, 0.5), self.references, self.quantiles
        )
        return base + (scores - medians) @ self.components

    def encode_anchored(
        self, real_space: np.ndarray, anchor: np.ndarray, margin: float = 0.0
    ) -> np.ndarray:
        values = _real_space_values(real_space, self.mean.shape)
        base = _real_space_values(anchor, self.mean.shape)
        delta_scores = (values - base) @ self.components.T
        medians = _interpolate_columns(
            np.full(self.dimension, 0.5), self.references, self.quantiles
        )
        probabilities = np.array(
            [
                np.interp(
                    medians[index] + delta_scores[index],
                    self.quantiles[:, index],
                    self.references,
                )
                for index in range(self.dimension)
            ]
        )
        unit = (probabilities - margin) / (1.0 - 2.0 * margin)
        if np.any((unit < 0.0) | (unit > 1.0)):
            raise ValueError("surface lies outside the anchored quantile margin")
        return unit

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            n_theta=np.array(self.n_theta),
            n_phi=np.array(self.n_phi),
            mean=self.mean,
            components=self.components,
            references=self.references,
            quantiles=self.quantiles,
            explained_variance=self.explained_variance,
            source_sha256=np.array(self.source_sha256),
        )


def fit_weighted_surface_transform(path: Path, dimension: int) -> WeightedSurfaceTransform:
    source = Path(path)
    with h5py.File(source, "r") as archive:
        if not {"data", "weights", "n_theta", "n_phi"} <= set(archive):
            raise ValueError("weighted surface data inventory is incomplete")
        data = np.asarray(archive["data"], dtype=float)
        weights = np.asarray(archive["weights"], dtype=float)
        n_theta = int(archive["n_theta"][()])
        n_phi = int(archive["n_phi"][()])
    _validate_training_data(data, weights, n_theta, n_phi, dimension)
    pca = WeightedPCA(dimension).fit(data, sample_weight=weights)
    components = np.asarray(pca.components_, dtype=float).copy()
    scores = (data - pca.mean_) @ components.T
    quantile = WeightedQuantileTransformer(subsample=None).fit(
        scores, sample_weight=weights
    )
    transform = WeightedSurfaceTransform(
        n_theta,
        n_phi,
        np.asarray(pca.mean_, dtype=float),
        components,
        np.asarray(quantile.references_, dtype=float),
        np.asarray(quantile.quantiles_, dtype=float),
        np.asarray(pca.explained_variance_ratio_, dtype=float),
        _file_sha256(source),
    )
    validate_transform(transform)
    return transform


def load_weighted_surface_transform(path: Path) -> WeightedSurfaceTransform:
    with np.load(path, allow_pickle=False) as archive:
        expected = {
            "n_theta",
            "n_phi",
            "mean",
            "components",
            "references",
            "quantiles",
            "explained_variance",
            "source_sha256",
        }
        if set(archive.files) != expected:
            raise ValueError("weighted surface transform inventory differs")
        transform = WeightedSurfaceTransform(
            int(archive["n_theta"]),
            int(archive["n_phi"]),
            np.asarray(archive["mean"]),
            np.asarray(archive["components"]),
            np.asarray(archive["references"]),
            np.asarray(archive["quantiles"]),
            np.asarray(archive["explained_variance"]),
            str(archive["source_sha256"]),
        )
    validate_transform(transform)
    return transform


def surface_from_unit(
    transform: WeightedSurfaceTransform,
    unit: np.ndarray,
    *,
    nfp: int,
    major_radius: float,
    minor_radius: float,
    mpol: int = 6,
    ntor: int = 6,
    margin: float = 0.0,
    exact_radii: bool = True,
    anchor: np.ndarray | None = None,
):
    from simsopt.geo import SurfaceRZFourier

    surface = SurfaceRZFourier.from_nphi_ntheta(
        nfp=nfp,
        mpol=mpol,
        ntor=ntor,
        range="half period",
        ntheta=transform.n_theta,
        nphi=transform.n_phi,
    )
    normalized = (
        transform.decode(unit, margin)
        if anchor is None
        else transform.decode_anchored(unit, anchor, margin)
    )
    surface.least_squares_fit(_cartesian_grid(surface, normalized, major_radius, minor_radius))
    if exact_radii:
        _enforce_radii(surface, major_radius, minor_radius)
    return surface


def normalized_surface_grid(surface, transform: WeightedSurfaceTransform) -> np.ndarray:
    from simsopt.geo import SurfaceRZFourier

    grid = SurfaceRZFourier.from_nphi_ntheta(
        nfp=surface.nfp,
        mpol=surface.mpol,
        ntor=surface.ntor,
        range="half period",
        ntheta=transform.n_theta,
        nphi=transform.n_phi,
    )
    sampled = SurfaceRZFourier(
        nfp=surface.nfp,
        stellsym=surface.stellsym,
        mpol=surface.mpol,
        ntor=surface.ntor,
        quadpoints_phi=grid.quadpoints_phi,
        quadpoints_theta=grid.quadpoints_theta,
        dofs=surface.dofs,
    )
    gamma = sampled.gamma()
    radius = np.hypot(gamma[:, :, 0], gamma[:, :, 1])
    major = surface.major_radius()
    minor = surface.minor_radius()
    return np.concatenate(
        (((radius - major) / minor).ravel(), (gamma[:, :, 2] / minor).ravel())
    )


def validate_transform(transform: WeightedSurfaceTransform) -> None:
    size = 2 * transform.n_theta * transform.n_phi
    if (
        min(transform.n_theta, transform.n_phi, transform.dimension) < 1
        or transform.mean.shape != (size,)
        or transform.components.shape != (transform.dimension, size)
        or transform.quantiles.shape != (len(transform.references), transform.dimension)
        or transform.explained_variance.shape != (transform.dimension,)
        or len(transform.source_sha256) != 64
    ):
        raise ValueError("weighted surface transform shapes differ")
    arrays = (
        transform.mean,
        transform.components,
        transform.references,
        transform.quantiles,
        transform.explained_variance,
    )
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("weighted surface transform contains nonfinite values")
    if np.any(np.diff(transform.references) <= 0.0) or np.any(
        np.diff(transform.quantiles, axis=0) < 0.0
    ):
        raise ValueError("weighted surface quantiles are not monotone")
    np.testing.assert_allclose(
        transform.components @ transform.components.T,
        np.eye(transform.dimension),
        rtol=0.0,
        atol=1.0e-12,
    )


def _validate_training_data(data, weights, n_theta, n_phi, dimension) -> None:
    if (
        data.ndim != 2
        or data.shape[1] != 2 * n_theta * n_phi
        or weights.shape != (len(data),)
        or not np.isfinite(data).all()
        or not np.isfinite(weights).all()
        or np.any(weights <= 0.0)
        or not 1 <= dimension <= min(data.shape)
    ):
        raise ValueError("weighted surface training data are invalid")


def _unit_values(unit, dimension, margin) -> np.ndarray:
    values = np.asarray(unit, dtype=float)
    if (
        values.shape != (dimension,)
        or not np.isfinite(values).all()
        or np.any((values < 0.0) | (values > 1.0))
        or not 0.0 <= margin < 0.5
    ):
        raise ValueError("surface coordinates or quantile margin are invalid")
    return values


def _real_space_values(values, shape) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError("real-space surface vector has the wrong shape")
    return array


def _interpolate_columns(values, references, quantiles) -> np.ndarray:
    return np.array(
        [
            np.interp(values[index], references, quantiles[:, index])
            for index in range(len(values))
        ]
    )


def _cartesian_grid(surface, normalized, major, minor) -> np.ndarray:
    count = surface.quadpoints_phi.size * surface.quadpoints_theta.size
    radius = normalized[:count].reshape((surface.quadpoints_phi.size, -1)) * minor + major
    vertical = normalized[count:].reshape((surface.quadpoints_phi.size, -1)) * minor
    angle = 2.0 * np.pi * surface.quadpoints_phi[:, None]
    return np.stack((radius * np.cos(angle), radius * np.sin(angle), vertical), axis=2)


def _enforce_radii(surface, major_radius: float, minor_radius: float) -> None:
    target_aspect = major_radius / minor_radius

    def residual(value):
        dofs = surface.x.copy()
        dofs[0] = value
        surface.x = dofs
        return surface.aspect_ratio() - target_aspect

    root = optimize.newton(residual, x0=major_radius)
    residual(root)
    surface.x = surface.x * (minor_radius / surface.minor_radius())


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
