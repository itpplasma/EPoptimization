"""Continuous fields from SIMPLE's two fast orbit classifiers.

The production quantities in this module never use an integer class, a
classifier threshold, a legacy monotonicity margin, or a radial max-minus-min.
They consume the raw finite-time scores written by SIMPLE:

``jpar``
    RMS relative drift rate of the parallel adiabatic invariant per achieved
    poloidal precession turn.
``rotation``
    Absolute drift between the first- and second-half rotation numbers of the
    banana-tip map.

For a score ``C(s, mu)`` (zero is best), an intact radial barrier is the best
regular surface along the escape path.  Since regularity is ``R = -C``, the
soft maximum of regularity is reported in score units as

``C_barrier(mu) = -tau log(sum_s q_s exp(-C(s, mu) / tau))``.

This is a weighted soft minimum of the raw score.  It equals the score when
all surfaces agree and approaches the best surface as ``tau`` tends to zero.
The final scalar is its integral against the sampled alpha-birth density in
``mu``.  Gaussian reconstruction in ``mu`` and a soft trapped-passing weight
avoid moving bins and hard masks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ClassifierScores:
    """One SIMPLE run's raw fast-classifier scores and diagnostics."""

    jpar_variation_rate: np.ndarray
    rotation_number_drift: np.ndarray
    precession_turns: np.ndarray
    jpar_sample_count: np.ndarray
    rotation_half_count: np.ndarray
    tip_count: np.ndarray
    legacy_jpar_spread: np.ndarray
    legacy_jpar_reference: np.ndarray
    legacy_topology_margin: np.ndarray
    legacy_status: np.ndarray
    trap_par: np.ndarray

    def __len__(self) -> int:
        return int(self.jpar_variation_rate.size)


@dataclass(frozen=True)
class SurfaceScoreField:
    """Kernel reconstruction of one raw score on fixed ``mu`` nodes."""

    values: np.ndarray
    resolved_coverage: np.ndarray
    birth_density: np.ndarray


@dataclass(frozen=True)
class BarrierMetric:
    """Birth-weighted barrier defect and the fields needed to audit it."""

    value: float
    barrier_field: np.ndarray
    birth_density: np.ndarray
    resolved_coverage: float
    surface_fields: tuple[SurfaceScoreField, ...]


def load_class_scores(run: str | Path) -> ClassifierScores:
    """Read the 12-column ``class_scores.dat`` schema from SIMPLE."""
    path = Path(run) / "class_scores.dat"
    table = np.loadtxt(path, ndmin=2)
    if table.shape[1] < 12:
        raise ValueError(
            f"{path} has {table.shape[1]} columns; expected at least 12: "
            "index, two raw "
            "scores, five resolution values, three legacy diagnostics, "
            "legacy status and trap_par"
        )
    expected = np.arange(1, table.shape[0] + 1)
    if not np.array_equal(table[:, 0].astype(int), expected):
        raise ValueError(f"{path} particle indices are not sequential")
    integer_columns = table[:, [4, 5, 6, 10]]
    if not np.array_equal(integer_columns, integer_columns.astype(int)):
        raise ValueError(f"{path} contains non-integral resolution metadata")
    return ClassifierScores(
        jpar_variation_rate=table[:, 1],
        rotation_number_drift=table[:, 2],
        precession_turns=table[:, 3],
        jpar_sample_count=table[:, 4].astype(int),
        rotation_half_count=table[:, 5].astype(int),
        tip_count=table[:, 6].astype(int),
        legacy_jpar_spread=table[:, 7],
        legacy_jpar_reference=table[:, 8],
        legacy_topology_margin=table[:, 9],
        legacy_status=table[:, 10].astype(int),
        trap_par=table[:, 11],
    )


def logistic(x: np.ndarray, *, width: float) -> np.ndarray:
    """Numerically stable smooth replacement for a zero-centred step."""
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("width must be finite and positive")
    values = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.tanh(values / (2.0 * width)))


def trapped_weight(trap_par: np.ndarray, *, width: float) -> np.ndarray:
    """Soft weight for the physically trapped side of ``trap_par = 0``."""
    return logistic(np.asarray(trap_par, dtype=float), width=width)


def score_and_resolution(
    scores: ClassifierScores, *, classifier: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return a raw score and its independently recorded resolution mask."""
    if classifier == "jpar":
        values = np.asarray(scores.jpar_variation_rate, dtype=float)
        resolved = np.asarray(scores.jpar_sample_count) > 0
    elif classifier == "rotation":
        values = np.asarray(scores.rotation_number_drift, dtype=float)
        resolved = np.asarray(scores.rotation_half_count) > 0
    else:
        raise ValueError(f"unknown fast classifier {classifier}")
    if np.any(values < 0.0) or not np.all(np.isfinite(values)):
        raise ValueError(f"{classifier} scores must be finite and non-negative")
    return values, resolved.astype(float)


def resolved_fraction(scores: ClassifierScores, *, classifier: str) -> float:
    """Fraction carrying a raw score; legacy class status is irrelevant."""
    _, resolved = score_and_resolution(scores, classifier=classifier)
    return float(np.mean(resolved))


def gaussian_kernel(
    mu: np.ndarray, nodes: np.ndarray, *, width: float
) -> np.ndarray:
    """Normal-density kernel with shape ``(nodes, particles)``."""
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("mu width must be finite and positive")
    mu = np.asarray(mu, dtype=float)
    nodes = np.asarray(nodes, dtype=float)
    if mu.ndim != 1 or nodes.ndim != 1 or nodes.size < 2:
        raise ValueError("mu and at least two fixed nodes must be one-dimensional")
    if not np.all(np.diff(nodes) > 0.0):
        raise ValueError("mu nodes must be strictly increasing")
    z = (nodes[:, None] - mu[None, :]) / width
    return np.exp(-0.5 * z * z) / (np.sqrt(2.0 * np.pi) * width)


def _sample_weights(count: int, supplied: np.ndarray | None) -> np.ndarray:
    if supplied is None:
        return np.ones(count, dtype=float)
    weights = np.asarray(supplied, dtype=float)
    if weights.shape != (count,):
        raise ValueError("sample weights and classifier scores have different lengths")
    if np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("sample weights must be finite and non-negative")
    if not np.any(weights > 0.0):
        raise ValueError("at least one sample weight must be positive")
    return weights


def surface_score_field(
    scores: ClassifierScores,
    mu: np.ndarray,
    nodes: np.ndarray,
    *,
    classifier: str,
    trapped_width: float,
    mu_width: float,
    sample_weights: np.ndarray | None = None,
) -> SurfaceScoreField:
    """Reconstruct ``C(s, mu)`` as a resolved, trapped weighted mean."""
    mu = np.asarray(mu, dtype=float)
    if mu.shape != (len(scores),):
        raise ValueError("mu and classifier scores have different lengths")
    values, resolved = score_and_resolution(scores, classifier=classifier)
    weights = _sample_weights(len(scores), sample_weights)
    phase_weight = weights * trapped_weight(scores.trap_par, width=trapped_width)
    kernel = gaussian_kernel(mu, nodes, width=mu_width)
    birth_density = kernel @ phase_weight
    resolved_density = kernel @ (phase_weight * resolved)
    numerator = kernel @ (phase_weight * resolved * values)
    field = np.full(np.asarray(nodes).shape, np.nan, dtype=float)
    coverage = np.zeros_like(field)
    np.divide(numerator, resolved_density, out=field, where=resolved_density > 0.0)
    np.divide(
        resolved_density,
        birth_density,
        out=coverage,
        where=birth_density > 0.0,
    )
    return SurfaceScoreField(field, coverage, birth_density)


def softmin_across_surfaces(
    values: np.ndarray,
    *,
    temperature: float,
    surface_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Weighted log-mean-exp soft minimum, independently at each node."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("surface fields must have shape (surfaces, nodes)")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("softmin temperature must be finite and positive")
    weights = _sample_weights(values.shape[0], surface_weights)
    weights = weights / weights.sum()
    result = np.full(values.shape[1], np.nan, dtype=float)
    valid = np.all(np.isfinite(values), axis=0)
    if not np.any(valid):
        return result
    selected = values[:, valid]
    minimum = np.min(selected, axis=0)
    shifted = np.exp(-(selected - minimum[None, :]) / temperature)
    result[valid] = minimum - temperature * np.log(weights @ shifted)
    return result


def birth_weighted_integral(
    nodes: np.ndarray, values: np.ndarray, density: np.ndarray
) -> float:
    """Integrate a field against a non-negative birth density."""
    nodes = np.asarray(nodes, dtype=float)
    values = np.asarray(values, dtype=float)
    density = np.asarray(density, dtype=float)
    if values.shape != nodes.shape or density.shape != nodes.shape:
        raise ValueError("nodes, values and density must have the same shape")
    if not np.all(np.isfinite(values)):
        return float("nan")
    if np.any(density < 0.0) or not np.all(np.isfinite(density)):
        raise ValueError("birth density must be finite and non-negative")
    normalisation = float(np.trapezoid(density, nodes))
    if normalisation <= 0.0:
        return float("nan")
    return float(np.trapezoid(values * density, nodes) / normalisation)


def continuous_barrier_metric(
    scores_by_surface: Sequence[ClassifierScores],
    mu_by_surface: Sequence[np.ndarray],
    nodes: np.ndarray,
    *,
    classifier: str,
    temperature: float,
    trapped_width: float,
    mu_width: float,
    sample_weights_by_surface: Sequence[np.ndarray] | None = None,
    surface_weights: np.ndarray | None = None,
) -> BarrierMetric:
    """Build and birth-integrate the continuous radial-barrier defect."""
    if len(scores_by_surface) != len(mu_by_surface) or not scores_by_surface:
        raise ValueError("scores and mu must name the same non-empty surface set")
    if sample_weights_by_surface is None:
        sample_weights_by_surface = [None] * len(scores_by_surface)
    if len(sample_weights_by_surface) != len(scores_by_surface):
        raise ValueError("sample weights must name every surface")
    fields = tuple(
        surface_score_field(
            scores,
            mu,
            nodes,
            classifier=classifier,
            trapped_width=trapped_width,
            mu_width=mu_width,
            sample_weights=weights,
        )
        for scores, mu, weights in zip(
            scores_by_surface, mu_by_surface, sample_weights_by_surface
        )
    )
    barrier = softmin_across_surfaces(
        np.stack([field.values for field in fields]),
        temperature=temperature,
        surface_weights=surface_weights,
    )
    birth_density = fields[0].birth_density
    value = birth_weighted_integral(nodes, barrier, birth_density)
    node_coverage = np.min(
        np.stack([field.resolved_coverage for field in fields]), axis=0
    )
    coverage = birth_weighted_integral(nodes, node_coverage, birth_density)
    return BarrierMetric(value, barrier, birth_density, coverage, fields)
