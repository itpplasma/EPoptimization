"""Continuous barrier metrics from SIMPLE's classifier margins.

The discrete barrier overlap in :mod:`barrier_overlap` is a sum over mu bins of
products of counting fractions built from integer class codes. Its gradient is
zero almost everywhere and a delta on the switching set, so no differentiation
tool can extract anything from it — Enzyme or a source-transformation engine
returns 0.0 silently.

Three stacked discontinuities produce that:

1. the trapped mask, a step in ``trap_par``,
2. mu bin membership, a step at each bin edge,
3. the class code itself, an integer in {0, 1, 2}.

The first two are mollified here with a logistic in ``trap_par`` and a smooth
bin kernel. The third cannot be mollified after the fact, because the code is
all that survives — but the classifier forms continuous quantities before
thresholding and SIMPLE now writes them to ``class_scores.dat``
(itpplasma/SIMPLE#513). Two are available:

``jpar``
    The spread of the parallel adiabatic invariant across banana tips,
    relative to ``tol_perpinv``. The classifier calls an orbit non-conserving
    when this exceeds one; the continuous score is the ratio itself.
``topology``
    The ideal-orbit monotonicity margin, negative when the tip sequence
    inverts. The continuous score is a logistic in the negated margin.

Both are finite-time quantities over ``nturns`` bounce periods, so the Lyapunov
exposure is a few bounce times rather than the 1e3-1e4 of a slowing-down trace.
That is what keeps tangents inside double precision and makes this the
differentiable route, not merely the smooth one.

Chaos score conventions: every score returned here is in [0, 1] and increasing
in "worse confinement", matching the discrete metric's orientation, so a lower
barrier overlap is better in both.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Absolute tolerance the classifier applies to the J_parallel spread, from
#: ``tol_perpinv`` in SIMPLE's ``check_orbit_type``.
TOL_PERPINV = 15.0

#: score_status values in class_scores.dat. Only 1 carries a topology margin;
#: 1 and 3 carry a J_parallel spread.
STATUS_UNRESOLVED = 0
STATUS_MARGIN = 1
STATUS_EARLY_STOCHASTIC = 2
STATUS_NO_MARGIN = 3


@dataclass(frozen=True)
class ClassifierScores:
    """One classification run's continuous margins."""

    jpar_spread: np.ndarray
    jpar_reference: np.ndarray
    topology_margin: np.ndarray
    status: np.ndarray
    radial_spread: np.ndarray
    tip_count: np.ndarray
    trap_par: np.ndarray

    def __len__(self) -> int:
        return int(self.jpar_spread.size)


def load_class_scores(run: str | Path) -> ClassifierScores:
    """Read ``class_scores.dat`` written by a classifying SIMPLE run."""
    path = Path(run) / "class_scores.dat"
    table = np.loadtxt(path, ndmin=2)
    if table.shape[1] < 8:
        raise ValueError(
            f"{path} has {table.shape[1]} columns; expected index, spread, "
            "reference, margin, status, radial spread, tip count and trap_par"
        )
    if not np.array_equal(
        table[:, 0].astype(int), np.arange(1, table.shape[0] + 1)
    ):
        raise ValueError(f"{path} particle indices are not sequential")
    return ClassifierScores(
        jpar_spread=table[:, 1],
        jpar_reference=table[:, 2],
        topology_margin=table[:, 3],
        status=table[:, 4].astype(int),
        radial_spread=table[:, 5],
        tip_count=table[:, 6].astype(int),
        trap_par=table[:, 7],
    )


def logistic(x: np.ndarray, *, width: float) -> np.ndarray:
    """Numerically safe logistic with a mollifier width."""
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("mollifier width must be finite and positive")
    return 0.5 * (1.0 + np.tanh(np.asarray(x, dtype=float) / (2.0 * width)))


def trapped_weight(trap_par: np.ndarray, *, width: float) -> np.ndarray:
    """Soft replacement for the ``trap_par > 0`` mask.

    ``trap_par`` is the normalised trapping parameter, positive for trapped
    orbits and crossing zero at the trapped-passing boundary. The hard mask
    makes the metric jump whenever a start point crosses that boundary as the
    geometry changes; the logistic makes it slide.
    """
    return logistic(np.asarray(trap_par, dtype=float), width=width)


#: Minimum tips for a radial excursion to exist.
MIN_TIPS = 2


def radial_chaos_score(
    scores: ClassifierScores, *, width: float, reference: float
) -> np.ndarray:
    """Continuous radial transport, in [0, 1].

    The excursion of the banana tips is the width of the radial band an orbit
    explores. A barrier holds the band narrow; a broken barrier lets it spread.
    Unlike the class margins this exists for any orbit with two tips, so it
    carries data where they do not.

    ``reference`` is the excursion at which an orbit counts as fully
    transporting. The score is a saturating ratio rather than a logistic
    because the excursion has a hard floor at zero and no natural threshold to
    sit symmetrically about.
    """
    if not np.isfinite(reference) or reference <= 0.0:
        raise ValueError("radial reference must be finite and positive")
    ratio = np.asarray(scores.radial_spread, dtype=float) / reference
    value = 1.0 - np.exp(-np.maximum(ratio, 0.0) / max(width, 1e-12))
    return np.where(scores.tip_count >= MIN_TIPS, value, 0.0)


def resolution_weight(scores: ClassifierScores, *, classifier: str) -> np.ndarray:
    """Whether an orbit carries the margin the score needs.

    Unresolved orbits are censored, not classified. Scoring them as chaotic
    would reward designs whose orbits merely resolve faster; scoring them as
    regular would hide real losses. They are excluded from numerator and
    denominator alike, and :func:`resolved_fraction` reports how many were
    dropped so the dilution stays visible.
    """
    if classifier == "jpar":
        return np.isin(
            scores.status, (STATUS_MARGIN, STATUS_NO_MARGIN)
        ).astype(float)
    if classifier == "topology":
        return (scores.status == STATUS_MARGIN).astype(float)
    if classifier == "radial":
        return (scores.tip_count >= MIN_TIPS).astype(float)
    raise ValueError(f"unknown classifier {classifier}")


def resolved_fraction(scores: ClassifierScores, *, classifier: str) -> float:
    return float(np.mean(resolution_weight(scores, classifier=classifier)))


def jpar_chaos_score(scores: ClassifierScores, *, width: float) -> np.ndarray:
    """Continuous J_parallel non-conservation, in [0, 1].

    The classifier's test is ``spread > tol_perpinv``; the score is a logistic
    in ``spread / tol_perpinv - 1`` so that a hard threshold is the zero-width
    limit. Orbits with no spread score 0 here and are removed by the
    resolution weight rather than by a value choice.
    """
    ratio = np.asarray(scores.jpar_spread, dtype=float) / TOL_PERPINV - 1.0
    value = logistic(ratio, width=width)
    return np.where(_has_jpar(scores), value, 0.0)


def topology_chaos_score(scores: ClassifierScores, *, width: float) -> np.ndarray:
    """Continuous ideal-orbit violation, in [0, 1].

    The classifier's test is ``margin < 0``; the score is a logistic in the
    negated margin. Orbits without a margin score 0 here and are removed by the
    resolution weight.
    """
    value = logistic(-np.asarray(scores.topology_margin, dtype=float), width=width)
    return np.where(scores.status == STATUS_MARGIN, value, 0.0)


def _has_jpar(scores: ClassifierScores) -> np.ndarray:
    return np.isin(scores.status, (STATUS_MARGIN, STATUS_NO_MARGIN))


def chaos_score(
    scores: ClassifierScores,
    *,
    classifier: str,
    width: float,
    radial_reference: float = 1.0,
) -> np.ndarray:
    if classifier == "jpar":
        return jpar_chaos_score(scores, width=width)
    if classifier == "topology":
        return topology_chaos_score(scores, width=width)
    if classifier == "radial":
        return radial_chaos_score(scores, width=width, reference=radial_reference)
    raise ValueError(f"unknown classifier {classifier}")


def bin_weights(mu: np.ndarray, edges: np.ndarray, *, width: float) -> np.ndarray:
    """Soft bin membership, shape (bins, particles).

    Each bin's weight is the difference of two logistics, which is the
    mollified indicator of the interval. Weights are a partition of unity in
    the zero-width limit, so the smooth metric reduces to the discrete one.
    """
    mu = np.asarray(mu, dtype=float)
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or not np.all(np.diff(edges) > 0.0):
        raise ValueError("mu bin edges must be increasing and at least two")
    lower = logistic(mu[None, :] - edges[:-1, None], width=width)
    upper = logistic(mu[None, :] - edges[1:, None], width=width)
    return lower - upper


def smooth_barrier_overlap(
    inner: ClassifierScores,
    outer: ClassifierScores,
    *,
    mu_inner: np.ndarray,
    mu_outer: np.ndarray,
    edges: np.ndarray,
    classifier: str,
    chaos_width: float,
    trapped_width: float,
    bin_width: float,
    radial_reference: float = 1.0,
) -> float:
    """Mollified barrier overlap.

    Same shape as the discrete metric — a sum over mu bins of the birth-chaotic
    probability times the barrier-breach fraction — with every indicator
    replaced by a smooth weight. As all three widths go to zero this converges
    to :func:`barrier_overlap.barrier_overlap_samples`.

    ``mu_inner`` and ``mu_outer`` are the perpendicular invariants from
    ``class_parts.dat``, the same labels the discrete metric bins on. They are
    passed in rather than taken from the score file because the parallel
    invariant that lives there is a different constant of motion.
    """
    inner_trapped = trapped_weight(inner.trap_par, width=trapped_width) * (
        resolution_weight(inner, classifier=classifier)
    )
    outer_trapped = trapped_weight(outer.trap_par, width=trapped_width) * (
        resolution_weight(outer, classifier=classifier)
    )
    inner_total = float(inner_trapped.sum())
    outer_total = float(outer_trapped.sum())
    if inner_total <= 0.0 or outer_total <= 0.0:
        return float("nan")

    inner_chaos = chaos_score(
        inner, classifier=classifier, width=chaos_width,
        radial_reference=radial_reference,
    )
    outer_chaos = chaos_score(
        outer, classifier=classifier, width=chaos_width,
        radial_reference=radial_reference,
    )

    mu_inner = np.asarray(mu_inner, dtype=float)
    mu_outer = np.asarray(mu_outer, dtype=float)
    if mu_inner.shape != inner.jpar_spread.shape:
        raise ValueError("inner mu and classifier scores have different lengths")
    if mu_outer.shape != outer.jpar_spread.shape:
        raise ValueError("outer mu and classifier scores have different lengths")
    inner_bins = bin_weights(mu_inner, edges, width=bin_width)
    outer_bins = bin_weights(mu_outer, edges, width=bin_width)

    birth = (inner_bins * (inner_trapped * inner_chaos)[None, :]).sum(axis=1)
    barrier_mass = (outer_bins * outer_trapped[None, :]).sum(axis=1)
    barrier_chaos = (outer_bins * (outer_trapped * outer_chaos)[None, :]).sum(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        breach = np.where(barrier_mass > 0.0, barrier_chaos / barrier_mass, 0.0)
    return float((birth / inner_total * breach).sum())
