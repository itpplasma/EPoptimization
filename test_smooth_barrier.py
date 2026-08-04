from __future__ import annotations

import numpy as np
import pytest

import smooth_barrier as sb
from barrier_overlap import barrier_overlap_samples

EDGES = np.linspace(0.0, 0.04, 5)


def scores(spread, margin, status, trap_par, reference=None):
    n = len(spread)
    return sb.ClassifierScores(
        jpar_spread=np.asarray(spread, dtype=float),
        jpar_reference=np.asarray(
            reference if reference is not None else np.full(n, 1000.0), dtype=float
        ),
        topology_margin=np.asarray(margin, dtype=float),
        status=np.asarray(status, dtype=int),
        trap_par=np.asarray(trap_par, dtype=float),
    )


# --- mollifier building blocks -------------------------------------------


def test_logistic_is_monotone_and_bounded() -> None:
    x = np.linspace(-10.0, 10.0, 101)
    y = sb.logistic(x, width=1.0)
    assert np.all(np.diff(y) > 0.0)
    assert y.min() > 0.0 and y.max() < 1.0
    assert sb.logistic(np.array([0.0]), width=1.0)[0] == pytest.approx(0.5)


def test_logistic_rejects_a_nonpositive_width() -> None:
    for bad in (0.0, -1.0, np.inf, np.nan):
        with pytest.raises(ValueError):
            sb.logistic(np.array([0.0]), width=bad)


def test_narrow_logistic_approaches_the_step_it_replaces() -> None:
    x = np.array([-1.0, 1.0])
    y = sb.logistic(x, width=1e-3)
    assert y[0] < 1e-6
    assert y[1] > 1.0 - 1e-6


def test_bin_weights_partition_unity_inside_the_range() -> None:
    mu = np.array([0.005, 0.015, 0.025, 0.035])
    weights = sb.bin_weights(mu, EDGES, width=1e-4)
    assert weights.shape == (4, 4)
    assert np.allclose(weights.sum(axis=0), 1.0, atol=1e-3)


def test_bin_weights_reject_unsorted_edges() -> None:
    with pytest.raises(ValueError):
        sb.bin_weights(np.array([0.1]), np.array([1.0, 0.0]), width=0.1)


def test_wider_bins_spread_a_point_across_neighbours() -> None:
    mu = np.array([0.0201])  # just past an edge
    sharp = sb.bin_weights(mu, EDGES, width=1e-5)[:, 0]
    soft = sb.bin_weights(mu, EDGES, width=5e-3)[:, 0]
    assert sharp.max() > 0.99
    assert soft.max() < 0.99
    assert np.count_nonzero(soft > 0.01) > np.count_nonzero(sharp > 0.01)


# --- chaos scores ---------------------------------------------------------


def test_jpar_score_brackets_the_classifier_threshold() -> None:
    s = scores([0.0, sb.TOL_PERPINV, 100.0], [0.0] * 3, [1, 1, 1], [1.0] * 3)
    value = sb.jpar_chaos_score(s, width=1e-3)
    assert value[0] < 0.01
    assert value[1] == pytest.approx(0.5, abs=1e-6)
    assert value[2] > 0.99


def test_unresolved_orbits_score_as_non_conserving() -> None:
    s = scores([0.0, 0.0], [0.0, 0.0], [sb.STATUS_UNRESOLVED, sb.STATUS_EARLY_STOCHASTIC], [1.0, 1.0])
    assert np.allclose(sb.jpar_chaos_score(s, width=0.1), 1.0)


def test_topology_score_follows_the_margin_sign() -> None:
    s = scores([0.0] * 3, [-1.0, 0.0, 1.0], [1, 1, 1], [1.0] * 3)
    value = sb.topology_chaos_score(s, width=1e-3)
    assert value[0] > 0.99
    assert value[1] == pytest.approx(0.5, abs=1e-6)
    assert value[2] < 0.01


def test_topology_score_falls_back_where_no_margin_exists() -> None:
    s = scores([100.0], [0.0], [sb.STATUS_NO_MARGIN], [1.0])
    assert sb.topology_chaos_score(s, width=1e-3)[0] > 0.99


def test_unknown_classifier_is_rejected() -> None:
    s = scores([0.0], [0.0], [1], [1.0])
    with pytest.raises(ValueError):
        sb.chaos_score(s, classifier="fractal", width=0.1)


# --- the convergence property that makes this a valid surrogate ----------


def _discrete_equivalent(s: sb.ClassifierScores, mu, classifier):
    """Integer classes implied by the same margins, for the discrete metric.

    Mirrors the fallbacks in the smooth scores exactly: an orbit with no
    J_parallel spread counts as non-conserving, and the topology score falls
    back to the J_parallel one where no margin was formed.
    """
    jpar = np.where(s.jpar_spread > sb.TOL_PERPINV, 2, 1)
    jpar = np.where(np.isin(s.status, (sb.STATUS_MARGIN, sb.STATUS_NO_MARGIN)), jpar, 2)
    if classifier == "jpar":
        code = jpar
    else:
        code = np.where(
            s.status == sb.STATUS_MARGIN,
            np.where(s.topology_margin < 0.0, 2, 1),
            jpar,
        )
    return mu, code, s.trap_par > 0.0


@pytest.mark.parametrize("classifier", ["jpar", "topology"])
def test_smooth_overlap_converges_to_the_discrete_metric(classifier) -> None:
    rng = np.random.default_rng(20260804)
    n = 400
    mu_in = rng.uniform(0.002, 0.038, n)
    mu_out = rng.uniform(0.002, 0.038, n)
    inner = scores(
        rng.uniform(0.0, 40.0, n), rng.uniform(-1.0, 1.0, n),
        rng.choice([1, 3], n), rng.uniform(-1.0, 1.0, n),
    )
    outer = scores(
        rng.uniform(0.0, 40.0, n), rng.uniform(-1.0, 1.0, n),
        rng.choice([1, 3], n), rng.uniform(-1.0, 1.0, n),
    )
    discrete = barrier_overlap_samples(
        *_discrete_equivalent(inner, mu_in, classifier),
        *_discrete_equivalent(outer, mu_out, classifier),
        edges=EDGES,
    )
    previous = None
    for width in (1e-1, 1e-2, 1e-3, 1e-5):
        smooth = sb.smooth_barrier_overlap(
            inner, outer, mu_inner=mu_in, mu_outer=mu_out, edges=EDGES,
            classifier=classifier,
            chaos_width=width * sb.TOL_PERPINV,
            trapped_width=width,
            bin_width=width * 0.04,
        )
        error = abs(smooth - discrete)
        if previous is not None:
            assert error <= previous + 1e-9, "narrowing the mollifier diverged"
        previous = error
    assert previous < 1e-3, f"did not converge to the discrete metric: {previous}"


def test_smooth_overlap_responds_where_the_discrete_one_cannot() -> None:
    """A sub-threshold drift change moves the smooth metric, not the discrete one."""
    n = 200
    mu = np.linspace(0.003, 0.037, n)
    trap = np.full(n, 1.0)
    base = scores(np.full(n, 5.0), np.zeros(n), np.full(n, 3), trap)
    nudged = scores(np.full(n, 7.0), np.zeros(n), np.full(n, 3), trap)

    discrete_base = barrier_overlap_samples(
        *_discrete_equivalent(base, mu, "jpar"),
        *_discrete_equivalent(base, mu, "jpar"), edges=EDGES)
    discrete_nudged = barrier_overlap_samples(
        *_discrete_equivalent(nudged, mu, "jpar"),
        *_discrete_equivalent(nudged, mu, "jpar"), edges=EDGES)
    assert discrete_base == discrete_nudged  # both below threshold: no signal

    kwargs = dict(
        mu_inner=mu, mu_outer=mu, edges=EDGES, classifier="jpar",
        chaos_width=sb.TOL_PERPINV * 0.2, trapped_width=0.1, bin_width=0.002,
    )
    smooth_base = sb.smooth_barrier_overlap(base, base, **kwargs)
    smooth_nudged = sb.smooth_barrier_overlap(nudged, nudged, **kwargs)
    assert smooth_nudged > smooth_base


def test_overlap_is_nan_without_trapped_weight() -> None:
    n = 8
    s = scores(np.zeros(n), np.zeros(n), np.full(n, 1), np.full(n, -50.0))
    value = sb.smooth_barrier_overlap(
        s, s, mu_inner=np.linspace(0.01, 0.03, n),
        mu_outer=np.linspace(0.01, 0.03, n), edges=EDGES, classifier="jpar",
        chaos_width=1.0, trapped_width=1e-3, bin_width=1e-3,
    )
    assert np.isnan(value)


def test_reader_rejects_nonsequential_indices(tmp_path) -> None:
    np.savetxt(tmp_path / "class_scores.dat", np.array([[2, 0.0, 1.0, 0.0, 1, 0.5]]))
    with pytest.raises(ValueError):
        sb.load_class_scores(tmp_path)


def test_reader_round_trips_a_written_table(tmp_path) -> None:
    rows = np.array([[1, 3.0, 100.0, -0.2, 1, 0.4], [2, 30.0, 90.0, 0.1, 3, -0.2]])
    np.savetxt(tmp_path / "class_scores.dat", rows)
    s = sb.load_class_scores(tmp_path)
    assert len(s) == 2
    assert s.jpar_spread[1] == pytest.approx(30.0)
    assert s.status.tolist() == [1, 3]
    assert s.trap_par[0] == pytest.approx(0.4)
