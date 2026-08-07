from __future__ import annotations

import numpy as np
import pytest

import smooth_barrier as sb


def scores(
    jpar,
    rotation,
    *,
    jpar_samples=None,
    rotation_samples=None,
    trap_par=None,
    legacy_status=None,
):
    jpar = np.asarray(jpar, dtype=float)
    rotation = np.asarray(rotation, dtype=float)
    n = jpar.size
    return sb.ClassifierScores(
        jpar_variation_rate=jpar,
        rotation_number_drift=rotation,
        precession_turns=np.ones(n),
        jpar_sample_count=np.asarray(
            jpar_samples if jpar_samples is not None else np.ones(n), dtype=int
        ),
        rotation_half_count=np.asarray(
            rotation_samples if rotation_samples is not None else np.ones(n),
            dtype=int,
        ),
        tip_count=np.full(n, 6, dtype=int),
        legacy_jpar_spread=np.zeros(n),
        legacy_jpar_reference=np.ones(n),
        legacy_topology_margin=np.zeros(n),
        legacy_status=np.asarray(
            legacy_status if legacy_status is not None else np.zeros(n), dtype=int
        ),
        trap_par=np.asarray(
            trap_par if trap_par is not None else np.zeros(n), dtype=float
        ),
    )


def test_logistic_is_monotone_bounded_and_centred() -> None:
    x = np.linspace(-10.0, 10.0, 101)
    y = sb.logistic(x, width=1.0)
    assert np.all(np.diff(y) > 0.0)
    assert y.min() > 0.0 and y.max() < 1.0
    assert sb.logistic(np.array([0.0]), width=1.0)[0] == pytest.approx(0.5)


def test_logistic_rejects_a_nonpositive_width() -> None:
    for bad in (0.0, -1.0, np.inf, np.nan):
        with pytest.raises(ValueError):
            sb.logistic(np.array([0.0]), width=bad)


def test_resolution_uses_raw_score_metadata_not_legacy_status() -> None:
    sample = scores(
        [1.0, 2.0, 3.0],
        [0.1, 0.2, 0.3],
        jpar_samples=[2, 0, 4],
        rotation_samples=[0, 2, 2],
        legacy_status=[1, 1, 3],
    )
    _, jpar = sb.score_and_resolution(sample, classifier="jpar")
    _, rotation = sb.score_and_resolution(sample, classifier="rotation")
    assert jpar.tolist() == [1.0, 0.0, 1.0]
    assert rotation.tolist() == [0.0, 1.0, 1.0]


def test_rotation_drift_uses_absolute_first_second_half_difference() -> None:
    sample = scores([0.1], [-0.3])
    values, _ = sb.score_and_resolution(sample, classifier="rotation")
    assert values == pytest.approx([0.3])


def test_surface_field_matches_a_hand_weighted_mean() -> None:
    sample = scores([1.0, 3.0], [0.0, 0.0])
    field = sb.surface_score_field(
        sample,
        np.array([0.0, 0.0]),
        np.array([-0.1, 0.1]),
        classifier="jpar",
        trapped_width=1.0,
        mu_width=0.2,
        sample_weights=np.array([1.0, 3.0]),
    )
    assert field.values == pytest.approx([2.5, 2.5])
    assert field.resolved_coverage == pytest.approx([1.0, 1.0])


def test_unresolved_sample_is_censored_and_reported_in_coverage() -> None:
    sample = scores(
        [2.0, 100.0],
        [0.0, 0.0],
        jpar_samples=[1, 0],
    )
    field = sb.surface_score_field(
        sample,
        np.array([0.0, 0.0]),
        np.array([-0.1, 0.1]),
        classifier="jpar",
        trapped_width=1.0,
        mu_width=0.2,
    )
    assert field.values == pytest.approx([2.0, 2.0])
    assert field.resolved_coverage == pytest.approx([0.5, 0.5])


def test_weighted_softmin_matches_log_mean_exp_oracle() -> None:
    values = np.array([[0.0, 4.0], [2.0, 4.0]])
    actual = sb.softmin_across_surfaces(values, temperature=1.0)
    expected_first = -np.log((1.0 + np.exp(-2.0)) / 2.0)
    assert actual == pytest.approx([expected_first, 4.0])


def test_birth_weighted_integral_matches_trapezoid_oracle() -> None:
    nodes = np.array([0.0, 1.0, 2.0])
    value = sb.birth_weighted_integral(
        nodes, np.array([0.0, 1.0, 2.0]), np.ones(3)
    )
    assert value == pytest.approx(1.0)


def test_constant_surface_scores_give_an_exact_barrier_metric() -> None:
    inner = scores(np.full(4, 1.0), np.full(4, 0.1))
    outer = scores(np.full(4, 3.0), np.full(4, 0.4))
    mu = np.linspace(-0.05, 0.05, 4)
    nodes = np.linspace(-0.1, 0.1, 5)
    metric = sb.continuous_barrier_metric(
        [inner, outer],
        [mu, mu],
        nodes,
        classifier="jpar",
        temperature=1.0,
        trapped_width=1.0,
        mu_width=0.05,
    )
    expected = -np.log((np.exp(-1.0) + np.exp(-3.0)) / 2.0)
    assert metric.value == pytest.approx(expected)
    assert metric.barrier_field == pytest.approx(np.full(nodes.size, expected))
    assert metric.resolved_coverage == pytest.approx(1.0)


def test_intact_low_drift_surface_reduces_the_barrier_defect() -> None:
    mu = np.linspace(-0.05, 0.05, 8)
    nodes = np.linspace(-0.1, 0.1, 9)
    low = scores(np.full(8, 0.1), np.zeros(8))
    high = scores(np.full(8, 5.0), np.zeros(8))
    kwargs = dict(
        nodes=nodes,
        classifier="jpar",
        temperature=0.2,
        trapped_width=1.0,
        mu_width=0.05,
    )
    intact = sb.continuous_barrier_metric([high, low], [mu, mu], **kwargs)
    broken = sb.continuous_barrier_metric([high, high], [mu, mu], **kwargs)
    assert intact.value < broken.value


def test_subthreshold_language_cannot_create_a_plateau() -> None:
    """Any raw-score change moves the metric; no class threshold is applied."""
    mu = np.linspace(-0.05, 0.05, 8)
    nodes = np.linspace(-0.1, 0.1, 9)
    first = scores(np.full(8, 0.10), np.zeros(8))
    second = scores(np.full(8, 0.11), np.zeros(8))
    kwargs = dict(
        mu_by_surface=[mu],
        nodes=nodes,
        classifier="jpar",
        temperature=0.2,
        trapped_width=1.0,
        mu_width=0.05,
    )
    value_first = sb.continuous_barrier_metric([first], **kwargs).value
    value_second = sb.continuous_barrier_metric([second], **kwargs).value
    assert value_second > value_first


def test_quadrature_result_is_invariant_to_weight_preserving_duplication() -> None:
    base = scores([1.0, 3.0], [0.1, 0.3])
    duplicate = scores([1.0, 3.0, 1.0, 3.0], [0.1, 0.3, 0.1, 0.3])
    nodes = np.array([-0.1, 0.1])
    base_field = sb.surface_score_field(
        base,
        np.array([0.0, 0.0]),
        nodes,
        classifier="jpar",
        trapped_width=1.0,
        mu_width=0.2,
        sample_weights=np.array([1.0, 3.0]),
    )
    duplicate_field = sb.surface_score_field(
        duplicate,
        np.zeros(4),
        nodes,
        classifier="jpar",
        trapped_width=1.0,
        mu_width=0.2,
        sample_weights=np.array([0.5, 1.5, 0.5, 1.5]),
    )
    assert duplicate_field.values == pytest.approx(base_field.values)
    assert duplicate_field.birth_density == pytest.approx(base_field.birth_density)


def test_reader_round_trips_the_continuous_schema(tmp_path) -> None:
    rows = np.array(
        [
            [1, 0.2, 0.03, 1.5, 4, 2, 7, 10.0, 100.0, -0.2, 1, 0.4],
            [2, 0.7, 0.08, 2.0, 5, 3, 9, 20.0, 90.0, 0.1, 3, -0.2],
        ]
    )
    np.savetxt(tmp_path / "class_scores.dat", rows)
    loaded = sb.load_class_scores(tmp_path)
    assert loaded.jpar_variation_rate == pytest.approx([0.2, 0.7])
    assert loaded.rotation_number_drift == pytest.approx([0.03, 0.08])
    assert loaded.jpar_sample_count.tolist() == [4, 5]
    assert loaded.legacy_status.tolist() == [1, 3]


def test_reader_rejects_the_old_thresholded_schema(tmp_path) -> None:
    np.savetxt(tmp_path / "class_scores.dat", np.zeros((1, 8)))
    with pytest.raises(ValueError, match="12"):
        sb.load_class_scores(tmp_path)
