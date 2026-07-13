import numpy as np

from spatial_barrier import (
    periodic_components,
    periodic_kernel_risk,
    radial_band_features,
    spatial_barrier_score,
    strongest_channel,
)


def test_periodic_components_join_wrapped_hole() -> None:
    mask = np.zeros((5, 6), dtype=bool)
    mask[2, [0, 5]] = True
    components, count = periodic_components(mask)
    assert count == 1
    assert components[2, 0] == components[2, 5] == 1


def test_kernel_risk_respects_periodicity_and_invalid_cells() -> None:
    nonideal = np.zeros((5, 6), dtype=bool)
    nonideal[2, 0] = True
    valid = np.ones_like(nonideal)
    valid[2, 1] = False
    risk = periodic_kernel_risk(nonideal, valid, np.ones_like(nonideal))
    assert risk[2, 5] > risk[2, 3]
    assert np.all((0.0 <= risk) & (risk <= 1.0))


def test_aligned_holes_form_channel_and_misaligned_holes_do_not() -> None:
    aligned = np.zeros((3, 8, 8), dtype=bool)
    aligned[:, 2:5, 3:6] = True
    risk = aligned.astype(float)
    aligned_score = strongest_channel(aligned, risk, np.ones((8, 8)))
    misaligned = aligned.copy()
    misaligned[1] = np.roll(misaligned[1], 4, axis=0)
    misaligned_score = strongest_channel(
        misaligned, misaligned.astype(float), np.ones((8, 8))
    )
    assert aligned_score > 0.0
    assert misaligned_score == 0.0


def test_duplicate_surface_does_not_change_bottleneck_capacity() -> None:
    mask = np.zeros((3, 8, 8), dtype=bool)
    mask[:, 1:4, 2:6] = True
    baseline = strongest_channel(mask, mask.astype(float), np.ones((8, 8)))
    refined = np.insert(mask, 1, mask[1], axis=0)
    score = strongest_channel(refined, refined.astype(float), np.ones((8, 8)))
    assert score == baseline


def test_spatial_score_uses_population_weights_and_reports_worst() -> None:
    topology = np.ones((3, 2, 2, 8, 8), dtype=int)
    topology[:, 0, 0, 1:5, 1:5] = 2
    topology[:, 1, 1, 2:4, 2:4] = 2
    valid = np.ones_like(topology, dtype=bool)
    population = np.array([[3.0, 0.0], [0.0, 1.0]])
    result = spatial_barrier_score(
        topology,
        valid,
        np.ones((8, 8)),
        population,
        sigma_cells=0.5,
    )
    assert result.channel_scores[0, 0] > result.channel_scores[1, 1] > 0.0
    assert result.score < result.worst_channel
    assert result.score > result.channel_scores[1, 1]


def test_radial_band_distinguishes_thin_and_thick_separators() -> None:
    surfaces = np.linspace(0.3, 0.8, 5)
    nonideal = np.ones((5, 6, 6), dtype=bool)
    thin = nonideal.copy()
    thin[2] = False
    thick = nonideal.copy()
    thick[1:4] = False
    weights = np.ones_like(nonideal, dtype=float) / 36.0

    thin_result = radial_band_features(thin, weights, surfaces)
    thick_result = radial_band_features(thick, weights, surfaces)

    assert np.isclose(thin_result.minimum_separator_width, 0.125)
    assert np.isclose(thick_result.minimum_separator_width, 0.375)
    assert thick_result.nonideal_volume < thin_result.nonideal_volume


def test_radial_band_reports_zero_separator_for_connected_hole() -> None:
    surfaces = np.array([0.3, 0.55, 0.8])
    nonideal = np.zeros((3, 6, 6), dtype=bool)
    nonideal[:, 2:4, 1:3] = True
    weights = np.ones_like(nonideal, dtype=float) / 36.0

    result = radial_band_features(nonideal, weights, surfaces)

    assert result.minimum_separator_width == 0.0
    assert result.escape_volume > 0.0


def test_unresolved_cells_do_not_supply_ideal_separator_width() -> None:
    surfaces = np.array([0.3, 0.55, 0.8])
    nonideal = np.zeros((3, 6, 6), dtype=bool)
    ideal = np.ones_like(nonideal)
    ideal[1] = False
    weights = np.ones_like(nonideal, dtype=float) / 36.0

    result = radial_band_features(nonideal, weights, surfaces, ideal=ideal)

    assert result.minimum_separator_width == 0.0


def test_radial_band_integrals_are_stable_under_exact_refinement() -> None:
    surfaces = np.array([0.3, 0.55, 0.8])
    nonideal = np.zeros((3, 6, 6), dtype=bool)
    nonideal[:, 1:4, 2:5] = True
    weights = np.ones_like(nonideal, dtype=float) / 36.0
    baseline = radial_band_features(nonideal, weights, surfaces)
    refined = radial_band_features(
        np.insert(nonideal, 1, nonideal[0], axis=0),
        np.insert(weights, 1, weights[0], axis=0),
        np.array([0.3, 0.425, 0.55, 0.8]),
    )

    assert refined.nonideal_volume == baseline.nonideal_volume
    assert refined.escape_volume == baseline.escape_volume
