import numpy as np

from classifier_proxy import classifier_surface_features


def test_classifier_prompt_features_use_valid_grid_without_loss_times() -> None:
    topology = np.ones((1, 1, 2, 4, 4), dtype=np.int8)
    topology[:, :, :, 1:3, 1:3] = 0
    jpar = topology.copy()
    particle_index = np.arange(topology.size).reshape(topology.shape)
    passing = np.zeros_like(topology, dtype=bool)
    passing[:, :, :, 0, :] = True
    weights = np.ones((2, 4, 4), dtype=float) / 16.0

    result = classifier_surface_features(
        topology, jpar, particle_index, passing, weights
    )

    assert result.unclassified_fraction == 0.25
    assert result.passing_fraction == 0.25
    assert result.jpar_nonideal_fraction == 0.0
    assert result.unclassified_component_fraction == 0.25
    assert result.unclassified_component_count == 1.0
    assert result.aliasing_half_range == 0.0
    np.testing.assert_allclose(result.shift_nonideal_fractions, [0.0, 0.0])
    np.testing.assert_allclose(result.shift_jpar_nonideal_fractions, [0.0, 0.0])


def test_classifier_prompt_features_exclude_forbidden_pitch_cells() -> None:
    topology = np.array([[[[[0, 1], [2, 0]]]]], dtype=np.int8)
    jpar = np.array([[[[[0, 2], [1, 0]]]]], dtype=np.int8)
    particle_index = np.array([[[[[0, 1], [2, -1]]]]])
    passing = np.zeros_like(topology, dtype=bool)
    weights = np.ones((1, 2, 2), dtype=float) / 4.0

    result = classifier_surface_features(
        topology, jpar, particle_index, passing, weights
    )

    assert result.unclassified_fraction == 1.0 / 3.0
    assert result.nonideal_fraction == 1.0 / 3.0
    assert result.jpar_nonideal_fraction == 1.0 / 3.0


def test_classifier_prompt_features_do_not_label_passing_as_prompt() -> None:
    topology = np.array([[[[[0, 0], [1, 2]]]]], dtype=np.int8)
    jpar = topology.copy()
    particle_index = np.arange(4).reshape(topology.shape)
    passing = np.array([[[[[True, False], [False, False]]]]])
    weights = np.ones((1, 2, 2), dtype=float) / 4.0

    result = classifier_surface_features(
        topology, jpar, particle_index, passing, weights
    )

    assert result.unclassified_fraction == 0.25
    assert result.passing_fraction == 0.25
