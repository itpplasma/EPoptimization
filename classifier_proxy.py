from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spatial_barrier import periodic_components


@dataclass(frozen=True)
class ClassifierSurfaceFeatures:
    unclassified_fraction: float
    nonideal_fraction: float
    jpar_nonideal_fraction: float
    passing_fraction: float
    unclassified_component_fraction: float
    unclassified_component_count: float
    aliasing_half_range: float
    shift_unclassified_fractions: np.ndarray
    shift_nonideal_fractions: np.ndarray
    shift_jpar_nonideal_fractions: np.ndarray


def _weighted_fraction(mask: np.ndarray, valid: np.ndarray, weight: np.ndarray) -> float:
    denominator = float(np.sum(weight[valid]))
    if denominator <= 0.0:
        raise ValueError("classifier grid has no valid weighted cells")
    return float(np.sum(weight[mask & valid]) / denominator)


def _prompt_components(
    prompt: np.ndarray,
    valid: np.ndarray,
    angular_weights: np.ndarray,
    population_weights: np.ndarray,
) -> tuple[float, float]:
    largest = 0.0
    denominator = 0.0
    count_total = 0.0
    count_weight = 0.0
    for mu_index in range(prompt.shape[0]):
        for sign_index in range(prompt.shape[1]):
            population = float(population_weights[mu_index, sign_index])
            channel_valid = valid[mu_index, sign_index]
            channel_weight = angular_weights * population
            denominator += float(np.sum(channel_weight[channel_valid]))
            components, count = periodic_components(
                prompt[mu_index, sign_index] & channel_valid
            )
            areas = [
                float(np.sum(channel_weight[components == item]))
                for item in range(1, count + 1)
            ]
            largest += max(areas, default=0.0)
            count_total += population * count
            count_weight += population
    if denominator <= 0.0 or count_weight <= 0.0:
        raise ValueError("classifier component weights must be positive")
    return largest / denominator, count_total / count_weight


def classifier_surface_features(
    topology: np.ndarray,
    jpar: np.ndarray,
    particle_index: np.ndarray,
    passing: np.ndarray,
    angular_weights: np.ndarray,
    population_weights: np.ndarray | None = None,
) -> ClassifierSurfaceFeatures:
    topology = np.asarray(topology)
    jpar = np.asarray(jpar)
    particle_index = np.asarray(particle_index)
    passing = np.asarray(passing, dtype=bool)
    if topology.ndim != 5 or particle_index.shape != topology.shape:
        raise ValueError("classifier arrays must have shape (mu, sign, shift, theta, zeta)")
    if passing.shape != topology.shape or jpar.shape != topology.shape:
        raise ValueError("passing, J-parallel, and topology arrays differ")
    weights = np.asarray(angular_weights, dtype=float)
    expected = (topology.shape[2], *topology.shape[-2:])
    if weights.shape != expected or np.any(weights < 0.0):
        raise ValueError("angular weights must have shape (shift, theta, zeta)")
    population = (
        np.ones(topology.shape[:2], dtype=float)
        if population_weights is None
        else np.asarray(population_weights, dtype=float)
    )
    if population.shape != topology.shape[:2] or np.any(population < 0.0):
        raise ValueError("population weights must have shape (mu, sign)")
    valid = particle_index >= 0
    shift_prompt = []
    shift_nonideal = []
    shift_jpar_nonideal = []
    shift_passing = []
    component_fraction = []
    component_count = []
    for shift in range(topology.shape[2]):
        shift_valid = valid[:, :, shift]
        shift_prompt_mask = (topology[:, :, shift] == 0) & ~passing[:, :, shift]
        weight = population[:, :, None, None] * weights[shift]
        shift_prompt.append(
            _weighted_fraction(shift_prompt_mask, shift_valid, weight)
        )
        shift_nonideal.append(
            _weighted_fraction(topology[:, :, shift] == 2, shift_valid, weight)
        )
        shift_jpar_nonideal.append(
            _weighted_fraction(jpar[:, :, shift] == 2, shift_valid, weight)
        )
        shift_passing.append(
            _weighted_fraction(passing[:, :, shift], shift_valid, weight)
        )
        largest, count = _prompt_components(
            shift_prompt_mask,
            shift_valid,
            weights[shift],
            population,
        )
        component_fraction.append(largest)
        component_count.append(count)
    prompt = np.asarray(shift_prompt)
    return ClassifierSurfaceFeatures(
        unclassified_fraction=float(np.mean(prompt)),
        nonideal_fraction=float(np.mean(shift_nonideal)),
        jpar_nonideal_fraction=float(np.mean(shift_jpar_nonideal)),
        passing_fraction=float(np.mean(shift_passing)),
        unclassified_component_fraction=float(np.mean(component_fraction)),
        unclassified_component_count=float(np.mean(component_count)),
        aliasing_half_range=float(0.5 * np.ptp(prompt)),
        shift_unclassified_fractions=prompt,
        shift_nonideal_fractions=np.asarray(shift_nonideal),
        shift_jpar_nonideal_fractions=np.asarray(shift_jpar_nonideal),
    )
