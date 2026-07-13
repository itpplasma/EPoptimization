from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, label


@dataclass(frozen=True)
class SpatialBarrierResult:
    score: float
    worst_channel: float
    channel_scores: np.ndarray
    risk: np.ndarray


@dataclass(frozen=True)
class RadialBandResult:
    nonideal_volume: float
    escape_volume: float
    minimum_separator_width: float
    normalized_separator_width: float


def periodic_kernel_risk(
    nonideal: np.ndarray,
    valid: np.ndarray,
    weights: np.ndarray,
    sigma_cells: float = 1.0,
) -> np.ndarray:
    nonideal = np.asarray(nonideal, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    weights = np.broadcast_to(np.asarray(weights, dtype=float), nonideal.shape)
    if nonideal.shape != valid.shape or nonideal.ndim < 2:
        raise ValueError("nonideal and valid must have equal shapes with angular axes")
    if sigma_cells <= 0.0 or np.any(weights < 0.0):
        raise ValueError("sigma and weights must be nonnegative")
    support = valid * weights
    axes = tuple(range(nonideal.ndim - 2, nonideal.ndim))
    sigma = tuple(
        0.0 if axis not in axes else sigma_cells for axis in range(nonideal.ndim)
    )
    numerator = gaussian_filter(nonideal * support, sigma=sigma, mode="wrap")
    denominator = gaussian_filter(support, sigma=sigma, mode="wrap")
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 0.0,
    )


def periodic_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("component mask must be two-dimensional")
    labels, count = label(mask)
    parent = np.arange(count + 1)

    def root(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def join(left: int, right: int) -> None:
        if left and right:
            parent[root(right)] = root(left)

    for column in range(mask.shape[1]):
        join(int(labels[0, column]), int(labels[-1, column]))
    for row in range(mask.shape[0]):
        join(int(labels[row, 0]), int(labels[row, -1]))
    roots = sorted({root(item) for item in range(1, count + 1)})
    remap = {item: index + 1 for index, item in enumerate(roots)}
    output = np.zeros_like(labels)
    for item in range(1, count + 1):
        output[labels == item] = remap[root(item)]
    return output, len(roots)


def periodic_volume_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError("volume component mask must be three-dimensional")
    labels, count = label(mask)
    parent = np.arange(count + 1)

    def root(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def join(left: int, right: int) -> None:
        if left and right:
            parent[root(right)] = root(left)

    for surface in range(mask.shape[0]):
        for column in range(mask.shape[2]):
            join(int(labels[surface, 0, column]), int(labels[surface, -1, column]))
        for row in range(mask.shape[1]):
            join(int(labels[surface, row, 0]), int(labels[surface, row, -1]))
    roots = sorted({root(item) for item in range(1, count + 1)})
    remap = {item: index + 1 for index, item in enumerate(roots)}
    output = np.zeros_like(labels)
    for item in range(1, count + 1):
        output[labels == item] = remap[root(item)]
    return output, len(roots)


def radial_cell_widths(surfaces: np.ndarray) -> np.ndarray:
    surfaces = np.asarray(surfaces, dtype=float)
    if surfaces.ndim != 1 or len(surfaces) < 2:
        raise ValueError("radial bands require at least two surfaces")
    intervals = np.diff(surfaces)
    if not np.all(np.isfinite(surfaces)) or np.any(intervals <= 0.0):
        raise ValueError("radial surfaces must be finite and strictly increasing")
    widths = np.empty_like(surfaces)
    widths[0] = 0.5 * intervals[0]
    widths[-1] = 0.5 * intervals[-1]
    widths[1:-1] = 0.5 * (intervals[:-1] + intervals[1:])
    return widths


def _spread_nonideal_cost(cost: np.ndarray, nonideal: np.ndarray) -> np.ndarray:
    components, count = periodic_components(nonideal)
    output = cost.copy()
    for item in range(1, count + 1):
        selected = components == item
        output[selected] = np.min(cost[selected])
    return output


def minimum_radial_separator_width(
    nonideal: np.ndarray,
    surfaces: np.ndarray,
    ideal: np.ndarray | None = None,
) -> float:
    nonideal = np.asarray(nonideal, dtype=bool)
    ideal = ~nonideal if ideal is None else np.asarray(ideal, dtype=bool)
    widths = radial_cell_widths(surfaces)
    if (
        nonideal.ndim != 3
        or nonideal.shape[0] != len(widths)
        or ideal.shape != nonideal.shape
        or np.any(nonideal & ideal)
    ):
        raise ValueError("radial mask must have shape (surface, theta, zeta)")
    cost = np.where(nonideal[0], 0.0, np.where(ideal[0], widths[0], np.inf))
    cost = _spread_nonideal_cost(cost, nonideal[0])
    for index in range(1, len(widths)):
        layer = np.where(
            nonideal[index], 0.0, np.where(ideal[index], widths[index], np.inf)
        )
        cost = cost + layer
        cost = _spread_nonideal_cost(cost, nonideal[index])
    minimum = float(np.min(cost))
    return minimum if np.isfinite(minimum) else 0.0


def radial_band_features(
    nonideal: np.ndarray,
    angular_weights: np.ndarray,
    surfaces: np.ndarray,
    ideal: np.ndarray | None = None,
) -> RadialBandResult:
    nonideal = np.asarray(nonideal, dtype=bool)
    weights = np.asarray(angular_weights, dtype=float)
    widths = radial_cell_widths(surfaces)
    if nonideal.ndim != 3 or weights.shape != nonideal.shape:
        raise ValueError("band arrays must have shape (surface, theta, zeta)")
    if nonideal.shape[0] != len(widths) or np.any(weights < 0.0):
        raise ValueError("band arrays and radial surfaces differ")
    totals = np.sum(weights, axis=(1, 2))
    if np.any(totals <= 0.0):
        raise ValueError("every radial surface needs positive angular weight")
    normalized = weights / totals[:, None, None]
    radial_range = float(np.sum(widths))
    volume = float(np.sum(widths[:, None, None] * normalized * nonideal))
    components, count = periodic_volume_components(nonideal)
    escape = np.zeros_like(nonideal)
    for item in range(1, count + 1):
        selected = components == item
        if np.any(selected[0]) and np.any(selected[-1]):
            escape |= selected
    escape_volume = float(np.sum(widths[:, None, None] * normalized * escape))
    separator = minimum_radial_separator_width(nonideal, surfaces, ideal)
    return RadialBandResult(
        nonideal_volume=volume / radial_range,
        escape_volume=escape_volume / radial_range,
        minimum_separator_width=separator,
        normalized_separator_width=separator / radial_range,
    )


def _component_capacities(
    components: np.ndarray, count: int, risk: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("angular weights must have positive sum")
    values = np.zeros(count + 1)
    for item in range(1, count + 1):
        selected = components == item
        values[item] = float(np.sum(weights[selected] * risk[selected]) / total)
    return values


def _overlap_capacity(
    left_components: np.ndarray,
    left_item: int,
    right_components: np.ndarray,
    right_item: int,
    left_risk: np.ndarray,
    right_risk: np.ndarray,
    weights: np.ndarray,
) -> float:
    overlap = (left_components == left_item) & (right_components == right_item)
    if not np.any(overlap):
        return 0.0
    return float(
        np.sum(weights[overlap] * np.minimum(left_risk[overlap], right_risk[overlap]))
        / np.sum(weights)
    )


def strongest_channel(
    nonideal: np.ndarray, risk: np.ndarray, weights: np.ndarray
) -> float:
    nonideal = np.asarray(nonideal, dtype=bool)
    risk = np.asarray(risk, dtype=float)
    weights = np.broadcast_to(np.asarray(weights, dtype=float), nonideal.shape)
    if nonideal.ndim != 3 or risk.shape != nonideal.shape:
        raise ValueError("channel inputs must have shape (surface, theta, zeta)")
    components = [periodic_components(layer) for layer in nonideal]
    first_labels, first_count = components[0]
    capacity = _component_capacities(first_labels, first_count, risk[0], weights[0])
    for surface in range(1, nonideal.shape[0]):
        left_labels, left_count = components[surface - 1]
        right_labels, right_count = components[surface]
        node = _component_capacities(
            right_labels, right_count, risk[surface], weights[surface]
        )
        edge_weights = 0.5 * (weights[surface - 1] + weights[surface])
        updated = np.zeros(right_count + 1)
        for right_item in range(1, right_count + 1):
            for left_item in range(1, left_count + 1):
                edge = _overlap_capacity(
                    left_labels,
                    left_item,
                    right_labels,
                    right_item,
                    risk[surface - 1],
                    risk[surface],
                    edge_weights,
                )
                updated[right_item] = max(
                    updated[right_item],
                    min(capacity[left_item], edge, node[right_item]),
                )
        capacity = updated
    return float(np.max(capacity, initial=0.0))


def spatial_barrier_score(
    topology: np.ndarray,
    valid: np.ndarray,
    angular_weights: np.ndarray,
    population_weights: np.ndarray,
    sigma_cells: float = 1.0,
) -> SpatialBarrierResult:
    topology = np.asarray(topology)
    valid = np.asarray(valid, dtype=bool)
    if topology.shape != valid.shape or topology.ndim != 5:
        raise ValueError(
            "topology and valid must have shape (surface, mu, sign, theta, zeta)"
        )
    angular_weights = np.broadcast_to(
        np.asarray(angular_weights, dtype=float),
        (topology.shape[0], *topology.shape[-2:]),
    )
    population_weights = np.broadcast_to(
        np.asarray(population_weights, dtype=float), topology.shape[1:3]
    )
    if np.any(population_weights < 0.0) or np.sum(population_weights) <= 0.0:
        raise ValueError("population weights must be nonnegative with positive sum")
    nonideal = (topology == 2) & valid
    risk = periodic_kernel_risk(
        nonideal,
        valid,
        angular_weights[:, None, None],
        sigma_cells,
    )
    scores = np.zeros(topology.shape[1:3])
    for mu_index in range(topology.shape[1]):
        for sign_index in range(topology.shape[2]):
            scores[mu_index, sign_index] = strongest_channel(
                nonideal[:, mu_index, sign_index],
                risk[:, mu_index, sign_index],
                angular_weights,
            )
    normalized = population_weights / np.sum(population_weights)
    return SpatialBarrierResult(
        score=float(np.sum(normalized * scores)),
        worst_channel=float(np.max(scores)),
        channel_scores=scores,
        risk=risk,
    )
