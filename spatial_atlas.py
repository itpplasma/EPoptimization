from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from spatial_barrier import SpatialBarrierResult, spatial_barrier_score


@dataclass(frozen=True)
class SpatialAtlasResult:
    score: float
    aliasing_half_range: float
    shift_scores: np.ndarray
    shift_worst_channels: np.ndarray
    surfaces: np.ndarray
    refinement_interval: tuple[float, float] | None
    refinement_changes: np.ndarray
    risk: np.ndarray


def _matching(reference: np.ndarray, candidate: np.ndarray, name: str) -> None:
    if reference.shape != candidate.shape or not np.allclose(
        reference, candidate, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError(f"atlas files have inconsistent {name}")


def load_spatial_atlas(paths: list[Path]) -> dict[str, np.ndarray]:
    if not paths:
        raise ValueError("at least one topology file is required")
    records = [dict(np.load(path)) for path in paths]
    records.sort(key=lambda item: float(item["surface"]))
    for name in ("lambda_values", "shifts", "signs"):
        reference = records[0][name]
        for record in records[1:]:
            _matching(reference, record[name], name)
    topology = np.stack([item["topology"] for item in records])
    particle_index = np.stack([item["particle_index"] for item in records])
    weights = np.stack([item["weights"] for item in records])
    if topology.shape != particle_index.shape:
        raise ValueError("topology and particle-index shapes differ")
    return {
        "lambda_values": records[0]["lambda_values"],
        "particle_index": particle_index,
        "shifts": records[0]["shifts"],
        "signs": records[0]["signs"],
        "surfaces": np.array([float(item["surface"]) for item in records]),
        "topology": topology,
        "weights": weights,
    }


def _refinement_changes(risk: np.ndarray, weights: np.ndarray) -> np.ndarray:
    mean_risk = np.mean(risk, axis=(1, 2))
    changes = np.zeros(risk.shape[0] - 1)
    for surface in range(len(changes)):
        interval_weights = 0.5 * (weights[surface] + weights[surface + 1])
        difference = np.abs(mean_risk[surface + 1] - mean_risk[surface])
        changes[surface] = float(np.sum(interval_weights * difference))
    return changes


def score_spatial_atlas(
    atlas: dict[str, np.ndarray], sigma_cells: float = 1.0
) -> SpatialAtlasResult:
    topology = atlas["topology"]
    weights = atlas["weights"]
    surfaces = atlas["surfaces"]
    if topology.ndim != 6:
        raise ValueError(
            "atlas topology must have shape (surface, mu, sign, shift, theta, zeta)"
        )
    shift_results: list[SpatialBarrierResult] = []
    for shift in range(topology.shape[3]):
        shift_topology = topology[:, :, :, shift]
        shift_results.append(
            spatial_barrier_score(
                shift_topology,
                shift_topology > 0,
                weights[:, shift],
                np.ones(topology.shape[1:3]),
                sigma_cells=sigma_cells,
            )
        )
    scores = np.array([item.score for item in shift_results])
    worst = np.array([item.worst_channel for item in shift_results])
    risk = np.stack([item.risk for item in shift_results], axis=3)
    mean_risk = np.mean(risk, axis=3)
    mean_weights = np.mean(weights, axis=1)
    changes = _refinement_changes(mean_risk, mean_weights)
    interval = None
    if len(changes):
        index = int(np.argmax(changes))
        interval = (float(surfaces[index]), float(surfaces[index + 1]))
    return SpatialAtlasResult(
        score=float(np.mean(scores)),
        aliasing_half_range=float(0.5 * np.ptp(scores)),
        shift_scores=scores,
        shift_worst_channels=worst,
        surfaces=surfaces,
        refinement_interval=interval,
        refinement_changes=changes,
        risk=risk,
    )
