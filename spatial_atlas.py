from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from spatial_barrier import (
    SpatialBarrierResult,
    radial_band_features,
    spatial_barrier_score,
)


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
    shift_nonideal_volumes: np.ndarray
    shift_escape_volumes: np.ndarray
    shift_minimum_separator_widths: np.ndarray
    shift_mean_separator_widths: np.ndarray
    shift_open_channel_fractions: np.ndarray


def _matching(reference: np.ndarray, candidate: np.ndarray, name: str) -> None:
    if reference.shape != candidate.shape or not np.allclose(
        reference, candidate, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError(f"atlas files have inconsistent {name}")


def _load_record(path: Path) -> dict[str, np.ndarray]:
    record = dict(np.load(path))
    if "jpar" in record:
        return record
    classes = np.loadtxt(path.parent / "class_parts.dat", ndmin=2)
    particle_index = record["particle_index"]
    selected = particle_index >= 0
    indices = particle_index[selected]
    if classes.shape[1] < 4 or np.max(indices, initial=-1) >= len(classes):
        raise ValueError("class_parts.dat cannot supply the J-parallel classifier")
    jpar = np.zeros(particle_index.shape, dtype=np.int8)
    jpar[selected] = classes[indices, 3].astype(np.int8)
    record["jpar"] = jpar
    return record


def load_spatial_atlas(paths: list[Path]) -> dict[str, np.ndarray]:
    if not paths:
        raise ValueError("at least one topology file is required")
    records = [_load_record(path) for path in paths]
    records.sort(key=lambda item: float(item["surface"]))
    for name in ("lambda_values", "shifts", "signs"):
        reference = records[0][name]
        for record in records[1:]:
            _matching(reference, record[name], name)
    for name in ("simple_sha256", "wout_sha256", "trace_time"):
        reference = str(records[0][name])
        if any(str(record[name]) != reference for record in records[1:]):
            raise ValueError(f"atlas files have inconsistent {name}")
    topology = np.stack([item["topology"] for item in records])
    jpar = np.stack([item["jpar"] for item in records])
    particle_index = np.stack([item["particle_index"] for item in records])
    weights = np.stack([item["weights"] for item in records])
    if topology.shape != particle_index.shape or jpar.shape != topology.shape:
        raise ValueError("classifier and particle-index shapes differ")
    expected_weights = (topology.shape[0], topology.shape[3], *topology.shape[-2:])
    if weights.shape != expected_weights:
        raise ValueError("atlas angular weights differ from classifier grids")
    return {
        "lambda_values": records[0]["lambda_values"],
        "particle_index": particle_index,
        "shifts": records[0]["shifts"],
        "signs": records[0]["signs"],
        "surfaces": np.array([float(item["surface"]) for item in records]),
        "topology": topology,
        "jpar": jpar,
        "weights": weights,
        "simple_sha256": records[0]["simple_sha256"],
        "wout_sha256": records[0]["wout_sha256"],
        "trace_time": records[0]["trace_time"],
    }


def _refinement_changes(risk: np.ndarray, weights: np.ndarray) -> np.ndarray:
    mean_risk = np.mean(risk, axis=(1, 2))
    changes = np.zeros(risk.shape[0] - 1)
    for surface in range(len(changes)):
        interval_weights = 0.5 * (weights[surface] + weights[surface + 1])
        difference = np.abs(mean_risk[surface + 1] - mean_risk[surface])
        changes[surface] = float(np.sum(interval_weights * difference))
    return changes


def _radial_band_summary(
    topology: np.ndarray, weights: np.ndarray, surfaces: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape = topology.shape[1:4]
    nonideal_volume = np.zeros(shape)
    escape_volume = np.zeros(shape)
    separator_width = np.zeros(shape)
    for mu_index in range(shape[0]):
        for sign_index in range(shape[1]):
            for shift_index in range(shape[2]):
                result = radial_band_features(
                    topology[:, mu_index, sign_index, shift_index] == 2,
                    weights[:, shift_index],
                    surfaces,
                )
                nonideal_volume[mu_index, sign_index, shift_index] = (
                    result.nonideal_volume
                )
                escape_volume[mu_index, sign_index, shift_index] = result.escape_volume
                separator_width[mu_index, sign_index, shift_index] = (
                    result.minimum_separator_width
                )
    return (
        np.mean(nonideal_volume, axis=(0, 1)),
        np.mean(escape_volume, axis=(0, 1)),
        np.min(separator_width, axis=(0, 1)),
        np.mean(separator_width, axis=(0, 1)),
        np.mean(separator_width <= 1.0e-14, axis=(0, 1)),
    )


def score_spatial_atlas(
    atlas: dict[str, np.ndarray], sigma_cells: float = 1.0, classifier: str = "topology"
) -> SpatialAtlasResult:
    if classifier not in ("topology", "jpar"):
        raise ValueError("classifier must be topology or jpar")
    topology = atlas[classifier]
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
    (
        nonideal_volume,
        escape_volume,
        minimum_separator_width,
        mean_separator_width,
        open_channel_fraction,
    ) = _radial_band_summary(topology, weights, surfaces)
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
        shift_nonideal_volumes=nonideal_volume,
        shift_escape_volumes=escape_volume,
        shift_minimum_separator_widths=minimum_separator_width,
        shift_mean_separator_widths=mean_separator_width,
        shift_open_channel_fractions=open_channel_fraction,
    )
