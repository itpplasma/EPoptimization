#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from classifier_proxy import classifier_surface_features
from spatial_atlas import load_spatial_atlas, score_spatial_atlas


SURFACE_NAMES = (
    "s0p25000",
    "s0p30000",
    "s0p42500",
    "s0p48750",
    "s0p55000",
    "s0p80000",
)
PROMPT_FEATURES = (
    "prompt_unclassified",
    "prompt_topology_nonideal",
    "prompt_jpar_nonideal",
)
LATE_FEATURES = (
    "late_topology_escape",
    "late_topology_nonideal",
    "late_jpar_escape",
    "late_jpar_nonideal",
)


def _surface_features(path: Path) -> dict[str, np.ndarray]:
    record = np.load(path)
    result = classifier_surface_features(
        record["topology"],
        record["jpar"],
        record["particle_index"],
        record["passing"],
        record["weights"],
    )
    return {
        "prompt_unclassified": result.shift_unclassified_fractions,
        "prompt_topology_nonideal": result.shift_nonideal_fractions,
        "prompt_jpar_nonideal": result.shift_jpar_nonideal_fractions,
    }


def extract_features(
    case: Path, surface_names: tuple[str, ...] = SURFACE_NAMES
) -> dict[str, np.ndarray | str]:
    paths = [case / "surfaces" / name / "topology.npz" for name in surface_names]
    if any(not path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise ValueError(f"incomplete classifier atlas: {missing}")
    atlas = load_spatial_atlas(paths)
    topology = score_spatial_atlas(atlas, classifier="topology")
    jpar = score_spatial_atlas(atlas, classifier="jpar")
    features: dict[str, np.ndarray | str] = {
        **_surface_features(paths[0]),
        "late_topology_escape": topology.shift_escape_volumes,
        "late_topology_nonideal": topology.shift_nonideal_volumes,
        "late_jpar_escape": jpar.shift_escape_volumes,
        "late_jpar_nonideal": jpar.shift_nonideal_volumes,
        "simple_sha256": str(atlas["simple_sha256"]),
        "wout_sha256": str(atlas["wout_sha256"]),
    }
    for name in (*PROMPT_FEATURES, *LATE_FEATURES):
        values = np.asarray(features[name], dtype=float)
        if values.shape != (2,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain two finite lattice shifts")
        features[name] = values
    return features


def _label_rows(path: Path) -> tuple[dict[str, dict], dict]:
    document = json.loads(path.read_text())
    if float(document["birth_surface"]) != 0.25:
        raise ValueError("calibration labels must use birth surface s = 0.25")
    rows = document["candidates"]
    for candidate, row in rows.items():
        if sorted(row["seeds"]) != [12345, 22345, 32345, 42345]:
            raise ValueError(f"candidate {candidate} does not use four frozen seeds")
    return rows, document


def _paired_delta(candidate: dict, reference: dict, name: str) -> np.ndarray:
    left = np.asarray(candidate[name], dtype=float)
    right = np.asarray(reference[name], dtype=float)
    if left.shape != (2,) or right.shape != (2,):
        raise ValueError(f"{name} does not contain two lattice shifts")
    return left - right


def _weighted_slope(x: np.ndarray, y: np.ndarray, se: np.ndarray) -> float:
    weights = 1.0 / np.square(se)
    denominator = float(np.sum(weights * np.square(x)))
    if denominator <= 0.0:
        raise ValueError("proxy feature has no variation")
    return float(np.sum(weights * x * y) / denominator)


def grouped_scalar_fit(
    x: np.ndarray, shift_delta: np.ndarray, y: np.ndarray, se: np.ndarray
) -> dict:
    if x.shape != y.shape or y.shape != se.shape:
        raise ValueError("feature, label, and uncertainty arrays differ")
    if shift_delta.shape != (len(x), 2) or len(x) < 4:
        raise ValueError("grouped fit needs four cases and two shifts per case")
    predictions = np.zeros_like(y)
    for holdout in range(len(y)):
        train = np.arange(len(y)) != holdout
        predictions[holdout] = _weighted_slope(x[train], y[train], se[train]) * x[holdout]
    correlation = float(spearmanr(predictions, y).statistic)
    significant = np.abs(y) >= 2.0 * se
    sign_correct = np.sign(predictions[significant]) == np.sign(y[significant])
    false_safe = significant & (y > 0.0) & (predictions <= 0.0)
    false_improvement = significant & (y >= 0.0) & (predictions < 0.0)
    resolved_pairs = []
    correct_pairs = []
    for left in range(len(y)):
        for right in range(left + 1, len(y)):
            resolution = 2.0 * np.hypot(se[left], se[right])
            if abs(y[left] - y[right]) < resolution:
                continue
            resolved_pairs.append((left, right))
            correct_pairs.append(
                np.sign(predictions[left] - predictions[right])
                == np.sign(y[left] - y[right])
            )
    shift_concordant = np.all(
        (np.sign(shift_delta) == np.sign(x)[:, None]) | (shift_delta == 0.0), axis=1
    )
    return {
        "slope": _weighted_slope(x, y, se),
        "loco_predictions": predictions.tolist(),
        "spearman": correlation,
        "standardized_rmse": float(np.sqrt(np.mean(np.square((predictions - y) / se)))),
        "sign_correct_fraction": float(np.mean(sign_correct)) if np.any(significant) else None,
        "false_safe_count": int(np.sum(false_safe)),
        "false_improvement_count": int(np.sum(false_improvement)),
        "resolved_ranking_pairs": [list(pair) for pair in resolved_pairs],
        "resolved_ranking_correct_fraction": (
            float(np.mean(correct_pairs)) if correct_pairs else None
        ),
        "shift_concordant": shift_concordant.tolist(),
        "passes": bool(
            np.all(sign_correct)
            and np.all(shift_concordant)
            and not np.any(false_safe)
            and (not correct_pairs or np.all(correct_pairs))
        ),
    }


def radial_convergence(levels: dict[str, np.ndarray]) -> dict:
    order = ("coarse", "medium", "fine")
    if any(level not in levels for level in order):
        raise ValueError("radial convergence needs coarse, medium, and fine levels")
    arrays = [np.asarray(levels[level], dtype=float) for level in order]
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("radial feature arrays differ")
    relative_changes = []
    for left, right in zip(arrays[:-1], arrays[1:], strict=True):
        relative_changes.append(
            np.abs(right - left) / np.maximum(np.abs(left), 1.0e-12)
        )
    deltas = [array[1:] - array[0] for array in arrays]
    sign_concordant = np.all(
        np.stack([np.sign(delta) == np.sign(deltas[-1]) for delta in deltas]),
        axis=0,
    )
    correlations = []
    for delta in deltas[:-1]:
        for shift in range(delta.shape[1]):
            correlations.append(
                float(spearmanr(delta[:, shift], deltas[-1][:, shift]).statistic)
            )
    return {
        "maximum_relative_change": float(np.max(relative_changes)),
        "ordering_spearman": correlations,
        "sign_concordant": sign_concordant.tolist(),
        "passes": bool(
            np.max(relative_changes) <= 0.05
            and np.all(sign_concordant)
            and np.all(np.isfinite(correlations))
            and np.min(correlations) >= 0.90
        ),
    }


def _radial_levels(
    level_surfaces: dict[str, tuple[str, ...]] | None,
) -> dict[str, tuple[str, ...]]:
    if level_surfaces is None:
        level_surfaces = {
            "coarse": ("s0p25000", "s0p30000", "s0p55000", "s0p80000"),
            "medium": (
                "s0p25000",
                "s0p30000",
                "s0p42500",
                "s0p55000",
                "s0p80000",
            ),
            "fine": SURFACE_NAMES,
        }
    if set(level_surfaces) != {"coarse", "medium", "fine"}:
        raise ValueError("radial levels must be coarse, medium, and fine")
    if any(not surfaces or surfaces[0] != "s0p25000" for surfaces in level_surfaces.values()):
        raise ValueError("every radial level must start at birth surface s = 0.25")
    if not (
        set(level_surfaces["coarse"]) < set(level_surfaces["medium"])
        and set(level_surfaces["medium"]) < set(level_surfaces["fine"])
    ):
        raise ValueError("radial levels must be strictly nested")
    return level_surfaces


def _collect_calibration(
    atlas_root: Path,
    reference: Path,
    label_rows: dict[str, dict],
    level_surfaces: dict[str, tuple[str, ...]],
) -> tuple[list[str], dict, dict, dict]:
    reference_by_level = {
        level: extract_features(reference, surfaces)
        for level, surfaces in level_surfaces.items()
    }
    reference_features = reference_by_level["fine"]
    cases = sorted(label_rows, key=int)
    changes: dict[str, list[np.ndarray]] = {
        name: [] for name in (*PROMPT_FEATURES, *LATE_FEATURES)
    }
    prompt = []
    prompt_se = []
    late = []
    late_se = []
    radial_values = {
        name: {
            level: [np.asarray(reference_by_level[level][name], dtype=float)]
            for level in level_surfaces
        }
        for name in LATE_FEATURES
    }
    for candidate in cases:
        candidate_root = atlas_root / f"candidate{int(candidate):03d}"
        candidate_by_level = {
            level: extract_features(candidate_root, surfaces)
            for level, surfaces in level_surfaces.items()
        }
        features = candidate_by_level["fine"]
        if features["wout_sha256"] != label_rows[candidate]["wout_sha256"]:
            raise ValueError(f"candidate {candidate} equilibrium differs from labels")
        if features["simple_sha256"] != reference_features["simple_sha256"]:
            raise ValueError(f"candidate {candidate} uses another classifier executable")
        for name in changes:
            changes[name].append(_paired_delta(features, reference_features, name))
        for name in LATE_FEATURES:
            for level in level_surfaces:
                radial_values[name][level].append(
                    np.asarray(candidate_by_level[level][name], dtype=float)
                )
        aggregate = label_rows[candidate]["aggregate"]
        prompt.append(float(aggregate["prompt"]["change"]))
        prompt_se.append(float(aggregate["prompt"]["paired_se"]))
        late.append(float(aggregate["late"]["change"]))
        late_se.append(float(aggregate["late"]["paired_se"]))
    labels = {
        "prompt_change": prompt,
        "prompt_paired_se": prompt_se,
        "late_change": late,
        "late_paired_se": late_se,
    }
    return cases, changes, radial_values, labels


def calibrate(
    atlas_root: Path,
    reference: Path,
    labels: Path,
    level_surfaces: dict[str, tuple[str, ...]] | None = None,
) -> dict:
    label_rows, label_document = _label_rows(labels)
    levels = _radial_levels(level_surfaces)
    cases, changes, radial_values, label_values = _collect_calibration(
        atlas_root, reference, label_rows, levels
    )
    fits = {}
    for name, target, uncertainty in (
        *((name, label_values["prompt_change"], label_values["prompt_paired_se"])
          for name in PROMPT_FEATURES),
        *((name, label_values["late_change"], label_values["late_paired_se"])
          for name in LATE_FEATURES),
    ):
        shift_delta = np.asarray(changes[name])
        fits[name] = grouped_scalar_fit(
            np.mean(shift_delta, axis=1),
            shift_delta,
            np.asarray(target),
            np.asarray(uncertainty),
        )
        if name in LATE_FEATURES:
            fits[name]["radial"] = radial_convergence(
                {
                    level: np.asarray(values)
                    for level, values in radial_values[name].items()
                }
            )
    topology_jpar_identical = all(
        np.array_equal(changes[topology], changes[jpar])
        for topology, jpar in (
            ("prompt_topology_nonideal", "prompt_jpar_nonideal"),
            ("late_topology_escape", "late_jpar_escape"),
            ("late_topology_nonideal", "late_jpar_nonideal"),
        )
    )
    return {
        "schema_name": "alpha-loss.classifier-proxy-calibration",
        "schema_version": 1,
        "birth_surface": label_document["birth_surface"],
        "radial_levels": {
            level: list(surfaces) for level, surfaces in levels.items()
        },
        "cases": cases,
        "features": {
            name: np.asarray(values).tolist() for name, values in changes.items()
        },
        "labels": label_values,
        "fits": fits,
        "topology_jpar_identical": topology_jpar_identical,
        "fractal_features": [],
    }


def predict_case(frozen: dict, candidate: Path, reference: Path) -> dict:
    if frozen.get("fractal_features") != []:
        raise ValueError("fractal features are forbidden")
    radial_surfaces = frozen.get("radial_surfaces")
    if (
        not isinstance(radial_surfaces, list)
        or len(radial_surfaces) < 2
        or radial_surfaces[0] != "s0p25000"
        or len(set(radial_surfaces)) != len(radial_surfaces)
    ):
        raise ValueError("frozen heads do not define a valid radial grid")
    surfaces = tuple(radial_surfaces)
    candidate_features = extract_features(candidate, surfaces)
    reference_features = extract_features(reference, surfaces)
    predictions = {}
    feature_deltas = {}
    for target in ("prompt", "late"):
        head = frozen[target]
        feature = head["feature"]
        delta = _paired_delta(candidate_features, reference_features, feature)
        feature_deltas[target] = delta.tolist()
        predictions[target] = (float(head["slope"]) * delta).tolist()
    return {
        "schema_name": "alpha-loss.classifier-proxy-prediction",
        "schema_version": 1,
        "wout_sha256": candidate_features["wout_sha256"],
        "feature_deltas": feature_deltas,
        "predictions": predictions,
        "fractal_features": [],
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    calibration = commands.add_parser("fit")
    calibration.add_argument("--atlas-root", type=Path, required=True)
    calibration.add_argument("--reference", type=Path, required=True)
    calibration.add_argument("--labels", type=Path, required=True)
    calibration.add_argument("--radial-levels", type=Path)
    calibration.add_argument("--out", type=Path, required=True)
    prediction = commands.add_parser("predict")
    prediction.add_argument("--frozen-heads", type=Path, required=True)
    prediction.add_argument("--candidate", type=Path, required=True)
    prediction.add_argument("--reference", type=Path, required=True)
    prediction.add_argument("--out", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "fit":
        level_surfaces = None
        if args.radial_levels:
            document = json.loads(args.radial_levels.read_text())
            level_surfaces = {
                level: tuple(str(surface) for surface in surfaces)
                for level, surfaces in document.items()
            }
        result = calibrate(
            args.atlas_root.resolve(),
            args.reference.resolve(),
            args.labels.resolve(),
            level_surfaces,
        )
    else:
        frozen = json.loads(args.frozen_heads.read_text())
        result = predict_case(
            frozen, args.candidate.resolve(), args.reference.resolve()
        )
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
