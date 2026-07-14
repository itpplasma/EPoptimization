#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def _target_gate(target: str, rows: list[dict], predictions: np.ndarray) -> dict:
    changes = np.asarray([row["aggregate"][target]["change"] for row in rows])
    errors = np.asarray([row["aggregate"][target]["paired_se"] for row in rows])
    significant = np.abs(changes) >= 2.0 * errors
    sign_correct = np.ones_like(predictions, dtype=bool)
    sign_correct[significant] = (
        np.sign(predictions[significant]) == np.sign(changes[significant, None])
    )
    false_safe = significant[:, None] & (changes[:, None] > 0.0) & (predictions <= 0.0)
    resolved_pairs = []
    correct_pairs = []
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            if abs(changes[left] - changes[right]) < 2.0 * np.hypot(
                errors[left], errors[right]
            ):
                continue
            resolved_pairs.append([left, right])
            correct_pairs.append(
                (
                    np.sign(predictions[left] - predictions[right])
                    == np.sign(changes[left] - changes[right])
                ).tolist()
            )
    correlations = [
        float(spearmanr(predictions[:, shift], changes).statistic)
        for shift in range(predictions.shape[1])
    ]
    return {
        "changes": changes.tolist(),
        "paired_se": errors.tolist(),
        "predictions": predictions.tolist(),
        "sign_correct": sign_correct.tolist(),
        "false_safe": false_safe.tolist(),
        "resolved_ranking_pairs": resolved_pairs,
        "resolved_ranking_correct": correct_pairs,
        "ordering_spearman": correlations,
        "passes": bool(
            np.all(sign_correct)
            and not np.any(false_safe)
            and (not correct_pairs or np.all(correct_pairs))
        ),
    }


def validate_predictions(labels: dict, predictions: dict[str, dict]) -> dict:
    if float(labels["birth_surface"]) != 0.25:
        raise ValueError("held-out labels must use birth surface s = 0.25")
    if set(labels["candidates"]) != set(predictions):
        raise ValueError("held-out label and prediction candidates differ")
    candidates = sorted(predictions, key=int)
    rows = [labels["candidates"][candidate] for candidate in candidates]
    arrays = {"prompt": [], "late": []}
    for candidate, row in zip(candidates, rows, strict=True):
        prediction = predictions[candidate]
        if prediction.get("fractal_features") != []:
            raise ValueError("fractal features are forbidden")
        if prediction["wout_sha256"] != row["wout_sha256"]:
            raise ValueError(f"candidate {candidate} equilibrium differs from labels")
        for target in arrays:
            values = np.asarray(prediction["predictions"][target], dtype=float)
            if values.shape != (2,) or not np.all(np.isfinite(values)):
                raise ValueError(f"candidate {candidate} {target} prediction is invalid")
            arrays[target].append(values)
    gates = {
        target: _target_gate(target, rows, np.asarray(values))
        for target, values in arrays.items()
    }
    return {
        "schema_name": "alpha-loss.classifier-proxy-heldout-validation",
        "schema_version": 1,
        "candidates": candidates,
        **gates,
        "passes": bool(all(gate["passes"] for gate in gates.values())),
        "fractal_features": [],
    }


def _load_predictions(values: list[str]) -> dict[str, dict]:
    result = {}
    for value in values:
        candidate, separator, path = value.partition("=")
        if not separator or not candidate or candidate in result:
            raise ValueError("predictions must be unique CANDIDATE=PATH values")
        result[candidate] = json.loads(Path(path).read_text())
    return result


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--frozen-heads", type=Path, required=True)
    root.add_argument("--labels", type=Path, required=True)
    root.add_argument("--prediction", action="append", required=True)
    root.add_argument("--out", type=Path, required=True)
    root.add_argument("--frozen-out", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    frozen = json.loads(args.frozen_heads.read_text())
    if frozen.get("fractal_features") != []:
        raise ValueError("fractal features are forbidden")
    result = validate_predictions(
        json.loads(args.labels.read_text()), _load_predictions(args.prediction)
    )
    frozen["heldout_status"] = "passed" if result["passes"] else "failed"
    frozen["heldout_validation"] = str(args.out.resolve())
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.frozen_out.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
