#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from classifier_proxy import classifier_surface_features


def _jpar(data, path: Path) -> np.ndarray:
    if "jpar" in data.files:
        return data["jpar"]
    classes = np.loadtxt(path.parent / "class_parts.dat", ndmin=2)
    particle_index = data["particle_index"]
    selected = particle_index >= 0
    indices = particle_index[selected]
    if classes.shape[1] < 4 or np.max(indices, initial=-1) >= len(classes):
        raise ValueError("class_parts.dat cannot supply the J-parallel classifier")
    jpar = np.zeros(particle_index.shape, dtype=np.int8)
    jpar[selected] = classes[indices, 3].astype(np.int8)
    return jpar


def evaluate(path: Path) -> dict:
    data = np.load(path)
    result = classifier_surface_features(
        data["topology"],
        _jpar(data, path),
        data["particle_index"],
        data["passing"],
        data["weights"],
    )
    return {
        "schema_name": "alpha-loss.classifier-prompt-features",
        "schema_version": 1,
        "surface": float(data["surface"]),
        "unclassified_fraction": result.unclassified_fraction,
        "nonideal_fraction": result.nonideal_fraction,
        "jpar_nonideal_fraction": result.jpar_nonideal_fraction,
        "passing_fraction": result.passing_fraction,
        "unclassified_component_fraction": result.unclassified_component_fraction,
        "unclassified_component_count": result.unclassified_component_count,
        "aliasing_half_range": result.aliasing_half_range,
        "shift_unclassified_fractions": (
            result.shift_unclassified_fractions.tolist()
        ),
        "shift_nonideal_fractions": result.shift_nonideal_fractions.tolist(),
        "shift_jpar_nonideal_fractions": (
            result.shift_jpar_nonideal_fractions.tolist()
        ),
        "simple_sha256": str(data["simple_sha256"]),
        "trace_time": float(data["trace_time"]),
        "wout_sha256": str(data["wout_sha256"]),
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--topology", type=Path, required=True)
    root.add_argument("--out", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    payload = evaluate(args.topology.resolve())
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
