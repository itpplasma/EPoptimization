#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


PROMPT_ORDER = (
    "prompt_topology_nonideal",
    "prompt_jpar_nonideal",
    "prompt_unclassified",
)
LATE_ORDER = (
    "late_topology_escape",
    "late_jpar_escape",
    "late_topology_nonideal",
    "late_jpar_nonideal",
)


def _select(
    fits: dict, order: tuple[str, ...], target: str, require_radial: bool = False
) -> dict:
    for name in order:
        result = fits[name]
        radial_passes = not require_radial or result.get("radial", {}).get("passes")
        if result["passes"] and radial_passes:
            return {
                "feature": name,
                "slope": float(result["slope"]),
                "training_spearman": float(result["spearman"]),
            }
    raise ValueError(f"no {target} classifier head passes the frozen training gates")


def freeze(calibration: dict) -> dict:
    if calibration.get("fractal_features") != []:
        raise ValueError("fractal features are forbidden")
    radial_surfaces = calibration.get("radial_levels", {}).get("fine")
    if (
        not isinstance(radial_surfaces, list)
        or len(radial_surfaces) < 2
        or radial_surfaces[0] != "s0p25000"
        or len(set(radial_surfaces)) != len(radial_surfaces)
    ):
        raise ValueError("calibration must define the converged fine radial grid")
    fits = calibration["fits"]
    prompt = _select(fits, PROMPT_ORDER, "prompt")
    late = _select(fits, LATE_ORDER, "late", require_radial=True)
    return {
        "schema_name": "alpha-loss.frozen-classifier-heads",
        "schema_version": 1,
        "birth_surface": 0.25,
        "trace_time": 0.02,
        "prompt": prompt,
        "late": late,
        "radial_surfaces": radial_surfaces,
        "gamma_c_role": "independent_guard",
        "fractal_features": [],
        "heldout_status": "pending",
        "horizon_status": "pending",
        "radial_status": "passed",
        "angular_status": "pending",
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--calibration", type=Path, required=True)
    root.add_argument("--out", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    calibration = json.loads(args.calibration.read_text())
    args.out.write_text(json.dumps(freeze(calibration), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
