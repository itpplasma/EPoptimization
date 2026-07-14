#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from spatial_atlas import (
    fixed_shell_nonideal_volumes,
    load_spatial_atlas,
    score_spatial_atlas,
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--topology", type=Path, nargs="+", required=True)
    root.add_argument("--out", type=Path, required=True)
    root.add_argument("--risk-out", type=Path, required=True)
    root.add_argument("--sigma-cells", type=float, default=1.0)
    root.add_argument("--classifier", choices=("topology", "jpar"), default="topology")
    root.add_argument("--shell-inner", type=float)
    root.add_argument("--shell-outer", type=float)
    return root


if __name__ == "__main__":
    args = parser().parse_args()
    atlas = load_spatial_atlas([path.resolve() for path in args.topology])
    result = score_spatial_atlas(
        atlas, sigma_cells=args.sigma_cells, classifier=args.classifier
    )
    payload = {
        "aliasing_half_range": result.aliasing_half_range,
        "refinement_changes": result.refinement_changes.tolist(),
        "refinement_interval": result.refinement_interval,
        "schema_name": "alpha-loss.spatial-barrier-score",
        "schema_version": 2,
        "classifier": args.classifier,
        "score": result.score,
        "shift_escape_volumes": result.shift_escape_volumes.tolist(),
        "shift_minimum_separator_widths": (
            result.shift_minimum_separator_widths.tolist()
        ),
        "shift_mean_separator_widths": result.shift_mean_separator_widths.tolist(),
        "shift_nonideal_volumes": result.shift_nonideal_volumes.tolist(),
        "shift_open_channel_fractions": (
            result.shift_open_channel_fractions.tolist()
        ),
        "shift_scores": result.shift_scores.tolist(),
        "shift_worst_channels": result.shift_worst_channels.tolist(),
        "sigma_cells": args.sigma_cells,
        "surfaces": result.surfaces.tolist(),
        "simple_sha256": str(atlas["simple_sha256"]),
        "trace_time": float(atlas["trace_time"]),
        "wout_sha256": str(atlas["wout_sha256"]),
    }
    if (args.shell_inner is None) != (args.shell_outer is None):
        raise ValueError("fixed shell requires both bounds")
    if args.shell_inner is not None:
        payload["shell_inner"] = args.shell_inner
        payload["shell_outer"] = args.shell_outer
        payload["shift_shell_nonideal_volumes"] = fixed_shell_nonideal_volumes(
            atlas,
            args.classifier,
            args.shell_inner,
            args.shell_outer,
        ).tolist()
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        args.risk_out,
        risk=result.risk,
        surfaces=result.surfaces,
        shifts=atlas["shifts"],
        lambda_values=atlas["lambda_values"],
        signs=atlas["signs"],
    )
