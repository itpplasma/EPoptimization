#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from spatial_atlas import load_spatial_atlas, score_spatial_atlas


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--topology", type=Path, nargs="+", required=True)
    root.add_argument("--out", type=Path, required=True)
    root.add_argument("--risk-out", type=Path, required=True)
    root.add_argument("--sigma-cells", type=float, default=1.0)
    return root


if __name__ == "__main__":
    args = parser().parse_args()
    atlas = load_spatial_atlas([path.resolve() for path in args.topology])
    result = score_spatial_atlas(atlas, sigma_cells=args.sigma_cells)
    payload = {
        "aliasing_half_range": result.aliasing_half_range,
        "refinement_changes": result.refinement_changes.tolist(),
        "refinement_interval": result.refinement_interval,
        "schema_name": "alpha-loss.spatial-barrier-score",
        "schema_version": 1,
        "score": result.score,
        "shift_scores": result.shift_scores.tolist(),
        "shift_worst_channels": result.shift_worst_channels.tolist(),
        "sigma_cells": args.sigma_cells,
        "surfaces": result.surfaces.tolist(),
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        args.risk_out,
        risk=result.risk,
        surfaces=result.surfaces,
        shifts=atlas["shifts"],
        lambda_values=atlas["lambda_values"],
        signs=atlas["signs"],
    )
