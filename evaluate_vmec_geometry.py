#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from Alan_objectives import MaxElongationPen, MirrorRatioPen


def evaluate(
    wout: Path,
    *,
    mirror_limit: float = 0.20,
    elongation_limit: float = 6.0,
) -> dict:
    from simsopt.mhd import Vmec

    if not 0.0 < mirror_limit < 1.0 or elongation_limit <= 1.0:
        raise ValueError("geometry limits are outside their physical ranges")
    vmec = Vmec(str(wout.resolve()), verbose=False)
    metrics = {
        "aspect": float(vmec.aspect()),
        "mean_iota": float(vmec.mean_iota()),
        "vacuum_well": float(vmec.vacuum_well()),
        "mirror_ratio": float(MirrorRatioPen(v=vmec, output_mirror=True)),
        "max_elongation": float(MaxElongationPen(vmec=vmec, return_elongation=True)),
    }
    if not np.isfinite(list(metrics.values())).all():
        raise ValueError("VMEC geometry metrics must be finite")
    constraints = [
        metrics["mirror_ratio"] / mirror_limit - 1.0,
        metrics["max_elongation"] / elongation_limit - 1.0,
    ]
    return {
        "schema_name": "alpha-loss.direct-geometry",
        "schema_version": 1,
        "wout": str(wout.resolve()),
        "metrics": metrics,
        "constraint_names": ["mirror_ratio", "max_elongation"],
        "constraint_limits": [mirror_limit, elongation_limit],
        "constraints": constraints,
        "feasible": bool(max(constraints) <= 0.0),
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--wout", type=Path, required=True)
    root.add_argument("--out", type=Path, required=True)
    root.add_argument("--mirror-limit", type=float, default=0.20)
    root.add_argument("--elongation-limit", type=float, default=6.0)
    return root


def main() -> None:
    args = parser().parse_args()
    document = evaluate(
        args.wout,
        mirror_limit=args.mirror_limit,
        elongation_limit=args.elongation_limit,
    )
    args.out.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
