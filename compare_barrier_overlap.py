#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from simple_barrier import paired_barrier_bootstrap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--classifier", choices=("jpar", "topology"), default="topology")
    args = parser.parse_args()
    result = paired_barrier_bootstrap(
        args.reference / "inner",
        args.reference / "outer",
        args.candidate / "inner",
        args.candidate / "outer",
        nbins=args.bins,
        col={"jpar": 3, "topology": 4}[args.classifier],
        replicates=args.replicates,
        seed=args.seed,
    )
    result["proxy_resolved"] = result["change"] <= -2.0 * result["paired_standard_error"]
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
