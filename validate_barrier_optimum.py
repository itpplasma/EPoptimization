#!/usr/bin/env python3
"""Direct-loss validation of the best barrier-overlap candidate.

The barrier overlap is a proxy. Whether it steered the optimizer anywhere real
is only answerable by tracing the winning configuration the expensive way, on
the same contract the direct campaign uses: 100 ms, 256 particles, s = 0.25.

Run this after an optimizer finishes. It reads the ledger, picks the best
feasible candidate, and evaluates the direct loss for it and for the anchor,
so the pair can be compared.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from simple_direct import direct_loss_metrics


def best_feasible_record(ledger: dict) -> dict:
    records = [row for row in ledger["records"] if row.get("observation")]
    if not records:
        raise ValueError("ledger holds no successful evaluations")
    feasible = [
        row
        for row in records
        if max(row["observation"]["constraints"], default=1.0) <= 0.0
    ]
    pool = feasible or records
    return min(pool, key=lambda row: row["observation"]["value"])


def wout_for(campaign_root: Path, method: str, evaluation_id: int) -> Path:
    case = f"candidate-{evaluation_id:08d}"
    wave = campaign_root / method / f"eval-{evaluation_id:03d}"
    path = wave / "candidates" / case / f"wout_{case}.nc"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def evaluate(wout: Path, args: argparse.Namespace) -> dict:
    return direct_loss_metrics(
        wout,
        ntestpart=args.particles,
        expected_simple_sha256=args.simple_sha256,
        trace_time=args.trace_time,
        prompt_time=args.prompt_time,
        sbeg=args.birth_surface,
        seed=args.seed,
        simple_executable=args.simple_executable,
        timeout_s=args.timeout,
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--campaign-root", type=Path, required=True)
    root.add_argument("--method", required=True)
    root.add_argument("--out", type=Path, required=True)
    root.add_argument("--simple-executable", type=Path, required=True)
    root.add_argument("--simple-sha256", required=True)
    root.add_argument("--particles", type=int, default=256)
    root.add_argument("--seed", type=int, default=12345)
    root.add_argument("--birth-surface", type=float, default=0.25)
    root.add_argument("--prompt-time", type=float, default=0.001)
    root.add_argument("--trace-time", type=float, default=0.1)
    root.add_argument("--timeout", type=float, default=7200.0)
    return root


def main() -> None:
    args = parser().parse_args()
    ledger = json.loads(
        (args.campaign_root / args.method / "ledger.json").read_text()
    )
    if ledger.get("metric") != "barrier-overlap":
        raise ValueError("ledger was not produced by a barrier-overlap run")
    best = best_feasible_record(ledger)
    anchor = ledger["records"][0]

    results = {}
    for label, record in (("best", best), ("anchor", anchor)):
        if not record.get("observation"):
            continue
        wout = wout_for(args.campaign_root, args.method, record["evaluation_id"])
        results[label] = {
            "evaluation_id": record["evaluation_id"],
            "barrier_overlap": record["observation"]["value"],
            "constraints": record["observation"]["constraints"],
            "direct": evaluate(wout, args),
        }

    document = {
        "schema_name": "alpha-loss.barrier-optimum-validation",
        "schema_version": 1,
        "method": args.method,
        "metric": ledger["metric"],
        "budget": ledger["budget"],
        "seed": ledger["seed"],
        "direct_contract": {
            "particles": args.particles,
            "birth_surface": args.birth_surface,
            "trace_time": args.trace_time,
            "prompt_time": args.prompt_time,
            "seed": args.seed,
        },
        "results": results,
    }
    if "best" in results and "anchor" in results:
        document["direct_loss_change"] = float(
            results["best"]["direct"]["total_loss"]
            - results["anchor"]["direct"]["total_loss"]
        )
        document["proxy_moved_the_direct_loss"] = bool(
            np.isfinite(document["direct_loss_change"])
            and document["direct_loss_change"] < 0.0
        )
    args.out.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
