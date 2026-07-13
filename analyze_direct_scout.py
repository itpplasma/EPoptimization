#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def analyze(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text())
    reference = loss_indicators(
        args.reference_times, args.prompt_time, args.trace_time
    )
    if len(reference["total"]) != args.particles:
        raise ValueError("reference particle count differs from the scout contract")
    rows = [analyze_case(case, args, reference) for case in manifest["cases"]]
    successful = [row for row in rows if row["status"] == "ok"]
    if not successful:
        raise RuntimeError("direct scout contains no successful traces")
    ranked = sorted(
        successful,
        key=lambda row: (row["total"]["change"], row["late"]["change"]),
    )
    document = {
        "schema_name": "alpha-loss.data-informed-direct-scout",
        "schema_version": 1,
        "particles": len(reference["total"]),
        "seed": args.seed,
        "prompt_time": args.prompt_time,
        "trace_time": args.trace_time,
        "late_reduction_target": args.late_target,
        "reference": window_counts(reference),
        "successful": len(successful),
        "failed": len(rows) - len(successful),
        "best_total_case": ranked[0]["case"],
        "best_joint_cases": [row["case"] for row in ranked[: args.keep]],
        "cases": rows,
    }
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def analyze_case(case: dict, args, reference: dict[str, np.ndarray]) -> dict:
    path = args.results / case["case"] / "result.json"
    result = json.loads(path.read_text())
    row = {"case": case["case"], "unit": case["unit"], "status": result["status"]}
    if result["status"] != "ok":
        row["failure_kind"] = result["failure_kind"]
        return row
    if result["particles"] != len(reference["total"]) or result["seed"] != args.seed:
        raise ValueError(f"trace contract differs for {case['case']}")
    candidate = loss_indicators(
        path.parent / "direct" / "times_lost.dat", args.prompt_time, args.trace_time
    )
    counts = window_counts(candidate)
    for name in ("total", "prompt", "late"):
        if counts[name]["count"] != result["direct"][f"{name}_count"]:
            raise ValueError(f"stored {name} count differs for {case['case']}")
        row[name] = paired_summary(reference[name], candidate[name])
        row[name].update(counts[name])
    row["objective"] = counts["total"]["loss"]
    row["threshold_score"] = result.get("threshold_score", result["objective"])
    row["wout_sha256"] = result["wout_sha256"]
    row["observation"] = {
        "value": row["objective"],
        "variance": row["total"]["paired_se"] ** 2,
        "constraints": [row["late"]["change"] + args.late_target],
        "constraint_variances": [row["late"]["paired_se"] ** 2],
    }
    return row


def loss_indicators(
    path: Path, prompt_time: float, trace_time: float
) -> dict[str, np.ndarray]:
    particles = np.loadtxt(path, ndmin=2)
    expected = np.arange(1, len(particles) + 1)
    if particles.shape[1] < 2 or not np.array_equal(particles[:, 0].astype(int), expected):
        raise ValueError(f"particle inventory differs in {path}")
    times = particles[:, 1]
    prompt = (times > 0.0) & (times <= prompt_time)
    late = (times > prompt_time) & (times < trace_time)
    return {"prompt": prompt, "late": late, "total": prompt | late}


def paired_summary(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    if reference.shape != candidate.shape or reference.ndim != 1:
        raise ValueError("paired particle windows differ")
    difference = candidate.astype(float) - reference.astype(float)
    standard_error = float(np.std(difference, ddof=1) / np.sqrt(len(difference)))
    return {"change": float(np.mean(difference)), "paired_se": standard_error}


def window_counts(windows: dict[str, np.ndarray]) -> dict[str, dict[str, float | int]]:
    return {
        name: {"count": int(values.sum()), "loss": float(values.mean())}
        for name, values in windows.items()
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--manifest", type=Path, required=True)
    root.add_argument("--results", type=Path, required=True)
    root.add_argument("--reference-times", type=Path, required=True)
    root.add_argument("--output", type=Path, required=True)
    root.add_argument("--particles", type=int, default=64)
    root.add_argument("--seed", type=int, default=12345)
    root.add_argument("--prompt-time", type=float, default=1.0e-3)
    root.add_argument("--trace-time", type=float, default=3.0e-1)
    root.add_argument("--keep", type=int, default=16)
    root.add_argument("--late-target", type=float, default=0.01)
    return root


def main() -> None:
    analyze(parser().parse_args())


if __name__ == "__main__":
    main()
