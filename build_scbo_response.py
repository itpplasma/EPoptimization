#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_direct_scout import loss_indicators, paired_summary


def build(args: argparse.Namespace) -> None:
    request = json.loads(args.request.read_text())
    if args.failure_kind:
        response = base_response(request, "failed", args.failure_kind)
    else:
        response = successful_response(request, args)
    args.out.write_text(
        json.dumps(response, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def successful_response(request: dict, args: argparse.Namespace) -> dict:
    result = json.loads(args.result.read_text())
    if result["status"] != "ok":
        raise ValueError("direct result is not successful")
    reference = loss_indicators(
        args.reference_times, args.prompt_time, args.trace_time
    )
    candidate = loss_indicators(
        args.result.parent / "direct" / "times_lost.dat",
        args.prompt_time,
        args.trace_time,
    )
    total = paired_summary(reference["total"], candidate["total"])
    late = paired_summary(reference["late"], candidate["late"])
    response = base_response(request, "ok", None)
    response["observation"] = {
        "value": total["change"],
        "variance": total["paired_se"] ** 2,
        "constraints": [late["change"] + args.late_target],
        "constraint_variances": [late["paired_se"] ** 2],
    }
    response["metrics"] = {
        "total_change": total["change"],
        "total_paired_se": total["paired_se"],
        "late_change": late["change"],
        "late_paired_se": late["paired_se"],
        "direct": result["direct"],
        "wout_sha256": result["wout_sha256"],
    }
    return response


def base_response(request: dict, status: str, failure_kind: str | None) -> dict:
    return {
        "candidate_id": int(request["candidate_id"]),
        "unit_x": [float(value) for value in request["unit_x"]],
        "observation": None,
        "status": status,
        "failure_kind": failure_kind,
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--request", type=Path, required=True)
    root.add_argument("--result", type=Path)
    root.add_argument("--reference-times", type=Path)
    root.add_argument("--out", type=Path, required=True)
    root.add_argument("--failure-kind")
    root.add_argument("--prompt-time", type=float, default=1.0e-3)
    root.add_argument("--trace-time", type=float, default=3.0e-1)
    root.add_argument("--late-target", type=float, default=0.01)
    return root


def validate(args: argparse.Namespace) -> None:
    if not args.failure_kind and (args.result is None or args.reference_times is None):
        raise ValueError("successful response requires result and reference times")


def main() -> None:
    args = parser().parse_args()
    validate(args)
    build(args)


if __name__ == "__main__":
    main()
