#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def jeffreys_variance(losses: int, particles: int) -> float:
    if particles < 1 or not 0 <= losses <= particles:
        raise ValueError("binomial counts are inconsistent")
    alpha = losses + 0.5
    beta = particles - losses + 0.5
    total = alpha + beta
    return float(alpha * beta / (total * total * (total + 1.0)))


def base_response(request: dict, status: str, failure_kind: str | None) -> dict:
    return {
        "candidate_id": int(request["candidate_id"]),
        "unit_x": [float(value) for value in request["unit_x"]],
        "observation": None,
        "status": status,
        "failure_kind": failure_kind,
    }


def successful_response(
    request: dict,
    result: dict,
    geometry: dict,
    *,
    particles: int,
    seed: int,
    birth_surface: float,
    trace_time: float,
) -> dict:
    direct = result.get("direct", {})
    expected = {
        "particles": particles,
        "seed": seed,
        "birth_surface": birth_surface,
        "trace_time": trace_time,
    }
    for key, value in expected.items():
        if key not in result or not np.isclose(result[key], value):
            raise ValueError(f"direct result has unexpected {key}")
    total_count = int(direct["total_count"])
    prompt_count = int(direct["prompt_count"])
    late_count = int(direct["late_count"])
    if total_count != prompt_count + late_count:
        raise ValueError("direct loss windows do not partition total loss")
    total_loss = float(direct["total_loss"])
    if not np.isclose(total_loss, total_count / particles):
        raise ValueError("direct total loss differs from the particle count")
    constraints = [float(value) for value in geometry["constraints"]]
    if geometry.get("schema_name") != "alpha-loss.direct-geometry":
        raise ValueError("geometry document has an unexpected schema")
    if len(constraints) != 2 or not np.isfinite(constraints).all():
        raise ValueError("geometry constraints are incomplete")

    response = base_response(request, "ok", None)
    response["observation"] = {
        "value": total_loss,
        "variance": jeffreys_variance(total_count, particles),
        "constraints": constraints,
        "constraint_variances": [0.0, 0.0],
    }
    response["metrics"] = {
        "objective_name": "direct_total_loss_fraction_100ms_s025",
        "constraint_names": geometry["constraint_names"],
        "constraint_limits": geometry["constraint_limits"],
        "direct": direct,
        "geometry": geometry["metrics"],
        "geometry_feasible": bool(geometry["feasible"]),
        "wout_sha256": result["wout_sha256"],
    }
    return response


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--request", type=Path, required=True)
    root.add_argument("--result", type=Path)
    root.add_argument("--geometry", type=Path)
    root.add_argument("--out", type=Path, required=True)
    root.add_argument("--failure-kind")
    root.add_argument("--particles", type=int, default=256)
    root.add_argument("--seed", type=int, default=12345)
    root.add_argument("--birth-surface", type=float, default=0.25)
    root.add_argument("--trace-time", type=float, default=0.1)
    return root


def main() -> None:
    args = parser().parse_args()
    request = json.loads(args.request.read_text())
    if args.failure_kind:
        response = base_response(request, "failed", args.failure_kind)
    else:
        if args.result is None or args.geometry is None:
            raise ValueError("successful response requires direct and geometry results")
        response = successful_response(
            request,
            json.loads(args.result.read_text()),
            json.loads(args.geometry.read_text()),
            particles=args.particles,
            seed=args.seed,
            birth_surface=args.birth_surface,
            trace_time=args.trace_time,
        )
    args.out.write_text(
        json.dumps(response, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
