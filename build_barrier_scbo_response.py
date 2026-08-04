#!/usr/bin/env python3
"""Turn a barrier-overlap evaluation into an optimizer response.

Objective: barrier overlap. Constraints: mirror ratio, maximum elongation, and
prompt loss, all in the ``value / limit - 1 <= 0`` convention the geometry
evaluator already uses. Prompt loss is a constraint rather than a weighted
penalty because prompt losses are invisible to the overlap metric: a
prompt-lost particle carries classifier code 0, not 2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def base_response(request: dict, status: str, failure_kind: str | None) -> dict:
    return {
        "candidate_id": int(request["candidate_id"]),
        "unit_x": [float(value) for value in request["unit_x"]],
        "observation": None,
        "status": status,
        "failure_kind": failure_kind,
    }


#: Objective names. "discrete" is the classifier-counting metric; the smooth
#: variants replace every indicator with a mollified weight and are the ones a
#: differentiation tool could act on.
OBJECTIVES = ("discrete", "smooth-jpar", "smooth-topology")


def select_objective(barrier: dict, objective: str) -> float:
    if objective == "discrete":
        return barrier["barrier_overlap"]
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective}")
    smooth = barrier.get("smooth") or {}
    if not smooth.get("available"):
        raise ValueError(
            f"objective {objective} needs class_scores.dat; the SIMPLE build "
            "must carry the continuous classifier margins"
        )
    key = "smooth_barrier_overlap_" + objective.removeprefix("smooth-")
    if key not in smooth:
        raise ValueError(f"barrier result carries no {key}")
    return smooth[key]


def successful_response(
    request: dict,
    result: dict,
    geometry: dict,
    *,
    objective: str,
    inner_surface: float,
    outer_surface: float,
    particles_per_surface: int,
    prompt_limit: float,
) -> dict:
    if result.get("schema_name") != "alpha-loss.barrier-overlap-result":
        raise ValueError("barrier result has an unexpected schema")
    if geometry.get("schema_name") != "alpha-loss.direct-geometry":
        raise ValueError("geometry document has an unexpected schema")
    if not 0.0 < prompt_limit <= 1.0:
        raise ValueError("prompt loss limit must lie in (0, 1]")

    barrier = result["barrier"]
    expected = {
        "s_inner": inner_surface,
        "s_outer": outer_surface,
        "particles_per_surface": particles_per_surface,
    }
    for key, value in expected.items():
        if key not in barrier or not np.isclose(barrier[key], value):
            raise ValueError(f"barrier result has unexpected {key}")

    overlap = float(select_objective(barrier, objective))
    if not np.isfinite(overlap):
        raise ValueError(f"objective {objective} is not finite")
    if barrier["trapped_inner"] <= 0 or barrier["trapped_outer"] <= 0:
        raise ValueError("barrier surfaces carry no trapped particles")

    prompt_loss = float(barrier["prompt_loss_inner"])
    geometry_constraints = [float(value) for value in geometry["constraints"]]
    if len(geometry_constraints) != 2:
        raise ValueError("geometry constraints are incomplete")
    constraints = geometry_constraints + [prompt_loss / prompt_limit - 1.0]
    if not np.isfinite(constraints).all():
        raise ValueError("constraints must be finite")

    response = base_response(request, "ok", None)
    response["observation"] = {
        "value": overlap,
        "constraints": constraints,
        "constraint_variances": [0.0, 0.0, 0.0],
    }
    response["metrics"] = {
        "objective_name": (
            f"{objective}_barrier_overlap_{barrier['classifier']}"
            f"_s{inner_surface:g}_to_s{outer_surface:g}"
        ),
        "objective_kind": objective,
        "discrete_barrier_overlap": barrier["barrier_overlap"],
        "constraint_names": list(geometry["constraint_names"]) + ["prompt_loss"],
        "constraint_limits": list(geometry["constraint_limits"]) + [prompt_limit],
        "barrier": barrier,
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
    root.add_argument("--inner-surface", type=float, default=0.25)
    root.add_argument("--outer-surface", type=float, default=0.6)
    root.add_argument("--particles-per-surface", type=int, default=1024)
    root.add_argument("--prompt-limit", type=float, default=0.05)
    root.add_argument("--objective", choices=OBJECTIVES, default="discrete")
    return root


def main() -> None:
    args = parser().parse_args()
    request = json.loads(args.request.read_text())
    if args.failure_kind:
        response = base_response(request, "failed", args.failure_kind)
    else:
        if args.result is None or args.geometry is None:
            raise ValueError("successful response requires barrier and geometry results")
        response = successful_response(
            request,
            json.loads(args.result.read_text()),
            json.loads(args.geometry.read_text()),
            objective=args.objective,
            inner_surface=args.inner_surface,
            outer_surface=args.outer_surface,
            particles_per_surface=args.particles_per_surface,
            prompt_limit=args.prompt_limit,
        )
    args.out.write_text(
        json.dumps(response, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
