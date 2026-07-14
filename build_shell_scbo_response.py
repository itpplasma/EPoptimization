#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_direct_scout import loss_indicators, paired_summary


def base_response(request: dict, status: str, failure_kind: str | None) -> dict:
    return {
        "candidate_id": int(request["candidate_id"]),
        "unit_x": [float(value) for value in request["unit_x"]],
        "observation": None,
        "status": status,
        "failure_kind": failure_kind,
    }


def paired_shell_change(candidate: dict, reference: dict, slope: float) -> np.ndarray:
    if candidate["classifier"] != "topology" or reference["classifier"] != "topology":
        raise ValueError("fixed shell requires the topology classifier")
    for document in (candidate, reference):
        if not np.isclose(document.get("shell_inner"), 0.675):
            raise ValueError("fixed shell inner boundary must be s = 0.675")
        if not np.isclose(document.get("shell_outer"), 0.8):
            raise ValueError("fixed shell outer boundary must be s = 0.8")
    for key in ("surfaces", "simple_sha256", "trace_time"):
        if candidate[key] != reference[key]:
            raise ValueError(f"candidate and reference shell {key} differ")
    left = np.asarray(candidate["shift_shell_nonideal_volumes"], dtype=float)
    right = np.asarray(reference["shift_shell_nonideal_volumes"], dtype=float)
    if left.shape != right.shape or left.ndim != 1 or left.size < 2:
        raise ValueError("fixed shell values differ or have fewer than two shifts")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("fixed shell values must be finite")
    return slope * (left - right)


def prompt_change(
    result: dict,
    candidate_times: Path,
    reference_times: Path,
    prompt_time: float,
) -> dict:
    if result.get("status") != "ok":
        raise ValueError("prompt direct result is not successful")
    direct = result["direct"]
    if not np.isclose(result["birth_surface"], 0.25):
        raise ValueError("prompt direct result must start at s = 0.25")
    if not np.isclose(result["prompt_time"], prompt_time):
        raise ValueError("prompt direct result uses another prompt boundary")
    trace_time = float(result["trace_time"])
    reference = loss_indicators(reference_times, prompt_time, trace_time)["prompt"]
    candidate = loss_indicators(candidate_times, prompt_time, trace_time)["prompt"]
    if len(reference) != len(candidate) or len(candidate) != int(direct["particles"]):
        raise ValueError("candidate and reference prompt particle counts differ")
    return paired_summary(reference, candidate)


def successful_response(
    request: dict,
    shell: dict,
    reference_shell: dict,
    shell_head: dict,
    prompt: dict,
    gamma_c: float,
    reference_gamma_c: float,
    prompt_tolerance: float,
    late_tolerance: float,
) -> dict:
    if shell_head.get("feature") != "topology_fixed_outer_shell_volume":
        raise ValueError("frozen late head is not the fixed outer shell volume")
    if shell_head.get("status") not in {"passed", "pilot"}:
        raise ValueError("fixed outer shell evidence does not permit optimization")
    slope = float(shell_head["slope"])
    if not np.isfinite(slope) or slope <= 0.0:
        raise ValueError("fixed outer shell slope must be positive")
    if min(prompt_tolerance, late_tolerance) < 0.0:
        raise ValueError("proxy tolerances must be nonnegative")
    if not np.isfinite(gamma_c) or not np.isfinite(reference_gamma_c):
        raise ValueError("Gamma-c values must be finite")
    late = paired_shell_change(shell, reference_shell, slope)
    prompt_mean = float(prompt["change"])
    prompt_variance = float(prompt["paired_se"]) ** 2
    objective_shifts = late + prompt_mean
    constraints = (late - late_tolerance).tolist()
    constraint_names = [
        f"late_shell_prediction_shift_{index}" for index in range(len(late))
    ]
    constraints.append(prompt_mean - prompt_tolerance)
    constraint_names.append("prompt_loss_change")
    gamma_limit = max(1.5 * reference_gamma_c, reference_gamma_c + 0.002)
    constraints.append(float(gamma_c - gamma_limit))
    constraint_names.append("gamma_c_s03")
    response = base_response(request, "ok", None)
    response["observation"] = {
        "value": float(np.mean(objective_shifts)),
        "variance": float(np.var(late, ddof=1) / len(late) + prompt_variance),
        "constraints": constraints,
        "constraint_variances": [0.0] * len(late)
        + [prompt_variance, 0.0],
    }
    response["metrics"] = {
        "constraint_names": constraint_names,
        "late_feature": shell_head["feature"],
        "late_predictions": late.tolist(),
        "late_tolerance": late_tolerance,
        "prompt_change": prompt_mean,
        "prompt_paired_se": float(prompt["paired_se"]),
        "prompt_tolerance": prompt_tolerance,
        "predicted_total_change": float(np.mean(objective_shifts)),
        "gamma_c_s03": gamma_c,
        "gamma_c_limit_s03": gamma_limit,
        "wout_sha256": shell["wout_sha256"],
    }
    return response


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--request", type=Path, required=True)
    root.add_argument("--shell", type=Path)
    root.add_argument("--reference-shell", type=Path)
    root.add_argument("--shell-head", type=Path)
    root.add_argument("--prompt-result", type=Path)
    root.add_argument("--reference-prompt-times", type=Path)
    root.add_argument("--gamma-c", type=float)
    root.add_argument("--reference-gamma-c", type=float)
    root.add_argument("--prompt-time", type=float, default=1.0e-3)
    root.add_argument("--prompt-tolerance", type=float, default=0.005)
    root.add_argument("--late-tolerance", type=float, default=0.0)
    root.add_argument("--failure-kind")
    root.add_argument("--out", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    request = json.loads(args.request.read_text())
    if args.failure_kind:
        response = base_response(request, "failed", args.failure_kind)
    else:
        paths = (
            args.shell,
            args.reference_shell,
            args.shell_head,
            args.prompt_result,
            args.reference_prompt_times,
        )
        if any(path is None for path in paths):
            raise ValueError("successful response requires every shell and prompt input")
        if args.gamma_c is None or args.reference_gamma_c is None:
            raise ValueError("successful response requires both Gamma-c values")
        prompt_result = json.loads(args.prompt_result.read_text())
        prompt = prompt_change(
            prompt_result,
            args.prompt_result.parent / "direct" / "times_lost.dat",
            args.reference_prompt_times,
            args.prompt_time,
        )
        shell = json.loads(args.shell.read_text())
        if prompt_result["wout_sha256"] != shell["wout_sha256"]:
            raise ValueError("prompt and shell evaluations use different equilibria")
        response = successful_response(
            request,
            shell,
            json.loads(args.reference_shell.read_text()),
            json.loads(args.shell_head.read_text()),
            prompt,
            args.gamma_c,
            args.reference_gamma_c,
            args.prompt_tolerance,
            args.late_tolerance,
        )
    args.out.write_text(
        json.dumps(response, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
