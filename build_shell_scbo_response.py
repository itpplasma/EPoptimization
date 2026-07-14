#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_direct_scout import paired_summary


PROMPT_END = 1.0e-4
EARLY_END = 1.0e-3


def base_response(request: dict, status: str, failure_kind: str | None) -> dict:
    response = {
        "candidate_id": int(request["candidate_id"]),
        "unit_x": [float(value) for value in request["unit_x"]],
        "observation": None,
        "pareto_observation": None,
        "status": status,
        "failure_kind": failure_kind,
    }
    if "generation" in request:
        response["generation"] = int(request["generation"])
    return response


def taxonomy_indicators(
    path: Path,
    prompt_end: float = PROMPT_END,
    early_end: float = EARLY_END,
) -> dict[str, np.ndarray]:
    if not 0.0 < prompt_end < early_end:
        raise ValueError("loss boundaries must satisfy 0 < prompt < early")
    particles = np.loadtxt(path, ndmin=2)
    expected = np.arange(1, len(particles) + 1)
    if particles.shape[1] < 2 or not np.array_equal(
        particles[:, 0].astype(int), expected
    ):
        raise ValueError(f"particle inventory differs in {path}")
    times = particles[:, 1]
    prompt = (times > 0.0) & (times < prompt_end)
    early = (times >= prompt_end) & (times < early_end)
    return {"prompt": prompt, "early": early, "short": prompt | early}


def paired_direct_changes(
    result: dict,
    candidate_times: Path,
    reference_times: Path,
    prompt_end: float,
    early_end: float,
) -> dict[str, dict[str, float]]:
    if result.get("status") != "ok":
        raise ValueError("short direct result is not successful")
    if not np.isclose(result["birth_surface"], 0.25):
        raise ValueError("short direct result must start at s = 0.25")
    if not np.isclose(result["prompt_time"], prompt_end):
        raise ValueError("short direct result uses another prompt boundary")
    if float(result["trace_time"]) <= early_end:
        raise ValueError("short direct trace must extend beyond the early boundary")
    reference = taxonomy_indicators(reference_times, prompt_end, early_end)
    candidate = taxonomy_indicators(candidate_times, prompt_end, early_end)
    particles = int(result["direct"]["particles"])
    if any(len(values) != particles for values in (*reference.values(), *candidate.values())):
        raise ValueError("candidate and reference short particle counts differ")
    return {
        name: paired_summary(reference[name], candidate[name])
        for name in ("prompt", "early", "short")
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


def validate_head(shell_head: dict) -> float:
    if shell_head.get("feature") != "topology_fixed_outer_shell_volume":
        raise ValueError("frozen late head is not the fixed outer shell volume")
    if shell_head.get("status") != "passed":
        raise ValueError("fixed outer shell evidence does not permit optimization")
    if shell_head.get("target") != "loss_1_100ms":
        raise ValueError("fixed outer shell head targets another loss window")
    slope = float(shell_head["slope"])
    if not np.isfinite(slope) or slope <= 0.0:
        raise ValueError("fixed outer shell slope must be positive")
    return slope


def successful_response(
    request: dict,
    shell: dict,
    reference_shell: dict,
    shell_head: dict,
    direct: dict[str, dict[str, float]],
    gamma_c: float,
    reference_gamma_c: float,
    prompt_tolerance: float,
    early_tolerance: float,
    late_tolerance: float,
) -> dict:
    slope = validate_head(shell_head)
    if min(prompt_tolerance, early_tolerance, late_tolerance) < 0.0:
        raise ValueError("loss tolerances must be nonnegative")
    if not np.isfinite(gamma_c) or not np.isfinite(reference_gamma_c):
        raise ValueError("Gamma-c values must be finite")
    late = paired_shell_change(shell, reference_shell, slope)
    late_mean = float(np.mean(late))
    late_variance = float(np.var(late, ddof=1) / len(late))
    prompt_mean = float(direct["prompt"]["change"])
    early_mean = float(direct["early"]["change"])
    short_variance = float(direct["short"]["paired_se"]) ** 2
    predicted_total = prompt_mean + early_mean + late_mean
    gamma_limit = max(1.5 * reference_gamma_c, reference_gamma_c + 0.002)
    scalar_constraints = [
        *(late - late_tolerance).tolist(),
        prompt_mean - prompt_tolerance,
        early_mean - early_tolerance,
        float(gamma_c - gamma_limit),
    ]
    scalar_names = [
        *(f"late_shell_prediction_shift_{index}" for index in range(len(late))),
        "prompt_loss_change",
        "early_loss_change",
        "gamma_c_s03",
    ]
    response = base_response(request, "ok", None)
    response["observation"] = {
        "value": predicted_total,
        "variance": short_variance + late_variance,
        "constraints": scalar_constraints,
        "constraint_variances": [
            *([0.0] * len(late)),
            float(direct["prompt"]["paired_se"]) ** 2,
            float(direct["early"]["paired_se"]) ** 2,
            0.0,
        ],
    }
    response["pareto_observation"] = {
        "values": [prompt_mean, early_mean, late_mean],
        "variances": [
            float(direct["prompt"]["paired_se"]) ** 2,
            float(direct["early"]["paired_se"]) ** 2,
            late_variance,
        ],
        "constraints": [float(gamma_c - gamma_limit)],
        "constraint_variances": [0.0],
    }
    response["metrics"] = {
        "scalar_constraint_names": scalar_names,
        "pareto_constraint_names": ["gamma_c_s03"],
        "pareto_objective_names": [
            "prompt_0_0p1ms_change",
            "early_0p1_1ms_change",
            "predicted_late_1_100ms_change",
        ],
        "prompt_end": PROMPT_END,
        "early_end": EARLY_END,
        "late_target_end": 0.1,
        "late_feature": shell_head["feature"],
        "late_predictions": late.tolist(),
        "late_tolerance": late_tolerance,
        "prompt_change": prompt_mean,
        "prompt_paired_se": float(direct["prompt"]["paired_se"]),
        "prompt_tolerance": prompt_tolerance,
        "early_change": early_mean,
        "early_paired_se": float(direct["early"]["paired_se"]),
        "early_tolerance": early_tolerance,
        "short_change": float(direct["short"]["change"]),
        "short_paired_se": float(direct["short"]["paired_se"]),
        "predicted_total_change": predicted_total,
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
    root.add_argument("--short-result", type=Path)
    root.add_argument("--reference-short-times", type=Path)
    root.add_argument("--gamma-c", type=float)
    root.add_argument("--reference-gamma-c", type=float)
    root.add_argument("--prompt-end", type=float, default=PROMPT_END)
    root.add_argument("--early-end", type=float, default=EARLY_END)
    root.add_argument("--prompt-tolerance", type=float, default=0.005)
    root.add_argument("--early-tolerance", type=float, default=0.005)
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
            args.short_result,
            args.reference_short_times,
        )
        if any(path is None for path in paths):
            raise ValueError("successful response requires every shell and short input")
        if args.gamma_c is None or args.reference_gamma_c is None:
            raise ValueError("successful response requires both Gamma-c values")
        short_result = json.loads(args.short_result.read_text())
        direct = paired_direct_changes(
            short_result,
            args.short_result.parent / "direct" / "times_lost.dat",
            args.reference_short_times,
            args.prompt_end,
            args.early_end,
        )
        shell = json.loads(args.shell.read_text())
        if short_result["wout_sha256"] != shell["wout_sha256"]:
            raise ValueError("short and shell evaluations use different equilibria")
        response = successful_response(
            request,
            shell,
            json.loads(args.reference_shell.read_text()),
            json.loads(args.shell_head.read_text()),
            direct,
            args.gamma_c,
            args.reference_gamma_c,
            args.prompt_tolerance,
            args.early_tolerance,
            args.late_tolerance,
        )
    args.out.write_text(
        json.dumps(response, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
