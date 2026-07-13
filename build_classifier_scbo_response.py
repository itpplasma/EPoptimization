#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


FEATURE_FIELDS = {
    "prompt_unclassified": ("prompt", "shift_unclassified_fractions"),
    "prompt_topology_nonideal": ("prompt", "shift_nonideal_fractions"),
    "prompt_jpar_nonideal": ("prompt", "shift_jpar_nonideal_fractions"),
    "late_topology_escape": ("topology", "shift_escape_volumes"),
    "late_topology_nonideal": ("topology", "shift_nonideal_volumes"),
    "late_jpar_escape": ("jpar", "shift_escape_volumes"),
    "late_jpar_nonideal": ("jpar", "shift_nonideal_volumes"),
}


def _paired_change(candidate: dict, reference: dict, key: str) -> np.ndarray:
    left = np.asarray(candidate[key], dtype=float)
    right = np.asarray(reference[key], dtype=float)
    if left.shape != right.shape or left.ndim != 1 or left.size < 2:
        raise ValueError(f"paired proxy field {key} differs or has fewer than two shifts")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError(f"paired proxy field {key} must be finite")
    return left - right


def _mean_variance(values: np.ndarray) -> float:
    return float(np.var(values, ddof=1) / len(values))


def _validate_inputs(
    topology: dict,
    reference_topology: dict,
    jpar: dict,
    reference_jpar: dict,
    prompt: dict,
    reference_prompt: dict,
) -> None:
    if topology["classifier"] != "topology" or jpar["classifier"] != "jpar":
        raise ValueError("candidate late-proxy classifiers differ")
    if (
        reference_topology["classifier"] != "topology"
        or reference_jpar["classifier"] != "jpar"
    ):
        raise ValueError("reference late-proxy classifiers differ")
    for candidate, reference in (
        (topology, reference_topology),
        (jpar, reference_jpar),
    ):
        if candidate["surfaces"] != reference["surfaces"]:
            raise ValueError("candidate and reference radial grids differ")
        if candidate["simple_sha256"] != reference["simple_sha256"]:
            raise ValueError("candidate and reference SIMPLE executables differ")
    if prompt["surface"] != 0.25 or reference_prompt["surface"] != 0.25:
        raise ValueError("prompt proxy must use birth surface s = 0.25")
    hashes = {topology["wout_sha256"], jpar["wout_sha256"], prompt["wout_sha256"]}
    reference_hashes = {
        reference_topology["wout_sha256"],
        reference_jpar["wout_sha256"],
        reference_prompt["wout_sha256"],
    }
    if len(hashes) != 1 or len(reference_hashes) != 1:
        raise ValueError("proxy heads do not use one equilibrium")
    documents = (
        topology,
        reference_topology,
        jpar,
        reference_jpar,
        prompt,
        reference_prompt,
    )
    if len({item["simple_sha256"] for item in documents}) != 1:
        raise ValueError("proxy heads do not use one SIMPLE executable")
    if len({float(item["trace_time"]) for item in documents}) != 1:
        raise ValueError("proxy heads do not use one trace time")


def successful_response(
    request: dict,
    topology: dict,
    reference_topology: dict,
    jpar: dict,
    reference_jpar: dict,
    prompt: dict,
    reference_prompt: dict,
    frozen_heads: dict,
    gamma_c: float,
    reference_gamma_c: float,
    prompt_tolerance: float,
) -> dict:
    if not np.isfinite(gamma_c) or not np.isfinite(reference_gamma_c):
        raise ValueError("Gamma-c values must be finite")
    if reference_gamma_c < 0.0 or prompt_tolerance < 0.0:
        raise ValueError("reference Gamma-c and prompt tolerance must be nonnegative")
    _validate_inputs(
        topology,
        reference_topology,
        jpar,
        reference_jpar,
        prompt,
        reference_prompt,
    )
    if frozen_heads.get("fractal_features") != []:
        raise ValueError("fractal features are forbidden")
    for gate in ("heldout_status", "horizon_status", "radial_status", "angular_status"):
        if frozen_heads.get(gate) != "passed":
            raise ValueError(f"frozen classifier head gate {gate} has not passed")
    documents = {
        "topology": (topology, reference_topology),
        "jpar": (jpar, reference_jpar),
        "prompt": (prompt, reference_prompt),
    }
    predictions = {}
    for target in ("late", "prompt"):
        head = frozen_heads[target]
        feature = head["feature"]
        if feature not in FEATURE_FIELDS:
            raise ValueError(f"unknown frozen classifier feature {feature}")
        source, field = FEATURE_FIELDS[feature]
        candidate, reference = documents[source]
        predictions[target] = float(head["slope"]) * _paired_change(
            candidate, reference, field
        )
    late_prediction = predictions["late"]
    prompt_prediction = predictions["prompt"]
    constraints = late_prediction.tolist()
    names = [f"late_prediction_shift_{index}" for index in range(len(late_prediction))]
    constraints.extend((prompt_prediction - prompt_tolerance).tolist())
    names.extend(
        f"prompt_prediction_shift_{index}" for index in range(len(prompt_prediction))
    )
    gamma_limit = max(1.5 * reference_gamma_c, reference_gamma_c + 0.002)
    constraints.append(float(gamma_c - gamma_limit))
    names.append("gamma_c_s03")
    response = base_response(request, "ok", None)
    response["observation"] = {
        "value": float(np.mean(late_prediction)),
        "variance": _mean_variance(late_prediction),
        "constraints": constraints,
        "constraint_variances": [0.0] * len(constraints),
    }
    response["metrics"] = {
        "constraint_names": names,
        "late_feature": frozen_heads["late"]["feature"],
        "late_predictions": late_prediction.tolist(),
        "prompt_feature": frozen_heads["prompt"]["feature"],
        "prompt_predictions": prompt_prediction.tolist(),
        "prompt_tolerance": prompt_tolerance,
        "gamma_c_s03": gamma_c,
        "gamma_c_limit_s03": gamma_limit,
        "wout_sha256": prompt["wout_sha256"],
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
    root.add_argument("--topology", type=Path)
    root.add_argument("--reference-topology", type=Path)
    root.add_argument("--jpar", type=Path)
    root.add_argument("--reference-jpar", type=Path)
    root.add_argument("--prompt", type=Path)
    root.add_argument("--reference-prompt", type=Path)
    root.add_argument("--frozen-heads", type=Path)
    root.add_argument("--gamma-c", type=float)
    root.add_argument("--reference-gamma-c", type=float)
    root.add_argument("--prompt-tolerance", type=float, default=0.005)
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
            args.topology,
            args.reference_topology,
            args.jpar,
            args.reference_jpar,
            args.prompt,
            args.reference_prompt,
            args.frozen_heads,
        )
        if any(path is None for path in paths):
            raise ValueError("successful response requires every proxy input")
        if args.gamma_c is None or args.reference_gamma_c is None:
            raise ValueError("successful response requires both Gamma-c values")
        documents = [json.loads(path.read_text()) for path in paths]
        response = successful_response(
            request,
            *documents,
            args.gamma_c,
            args.reference_gamma_c,
            args.prompt_tolerance,
        )
    args.out.write_text(json.dumps(response, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
