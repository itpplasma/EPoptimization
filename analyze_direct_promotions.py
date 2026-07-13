#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_direct_scout import loss_indicators, paired_summary, window_counts


SIMPLE_SHA256 = "cb4a8970c406a817fce12fb12d129141b43ef3046562edb66bc5f447da7fc937"
BASE_WOUT_SHA256 = "f7a37a943a0067e5550b01945a7f91e82d13fd0dd2a3eb95afb55b641e9ffba3"


def read_manifest(path: Path) -> dict[int, list[dict]]:
    candidates = defaultdict(list)
    for line in path.read_text().splitlines():
        case, candidate, _wout, wout_sha, seed = line.split("\t")
        candidates[int(candidate)].append(
            {"case": case, "wout_sha256": wout_sha, "seed": int(seed)}
        )
    if not candidates:
        raise ValueError("promotion manifest is empty")
    return dict(candidates)


def validate_result(
    path: Path,
    particles: int,
    seed: int,
    wout_sha256: str,
    birth_surface: float,
) -> dict:
    result = json.loads(path.read_text())
    expected = {
        "status": "ok",
        "particles": particles,
        "seed": seed,
        "wout_sha256": wout_sha256,
        "simple_sha256": SIMPLE_SHA256,
        "birth_surface": birth_surface,
        "prompt_time": 0.001,
        "trace_time": 0.3,
    }
    mismatches = {
        key: (result.get(key), value)
        for key, value in expected.items()
        if result.get(key) != value
    }
    if mismatches:
        raise ValueError(f"invalid direct result contract in {path}: {mismatches}")
    return result


def load_case(
    path: Path,
    particles: int,
    seed: int,
    wout_sha256: str,
    birth_surface: float,
) -> dict:
    result = validate_result(
        path / "result.json", particles, seed, wout_sha256, birth_surface
    )
    windows = loss_indicators(path / "direct" / "times_lost.dat", 0.001, 0.3)
    if len(windows["total"]) != particles:
        raise ValueError(f"particle count differs in {path}")
    counts = window_counts(windows)
    for name in ("total", "prompt", "late"):
        if result["direct"][f"{name}_count"] != counts[name]["count"]:
            raise ValueError(f"stored {name} count differs in {path}")
    return windows


def compare_windows(reference: dict, candidate: dict) -> dict:
    return {
        name: {
            **paired_summary(reference[name], candidate[name]),
            "reference": window_counts({name: reference[name]})[name],
            "candidate": window_counts({name: candidate[name]})[name],
        }
        for name in ("total", "prompt", "late")
    }


def concatenate(cases: list[dict], name: str) -> np.ndarray:
    return np.concatenate([case[name] for case in cases])


def summarize_candidate(
    entries: list[dict], args: argparse.Namespace
) -> dict:
    references = []
    candidates = []
    per_seed = {}
    for entry in sorted(entries, key=lambda item: item["seed"]):
        seed = entry["seed"]
        reference_case = args.reference_case_template.format(seed=seed)
        if Path(reference_case).name != reference_case:
            raise ValueError("reference case template must produce one directory name")
        reference = load_case(
            args.reference_results / reference_case,
            args.particles,
            seed,
            args.base_wout_sha256,
            args.birth_surface,
        )
        candidate = load_case(
            args.results / entry["case"],
            args.particles,
            seed,
            entry["wout_sha256"],
            args.birth_surface,
        )
        references.append(reference)
        candidates.append(candidate)
        per_seed[str(seed)] = compare_windows(reference, candidate)
    aggregate = compare_windows(
        {name: concatenate(references, name) for name in references[0]},
        {name: concatenate(candidates, name) for name in candidates[0]},
    )
    aggregate["total_passes"] = gate(aggregate["total"], args.total_target)
    aggregate["late_passes"] = gate(aggregate["late"], args.late_target)
    return {
        "seeds": sorted(entry["seed"] for entry in entries),
        "wout_sha256": entries[0]["wout_sha256"],
        "per_seed": per_seed,
        "aggregate": aggregate,
        "promoted": aggregate["total_passes"] and aggregate["late_passes"],
    }


def gate(summary: dict, target: float) -> bool:
    return summary["change"] <= -max(target, 2.0 * summary["paired_se"])


def analyze(args: argparse.Namespace) -> dict:
    manifest = read_manifest(args.manifest)
    expected_seeds = set(args.seeds)
    for candidate, entries in manifest.items():
        if {entry["seed"] for entry in entries} != expected_seeds:
            raise ValueError(f"candidate {candidate} does not contain the fixed seeds")
        if len({entry["wout_sha256"] for entry in entries}) != 1:
            raise ValueError(f"candidate {candidate} has multiple equilibrium hashes")
    candidates = {
        str(candidate): summarize_candidate(entries, args)
        for candidate, entries in sorted(manifest.items())
    }
    ranked = sorted(
        candidates,
        key=lambda key: (
            candidates[key]["aggregate"]["total"]["change"],
            candidates[key]["aggregate"]["late"]["change"],
        ),
    )
    return {
        "schema_name": "alpha-loss.direct-multiseed-promotion",
        "schema_version": 1,
        "birth_surface": args.birth_surface,
        "particles_per_seed": args.particles,
        "seeds": args.seeds,
        "total_reduction_target": args.total_target,
        "late_reduction_target": args.late_target,
        "ranked_candidates": ranked,
        "promoted_candidates": [key for key in ranked if candidates[key]["promoted"]],
        "candidates": candidates,
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--manifest", type=Path, required=True)
    root.add_argument("--results", type=Path, required=True)
    root.add_argument("--reference-results", type=Path, required=True)
    root.add_argument("--output", type=Path, required=True)
    root.add_argument("--particles", type=int, default=1024)
    root.add_argument("--seeds", type=int, nargs="+", default=[12345, 22345, 32345, 42345])
    root.add_argument("--base-wout-sha256", default=BASE_WOUT_SHA256)
    root.add_argument("--birth-surface", type=float, default=0.3)
    root.add_argument("--reference-case-template", default="seed{seed}_base")
    root.add_argument("--total-target", type=float, default=0.02)
    root.add_argument("--late-target", type=float, default=0.01)
    return root


def main() -> None:
    args = parser().parse_args()
    args.output.write_text(
        json.dumps(analyze(args), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
