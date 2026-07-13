from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_direct_promotions import BASE_WOUT_SHA256, SIMPLE_SHA256, analyze, gate


def write_case(path: Path, times: list[float], seed: int, wout_hash: str) -> None:
    direct = path / "direct"
    direct.mkdir(parents=True)
    np.savetxt(direct / "times_lost.dat", list(enumerate(times, start=1)))
    prompt = sum(0.0 < value <= 0.001 for value in times)
    late = sum(0.001 < value < 0.3 and not np.isclose(value, 0.3, atol=1e-12) for value in times)
    result = {
        "status": "ok",
        "particles": len(times),
        "seed": seed,
        "wout_sha256": wout_hash,
        "simple_sha256": SIMPLE_SHA256,
        "birth_surface": 0.3,
        "prompt_time": 0.001,
        "trace_time": 0.3,
        "direct": {"prompt_count": prompt, "late_count": late, "total_count": prompt + late},
    }
    (path / "result.json").write_text(json.dumps(result))


def test_analyze_concatenates_seed_pairs_and_applies_both_gates(tmp_path) -> None:
    manifest = tmp_path / "manifest.tsv"
    results = tmp_path / "results"
    references = tmp_path / "references"
    candidate_hash = "a" * 64
    seeds = [1, 2]
    lines = []
    for seed in seeds:
        reference = [0.02, 0.03, 0.04, 0.05, 0.3, 0.3, 0.3, 0.3 - 1e-16]
        candidate = [0.3] * 7 + [0.3 - 1e-16]
        write_case(references / f"seed{seed}_base", reference, seed, BASE_WOUT_SHA256)
        case = f"candidate-7-seed{seed}"
        write_case(results / case, candidate, seed, candidate_hash)
        lines.append(f"{case}\t7\tunused\t{candidate_hash}\t{seed}")
    manifest.write_text("\n".join(lines) + "\n")
    args = argparse.Namespace(
        manifest=manifest,
        results=results,
        reference_results=references,
        output=tmp_path / "out.json",
        particles=8,
        seeds=seeds,
        base_wout_sha256=BASE_WOUT_SHA256,
        total_target=0.02,
        late_target=0.01,
    )

    result = analyze(args)

    aggregate = result["candidates"]["7"]["aggregate"]
    assert aggregate["total"]["change"] == -0.5
    assert aggregate["late"]["change"] == -0.5
    assert result["promoted_candidates"] == ["7"]


def test_gate_requires_material_and_statistical_reduction() -> None:
    assert not gate({"change": -0.019, "paired_se": 0.001}, 0.02)
    assert not gate({"change": -0.021, "paired_se": 0.02}, 0.02)
    assert gate({"change": -0.04, "paired_se": 0.02}, 0.02)
