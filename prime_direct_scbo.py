#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from build_direct_scbo_response import jeffreys_variance
from evaluate_vmec_geometry import evaluate as evaluate_geometry


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observation(value: float, variance: float, geometry: dict) -> dict:
    return {
        "value": float(value),
        "variance": float(variance),
        "constraints": [float(value) for value in geometry["constraints"]],
        "constraint_variances": [0.0, 0.0],
    }


def evaluation_row(candidate_id: int, unit_x: list[float], payload: dict) -> dict:
    return {
        "candidate_id": candidate_id,
        "unit_x": [float(value) for value in unit_x],
        "observation": payload,
        "status": "ok",
        "failure_kind": None,
        "started_seconds": 0.0,
        "completed_seconds": 0.0,
        "worker_pid": 0,
        "worker_thread": 0,
    }


def prime(args: argparse.Namespace) -> None:
    from simsopt_dfo.scbo_checkpoint import (
        issue_candidates,
        prime_state,
        write_checkpoint,
    )

    source = json.loads(args.validation_summary.read_text())
    candidates = source["candidates"]
    if not candidates:
        raise ValueError("validation summary contains no candidates")
    campaign = args.validation_summary.resolve().parents[2]
    reference_fraction = float(candidates[0]["windows"]["total"]["reference_fraction"])
    reference_particles = int(candidates[0]["windows"]["total"]["particles"])
    reference_losses = round(reference_fraction * reference_particles)
    reference_geometry = evaluate_geometry(
        args.reference_wout,
        mirror_limit=args.mirror_limit,
        elongation_limit=args.elongation_limit,
    )
    rows = [
        evaluation_row(
            0,
            [0.5] * len(candidates[0]["unit_x"]),
            observation(
                reference_fraction,
                jeffreys_variance(reference_losses, reference_particles),
                reference_geometry,
            ),
        )
    ]
    provenance = [
        {
            "candidate_id": 0,
            "source": "reference",
            "source_candidate_id": 0,
            "wout_sha256": file_sha256(args.reference_wout),
            "particles": reference_particles,
            "total_loss": reference_fraction,
            "geometry": reference_geometry,
        }
    ]
    for candidate_id, candidate in enumerate(candidates, start=1):
        wout = campaign / candidate["case_root"] / candidate["wout"]
        if file_sha256(wout) != candidate["wout_sha256"]:
            raise ValueError(f"candidate {candidate['candidate_id']} wout hash differs")
        total = candidate["windows"]["total"]
        particles = int(total["particles"])
        value = float(total["candidate_fraction"])
        losses = round(value * particles)
        seed_variance = float(total["seed_se"]) ** 2
        geometry = evaluate_geometry(
            wout,
            mirror_limit=args.mirror_limit,
            elongation_limit=args.elongation_limit,
        )
        rows.append(
            evaluation_row(
                candidate_id,
                candidate["unit_x"],
                observation(
                    value,
                    max(seed_variance, jeffreys_variance(losses, particles)),
                    geometry,
                ),
            )
        )
        provenance.append(
            {
                "candidate_id": candidate_id,
                "source": "completed_multiseed_direct_validation",
                "source_candidate_id": int(candidate["candidate_id"]),
                "wout_sha256": candidate["wout_sha256"],
                "particles": particles,
                "total_loss": value,
                "geometry": geometry,
            }
        )

    configuration = {
        "dimension": len(rows[0]["unit_x"]),
        "constraint_count": 2,
        "budget": len(rows) + args.new_calls,
        "seed": args.seed,
        "workers": args.workers,
        "initial_points": len(rows),
        "initial_length": args.initial_length,
    }
    state = prime_state(configuration, rows)
    state, requests = issue_candidates(state)
    write_checkpoint(args.state, state)
    args.requests.write_text(
        json.dumps({"requests": requests}, indent=2, sort_keys=True) + "\n"
    )
    args.priming.write_text(
        json.dumps(
            {
                "schema_name": "alpha-loss.direct-scbo-priming",
                "schema_version": 1,
                "configuration": configuration,
                "source_summary_sha256": file_sha256(args.validation_summary),
                "constraint_names": ["mirror_ratio", "max_elongation"],
                "constraint_limits": [args.mirror_limit, args.elongation_limit],
                "observations": provenance,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--validation-summary", type=Path, required=True)
    root.add_argument("--reference-wout", type=Path, required=True)
    root.add_argument("--state", type=Path, required=True)
    root.add_argument("--requests", type=Path, required=True)
    root.add_argument("--priming", type=Path, required=True)
    root.add_argument("--new-calls", type=int, default=128)
    root.add_argument("--workers", type=int, default=8)
    root.add_argument("--seed", type=int, default=7301)
    root.add_argument("--initial-length", type=float, default=0.4)
    root.add_argument("--mirror-limit", type=float, default=0.20)
    root.add_argument("--elongation-limit", type=float, default=6.0)
    return root


def main() -> None:
    prime(parser().parse_args())


if __name__ == "__main__":
    main()
