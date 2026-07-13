#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def prime(args: argparse.Namespace) -> None:
    from simsopt_dfo.scbo_checkpoint import (
        issue_candidates,
        prime_state,
        write_checkpoint,
    )

    scout = json.loads(args.scout.read_text())
    extra = load_extra_responses(args.extra_responses)
    rows = (
        local_priming_rows(scout, extra)
        if args.local_only
        else priming_rows(scout, extra)
    )
    dimension = len(rows[0]["unit_x"])
    configuration = {
        "dimension": dimension,
        "constraint_count": 1,
        "budget": len(rows) + args.new_calls,
        "seed": args.seed,
        "workers": args.workers,
        "initial_points": args.workers if args.local_only else max(2 * dimension, args.workers),
    }
    state = prime_state(configuration, rows)
    state, requests = issue_candidates(state)
    write_checkpoint(args.state, state)
    args.requests.write_text(
        json.dumps({"requests": requests}, indent=2, sort_keys=True) + "\n"
    )


def priming_rows(scout: dict, extra: list[dict] | None = None) -> list[dict]:
    dimension = len(scout["cases"][0]["unit"])
    target = float(scout["late_reduction_target"])
    rows = [
        evaluation_row(
            0,
            [0.5] * dimension,
            {
                "value": 0.0,
                "variance": 0.0,
                "constraints": [target],
                "constraint_variances": [0.0],
            },
        )
    ]
    for candidate_id, case in enumerate(scout["cases"], start=1):
        if case["status"] == "ok":
            observation = dict(case["observation"])
            observation["value"] = case["total"]["change"]
            rows.append(evaluation_row(candidate_id, case["unit"], observation))
        else:
            rows.append(
                evaluation_row(
                    candidate_id,
                    case["unit"],
                    None,
                    failure_kind=case["failure_kind"],
                )
            )
    if extra:
        expected = list(range(len(rows), len(rows) + len(extra)))
        if [row["candidate_id"] for row in extra] != expected:
            raise ValueError("extra SCBO responses are not contiguous with the scout")
        rows.extend(extra)
    return rows


def load_extra_responses(root: Path | None) -> list[dict]:
    if root is None:
        return []
    rows = [json.loads(path.read_text()) for path in Path(root).glob("*/response.json")]
    return sorted(rows, key=lambda row: row["candidate_id"])


def local_priming_rows(scout: dict, extra: list[dict]) -> list[dict]:
    if not extra:
        raise ValueError("local SCBO priming requires completed responses")
    dimension = len(extra[0]["unit_x"])
    rows = [
        evaluation_row(
            0,
            [0.5] * dimension,
            {
                "value": 0.0,
                "variance": 0.0,
                "constraints": [float(scout["late_reduction_target"])],
                "constraint_variances": [0.0],
            },
        )
    ]
    for candidate_id, response in enumerate(extra, start=1):
        row = dict(response)
        row["candidate_id"] = candidate_id
        rows.append(row)
    return rows


def evaluation_row(candidate_id, unit_x, observation, failure_kind=None) -> dict:
    return {
        "candidate_id": candidate_id,
        "unit_x": unit_x,
        "observation": observation,
        "status": "ok" if observation is not None else "failed",
        "failure_kind": failure_kind,
        "started_seconds": 0.0,
        "completed_seconds": 0.0,
        "worker_pid": 0,
        "worker_thread": 0,
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--scout", type=Path, required=True)
    root.add_argument("--state", type=Path, required=True)
    root.add_argument("--requests", type=Path, required=True)
    root.add_argument("--seed", type=int, required=True)
    root.add_argument("--new-calls", type=int, default=128)
    root.add_argument("--workers", type=int, default=8)
    root.add_argument("--extra-responses", type=Path)
    root.add_argument("--local-only", action="store_true")
    return root


def main() -> None:
    prime(parser().parse_args())


if __name__ == "__main__":
    main()
