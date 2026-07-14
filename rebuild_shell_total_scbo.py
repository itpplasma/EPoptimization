#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def total_loss_row(response: dict) -> dict:
    row = dict(response)
    observation = response.get("observation")
    if observation is None:
        return row
    row["observation"] = {
        "value": float(observation["value"]),
        "variance": float(observation["variance"]),
        "constraints": [-1.0],
        "constraint_variances": [0.0],
    }
    return row


def load_responses(roots: list[Path]) -> list[dict]:
    by_id = {}
    for root in roots:
        for path in root.glob("*/response.json"):
            response = json.loads(path.read_text())
            candidate_id = int(response["candidate_id"])
            if candidate_id in by_id and by_id[candidate_id] != response:
                raise ValueError(f"conflicting responses for candidate {candidate_id}")
            by_id[candidate_id] = response
    rows = [total_loss_row(by_id[candidate_id]) for candidate_id in sorted(by_id)]
    if [row["candidate_id"] for row in rows] != list(range(len(rows))):
        raise ValueError("shell responses must have contiguous candidate IDs")
    return rows


def rebuild(args: argparse.Namespace) -> None:
    from simsopt_dfo.scbo_checkpoint import issue_candidates, prime_state, write_checkpoint

    rows = load_responses(args.responses)
    if not rows or len(rows) > args.budget:
        raise ValueError("completed response count must be between one and the budget")
    configuration = {
        "dimension": len(rows[0]["unit_x"]),
        "constraint_count": 1,
        "budget": args.budget,
        "seed": args.seed,
        "workers": args.workers,
        "initial_points": max(len(rows), 2 * len(rows[0]["unit_x"]), args.workers),
        "initial_length": args.initial_length,
    }
    state = prime_state(configuration, rows)
    state, requests = issue_candidates(state)
    write_checkpoint(args.state, state)
    args.requests.write_text(
        json.dumps({"requests": requests}, indent=2, sort_keys=True) + "\n"
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--responses", type=Path, nargs="+", required=True)
    root.add_argument("--state", type=Path, required=True)
    root.add_argument("--requests", type=Path, required=True)
    root.add_argument("--budget", type=int, required=True)
    root.add_argument("--seed", type=int, required=True)
    root.add_argument("--workers", type=int, default=8)
    root.add_argument("--initial-length", type=float, default=0.2)
    return root


def main() -> None:
    rebuild(parser().parse_args())


if __name__ == "__main__":
    main()
