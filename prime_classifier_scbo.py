#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_responses(root: Path) -> list[dict]:
    rows = [json.loads(path.read_text()) for path in root.glob("*/response.json")]
    rows.sort(key=lambda row: int(row["candidate_id"]))
    if [int(row["candidate_id"]) for row in rows] != list(range(len(rows))):
        raise ValueError("classifier priming responses must have contiguous IDs from zero")
    successful = [row for row in rows if row["observation"] is not None]
    if not successful:
        raise ValueError("classifier priming requires a successful response")
    constraint_counts = {
        len(row["observation"]["constraints"]) for row in successful
    }
    if len(constraint_counts) != 1:
        raise ValueError("classifier priming responses have different constraint counts")
    return rows


def configuration(args: argparse.Namespace, rows: list[dict]) -> dict:
    successful = next(row for row in rows if row["observation"] is not None)
    dimension = len(successful["unit_x"])
    initial_points = max(len(rows), args.workers, 2 * dimension)
    return {
        "dimension": dimension,
        "constraint_count": len(successful["observation"]["constraints"]),
        "budget": len(rows) + args.new_calls,
        "seed": args.seed,
        "workers": args.workers,
        "initial_points": initial_points,
        "initial_length": args.initial_length,
    }


def prime(args: argparse.Namespace) -> None:
    from simsopt_dfo.scbo_checkpoint import issue_candidates, prime_state, write_checkpoint

    rows = load_responses(args.responses)
    state = prime_state(configuration(args, rows), rows)
    state, requests = issue_candidates(state)
    write_checkpoint(args.state, state)
    args.requests.write_text(
        json.dumps({"requests": requests}, indent=2, sort_keys=True) + "\n"
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--responses", type=Path, required=True)
    root.add_argument("--state", type=Path, required=True)
    root.add_argument("--requests", type=Path, required=True)
    root.add_argument("--seed", type=int, required=True)
    root.add_argument("--new-calls", type=int, default=128)
    root.add_argument("--workers", type=int, default=8)
    root.add_argument("--initial-length", type=float, default=0.2)
    return root


def main() -> None:
    prime(parser().parse_args())


if __name__ == "__main__":
    main()
