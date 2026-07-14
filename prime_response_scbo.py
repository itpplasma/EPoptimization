#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_rows(root: Path) -> list[dict]:
    rows = [json.loads(path.read_text()) for path in root.glob("*/response.json")]
    rows.sort(key=lambda row: row["candidate_id"])
    if [row["candidate_id"] for row in rows] != list(range(len(rows))):
        raise ValueError("SCBO priming responses must have contiguous candidate IDs")
    if not rows or any(row["observation"] is None for row in rows):
        raise ValueError("SCBO priming requires successful scalar observations")
    return rows


def prime(args: argparse.Namespace) -> None:
    from simsopt_dfo.scbo_checkpoint import issue_candidates, prime_state, write_checkpoint

    rows = load_rows(args.responses)
    observation = rows[0]["observation"]
    configuration = {
        "dimension": len(rows[0]["unit_x"]),
        "constraint_count": len(observation["constraints"]),
        "budget": len(rows) + args.new_calls,
        "seed": args.seed,
        "workers": args.workers,
        "initial_points": max(2 * len(rows[0]["unit_x"]), args.workers),
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
    root.add_argument("--responses", type=Path, required=True)
    root.add_argument("--state", type=Path, required=True)
    root.add_argument("--requests", type=Path, required=True)
    root.add_argument("--seed", type=int, required=True)
    root.add_argument("--new-calls", type=int, required=True)
    root.add_argument("--workers", type=int, default=8)
    root.add_argument("--initial-length", type=float, default=0.2)
    return root


def main() -> None:
    prime(parser().parse_args())


if __name__ == "__main__":
    main()
