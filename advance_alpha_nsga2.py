#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def pending_responses(state: dict, root: Path) -> list[dict]:
    pending = {row["candidate_id"] for row in state["pending"]}
    rows = [json.loads(path.read_text()) for path in root.glob("*/response.json")]
    selected = sorted(
        (row for row in rows if row["candidate_id"] in pending),
        key=lambda row: row["candidate_id"],
    )
    if {row["candidate_id"] for row in selected} != pending:
        raise ValueError("NSGA-II response wave does not cover every pending candidate")
    return selected


def advance(args: argparse.Namespace) -> None:
    from simsopt_dfo.external_nsga2_checkpoint import (
        complete_candidates,
        issue_candidates,
        write_checkpoint,
    )

    state = json.loads(args.state.read_text())
    state = complete_candidates(state, pending_responses(state, args.responses))
    state, requests = issue_candidates(state)
    write_checkpoint(args.state, state)
    args.requests.write_text(
        json.dumps({"requests": requests}, indent=2, sort_keys=True) + "\n"
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--state", type=Path, required=True)
    root.add_argument("--responses", type=Path, required=True)
    root.add_argument("--requests", type=Path, required=True)
    return root


def main() -> None:
    advance(parser().parse_args())


if __name__ == "__main__":
    main()
