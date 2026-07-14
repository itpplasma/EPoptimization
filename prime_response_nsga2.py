#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from prime_response_scbo import load_rows


def prime(args: argparse.Namespace) -> None:
    from simsopt_dfo.external_nsga2_checkpoint import (
        issue_candidates,
        prime_state,
        write_checkpoint,
    )

    rows = load_rows(args.responses)
    observation = rows[0]["pareto_observation"]
    configuration = {
        "dimension": len(rows[0]["unit_x"]),
        "objective_count": len(observation["values"]),
        "constraint_count": len(observation["constraints"]),
        "population_size": args.population,
        "offspring_generations": args.generations,
        "seed": args.seed,
        "workers": args.workers,
        "objective_scales": args.objective_scales,
        "crossover_probability": 0.9,
        "crossover_index": 15.0,
        "mutation_probability": 1.0 / len(rows[0]["unit_x"]),
        "mutation_index": 20.0,
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
    root.add_argument("--objective-scales", type=float, nargs="+", required=True)
    root.add_argument("--population", type=int, default=16)
    root.add_argument("--generations", type=int, default=4)
    root.add_argument("--workers", type=int, default=8)
    return root


def main() -> None:
    prime(parser().parse_args())


if __name__ == "__main__":
    main()
