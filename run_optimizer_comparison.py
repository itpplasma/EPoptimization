#!/usr/bin/env python3
"""Run one raw-Fourier optimizer against externally scheduled direct evaluations."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from raw_fourier_surface import raw_coordinate_contract, write_raw_candidate


class ClusterObjective:
    def __init__(self, args: argparse.Namespace, contract: dict):
        self.args = args
        self.contract = contract
        self.root = args.root.resolve()
        self.method_root = self.root / args.method
        self.method_root.mkdir(parents=True, exist_ok=False)
        self.ledger_path = self.method_root / "ledger.json"
        self.records: list[dict] = []

    def __call__(self, unit_x) -> float:
        if len(self.records) >= self.args.budget:
            return min(row["penalized_value"] for row in self.records)
        candidate_id = len(self.records)
        unit = np.clip(np.asarray(unit_x, dtype=float), 0.0, 1.0)
        wave = f"{self.args.method}/eval-{candidate_id:03d}"
        wave_root = self.root / wave
        case_name = f"candidate-{candidate_id:08d}"
        case_root = wave_root / "candidates" / case_name
        generated = write_raw_candidate(
            self.args.base_input,
            self.contract,
            unit,
            candidate_id,
            case_root,
        )
        if generated["status"] != "ready":
            response = {
                "candidate_id": candidate_id,
                "unit_x": unit.tolist(),
                "status": "failed",
                "failure_kind": generated["failure_kind"],
                "observation": None,
            }
            (case_root / "response.json").write_text(
                json.dumps(response, indent=2, sort_keys=True) + "\n"
            )
            return self._record(response, None)
        relative = f"{case_name}/{generated['input']}"
        (wave_root / "manifest.tsv").write_text(
            f"{case_name}\t{relative}\t{generated['input_sha256']}\n"
        )
        (wave_root / "logs").mkdir()
        remote_wave = f"{self.args.remote}/campaign/{wave}"
        _run(["ssh", self.args.host, "mkdir", "-p", f"{remote_wave}/logs"])
        _run(
            [
                "rsync",
                "-az",
                "--exclude",
                "logs/",
                f"{wave_root}/",
                f"{self.args.host}:{remote_wave}/",
            ]
        )
        submitted = subprocess.run(
            [
                "ssh",
                self.args.host,
                "cd",
                self.args.remote,
                "&&",
                "sbatch",
                "--wait",
                "--parsable",
                f"--export=ALL,METHOD={self.args.method},EVAL={candidate_id:03d}",
                f"--output={remote_wave}/logs/%j.out",
                "run_single.sbatch",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        job_id = submitted.stdout.strip().split(";", 1)[0]
        (wave_root / "job.id").write_text(job_id + "\n")
        _run(
            [
                "rsync",
                "-az",
                f"{self.args.host}:{remote_wave}/candidates/",
                f"{wave_root}/candidates/",
            ]
        )
        response_path = case_root / "response.json"
        if not response_path.exists():
            response = {
                "candidate_id": candidate_id,
                "unit_x": unit.tolist(),
                "status": "failed",
                "failure_kind": "scheduler_failure",
                "observation": None,
            }
        else:
            response = json.loads(response_path.read_text())
        return self._record(response, job_id)

    def _record(self, response: dict, job_id: str | None) -> float:
        observation = response.get("observation")
        if observation is None:
            penalty = 2.0
        else:
            violation = np.maximum(np.asarray(observation["constraints"]), 0.0)
            penalty = float(observation["value"] + 10.0 * np.dot(violation, violation))
        record = {
            "evaluation_id": len(self.records),
            "job_id": job_id,
            "unit_x": response["unit_x"],
            "status": response["status"],
            "failure_kind": response.get("failure_kind"),
            "observation": observation,
            "penalized_value": penalty,
        }
        self.records.append(record)
        _atomic_json(
            self.ledger_path,
            {
                "schema_name": "alpha-loss.optimizer-comparison-ledger",
                "schema_version": 1,
                "method": self.args.method,
                "budget": self.args.budget,
                "seed": self.args.seed,
                "records": self.records,
            },
        )
        return penalty


def run_turbo(
    objective: ClusterObjective, args: argparse.Namespace, dimension: int
) -> dict:
    from simsopt_dfo.scbo_checkpoint import (
        complete_candidates,
        issue_candidates,
        prime_state,
        write_checkpoint,
    )

    anchor = np.full(dimension, 0.5)
    objective(anchor)
    response = _response_from_record(objective.records[0])
    state = prime_state(
        {
            "dimension": dimension,
            "constraint_count": 2,
            "budget": args.budget,
            "seed": args.seed,
            "workers": 1,
            "initial_points": min(2 * dimension, args.budget - 1),
            "initial_length": 0.8,
        },
        [response],
    )
    checkpoint = objective.method_root / "state.json"
    while len(objective.records) < args.budget:
        state, requests = issue_candidates(state)
        request = requests[0]
        objective(request["unit_x"])
        state = complete_candidates(
            state, [_response_from_record(objective.records[-1])]
        )
        write_checkpoint(checkpoint, state)
    best = min(
        (row for row in objective.records if row["observation"] is not None),
        key=lambda row: row["penalized_value"],
    )
    return {"best": best, "trust_region": state["trust_region"]}


def run_dual_annealing(
    objective: ClusterObjective, args: argparse.Namespace, dimension: int
) -> dict:
    from scipy.optimize import dual_annealing

    result = dual_annealing(
        objective,
        [(0.0, 1.0)] * dimension,
        x0=np.full(dimension, 0.5),
        seed=args.seed,
        maxiter=10_000,
        maxfun=args.budget,
        no_local_search=False,
    )
    return {
        "best": min(objective.records, key=lambda row: row["penalized_value"]),
        "termination": {"message": str(result.message), "nfev": int(result.nfev)},
    }


def run_bobyqa(
    objective: ClusterObjective, args: argparse.Namespace, dimension: int
) -> dict:
    import pybobyqa

    result = pybobyqa.solve(
        objective,
        np.full(dimension, 0.5),
        bounds=(np.zeros(dimension), np.ones(dimension)),
        maxfun=args.budget,
        rhobeg=0.2,
        rhoend=0.01,
        seek_global_minimum=False,
        print_progress=False,
    )
    return {
        "best": min(objective.records, key=lambda row: row["penalized_value"]),
        "termination": {"message": str(result.msg), "nfev": int(result.nf)},
    }


def _response_from_record(record: dict) -> dict:
    return {
        "candidate_id": record["evaluation_id"],
        "unit_x": record["unit_x"],
        "status": record["status"],
        "failure_kind": record["failure_kind"],
        "observation": record["observation"],
    }


def _atomic_json(path: Path, document: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument(
        "--method", choices=("turbo", "dual-annealing", "bobyqa"), required=True
    )
    root.add_argument("--root", type=Path, required=True)
    root.add_argument("--base-input", type=Path, required=True)
    root.add_argument("--budget", type=int, default=64)
    root.add_argument("--seed", type=int, required=True)
    root.add_argument("--host", default="acluster")
    root.add_argument("--remote", default="/home/ert/runs/alpha-optimizer-methods-v1")
    return root


def main() -> None:
    args = parser().parse_args()
    contract = raw_coordinate_contract(args.base_input)
    args.root.mkdir(parents=True, exist_ok=True)
    contract_path = args.root / "raw-fourier-contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("raw Fourier comparison contract changed")
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    objective = ClusterObjective(args, contract)
    runners = {
        "turbo": run_turbo,
        "dual-annealing": run_dual_annealing,
        "bobyqa": run_bobyqa,
    }
    result = runners[args.method](objective, args, len(contract["names"]))
    _atomic_json(
        objective.method_root / "result.json",
        {
            "schema_name": "alpha-loss.optimizer-comparison-result",
            "schema_version": 1,
            "method": args.method,
            "budget": args.budget,
            "seed": args.seed,
            **result,
        },
    )


if __name__ == "__main__":
    main()
