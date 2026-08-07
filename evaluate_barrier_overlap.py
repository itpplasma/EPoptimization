#!/usr/bin/env python3
"""Evaluate continuous fast-classifier metrics for one equilibrium."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from barrier_overlap import barrier_metrics, file_sha256


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--wout", type=Path, required=True)
    root.add_argument("--wout-sha256", required=True)
    root.add_argument("--out", type=Path, required=True)
    root.add_argument("--simple-executable", type=Path, required=True)
    root.add_argument("--simple-sha256", required=True)
    root.add_argument("--work-root", type=Path)
    root.add_argument("--surfaces", default="0.25,0.4,0.55,0.7")
    root.add_argument("--ntheta", type=int, default=8)
    root.add_argument("--nzeta", type=int, default=8)
    root.add_argument("--npitch", type=int, default=16)
    root.add_argument("--pitch-max", type=float, default=1.0)
    root.add_argument("--mu-nodes", type=int, default=24)
    root.add_argument("--trace-time", type=float, default=0.02)
    root.add_argument("--prompt-time", type=float, default=0.001)
    root.add_argument("--nturns", type=int, default=8)
    root.add_argument("--seed", type=int, default=12345)
    root.add_argument("--trapped-width", type=float, default=0.15)
    root.add_argument("--mu-width-factor", type=float, default=0.75)
    root.add_argument("--jpar-temperature", type=float, default=0.1)
    root.add_argument("--rotation-temperature", type=float, default=0.02)
    root.add_argument("--timeout", type=float, default=3600.0)
    return root


def main() -> None:
    args = parser().parse_args()
    surfaces = [float(value) for value in args.surfaces.split(",") if value.strip()]
    wout = args.wout.resolve()
    observed = file_sha256(wout)
    if observed != args.wout_sha256:
        raise ValueError("wout hash does not match the scheduled candidate")
    metrics = barrier_metrics(
        wout,
        expected_simple_sha256=args.simple_sha256,
        surfaces=surfaces,
        ntheta=args.ntheta,
        nzeta=args.nzeta,
        npitch=args.npitch,
        pitch_max=args.pitch_max,
        nmu=args.mu_nodes,
        trace_time=args.trace_time,
        prompt_time=args.prompt_time,
        nturns=args.nturns,
        seed=args.seed,
        continuous_settings={
            "trapped_width": args.trapped_width,
            "mu_width_factor": args.mu_width_factor,
            "temperature_jpar": args.jpar_temperature,
            "temperature_rotation": args.rotation_temperature,
        },
        simple_executable=args.simple_executable,
        work_root=args.work_root,
        timeout_s=args.timeout,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_name": "alpha-loss.continuous-fast-classifier-result",
        "schema_version": 2,
        "barrier": metrics,
        "wout_sha256": observed,
    }
    (args.out / "result.json").write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
