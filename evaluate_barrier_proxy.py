#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
from datetime import datetime, timezone
from pathlib import Path

import simple_barrier


def evaluate(args: argparse.Namespace) -> None:
    if args.particles <= 0 or args.bins <= 0:
        raise ValueError("particles and bins must be positive")
    if not 0.0 < args.inner_surface < args.outer_surface < 1.0:
        raise ValueError("proxy surfaces must satisfy 0 < inner < outer < 1")
    executable = args.simple_executable.resolve()
    executable_hash = simple_barrier.file_sha256(executable)
    if executable_hash != args.simple_sha256:
        raise ValueError(f"unexpected SIMPLE executable hash {executable_hash}")
    wout = args.wout.resolve()
    wout_hash = simple_barrier.file_sha256(wout)
    if wout_hash != args.wout_sha256:
        raise ValueError(f"unexpected equilibrium hash {wout_hash}")
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rz_scale, b_scale = simple_barrier.reactor_scale(wout)
    metrics = simple_barrier.barrier_metrics(
        wout,
        s_inner=args.inner_surface,
        s_outer=args.outer_surface,
        ntestpart=args.particles,
        rz_scale=rz_scale,
        b_scale=b_scale,
        facE_al=1.0,
        trace_time=args.trace_time,
        seed=args.seed,
        classifier="topology",
        overlap_bins=args.bins,
        simple_executable=executable,
        keep_workdir=True,
        timeout_s=args.timeout,
    )
    workdir = Path(str(metrics.pop("workdir")))
    classification = output / "classification"
    shutil.move(str(workdir), classification)
    for surface in ("inner", "outer"):
        write_execution_record(
            classification / surface,
            executable,
            executable_hash,
            wout_hash,
            args.seed,
        )
    result = {
        "status": "ok",
        "wout_sha256": wout_hash,
        "simple_sha256": executable_hash,
        "particles": args.particles,
        "seed": args.seed,
        "inner_surface": args.inner_surface,
        "outer_surface": args.outer_surface,
        "overlap_bins": args.bins,
        "trace_time": args.trace_time,
        "barrier": metrics,
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--wout", type=Path, required=True)
    root.add_argument("--out", type=Path, required=True)
    root.add_argument("--simple-executable", type=Path, required=True)
    root.add_argument("--simple-sha256", required=True)
    root.add_argument("--wout-sha256", required=True)
    root.add_argument("--particles", type=int, default=3000)
    root.add_argument("--seed", type=int, default=12345)
    root.add_argument("--inner-surface", type=float, default=0.3)
    root.add_argument("--outer-surface", type=float, default=0.6)
    root.add_argument("--bins", type=int, default=16)
    root.add_argument("--trace-time", type=float, default=2.0e-2)
    root.add_argument("--timeout", type=float, default=7200.0)
    return root


def write_execution_record(
    run: Path,
    executable: Path,
    executable_hash: str,
    wout_hash: str,
    seed: int,
) -> None:
    input_hash = simple_barrier.file_sha256(run / "simple.in")
    (run / "output_checksums.txt").write_text(
        f"{executable_hash}  {executable}\n"
        f"{wout_hash}  wout.nc\n"
        f"{input_hash}  simple.in\n"
    )
    (run / "execution.txt").write_text(
        f"simple_bin={executable}\n"
        f"host={socket.gethostname()}\n"
        f"slurm_job={os.environ.get('SLURM_JOB_ID', '')}\n"
        f"seed={seed}\n"
        f"completed={datetime.now(timezone.utc).astimezone().isoformat()}\n"
    )


def main() -> None:
    evaluate(parser().parse_args())


if __name__ == "__main__":
    main()
