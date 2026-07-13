#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

import simple_barrier
from evaluate_barrier_proxy import write_execution_record
from loss_threshold_objective import threshold_crossing, threshold_objective


def evaluate(args: argparse.Namespace) -> None:
    validate(args)
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
    metrics = simple_barrier.direct_loss_metrics(
        wout,
        ntestpart=args.particles,
        expected_simple_sha256=executable_hash,
        trace_time=args.trace_time,
        prompt_time=args.prompt_time,
        sbeg=args.birth_surface,
        seed=args.seed,
        simple_executable=executable,
        keep_workdir=True,
        timeout_s=args.timeout,
    )
    workdir = Path(str(metrics.pop("workdir")))
    direct = output / "direct"
    shutil.move(str(workdir), direct)
    write_execution_record(direct, executable, executable_hash, wout_hash, args.seed)
    curve = np.loadtxt(direct / "confined_fraction.dat", ndmin=2)
    crossing = threshold_crossing(curve, args.loss_threshold)
    objective = threshold_objective(
        metrics["total_loss"],
        args.trace_time,
        args.loss_threshold,
        crossing_time=crossing,
        epsilon=args.epsilon,
    )
    result = {
        "status": "ok",
        "wout_sha256": wout_hash,
        "simple_sha256": executable_hash,
        "particles": args.particles,
        "seed": args.seed,
        "birth_surface": args.birth_surface,
        "prompt_time": args.prompt_time,
        "trace_time": args.trace_time,
        "loss_threshold": args.loss_threshold,
        "epsilon": args.epsilon,
        "objective": objective,
        "crossing_time": crossing,
        "full_trace_completed": True,
        "late_constraint_available": True,
        "direct": metrics,
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def validate(args: argparse.Namespace) -> None:
    if args.particles <= 0:
        raise ValueError("particles must be positive")
    if not 0.0 < args.birth_surface < 1.0:
        raise ValueError("birth surface must satisfy 0 < s < 1")
    if not 0.0 < args.prompt_time < args.trace_time:
        raise ValueError("prompt time must precede the trace endpoint")
    if not 0.0 < args.loss_threshold < 1.0 or args.epsilon <= 0.0:
        raise ValueError("loss threshold and epsilon are outside their ranges")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--wout", type=Path, required=True)
    root.add_argument("--out", type=Path, required=True)
    root.add_argument("--simple-executable", type=Path, required=True)
    root.add_argument("--simple-sha256", required=True)
    root.add_argument("--wout-sha256", required=True)
    root.add_argument("--particles", type=int, default=1024)
    root.add_argument("--seed", type=int, default=12345)
    root.add_argument("--birth-surface", type=float, default=0.3)
    root.add_argument("--prompt-time", type=float, default=1.0e-3)
    root.add_argument("--trace-time", type=float, default=3.0e-1)
    root.add_argument("--loss-threshold", type=float, default=0.38)
    root.add_argument("--epsilon", type=float, default=1.0e-6)
    root.add_argument("--timeout", type=float, default=86400.0)
    return root


def main() -> None:
    evaluate(parser().parse_args())


if __name__ == "__main__":
    main()
