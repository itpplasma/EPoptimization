#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import simple_barrier
from evaluate_barrier_proxy import write_execution_record


def evaluate(args: argparse.Namespace) -> None:
    if args.particles <= 0:
        raise ValueError("particles must be positive")
    if not 0.0 < args.birth_surface < 1.0:
        raise ValueError("birth surface must satisfy 0 < s < 1")
    if not 0.0 < args.prompt_time < args.trace_time:
        raise ValueError("loss windows must satisfy 0 < prompt < trace")
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
    result = {
        "status": "ok",
        "wout_sha256": wout_hash,
        "simple_sha256": executable_hash,
        "particles": args.particles,
        "seed": args.seed,
        "birth_surface": args.birth_surface,
        "prompt_time": args.prompt_time,
        "trace_time": args.trace_time,
        "direct": metrics,
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
    root.add_argument("--particles", type=int, default=1024)
    root.add_argument("--seed", type=int, default=12345)
    root.add_argument("--birth-surface", type=float, default=0.3)
    root.add_argument("--prompt-time", type=float, default=1.0e-3)
    root.add_argument("--trace-time", type=float, default=3.0e-1)
    root.add_argument("--timeout", type=float, default=86400.0)
    return root


def main() -> None:
    evaluate(parser().parse_args())


if __name__ == "__main__":
    main()
