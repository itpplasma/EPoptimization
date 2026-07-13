#!/usr/bin/env python3
import argparse
from pathlib import Path

from spatial_grid import run_surface_classification


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--wout", type=Path, required=True)
    root.add_argument("--design", type=Path, required=True)
    root.add_argument("--out", type=Path, required=True)
    root.add_argument("--simple-executable", type=Path, required=True)
    root.add_argument("--trace-time", type=float, default=0.02)
    root.add_argument("--timeout", type=float, default=3600.0)
    return root


if __name__ == "__main__":
    args = parser().parse_args()
    run_surface_classification(
        args.wout.resolve(),
        args.design.resolve(),
        args.out.resolve(),
        args.simple_executable.resolve(),
        trace_time=args.trace_time,
        timeout_seconds=args.timeout,
    )
