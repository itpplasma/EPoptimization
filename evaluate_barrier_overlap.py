#!/usr/bin/env python3
"""Evaluate the barrier-overlap proxy for one equilibrium."""

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
    root.add_argument("--inner-surface", type=float, default=0.25)
    root.add_argument("--outer-surface", type=float, default=0.6)
    root.add_argument("--ntheta", type=int, default=8)
    root.add_argument("--nzeta", type=int, default=8)
    root.add_argument("--npitch", type=int, default=16)
    root.add_argument("--pitch-max", type=float, default=0.6)
    root.add_argument("--mu-bins", type=int, default=16)
    root.add_argument("--trace-time", type=float, default=0.02)
    root.add_argument("--prompt-time", type=float, default=0.001)
    root.add_argument("--nturns", type=int, default=8)
    root.add_argument("--seed", type=int, default=12345)
    root.add_argument("--classifier", default="topology", choices=("topology", "jpar"))
    root.add_argument("--smooth-chaos-width", type=float, default=0.25)
    root.add_argument("--smooth-trapped-width", type=float, default=0.15)
    root.add_argument("--smooth-bin-width", type=float, default=0.5)
    root.add_argument("--smooth-radial-reference", type=float, default=0.05)
    root.add_argument("--timeout", type=float, default=3600.0)
    return root


def main() -> None:
    args = parser().parse_args()
    wout = args.wout.resolve()
    observed = file_sha256(wout)
    if observed != args.wout_sha256:
        raise ValueError("wout hash does not match the scheduled candidate")
    metrics = barrier_metrics(
        wout,
        expected_simple_sha256=args.simple_sha256,
        s_inner=args.inner_surface,
        s_outer=args.outer_surface,
        ntheta=args.ntheta,
        nzeta=args.nzeta,
        npitch=args.npitch,
        pitch_max=args.pitch_max,
        nbins=args.mu_bins,
        trace_time=args.trace_time,
        prompt_time=args.prompt_time,
        nturns=args.nturns,
        seed=args.seed,
        classifier=args.classifier,
        smooth_widths={
            "chaos": args.smooth_chaos_width,
            "trapped": args.smooth_trapped_width,
            "bin": args.smooth_bin_width,
            "radial_reference": args.smooth_radial_reference,
        },
        simple_executable=args.simple_executable,
        timeout_s=args.timeout,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_name": "alpha-loss.barrier-overlap-result",
        "schema_version": 1,
        "barrier": metrics,
        "wout_sha256": observed,
    }
    (args.out / "result.json").write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
