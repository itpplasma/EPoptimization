#!/usr/bin/env python
"""Render the SIMPLE-vs-NEAT comparison figure.

Runs the direct simple.x reference, NEAT at matched resolution, and NEAT at
its default resolution (multharm=3, ns_s=ns_tp=3, npoiper2=128) on one
equilibrium, then plots the loss-fraction history and the per-orbit loss-time
agreement. Runner conventions live in compare_simple_neat.py.
"""

import argparse
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_simple_neat import loss_stats, run_direct, run_neat

import simple_barrier


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wout", required=True)
    ap.add_argument("--sbeg", type=float, default=0.25)
    ap.add_argument("--nparticles", type=int, default=256)
    ap.add_argument("--trace-time", type=float, default=1e-2)
    ap.add_argument("--nsamples", type=int, default=1000)
    ap.add_argument("--scale", choices=["reactor", "none"], default="none")
    ap.add_argument("--face-al", type=float, default=1.0)
    ap.add_argument("--timeout", type=float, default=7200.0)
    ap.add_argument("--out", default="compare_simple_neat.png")
    args = ap.parse_args()

    wout = Path(args.wout).expanduser()
    if args.scale == "reactor":
        rz, b = simple_barrier.reactor_scale(wout)
    else:
        rz, b = 1.0, 1.0

    base = Path(tempfile.mkdtemp(prefix="plot_compare_"))
    common = dict(sbeg=args.sbeg, n=args.nparticles, trace_time=args.trace_time,
                  nsamples=args.nsamples, rz=rz, b=b)
    direct = run_direct(wout, **common, face=args.face_al,
                        workdir=base / "direct", timeout_s=args.timeout)
    neat_matched = run_neat(wout, **common, matched=True, timeout_s=args.timeout)
    neat_default = run_neat(wout, **common, matched=False, timeout_s=args.timeout)

    runs = [
        ("direct simple.x", direct, dict(color="C0", ls="-", lw=2.5)),
        ("NEAT, matched resolution", neat_matched, dict(color="C1", ls="--", lw=1.5)),
        ("NEAT, default resolution", neat_default, dict(color="C2", ls="-.", lw=1.5)),
    ]

    fig, (ax_loss, ax_orbit) = plt.subplots(1, 2, figsize=(10.5, 4.3))

    for label, run, style in runs:
        stats = loss_stats(run, args.trace_time, 1e-4)
        loss = 1.0 - run["conf_pass"] - run["conf_trap"]
        pos = run["time"] > 0
        ax_loss.semilogx(run["time"][pos], loss[pos],
                         label=f"{label} ({stats['n_lost']} lost)", **style)
    ax_loss.set_xlabel("t [s]")
    ax_loss.set_ylabel("loss fraction")
    ax_loss.legend(loc="upper left", fontsize=9)
    ax_loss.grid(alpha=0.3)

    mask_d = loss_stats(direct, args.trace_time, 1e-4)["lost_mask"]
    for label, run, style in runs[1:]:
        mask_n = loss_stats(run, args.trace_time, 1e-4)["lost_mask"]
        both = mask_d & mask_n
        disagree = int((mask_d != mask_n).sum())
        ax_orbit.loglog(direct["times_lost"][both], run["times_lost"][both],
                        "o", ms=4, alpha=0.6, color=style["color"],
                        label=f"{label.replace('NEAT, ', '')} "
                              f"(n={int(both.sum())}, class. disagree={disagree})")
    lims = [args.trace_time * 1e-4, args.trace_time]
    ax_orbit.plot(lims, lims, color="0.5", lw=1, zorder=0)
    ax_orbit.set_xlim(lims); ax_orbit.set_ylim(lims)
    ax_orbit.set_xlabel(r"$t_\mathrm{lost}$ direct simple.x [s]")
    ax_orbit.set_ylabel(r"$t_\mathrm{lost}$ NEAT [s]")
    ax_orbit.legend(loc="upper left", fontsize=9)
    ax_orbit.grid(alpha=0.3, which="both")

    scale_txt = "native scale" if args.scale == "none" else "ARIES-CS scale"
    fig.suptitle(
        f"{wout.name}: {args.nparticles} alphas from s={args.sbeg:g}, "
        f"{scale_txt}, trace {args.trace_time:g} s"
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
