#!/usr/bin/env python
"""Compare SIMPLE loss runs: direct simple.x namelist vs NEAT wrapper.

Both runners trace the same deterministic alpha population from one starting
surface of the same wout file, so per-particle loss times are comparable
orbit by orbit. 'direct' uses the sensopt loss conventions (multharm=7,
ns_s=ns_tp=5, npoiper2=256, integmode=1, relerr=1e-13); 'neat' goes through
neat.fields.Simple + neat.tracing.ParticleEnsembleOrbit_Simple, either with
NEAT's default resolution (multharm=3, ns_s=ns_tp=3, npoiper2=128) or with
--matched resolution. Reports prompt/total loss fractions, the maximum
confined-fraction deviation, and per-particle loss-time agreement.

Example:
  .venv/bin/python scripts/compare_simple_neat.py \
      --wout ~/proj/proxima-simple-classification/data/wout_23_1900_fix_bdry.nc \
      --nparticles 128 --trace-time 1e-3 --matched
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import simple_barrier

LOSS_NAMELIST = """&config
  netcdffile = 'wout.nc'
  notrace_passing = 0
  ntestpart = {n}
  trace_time = {ttime}
  ntimstep = {nsamples}
  sbeg = {sbeg}d0
  num_surf = 1
  contr_pp = -1.0d10
  n_e = 2
  n_d = 4
  facE_al = {face}
  npoiper = 100
  npoiper2 = 256
  ns_s = 5
  ns_tp = 5
  multharm = 7
  isw_field_type = 2
  integmode = 1
  relerr = 1d-13
  nturns = 8
  tcut = -1d0
  vmec_RZ_scale = {rz}
  vmec_B_scale = {b}
  fast_class = .False.
  swcoll = .False.
  deterministic = .True.
/
"""


def run_direct(wout, *, sbeg, n, trace_time, nsamples, rz, b, face, workdir, timeout_s):
    import shutil

    workdir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(wout, workdir / "wout.nc")
    (workdir / "simple.in").write_text(
        LOSS_NAMELIST.format(
            n=n,
            ttime=f"{trace_time:.16g}".replace("e", "d"),
            nsamples=nsamples,
            sbeg=sbeg,
            face=f"{face:.16g}".replace("e", "d"),
            rz=f"{rz:.16g}".replace("e", "d"),
            b=f"{b:.16g}".replace("e", "d"),
        )
    )
    simple_x = simple_barrier.find_simple_x()
    completed = subprocess.run(
        [str(simple_x)], cwd=str(workdir), capture_output=True, text=True,
        timeout=timeout_s,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"simple.x failed:\n{completed.stdout}\n{completed.stderr}")
    confined = np.loadtxt(workdir / "confined_fraction.dat", ndmin=2)
    times_lost = np.loadtxt(workdir / "times_lost.dat", ndmin=2)
    return {
        "time": confined[:, 0],
        "conf_pass": confined[:, 1],
        "conf_trap": confined[:, 2],
        "times_lost": times_lost[:, 1],
        "trap_par": times_lost[:, 2],
    }


def run_neat(wout, *, sbeg, n, trace_time, nsamples, rz, b, matched, timeout_s):
    from neat.fields import Simple
    from neat.tracing import ChargedParticleEnsemble, ParticleEnsembleOrbit_Simple

    field = Simple(
        wout_filename=str(wout),
        B_scale=b,
        Aminor_scale=rz,
        multharm=7 if matched else 3,
        ns_s=5 if matched else 3,
        ns_tp=5 if matched else 3,
        simple_executable=str(simple_barrier.find_simple_x()),
    )
    particles = ChargedParticleEnsemble(r_initial=sbeg)
    orbits = ParticleEnsembleOrbit_Simple(
        particles,
        field,
        tfinal=trace_time,
        nsamples=nsamples,
        nparticles=n,
        notrace_passing=0,
        npoiper=100,
        npoiper2=256 if matched else 128,
        nper=1000,
    )
    return {
        "time": orbits.time,
        "conf_pass": orbits.confpart_pass,
        "conf_trap": orbits.confpart_trap,
        "times_lost": orbits.lost_times_of_particles,
        "trap_par": None,
    }


def loss_stats(run, trace_time, t_prompt):
    loss = 1.0 - run["conf_pass"] - run["conf_trap"]
    lost_mask = (run["times_lost"] > 0) & (run["times_lost"] < trace_time)
    return {
        "loss_total": float(loss[-1]),
        "loss_prompt": float(np.interp(t_prompt, run["time"], loss)),
        "n_lost": int(lost_mask.sum()),
        "lost_mask": lost_mask,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wout", required=True)
    ap.add_argument("--sbeg", type=float, default=0.3)
    ap.add_argument("--nparticles", type=int, default=128)
    ap.add_argument("--trace-time", type=float, default=1e-3)
    ap.add_argument("--nsamples", type=int, default=1000)
    ap.add_argument("--t-prompt", type=float, default=1e-4)
    ap.add_argument("--scale", choices=["reactor", "none"], default="reactor")
    ap.add_argument("--face-al", type=float, default=1.0)
    ap.add_argument("--matched", action="store_true",
                    help="run NEAT at the direct runner's resolution")
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--keep-workdir", action="store_true")
    args = ap.parse_args()

    wout = Path(args.wout).expanduser()
    if args.scale == "reactor":
        rz, b = simple_barrier.reactor_scale(wout)
    else:
        rz, b = 1.0, 1.0
    print(f"wout={wout.name} sbeg={args.sbeg} n={args.nparticles} "
          f"trace_time={args.trace_time:g}s rz_scale={rz:.4f} b_scale={b:.4f} "
          f"neat_resolution={'matched' if args.matched else 'default'}")

    base = Path(tempfile.mkdtemp(prefix="compare_simple_neat_"))
    direct = run_direct(
        wout, sbeg=args.sbeg, n=args.nparticles, trace_time=args.trace_time,
        nsamples=args.nsamples, rz=rz, b=b, face=args.face_al,
        workdir=base / "direct", timeout_s=args.timeout,
    )
    neat = run_neat(
        wout, sbeg=args.sbeg, n=args.nparticles, trace_time=args.trace_time,
        nsamples=args.nsamples, rz=rz, b=b, matched=args.matched,
        timeout_s=args.timeout,
    )

    sd = loss_stats(direct, args.trace_time, args.t_prompt)
    sn = loss_stats(neat, args.trace_time, args.t_prompt)

    print(f"{'':14s} {'direct':>10s} {'neat':>10s}")
    for key in ("loss_total", "loss_prompt", "n_lost"):
        print(f"{key:14s} {sd[key]:10.4f} {sn[key]:10.4f}")

    loss_d = 1.0 - direct["conf_pass"] - direct["conf_trap"]
    loss_n = np.interp(direct["time"], neat["time"], 1.0 - neat["conf_pass"] - neat["conf_trap"])
    print(f"max |loss_direct(t) - loss_neat(t)| = {np.max(np.abs(loss_d - loss_n)):.4g}")

    if len(direct["times_lost"]) == len(neat["times_lost"]):
        agree = sd["lost_mask"] == sn["lost_mask"]
        both_lost = sd["lost_mask"] & sn["lost_mask"]
        print(f"per-particle lost/confined agreement: {agree.sum()}/{len(agree)}")
        if both_lost.sum():
            dt = np.abs(direct["times_lost"][both_lost] - neat["times_lost"][both_lost])
            print(f"loss-time deltas among both-lost: median {np.median(dt):.3g}s, "
                  f"max {np.max(dt):.3g}s (n={both_lost.sum()})")
    else:
        print("particle counts differ; skipping per-particle comparison")

    if args.keep_workdir:
        print("direct workdir:", base / "direct")
    else:
        import shutil

        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
