"""Barrier-overlap confinement proxy from SIMPLE fast classification.

Ported from itpplasma/sensopt (benchmarks/cluster/cluster_eval.py, analyze.py).
Two trapped-only fast-classification runs (birth surface, barrier surface)
yield the mu-resolved barrier-breach fraction: the fraction of birth-chaotic
trapped particles whose magnetic moment lands in a mu region that is also
chaotic at the barrier surface. Lower is better; sensopt found this tracks
traced late loss far better than any single-surface chaotic fraction.

class_parts.dat columns: idx, s, perp_inv, jpar, topology, fractal with codes
0=prompt-loss, 1=regular/ideal, 2=chaotic/non-ideal.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

CLASS_COL = {"jpar": 3, "topology": 4}

A_TARGET = 1.704
B_TARGET = 5.865

CLASSIFY_NAMELIST = """&config
  netcdffile = 'wout.nc'
  notrace_passing = 1
  ntestpart = {n}
  trace_time = {ttime}
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
  class_plot = .True.
  fast_class = .True.
  swcoll = .False.
  deterministic = .True.
/
"""


def find_simple_x(explicit: str | os.PathLike | None = None) -> Path:
    if explicit is not None:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    env = os.environ.get("SIMPLE_X") or os.environ.get("SIMPLE_EXE")
    if env:
        path = Path(env)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    for candidate in (
        Path.home() / "code" / "SIMPLE" / "build" / "simple.x",
        Path.home() / "bin" / "simple.x",
    ):
        if candidate.exists():
            return candidate
    which = shutil.which("simple.x")
    if which:
        return Path(which)
    raise FileNotFoundError(
        "Could not locate SIMPLE executable. Set SIMPLE_X/SIMPLE_EXE or pass a path."
    )


def reactor_scale(wout_path: str | os.PathLike) -> tuple[float, float]:
    """(vmec_RZ_scale, vmec_B_scale) putting this wout at the ARIES-CS point."""
    from scipy.io import netcdf_file

    with netcdf_file(str(wout_path), "r", mmap=False) as d:
        a = float(d.variables["Aminor_p"][()])
        b = float(d.variables["volavgB"][()])
    return A_TARGET / a, B_TARGET / abs(b)


def load_classification(run: Path, col: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cp = np.loadtxt(run / "class_parts.dat", ndmin=2)
    particles = np.loadtxt(run / "times_lost.dat", ndmin=2)
    if cp.shape[0] != particles.shape[0]:
        raise ValueError(f"classification particle count mismatch in {run}")
    if not np.array_equal(cp[:, 0].astype(int), particles[:, 0].astype(int)):
        raise ValueError(f"classification particle indices mismatch in {run}")
    return cp[:, 2], cp[:, col].astype(int), particles[:, 2] >= 0.0


def chaotic_fraction(run: Path, col: int = 4) -> tuple[float, float, int]:
    """Return (chaotic_trapped, chaotic_all, n_trapped)."""
    _, topo, trapped = load_classification(run, col)
    n_tr = int(trapped.sum())
    chaotic_trapped = float((topo[trapped] == 2).sum() / n_tr) if n_tr else float("nan")
    return chaotic_trapped, float((topo == 2).mean()), n_tr


def barrier_overlap(inner: Path, outer: Path, nbins: int = 16, col: int = 4) -> float:
    """mu-resolved barrier-breach fraction (lower is better).

    The fraction of birth-surface particles that are chaotic at birth AND land
    in a magnetic-moment region that is also chaotic at the barrier surface.
    mu is class_parts column 2 (conserved along the orbit, so comparable across
    surfaces); bins are taken on the mu range common to both surfaces.
    """
    mu_i, topo_i, tr_i = load_classification(inner, col)
    mu_o, topo_o, tr_o = load_classification(outer, col)
    if tr_i.sum() == 0 or tr_o.sum() == 0:
        return float("nan")
    lo = max(mu_i[tr_i].min(), mu_o[tr_o].min())
    hi = min(mu_i[tr_i].max(), mu_o[tr_o].max())
    edges = np.linspace(lo, hi, nbins + 1)
    n_inner = int(tr_i.sum())
    m = 0.0
    for index, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        upper_i = mu_i <= b if index == nbins - 1 else mu_i < b
        upper_o = mu_o <= b if index == nbins - 1 else mu_o < b
        ii = tr_i & (mu_i >= a) & upper_i
        oo = tr_o & (mu_o >= a) & upper_o
        if ii.sum() == 0 or oo.sum() == 0:
            continue
        p_birth_chaotic = (topo_i[ii] == 2).sum() / n_inner
        f_barrier_breach = (topo_o[oo] == 2).sum() / oo.sum()
        m += p_birth_chaotic * f_barrier_breach
    return float(m)


def _fortran_d(value: float) -> str:
    return f"{float(value):.16g}".replace("e", "d").replace("E", "d")


def run_classification(
    wout_path: str | os.PathLike,
    *,
    sbeg: float,
    ntestpart: int,
    rz_scale: float,
    b_scale: float,
    facE_al: float = 1.0,
    trace_time: float = 2.0e-2,
    workdir: Path,
    simple_executable: str | os.PathLike | None = None,
    timeout_s: float = 3600.0,
) -> Path:
    simple_x = find_simple_x(simple_executable)
    workdir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(wout_path, workdir / "wout.nc")
    (workdir / "simple.in").write_text(
        CLASSIFY_NAMELIST.format(
            n=int(ntestpart),
            ttime=_fortran_d(trace_time),
            sbeg=sbeg,
            face=_fortran_d(facE_al),
            rz=_fortran_d(rz_scale),
            b=_fortran_d(b_scale),
        )
    )
    completed = subprocess.run(
        [str(simple_x)],
        cwd=str(workdir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    (workdir / "simple_stdout.txt").write_text(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(
            f"SIMPLE failed with exit code {completed.returncode}, "
            f"see {workdir / 'simple_stdout.txt'}"
        )
    if not (workdir / "class_parts.dat").exists():
        raise FileNotFoundError(workdir / "class_parts.dat")
    return workdir


def barrier_metrics(
    wout_path: str | os.PathLike,
    *,
    s_inner: float = 0.3,
    s_outer: float = 0.6,
    ntestpart: int = 3000,
    rz_scale: float,
    b_scale: float,
    facE_al: float = 1.0,
    trace_time: float = 2.0e-2,
    classifier: str = "topology",
    simple_executable: str | os.PathLike | None = None,
    keep_workdir: bool = False,
    timeout_s: float = 3600.0,
) -> dict:
    col = CLASS_COL[classifier]
    base = Path(tempfile.mkdtemp(prefix="barrier_"))
    try:
        runs = {}
        for label, sbeg in (("inner", s_inner), ("outer", s_outer)):
            runs[label] = run_classification(
                wout_path,
                sbeg=sbeg,
                ntestpart=ntestpart,
                rz_scale=rz_scale,
                b_scale=b_scale,
                facE_al=facE_al,
                trace_time=trace_time,
                workdir=base / label,
                simple_executable=simple_executable,
                timeout_s=timeout_s,
            )
        overlap = barrier_overlap(runs["inner"], runs["outer"], col=col)
        chaotic_in, _, n_trapped_in = chaotic_fraction(runs["inner"], col=col)
        chaotic_out, _, n_trapped_out = chaotic_fraction(runs["outer"], col=col)
        conf = np.loadtxt(runs["inner"] / "confined_fraction.dat", ndmin=2)
        proxy_loss = float(1.0 - conf[-1, 1] - conf[-1, 2]) if conf.size else float("nan")
        return {
            "barrier_overlap": overlap,
            "chaotic_trapped_inner": chaotic_in,
            "chaotic_trapped_outer": chaotic_out,
            "n_trapped_inner": n_trapped_in,
            "n_trapped_outer": n_trapped_out,
            "proxy_loss_inner": proxy_loss,
            "vmec_RZ_scale": float(rz_scale),
            "vmec_B_scale": float(b_scale),
            "facE_al": float(facE_al),
            "workdir": str(base) if keep_workdir else "",
        }
    finally:
        if not keep_workdir:
            shutil.rmtree(base, ignore_errors=True)
