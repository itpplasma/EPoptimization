"""Barrier-overlap proxy from SIMPLE fast orbit classification.

The metric is the mu-resolved barrier-breach fraction: the share of
birth-surface trapped particles that are chaotic at birth and whose magnetic
moment lands in a mu region that is also chaotic at the barrier surface. Lower
is better.

Three corrections against the 2026-07 campaign, which concluded the fast
classifier was slower than direct loss tracing:

1. ``class_plot`` stays ``.False.``. The early exit in SIMPLE's
   classification.f90 requires ``.not. class_plot``; with it set, no orbit ever
   exits early and every trace runs to ``trace_time``. The old namelist had it
   ``.True.`` in every commit of its history.
2. ``tcut = -1d0`` with ``fast_class = .True.``. This needs SIMPLE with the
   standalone-dispatch fix (PR #512): before it, that combination silently
   disabled the classifier and wrote no ``class_parts.dat``. It is the only
   configuration that classifies without the Minkowski fractal cut, which
   fires at ``kt == ntcut``.
3. ``notrace_passing = 1`` on *both* surfaces. The overlap masks everything by
   the trapped flag, so passing particles at the birth surface were traced for
   nothing and cost three quarters of the old budget. ``trap_par`` is computed
   before the skip and written for every particle, so the trapped mask is
   unaffected.

Starting points are a deterministic product grid rather than random draws, so
the same phase-space points are compared across candidate designs.

``class_parts.dat`` columns: idx, s, perp_inv, jpar, topology, fractal, with
codes 0 = prompt loss or unresolved, 1 = regular/ideal, 2 = chaotic/non-ideal.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

CLASS_COLUMN = {"jpar": 3, "topology": 4}

A_TARGET = 1.704
B_TARGET = 5.865

#: Boozer field, midpoint symplectic integrator. Pinned for every run.
ISW_FIELD_TYPE_BOOZER = 2
INTEGMODE_MIDPOINT = 3

CLASSIFY_NAMELIST = """&config
  netcdffile = 'wout.nc'
  startmode = 2
  num_surf = 1
  notrace_passing = 1
  ntestpart = {n}
  trace_time = {ttime}
  sbeg = {sbeg}d0
  contr_pp = -1.0d10
  n_e = 2
  n_d = 4
  facE_al = 1.0d0
  npoiper = 100
  npoiper2 = 256
  ns_s = 5
  ns_tp = 5
  multharm = 5
  isw_field_type = {field_type}
  integmode = {integmode}
  relerr = 1d-13
  nturns = {nturns}
  tcut = -1d0
  vmec_RZ_scale = {rz}
  vmec_B_scale = {b}
  class_plot = .False.
  fast_class = .True.
  swcoll = .False.
  deterministic = .True.
  ran_seed = {seed}
/
"""


def find_simple_x(explicit: str | os.PathLike | None = None) -> Path:
    if explicit is not None:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    environment = os.environ.get("SIMPLE_X") or os.environ.get("SIMPLE_EXE")
    if environment:
        path = Path(environment)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    for candidate in (
        Path.home() / "code" / "SIMPLE" / "build" / "simple.x",
        Path.home() / "bin" / "simple.x",
    ):
        if candidate.exists():
            return candidate
    installed = shutil.which("simple.x")
    if installed:
        return Path(installed)
    raise FileNotFoundError("Could not locate SIMPLE executable")


def file_sha256(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reactor_scale(wout_path: str | os.PathLike) -> tuple[float, float]:
    """Return scales that put a VMEC equilibrium at the ARIES-CS point."""
    from scipy.io import netcdf_file

    with netcdf_file(str(wout_path), "r", mmap=False) as dataset:
        minor_radius = float(dataset.variables["Aminor_p"][()])
        mean_field = float(dataset.variables["volavgB"][()])
    return A_TARGET / minor_radius, B_TARGET / abs(mean_field)


def pitch_grid(count: int, *, pitch_max: float = 0.6) -> np.ndarray:
    """Signed pitch values whose squares are uniform in quantile.

    Uniform spacing in ``lambda**2`` spreads points evenly in magnetic moment
    at fixed field strength, which is the variable the overlap metric bins.

    ``pitch_max`` bounds the grid because a particle is trapped only where
    ``lambda**2 < 1 - B/B_max``, and the overlap metric discards everything
    passing. Spanning the full unit interval put seven eighths of the starts
    outside the trapped region, leaving about eight trapped particles per mu
    bin. The bound is a fixed constant rather than a per-candidate trapping
    boundary so that every design is sampled at identical phase-space points.
    """
    if count <= 0 or count % 2 != 0:
        raise ValueError("pitch count must be positive and even")
    if not 0.0 < pitch_max <= 1.0:
        raise ValueError("pitch bound must lie in (0, 1]")
    half = count // 2
    magnitude = pitch_max * np.sqrt((np.arange(half, dtype=float) + 0.5) / half)
    return np.sort(np.concatenate((-magnitude, magnitude)))


def starting_grid(
    surface: float,
    *,
    ntheta: int,
    nzeta: int,
    npitch: int,
    nfp: int,
    pitch_max: float = 0.6,
) -> np.ndarray:
    """Deterministic (theta, zeta, pitch) product grid in VMEC coordinates.

    Columns match SIMPLE's ``start.dat``: s, theta, varphi, v/v0, v_par/v.
    Angles are cell-centred so no point sits on a symmetry plane, and zeta
    spans one field period since the equilibrium repeats.
    """
    if not 0.0 < surface < 1.0:
        raise ValueError("birth surface must lie inside the plasma")
    if min(ntheta, nzeta, npitch) <= 0 or nfp <= 0:
        raise ValueError("grid counts and field-period number must be positive")
    theta = 2.0 * np.pi * (np.arange(ntheta) + 0.5) / ntheta
    zeta = 2.0 * np.pi * (np.arange(nzeta) + 0.5) / (nfp * nzeta)
    pitch = pitch_grid(npitch, pitch_max=pitch_max)
    mesh = np.stack(
        np.meshgrid(theta, zeta, pitch, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    rows = np.empty((mesh.shape[0], 5), dtype=float)
    rows[:, 0] = surface
    rows[:, 1] = mesh[:, 0]
    rows[:, 2] = mesh[:, 1]
    rows[:, 3] = 1.0
    rows[:, 4] = mesh[:, 2]
    return rows


def field_periods(wout_path: str | os.PathLike) -> int:
    from scipy.io import netcdf_file

    with netcdf_file(str(wout_path), "r", mmap=False) as dataset:
        return int(dataset.variables["nfp"][()])


def load_classification(
    run: Path, column: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (perp_inv, class code, trapped mask) for one classification run.

    ``trap_par`` is column 3 of ``times_lost.dat`` and is written for every
    particle, including the passing ones skipped by ``notrace_passing``.
    """
    classes = np.loadtxt(run / "class_parts.dat", ndmin=2)
    particles = np.loadtxt(run / "times_lost.dat", ndmin=2)
    if classes.shape[0] != particles.shape[0]:
        raise ValueError(f"classification particle count mismatch in {run}")
    if not np.array_equal(
        classes[:, 0].astype(int), particles[:, 0].astype(int)
    ):
        raise ValueError(f"classification particle indices mismatch in {run}")
    if classes.shape[1] <= column or particles.shape[1] < 3:
        raise ValueError(f"classification output is missing columns in {run}")
    return classes[:, 2], classes[:, column].astype(int), particles[:, 2] > 0.0


def barrier_overlap_samples(
    mu_inner: np.ndarray,
    class_inner: np.ndarray,
    trapped_inner: np.ndarray,
    mu_outer: np.ndarray,
    class_outer: np.ndarray,
    trapped_outer: np.ndarray,
    *,
    edges: np.ndarray,
) -> float:
    """Sum over mu bins of P(chaotic at birth) * F(chaotic at barrier).

    ``edges`` is supplied by the caller rather than derived from the samples.
    Deriving them from the common mu range, as the 2026-07 version did, moves
    the bin boundaries with the candidate geometry and makes the metric
    discontinuous in the design variables.
    """
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or not np.all(np.diff(edges) > 0.0):
        raise ValueError("mu bin edges must be increasing and at least two")
    if trapped_inner.sum() == 0 or trapped_outer.sum() == 0:
        return float("nan")
    inner_total = int(trapped_inner.sum())
    nbins = edges.size - 1
    overlap = 0.0
    for index in range(nbins):
        low, high = edges[index], edges[index + 1]
        last = index == nbins - 1
        in_inner = (mu_inner >= low) & (
            mu_inner <= high if last else mu_inner < high
        )
        in_outer = (mu_outer >= low) & (
            mu_outer <= high if last else mu_outer < high
        )
        inner = trapped_inner & in_inner
        outer = trapped_outer & in_outer
        if inner.sum() == 0 or outer.sum() == 0:
            continue
        birth_chaotic = float((class_inner[inner] == 2).sum()) / inner_total
        barrier_breach = float((class_outer[outer] == 2).sum()) / int(outer.sum())
        overlap += birth_chaotic * barrier_breach
    return float(overlap)


def fixed_mu_edges(b_scale: float, *, nbins: int) -> np.ndarray:
    """Design-independent mu bin edges.

    ``perp_inv`` is ``v_perp**2 / B`` in SIMPLE's normalised units, so for a
    normalised speed of one it is bounded by ``1 / B_min``. Anchoring the
    edges to the reactor field rather than to the sample range keeps the bins
    identical across candidates, which is what the optimizer needs.
    """
    if nbins <= 0 or not np.isfinite(b_scale) or b_scale <= 0.0:
        raise ValueError("bin count and field scale must be positive")
    return np.linspace(0.0, 1.0 / B_TARGET, nbins + 1)


def classification_loss_metrics(
    run: Path, *, prompt_time: float, trace_time: float
) -> dict[str, float | int]:
    """Loss fractions from a classification run.

    Passing particles are skipped and recorded with ``times_lost = -1``, so
    they count as confined. That is a bias only to the extent that passing
    particles are actually lost, which is negligible at reactor scale on this
    time scale, and it is what makes the run cheap.
    """
    particles = np.loadtxt(run / "times_lost.dat", ndmin=2)
    if particles.ndim != 2 or particles.shape[1] < 2:
        raise ValueError(f"invalid times_lost.dat in {run}")
    if not 0.0 < prompt_time < trace_time:
        raise ValueError("loss windows require 0 < prompt < trace")
    times = particles[:, 1]
    prompt = (times > 0.0) & (times <= prompt_time)
    late = (times > prompt_time) & (times < trace_time)
    count = times.size
    return {
        "particles": int(count),
        "prompt_count": int(prompt.sum()),
        "late_count": int(late.sum()),
        "prompt_loss": float(prompt.mean()),
        "late_loss": float(late.mean()),
        "total_loss": float((prompt | late).mean()),
    }


def run_classification(
    wout_path: str | os.PathLike,
    *,
    surface: float,
    starts: np.ndarray,
    rz_scale: float,
    b_scale: float,
    trace_time: float,
    nturns: int,
    seed: int,
    workdir: Path,
    simple_executable: str | os.PathLike | None = None,
    timeout_s: float = 3600.0,
) -> Path:
    starts = np.asarray(starts, dtype=float)
    if starts.ndim != 2 or starts.shape[1] != 5:
        raise ValueError("starting points must have five columns")
    simple_x = find_simple_x(simple_executable)
    workdir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(wout_path, workdir / "wout.nc")
    np.savetxt(workdir / "start.dat", starts, fmt="%.17e")
    (workdir / "simple.in").write_text(
        CLASSIFY_NAMELIST.format(
            n=starts.shape[0],
            ttime=_fortran_d(trace_time),
            sbeg=_fortran_d(surface),
            field_type=ISW_FIELD_TYPE_BOOZER,
            integmode=INTEGMODE_MIDPOINT,
            nturns=int(nturns),
            rz=_fortran_d(rz_scale),
            b=_fortran_d(b_scale),
            seed=int(seed),
        )
    )
    log_path = workdir / "simple_stdout.txt"
    with log_path.open("w") as log:
        completed = subprocess.run(
            [str(simple_x)],
            cwd=str(workdir),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"SIMPLE failed with exit code {completed.returncode}; see {log_path}"
        )
    if not (workdir / "class_parts.dat").exists():
        raise FileNotFoundError(
            f"{workdir / 'class_parts.dat'} is missing; SIMPLE must carry the "
            "fast_class standalone dispatch fix"
        )
    return workdir


def barrier_metrics(
    wout_path: str | os.PathLike,
    *,
    expected_simple_sha256: str,
    s_inner: float,
    s_outer: float,
    ntheta: int,
    nzeta: int,
    npitch: int,
    pitch_max: float,
    nbins: int,
    trace_time: float,
    prompt_time: float,
    nturns: int,
    seed: int,
    classifier: str = "topology",
    simple_executable: str | os.PathLike | None = None,
    keep_workdir: bool = False,
    timeout_s: float = 3600.0,
) -> dict:
    if classifier not in CLASS_COLUMN:
        raise ValueError(f"unknown classifier {classifier}")
    if not 0.0 < s_inner < s_outer < 1.0:
        raise ValueError("barrier surfaces must satisfy 0 < inner < outer < 1")
    simple_x = find_simple_x(simple_executable)
    binary_hash = file_sha256(simple_x)
    if binary_hash != expected_simple_sha256:
        raise ValueError("unexpected SIMPLE executable hash")
    rz_scale, b_scale = reactor_scale(wout_path)
    nfp = field_periods(wout_path)
    edges = fixed_mu_edges(b_scale, nbins=nbins)
    base = Path(tempfile.mkdtemp(prefix="barrier_"))
    try:
        runs = {}
        for label, surface in (("inner", s_inner), ("outer", s_outer)):
            runs[label] = run_classification(
                wout_path,
                surface=surface,
                starts=starting_grid(
                    surface,
                    ntheta=ntheta,
                    nzeta=nzeta,
                    npitch=npitch,
                    nfp=nfp,
                    pitch_max=pitch_max,
                ),
                rz_scale=rz_scale,
                b_scale=b_scale,
                trace_time=trace_time,
                nturns=nturns,
                seed=seed,
                workdir=base / label,
                simple_executable=simple_x,
                timeout_s=timeout_s,
            )
        column = CLASS_COLUMN[classifier]
        inner = load_classification(runs["inner"], column)
        outer = load_classification(runs["outer"], column)
        overlap = barrier_overlap_samples(*inner, *outer, edges=edges)
        losses = classification_loss_metrics(
            runs["inner"], prompt_time=prompt_time, trace_time=trace_time
        )
        return {
            "barrier_overlap": overlap,
            "classifier": classifier,
            "prompt_loss_inner": losses["prompt_loss"],
            "short_loss_inner": losses["total_loss"],
            "loss_metrics_inner": losses,
            "trapped_inner": int(inner[2].sum()),
            "trapped_outer": int(outer[2].sum()),
            "chaotic_trapped_inner": _chaotic_fraction(inner),
            "chaotic_trapped_outer": _chaotic_fraction(outer),
            "particles_per_surface": ntheta * nzeta * npitch,
            "pitch_max": float(pitch_max),
            "mu_bin_edges": edges.tolist(),
            "s_inner": float(s_inner),
            "s_outer": float(s_outer),
            "nfp": nfp,
            "trace_time": float(trace_time),
            "nturns": int(nturns),
            "seed": int(seed),
            "integmode": INTEGMODE_MIDPOINT,
            "isw_field_type": ISW_FIELD_TYPE_BOOZER,
            "vmec_RZ_scale": float(rz_scale),
            "vmec_B_scale": float(b_scale),
            "simple_sha256": binary_hash,
            "wout_sha256": file_sha256(wout_path),
            "workdir": str(base) if keep_workdir else "",
        }
    finally:
        if not keep_workdir:
            shutil.rmtree(base, ignore_errors=True)


def _chaotic_fraction(sample: tuple[np.ndarray, np.ndarray, np.ndarray]) -> float:
    _, classes, trapped = sample
    count = int(trapped.sum())
    if count == 0:
        return float("nan")
    return float((classes[trapped] == 2).sum()) / count


def _fortran_d(value: float) -> str:
    return f"{float(value):.16g}".replace("e", "d").replace("E", "d")
