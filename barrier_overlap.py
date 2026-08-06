"""Continuous radial-barrier proxy from SIMPLE fast orbit classification.

Production uses only the raw J_parallel variation and tip-map rotation-drift
scores.  It reconstructs each score on fixed magnetic-moment nodes over several
radial surfaces, takes a soft minimum along the escape path, and integrates the
result against deterministic alpha-birth quadrature weights. Lower is better.

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

The old two-surface integer overlap helpers remain below only to read archived
campaigns.  :func:`barrier_metrics` never reads an integer class or invokes the
Minkowski classifier.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Sequence

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
  num_surf = {num_surf}
  notrace_passing = 1
  ntestpart = {n}
  trace_time = {ttime}
  sbeg = {sbeg}
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


def pitch_quadrature(
    count: int, *, pitch_max: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and normalized weights for isotropic pitch.

    Fusion alpha birth is uniform in ``v_parallel / v``.  Fixed quadrature
    nodes therefore represent that distribution without random sampling and
    without deriving a grid from the candidate geometry.
    """
    if count <= 0:
        raise ValueError("pitch count must be positive")
    if not 0.0 < pitch_max <= 1.0:
        raise ValueError("pitch bound must lie in (0, 1]")
    nodes, weights = np.polynomial.legendre.leggauss(count)
    nodes = pitch_max * nodes
    weights = pitch_max * weights
    return nodes, weights / weights.sum()


def pitch_grid(count: int, *, pitch_max: float = 1.0) -> np.ndarray:
    """Pitch nodes from :func:`pitch_quadrature`."""
    return pitch_quadrature(count, pitch_max=pitch_max)[0]


def starting_grid(
    surface: float,
    *,
    ntheta: int,
    nzeta: int,
    npitch: int,
    nfp: int,
    pitch_max: float = 1.0,
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
    mesh = np.stack(np.meshgrid(theta, zeta, pitch, indexing="ij"), axis=-1).reshape(
        -1, 3
    )
    rows = np.empty((mesh.shape[0], 5), dtype=float)
    rows[:, 0] = surface
    rows[:, 1] = mesh[:, 0]
    rows[:, 2] = mesh[:, 1]
    rows[:, 3] = 1.0
    rows[:, 4] = mesh[:, 2]
    return rows


def starting_weights(ntheta: int, nzeta: int, npitch: int) -> np.ndarray:
    """Normalized alpha-birth weights in ``starting_grid`` row order."""
    if min(ntheta, nzeta, npitch) <= 0:
        raise ValueError("grid counts must be positive")
    _, pitch_weights = pitch_quadrature(npitch)
    weights = np.tile(pitch_weights, ntheta * nzeta)
    return weights / (ntheta * nzeta)


#: SIMPLE splines the equilibrium with ns_s = ns_tp = 5, so a wout with too
#: few flux surfaces overruns the spline construction and aborts the process
#: inside spline_vmec_data rather than returning an error.
MINIMUM_FLUX_SURFACES = 2 * 5 + 1


def field_periods(wout_path: str | os.PathLike) -> int:
    from scipy.io import netcdf_file

    with netcdf_file(str(wout_path), "r", mmap=False) as dataset:
        return int(dataset.variables["nfp"][()])


def flux_surface_count(wout_path: str | os.PathLike) -> int:
    from scipy.io import netcdf_file

    with netcdf_file(str(wout_path), "r", mmap=False) as dataset:
        return int(dataset.variables["ns"][()])


def check_radial_resolution(wout_path: str | os.PathLike) -> int:
    """Reject equilibria too coarse for SIMPLE's radial splines."""
    surfaces = flux_surface_count(wout_path)
    if surfaces < MINIMUM_FLUX_SURFACES:
        raise ValueError(
            f"{wout_path} has ns={surfaces}, below the {MINIMUM_FLUX_SURFACES} "
            "flux surfaces SIMPLE needs for ns_s = ns_tp = 5; raise NS_ARRAY "
            "in the VMEC input"
        )
    return surfaces


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
    if not np.array_equal(classes[:, 0].astype(int), particles[:, 0].astype(int)):
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
        in_inner = (mu_inner >= low) & (mu_inner <= high if last else mu_inner < high)
        in_outer = (mu_outer >= low) & (mu_outer <= high if last else mu_outer < high)
        inner = trapped_inner & in_inner
        outer = trapped_outer & in_outer
        if inner.sum() == 0 or outer.sum() == 0:
            continue
        birth_chaotic = float((class_inner[inner] == 2).sum()) / inner_total
        barrier_breach = float((class_outer[outer] == 2).sum()) / int(outer.sum())
        overlap += birth_chaotic * barrier_breach
    return float(overlap)


#: SIMPLE carries the field in Gauss, so ``perp_inv = v_perp**2 / B`` for a
#: normalised speed sits near ``1 / (B[T] * 1e4)`` — of order 1e-5 at reactor
#: field, not of order 0.1.
GAUSS_PER_TESLA = 1.0e4

#: Fraction of the reference mu the bin band spans on either side. Trapped
#: particles occupy roughly ``1/B_max`` to ``1/B_min``, a factor set by the
#: mirror ratio; +-30% covers a mirror ratio of 0.2 with margin.
MU_BAND_FRACTION = 0.3


def reference_mu() -> float:
    """Perpendicular invariant of a trapped particle at the reactor field."""
    return 1.0 / (B_TARGET * GAUSS_PER_TESLA)


def fixed_mu_edges(b_scale: float, *, nbins: int) -> np.ndarray:
    """Design-independent mu bin edges.

    ``perp_inv`` is ``v_perp**2 / B`` in SIMPLE's units, so for a normalised
    speed of one it is of order ``1 / B_min`` with B in Gauss. Anchoring the
    edges to the reactor field rather than to the sample range keeps the bins
    identical across candidates, which is what the optimizer needs.

    The band is centred on the reference mu rather than starting at zero.
    Trapped particles occupy a narrow band set by the mirror ratio, so bins
    running from zero put every one of them in the first bin and throw away
    all the mu resolution the metric is built on.
    """
    if nbins <= 0 or not np.isfinite(b_scale) or b_scale <= 0.0:
        raise ValueError("bin count and field scale must be positive")
    centre = reference_mu()
    return np.linspace(
        centre * (1.0 - MU_BAND_FRACTION), centre * (1.0 + MU_BAND_FRACTION), nbins + 1
    )


def fixed_mu_nodes(count: int) -> np.ndarray:
    """Design-independent quadrature nodes for continuous ``mu`` fields."""
    if count < 3:
        raise ValueError("at least three mu nodes are required")
    centre = reference_mu()
    return np.linspace(
        centre * (1.0 - MU_BAND_FRACTION),
        centre * (1.0 + MU_BAND_FRACTION),
        count,
    )


def radial_quadrature_weights(surfaces: Sequence[float]) -> np.ndarray:
    """Normalized trapezoid weights on an increasing radial path."""
    surfaces = np.asarray(surfaces, dtype=float)
    if surfaces.ndim != 1 or surfaces.size < 2:
        raise ValueError("at least two radial surfaces are required")
    if not np.all(np.diff(surfaces) > 0.0):
        raise ValueError("radial surfaces must be strictly increasing")
    weights = np.empty_like(surfaces)
    weights[0] = 0.5 * (surfaces[1] - surfaces[0])
    weights[-1] = 0.5 * (surfaces[-1] - surfaces[-2])
    weights[1:-1] = 0.5 * (surfaces[2:] - surfaces[:-2])
    return weights / weights.sum()


def classification_loss_metrics(
    run: Path,
    *,
    prompt_time: float,
    trace_time: float,
    sample_weights: np.ndarray | None = None,
    rows: slice | None = None,
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
    if rows is not None:
        particles = particles[rows]
    times = particles[:, 1]
    prompt = (times > 0.0) & (times <= prompt_time)
    late = (times > prompt_time) & (times < trace_time)
    count = times.size
    weights = (
        np.ones(count, dtype=float)
        if sample_weights is None
        else np.asarray(sample_weights, dtype=float)
    )
    if weights.shape != (count,) or np.any(weights < 0.0):
        raise ValueError("loss sample weights are invalid")
    if not np.all(np.isfinite(weights)) or weights.sum() <= 0.0:
        raise ValueError("loss sample weights must be finite with positive mass")
    return {
        "particles": int(count),
        "prompt_count": int(prompt.sum()),
        "late_count": int(late.sum()),
        "prompt_loss": float(np.average(prompt, weights=weights)),
        "late_loss": float(np.average(late, weights=weights)),
        "total_loss": float(np.average(prompt | late, weights=weights)),
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
    surface_bounds: Sequence[float] | None = None,
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
    bounds = [surface] if surface_bounds is None else list(surface_bounds)
    if not bounds:
        raise ValueError("at least one classifier surface is required")
    namelist_surfaces = [bounds[0]] if len(bounds) == 1 else [bounds[0], bounds[-1]]
    sbeg = ", ".join(
        f"{float(value):.16e}".replace("e", "d") for value in namelist_surfaces
    )
    (workdir / "simple.in").write_text(
        CLASSIFY_NAMELIST.format(
            n=starts.shape[0],
            ttime=_fortran_d(trace_time),
            num_surf=len(namelist_surfaces),
            sbeg=sbeg,
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


DEFAULT_CONTINUOUS_SETTINGS = {
    "trapped_width": 0.15,
    "mu_width_factor": 0.75,
    "temperature_jpar": 0.1,
    "temperature_rotation": 0.02,
}


def load_perp_invariant(run: Path) -> np.ndarray:
    """Read magnetic moment without consulting any integer class column."""
    table = np.loadtxt(run / "class_parts.dat", ndmin=2)
    if table.shape[1] < 3:
        raise ValueError(f"class_parts.dat is missing magnetic moment in {run}")
    expected = np.arange(1, table.shape[0] + 1)
    if not np.array_equal(table[:, 0].astype(int), expected):
        raise ValueError(
            f"class_parts.dat particle indices are not sequential in {run}"
        )
    return table[:, 2]


def barrier_metrics(
    wout_path: str | os.PathLike,
    *,
    expected_simple_sha256: str,
    surfaces: Sequence[float],
    ntheta: int,
    nzeta: int,
    npitch: int,
    pitch_max: float,
    nmu: int,
    trace_time: float,
    prompt_time: float,
    nturns: int,
    seed: int,
    continuous_settings: dict | None = None,
    simple_executable: str | os.PathLike | None = None,
    work_root: str | os.PathLike | None = None,
    keep_workdir: bool = False,
    timeout_s: float = 3600.0,
) -> dict:
    """Run raw fast classifiers and form continuous radial-barrier fields."""
    surfaces = np.asarray(surfaces, dtype=float)
    radial_weights = radial_quadrature_weights(surfaces)
    if surfaces[0] <= 0.0 or surfaces[-1] >= 1.0:
        raise ValueError("barrier surfaces must lie strictly inside the plasma")
    simple_x = find_simple_x(simple_executable)
    binary_hash = file_sha256(simple_x)
    if binary_hash != expected_simple_sha256:
        raise ValueError("unexpected SIMPLE executable hash")
    check_radial_resolution(wout_path)
    rz_scale, b_scale = reactor_scale(wout_path)
    nfp = field_periods(wout_path)
    nodes = fixed_mu_nodes(nmu)
    particle_weights = starting_weights(ntheta, nzeta, npitch)
    work_parent = None if work_root is None else Path(work_root)
    if work_parent is not None:
        work_parent.mkdir(parents=True, exist_ok=True)
    base = Path(tempfile.mkdtemp(prefix="continuous_barrier_", dir=work_parent))
    total_started = time.monotonic()
    try:
        particles_per_surface = ntheta * nzeta * npitch
        starts = np.concatenate(
            [
                starting_grid(
                    float(surface),
                    ntheta=ntheta,
                    nzeta=nzeta,
                    npitch=npitch,
                    nfp=nfp,
                    pitch_max=pitch_max,
                )
                for surface in surfaces
            ]
        )
        surface_slices = [
            slice(index * particles_per_surface, (index + 1) * particles_per_surface)
            for index in range(len(surfaces))
        ]
        started = time.monotonic()
        run = run_classification(
            wout_path,
            surface=float(surfaces[0]),
            surface_bounds=(float(surfaces[0]), float(surfaces[-1])),
            starts=starts,
            rz_scale=rz_scale,
            b_scale=b_scale,
            trace_time=trace_time,
            nturns=nturns,
            seed=seed,
            workdir=base / "all_surfaces",
            simple_executable=simple_x,
            timeout_s=timeout_s,
        )
        classification_seconds = time.monotonic() - started
        continuous = _continuous_metrics(
            [run],
            particle_weights,
            nodes=nodes,
            radial_weights=radial_weights,
            settings=continuous_settings,
            surface_slices=surface_slices,
        )
        losses = classification_loss_metrics(
            run,
            prompt_time=prompt_time,
            trace_time=trace_time,
            sample_weights=particle_weights,
            rows=surface_slices[0],
        )
        return {
            "continuous": continuous,
            "classifiers": ["jpar", "rotation"],
            "prompt_loss_birth": losses["prompt_loss"],
            "short_loss_birth": losses["total_loss"],
            "loss_metrics_birth": losses,
            "trapped_count_by_surface": continuous["trapped_count_by_surface"],
            "soft_trapped_mass_by_surface": continuous["soft_trapped_mass_by_surface"],
            "particles_per_surface": particles_per_surface,
            "pitch_max": float(pitch_max),
            "sample_distribution": "isotropic-pitch-gauss-legendre-equal-angle",
            "mu_nodes": nodes.tolist(),
            "surfaces": surfaces.tolist(),
            "radial_weights": radial_weights.tolist(),
            "s_inner": float(surfaces[0]),
            "s_outer": float(surfaces[-1]),
            "nfp": nfp,
            "trace_time": float(trace_time),
            "nturns": int(nturns),
            "seed": int(seed),
            "integmode": INTEGMODE_MIDPOINT,
            "isw_field_type": ISW_FIELD_TYPE_BOOZER,
            "vmec_RZ_scale": float(rz_scale),
            "vmec_B_scale": float(b_scale),
            "classification_mode": "batched-surfaces",
            "classification_seconds": classification_seconds,
            "seconds_total": time.monotonic() - total_started,
            "simple_sha256": binary_hash,
            "wout_sha256": file_sha256(wout_path),
            "workdir": str(base) if keep_workdir else "",
        }
    finally:
        if not keep_workdir:
            shutil.rmtree(base, ignore_errors=True)


def _continuous_metrics(
    runs: Sequence[Path],
    particle_weights: np.ndarray,
    *,
    nodes: np.ndarray,
    radial_weights: np.ndarray,
    settings: dict | None,
    surface_slices: Sequence[slice] | None = None,
) -> dict:
    """Aggregate only raw J_parallel and rotation-drift score fields."""
    from smooth_barrier import (
        birth_weighted_integral,
        continuous_barrier_metric,
        load_class_scores,
        resolved_fraction,
        trapped_weight,
    )

    configured = dict(DEFAULT_CONTINUOUS_SETTINGS)
    if settings:
        unknown = set(settings) - set(configured)
        if unknown:
            raise ValueError(f"unknown continuous settings: {sorted(unknown)}")
        configured.update(settings)
    if any(not np.isfinite(value) or value <= 0.0 for value in configured.values()):
        raise ValueError("continuous settings must be finite and positive")

    if surface_slices is None:
        score_runs = [load_class_scores(run) for run in runs]
        mu_runs = [load_perp_invariant(run) for run in runs]
    else:
        if len(runs) != 1:
            raise ValueError("batched surface slices require exactly one SIMPLE run")
        all_scores = load_class_scores(runs[0])
        all_mu = load_perp_invariant(runs[0])
        score_runs = [all_scores.subset(rows) for rows in surface_slices]
        mu_runs = [all_mu[rows] for rows in surface_slices]
    if any(len(scores) != particle_weights.size for scores in score_runs):
        raise ValueError("particle quadrature and classifier output sizes differ")
    spacing = float(nodes[1] - nodes[0])
    mu_width = configured["mu_width_factor"] * spacing
    result = {}
    for classifier in ("jpar", "rotation"):
        metric = continuous_barrier_metric(
            score_runs,
            mu_runs,
            nodes,
            classifier=classifier,
            temperature=configured[f"temperature_{classifier}"],
            trapped_width=configured["trapped_width"],
            mu_width=mu_width,
            sample_weights_by_surface=[particle_weights] * len(score_runs),
            surface_weights=radial_weights,
        )
        birth_mean = birth_weighted_integral(
            nodes,
            metric.surface_fields[0].values,
            metric.birth_density,
        )
        result[classifier] = {
            "birth_mean": _finite_or_none(birth_mean),
            "barrier_defect": _finite_or_none(metric.value),
            "resolved_coverage": _finite_or_none(metric.resolved_coverage),
            "resolved_fraction_by_surface": [
                resolved_fraction(scores, classifier=classifier)
                for scores in score_runs
            ],
            "barrier_field": _array_for_json(metric.barrier_field),
            "birth_density": _array_for_json(metric.birth_density),
            "surface_fields": [
                {
                    "values": _array_for_json(field.values),
                    "resolved_coverage": _array_for_json(field.resolved_coverage),
                }
                for field in metric.surface_fields
            ],
        }
    soft_trapped = [
        float(
            np.sum(
                particle_weights
                * trapped_weight(scores.trap_par, width=configured["trapped_width"])
            )
        )
        for scores in score_runs
    ]
    return {
        "settings": {**configured, "mu_width": mu_width},
        "jpar": result["jpar"],
        "rotation": result["rotation"],
        "trapped_count_by_surface": [
            int(np.sum(scores.trap_par > 0.0)) for scores in score_runs
        ],
        "soft_trapped_mass_by_surface": soft_trapped,
    }


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _array_for_json(values: np.ndarray) -> list[float | None]:
    return [_finite_or_none(value) for value in np.asarray(values, dtype=float)]


def _fortran_d(value: float) -> str:
    return f"{float(value):.16g}".replace("e", "d").replace("E", "d")
