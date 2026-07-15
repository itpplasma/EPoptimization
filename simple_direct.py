"""Pinned direct SIMPLE loss evaluation for the optimizer."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

A_TARGET = 1.704
B_TARGET = 5.865

DIRECT_NAMELIST = """&config
  netcdffile = 'wout.nc'
  startmode = 1
  num_surf = 1
  notrace_passing = 0
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


def loss_windows(
    lost_times: np.ndarray,
    *,
    prompt_time: float,
    final_time: float,
) -> dict[str, float | int]:
    times = np.asarray(lost_times, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("lost_times must be a nonempty vector")
    if not 0.0 < prompt_time < final_time:
        raise ValueError("loss windows require 0 < prompt < final")
    prompt = (times > 0.0) & (times <= prompt_time)
    late = (times > prompt_time) & (times < final_time)
    total = prompt | late
    particles = times.size
    return {
        "particles": int(particles),
        "prompt_count": int(prompt.sum()),
        "late_count": int(late.sum()),
        "total_count": int(total.sum()),
        "prompt_loss": float(prompt.mean()),
        "late_loss": float(late.mean()),
        "total_loss": float(total.mean()),
    }


def direct_loss_metrics(
    wout_path: str | os.PathLike,
    *,
    ntestpart: int,
    expected_simple_sha256: str,
    trace_time: float,
    prompt_time: float,
    sbeg: float,
    seed: int,
    simple_executable: str | os.PathLike | None = None,
    keep_workdir: bool = False,
    timeout_s: float = 900.0,
) -> dict[str, float | int | str]:
    if ntestpart <= 0 or not 0.0 < sbeg < 1.0:
        raise ValueError("particle count and birth surface are invalid")
    if trace_time <= prompt_time:
        raise ValueError("trace time must exceed prompt time")
    simple_x = find_simple_x(simple_executable)
    binary_hash = file_sha256(simple_x)
    if binary_hash != expected_simple_sha256:
        raise ValueError("unexpected SIMPLE executable hash")
    rz_scale, b_scale = reactor_scale(wout_path)
    workdir = Path(tempfile.mkdtemp(prefix="direct_loss_"))
    try:
        shutil.copyfile(wout_path, workdir / "wout.nc")
        (workdir / "simple.in").write_text(
            DIRECT_NAMELIST.format(
                n=int(ntestpart),
                ttime=_fortran_d(trace_time),
                sbeg=_fortran_d(sbeg),
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
        particle_path = workdir / "times_lost.dat"
        curve_path = workdir / "confined_fraction.dat"
        particles = np.loadtxt(particle_path, ndmin=2)
        curve = np.loadtxt(curve_path, ndmin=2)
        if particles.shape[0] != ntestpart or particles.shape[1] < 2:
            raise ValueError(f"unexpected times_lost shape {particles.shape}")
        if curve.shape[0] == 0 or curve.shape[1] < 3:
            raise ValueError(f"unexpected confined_fraction shape {curve.shape}")
        expected_ids = np.arange(1, ntestpart + 1)
        if not np.array_equal(particles[:, 0].astype(int), expected_ids):
            raise ValueError("particle indices are not sequential")
        trace_endpoint = float(curve[-1, 0])
        if not np.isclose(trace_endpoint, trace_time, rtol=0.0, atol=1.0e-12):
            raise ValueError("SIMPLE trace endpoint differs from the contract")
        metrics = loss_windows(
            particles[:, 1], prompt_time=prompt_time, final_time=trace_endpoint
        )
        curve_loss_count = int(
            round((1.0 - float(curve[-1, 1]) - float(curve[-1, 2])) * ntestpart)
        )
        if metrics["total_count"] != curve_loss_count:
            raise ValueError("particle losses disagree with the confinement curve")
        metrics.update(
            {
                "vmec_RZ_scale": float(rz_scale),
                "vmec_B_scale": float(b_scale),
                "facE_al": 1.0,
                "simple_sha256": binary_hash,
                "wout_sha256": file_sha256(wout_path),
                "seed": int(seed),
                "trace_endpoint": trace_endpoint,
                "workdir": str(workdir) if keep_workdir else "",
            }
        )
        return metrics
    finally:
        if not keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)


def _fortran_d(value: float) -> str:
    return f"{float(value):.16g}".replace("e", "d").replace("E", "d")
