from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

import simple_barrier


B_TARGET = 5.865
LAMBDA_TOLERANCE = 2.0e-6


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pitch_quantile_lambdas(count: int) -> np.ndarray:
    if count <= 0:
        raise ValueError("mu count must be positive")
    absolute_pitch = (np.arange(count, dtype=float) + 0.5) / count
    return np.sort(1.0 - absolute_pitch**2)


def angular_grid(
    ntheta: int, nzeta: int, nfp: int, shift: float
) -> tuple[np.ndarray, np.ndarray]:
    if ntheta <= 0 or nzeta <= 0 or nfp <= 0:
        raise ValueError("angular counts and nfp must be positive")
    theta = 2.0 * np.pi * (np.arange(ntheta) + shift) / ntheta
    zeta = 2.0 * np.pi * (np.arange(nzeta) + shift) / (nfp * nzeta)
    return np.meshgrid(theta, zeta, indexing="ij")


def fourier_field(
    cosine: np.ndarray,
    sine: np.ndarray,
    xm: np.ndarray,
    xn: np.ndarray,
    theta: np.ndarray,
    zeta: np.ndarray,
) -> np.ndarray:
    phase = xm[:, None, None] * theta - xn[:, None, None] * zeta
    value = np.zeros_like(theta, dtype=float)
    if cosine.size:
        value += np.sum(cosine[:, None, None] * np.cos(phase), axis=0)
    if sine.size:
        value += np.sum(sine[:, None, None] * np.sin(phase), axis=0)
    return value


def _surface_xyz(bx, surface: int, theta: np.ndarray, zeta: np.ndarray) -> np.ndarray:
    xm = np.asarray(bx.xm_b)
    xn = np.asarray(bx.xn_b)
    empty = np.empty(0)
    r = fourier_field(bx.rmnc_b[:, surface], empty, xm, xn, theta, zeta)
    z = fourier_field(empty, bx.zmns_b[:, surface], xm, xn, theta, zeta)
    nu = fourier_field(empty, bx.numns_b[:, surface], xm, xn, theta, zeta)
    phi = zeta - nu
    return np.stack((r * np.cos(phi), r * np.sin(phi), z), axis=0)


def surface_weights(xyz: np.ndarray) -> np.ndarray:
    dtheta = np.roll(xyz, -1, axis=1) - np.roll(xyz, 1, axis=1)
    dzeta = np.roll(xyz, -1, axis=2) - np.roll(xyz, 1, axis=2)
    jacobian = np.linalg.norm(
        np.cross(dtheta, dzeta, axisa=0, axisb=0, axisc=0), axis=0
    )
    if not np.all(np.isfinite(jacobian)) or np.sum(jacobian) <= 0.0:
        raise ValueError("invalid surface Jacobian")
    return jacobian / np.sum(jacobian)


def _boozer(wout: Path, surfaces: np.ndarray, mpol: int, ntor: int):
    from simsopt.mhd import Boozer, Vmec

    transform = Boozer(Vmec(str(wout), verbose=False), mpol=mpol, ntor=ntor)
    transform.register(surfaces)
    transform.run()
    if transform.bx.bmnc_b.shape[1] != len(surfaces):
        raise ValueError("Boozer transform did not return every requested surface")
    return transform.bx


def _to_reference(wout: Path, points: np.ndarray) -> np.ndarray:
    import pysimple

    pysimple.init(
        wout,
        isw_field_type=2,
        ntestpart=1,
        num_surf=1,
        sbeg=np.array([float(points[0, 0])]),
    )
    reference = np.zeros_like(points, order="F")
    backend = pysimple._fortran_backend
    backend.params_wrapper.integ_traj_to_ref(np.asfortranarray(points), reference)
    return reference


def _design_surface(
    wout: Path,
    bx,
    surface_index: int,
    surface: float,
    ntheta: int,
    nzeta: int,
    lambdas: np.ndarray,
    shifts: np.ndarray,
    b_scale: float,
) -> dict[str, np.ndarray]:
    shape = (len(lambdas), 2, len(shifts), ntheta, nzeta)
    particle_index = np.full(shape, -1, dtype=int)
    starts = []
    b_fields = []
    weights = []
    signs = np.array([1.0, -1.0])
    for shift_index, shift in enumerate(shifts):
        theta, zeta = angular_grid(ntheta, nzeta, int(bx.nfp), float(shift))
        b_field = (
            fourier_field(
                bx.bmnc_b[:, surface_index],
                bx.bmns_b[:, surface_index] if bx.bmns_b.size else np.empty(0),
                np.asarray(bx.xm_b),
                np.asarray(bx.xn_b),
                theta,
                zeta,
            )
            * b_scale
        )
        b_fields.append(b_field)
        weights.append(surface_weights(_surface_xyz(bx, surface_index, theta, zeta)))
        for mu_index, lambda_value in enumerate(lambdas):
            for sign_index, sign in enumerate(signs):
                for angular_index in np.ndindex(theta.shape):
                    item = [
                        surface,
                        theta[angular_index],
                        zeta[angular_index],
                        1.0,
                        sign * 0.5,
                    ]
                    particle_index[
                        mu_index, sign_index, shift_index, *angular_index
                    ] = len(starts)
                    starts.append(item)
    points = np.asarray(starts, dtype=float).T
    reference = _to_reference(wout, points).T
    return {
        "b": np.asarray(b_fields),
        "lambda": lambdas,
        "particle_index": particle_index,
        "shifts": shifts,
        "signs": signs,
        "start": reference,
        "surface": np.array(surface),
        "weights": np.asarray(weights),
    }


def generate_design(
    wout: Path,
    out: Path,
    surfaces: np.ndarray,
    ntheta: int = 16,
    nzeta: int = 16,
    nmu: int = 9,
    shifts: np.ndarray = np.array([0.0, 0.5]),
    mpol: int = 16,
    ntor: int = 16,
) -> dict:
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    bx = _boozer(wout, surfaces, mpol, ntor)
    _, b_scale = simple_barrier.reactor_scale(wout)
    lambdas = pitch_quantile_lambdas(nmu)
    records = []
    for index, surface in enumerate(surfaces):
        name = f"s{surface:.5f}".replace(".", "p")
        directory = out / name
        directory.mkdir()
        design = _design_surface(
            wout, bx, index, float(surface), ntheta, nzeta, lambdas, shifts, b_scale
        )
        np.savez_compressed(directory / "design.npz", **design)
        np.savetxt(directory / "start.dat", design["start"], fmt="%.17e")
        records.append(
            {
                "design_sha256": file_sha256(directory / "design.npz"),
                "name": name,
                "particles": len(design["start"]),
                "start_sha256": file_sha256(directory / "start.dat"),
                "surface": float(surface),
            }
        )
    manifest = {
        "b_scale": b_scale,
        "coordinate_convention": "SIMPLE Boozer angles mapped to VMEC reference starts",
        "invariant_calibration": "same executable, same point, 10 microsecond trace",
        "lambda_definition": "mu * 5.865 T / E",
        "mpol": mpol,
        "nmu": nmu,
        "ntheta": ntheta,
        "ntor": ntor,
        "nzeta": nzeta,
        "records": records,
        "schema_name": "alpha-loss.spatial-topology-design",
        "schema_version": 1,
        "shifts": shifts.tolist(),
        "wout_sha256": file_sha256(wout),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def spatial_classify_namelist(
    *, particle_count: int, surface: float, wout: Path, trace_time: float
) -> str:
    trace = simple_barrier._fortran_d(trace_time)
    namelist = simple_barrier.CLASSIFY_NAMELIST.format(
        n=particle_count,
        notrace_passing=0,
        ttime=trace,
        sbeg=surface,
        face=simple_barrier._fortran_d(1.0),
        rz=simple_barrier._fortran_d(simple_barrier.reactor_scale(wout)[0]),
        b=simple_barrier._fortran_d(simple_barrier.reactor_scale(wout)[1]),
        seed=12345,
    )
    return namelist.replace("&config\n", "&config\n  startmode = 2\n", 1).replace(
        "class_plot = .True.", "class_plot = .False."
    )


def calibrate_fixed_invariant(
    design, calibration_classes: np.ndarray
) -> dict[str, np.ndarray]:
    particle_index = design["particle_index"]
    particle_count = len(design["start"])
    expected_indices = np.arange(1, particle_count + 1)
    if not np.array_equal(calibration_classes[:, 0].astype(int), expected_indices):
        raise ValueError("calibration particle indices are incomplete or unordered")
    pitch_squared = design["start"][:, 4] ** 2
    field_per_particle = (1.0 - pitch_squared) / calibration_classes[:, 2]
    field_grid = field_per_particle[particle_index]
    field = np.mean(field_grid, axis=(0, 1))
    if np.max(np.abs(field_grid - field[None, None])) > 1.0e-6:
        raise ValueError("calibration field differs across duplicate grid points")

    shape = particle_index.shape
    lambda_grid = np.broadcast_to(design["lambda"][:, None, None, None, None], shape)
    sign_grid = np.broadcast_to(design["signs"][None, :, None, None, None], shape)
    target_pitch_squared = 1.0 - lambda_grid * field_grid / (B_TARGET * 1.0e4)
    selected = target_pitch_squared >= 0.0
    calibration_rows = particle_index[selected]
    starts = design["start"][calibration_rows].copy()
    starts[:, 4] = sign_grid[selected] * np.sqrt(target_pitch_squared[selected])
    final_index = np.full(shape, -1, dtype=int)
    final_index[selected] = np.arange(len(starts))
    expected_lambda = lambda_grid[selected]
    return {
        "b": field / 1.0e4,
        "expected_lambda": expected_lambda,
        "particle_index": final_index,
        "start": starts,
    }


def _run_simple(
    directory: Path,
    wout: Path,
    starts: np.ndarray,
    namelist: str,
    simple_executable: Path,
    timeout_seconds: float,
) -> np.ndarray:
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(wout, directory / "wout.nc")
    np.savetxt(directory / "start.dat", starts, fmt="%.17e")
    (directory / "simple.in").write_text(namelist)
    completed = subprocess.run(
        [str(simple_executable)],
        cwd=directory,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_seconds,
        check=False,
    )
    (directory / "simple.stdout").write_text(completed.stdout)
    if completed.returncode != 0 or not (directory / "class_parts.dat").exists():
        raise RuntimeError(f"SIMPLE classification failed with {completed.returncode}")
    return np.loadtxt(directory / "class_parts.dat", ndmin=2)


def run_surface_classification(
    wout: Path,
    design_dir: Path,
    out: Path,
    simple_executable: Path,
    trace_time: float = 0.02,
    timeout_seconds: float = 3600.0,
) -> Path:
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    design = np.load(design_dir / "design.npz")
    calibration_count = len(design["start"])
    calibration_time = min(trace_time, 1.0e-5)
    calibration_namelist = spatial_classify_namelist(
        particle_count=calibration_count,
        surface=float(design["surface"]),
        wout=wout,
        trace_time=calibration_time,
    )
    calibration_classes = _run_simple(
        out / "invariant_calibration",
        wout,
        design["start"],
        calibration_namelist,
        simple_executable,
        timeout_seconds,
    )
    calibrated = calibrate_fixed_invariant(design, calibration_classes)
    particle_count = len(calibrated["start"])
    namelist = spatial_classify_namelist(
        particle_count=particle_count,
        surface=float(design["surface"]),
        wout=wout,
        trace_time=trace_time,
    )
    classes = _run_simple(
        out,
        wout,
        calibrated["start"],
        namelist,
        simple_executable,
        timeout_seconds,
    )
    times = np.loadtxt(out / "times_lost.dat", ndmin=2)
    expected_indices = np.arange(1, particle_count + 1)
    if not np.array_equal(classes[:, 0].astype(int), expected_indices):
        raise ValueError("class_parts.dat particle indices are incomplete or unordered")
    if not np.array_equal(times[:, 0].astype(int), expected_indices):
        raise ValueError("times_lost.dat particle indices are incomplete or unordered")
    observed_lambda = classes[:, 2] * B_TARGET * 1.0e4
    lambda_error = float(
        np.max(np.abs(observed_lambda - calibrated["expected_lambda"]))
    )
    if lambda_error > LAMBDA_TOLERANCE:
        raise ValueError(
            f"fixed mu B0 / E changed by {lambda_error:.6g}; "
            f"tolerance is {LAMBDA_TOLERANCE:.6g}"
        )
    topology = np.zeros(calibrated["particle_index"].shape, dtype=np.int8)
    jpar = np.zeros_like(topology)
    passing = np.zeros_like(topology, dtype=bool)
    lost = np.zeros_like(topology, dtype=bool)
    trap_parameter = np.full_like(topology, np.nan, dtype=float)
    selected = calibrated["particle_index"] >= 0
    indices = calibrated["particle_index"][selected]
    topology[selected] = classes[indices, 4].astype(np.int8)
    jpar[selected] = classes[indices, 3].astype(np.int8)
    passing[selected] = times[indices, 2] < 0.0
    lost[selected] = times[indices, 1] < trace_time * (1.0 - 1.0e-12)
    trap_parameter[selected] = times[indices, 2]
    np.savez_compressed(
        out / "topology.npz",
        topology=topology,
        jpar=jpar,
        b=calibrated["b"],
        lambda_values=design["lambda"],
        lambda_error_max=np.array(lambda_error),
        lost=lost,
        particle_index=calibrated["particle_index"],
        passing=passing,
        signs=design["signs"],
        simple_sha256=np.array(file_sha256(simple_executable)),
        shifts=design["shifts"],
        surface=design["surface"],
        trap_parameter=trap_parameter,
        trace_time=np.array(trace_time),
        weights=design["weights"],
        wout_sha256=np.array(file_sha256(wout)),
    )
    return out / "topology.npz"
