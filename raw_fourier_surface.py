"""Non-data-informed low-order Fourier coordinates for optimizer comparisons."""

from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path

import numpy as np
from scipy import optimize

_MODE = re.compile(r"(?:.*:)?(rc|zs)\((\d+),(-?\d+)\)$")


def load_boundary(base_input: Path):
    """Read the boundary straight from the VMEC input namelist.

    ``simsopt.mhd.Vmec`` would also serve, but it requires the compiled VMEC2000
    Python extension and mpi4py. Nothing here runs VMEC — equilibria are solved
    by the standalone ``xvmec`` binary on the worker — so the pure-Python
    surface reader is enough and keeps the driver free of that toolchain.
    """
    from simsopt.geo import SurfaceRZFourier

    return SurfaceRZFourier.from_vmec_input(str(Path(base_input).resolve()))


def write_vmec_input(base_input: Path, surface, path: Path) -> None:
    """Write a VMEC input carrying ``surface`` and the base file's settings.

    Only the boundary coefficient block is replaced. Resolution, pressure and
    current profiles, and solver controls are inherited verbatim from the base
    input, which ``SurfaceRZFourier.get_nml`` alone would drop.
    """
    boundary = [
        line
        for line in surface.get_nml().splitlines()
        if line.strip().startswith(("RBC(", "ZBS("))
    ]
    if not boundary:
        raise ValueError("surface namelist carries no boundary coefficients")
    out: list[str] = []
    inserted = False
    for line in Path(base_input).read_text().splitlines():
        if line.strip().startswith(("RBC(", "ZBS(")):
            if not inserted:
                out.extend(boundary)
                inserted = True
            continue
        out.append(line.rstrip())
    if not inserted:
        raise ValueError(f"{base_input} carries no boundary block to replace")
    Path(path).write_text("\n".join(out).rstrip() + "\n")


def raw_coordinate_contract(base_input: Path, max_mode: int = 2) -> dict:
    surface = load_boundary(base_input)
    surface.fix_all()
    surface.fixed_range(
        mmin=0,
        mmax=max_mode,
        nmin=-max_mode,
        nmax=max_mode,
        fixed=False,
    )
    surface.fix("rc(0,0)")
    names = [name.split(":", 1)[-1] for name in surface.dof_names]
    center = np.asarray(surface.x, dtype=float)
    scales = np.asarray([_mode_scale(name) for name in names])
    return {
        "schema_name": "alpha-loss.raw-fourier-contract",
        "schema_version": 1,
        "base_input_sha256": file_sha256(base_input),
        "max_mode": max_mode,
        "major_radius": float(surface.major_radius()),
        "minor_radius": float(surface.minor_radius()),
        "names": names,
        "center": center.tolist(),
        "half_width": scales.tolist(),
    }


def write_raw_candidate(
    base_input: Path,
    contract: dict,
    unit_x: np.ndarray,
    candidate_id: int,
    output: Path,
) -> dict:
    unit = np.asarray(unit_x, dtype=float)
    dimension = len(contract["names"])
    if (
        unit.shape != (dimension,)
        or not np.isfinite(unit).all()
        or np.any((unit < 0.0) | (unit > 1.0))
    ):
        raise ValueError("raw Fourier coordinates must lie in the unit box")
    if file_sha256(base_input) != contract["base_input_sha256"]:
        raise ValueError("raw Fourier base input hash differs")
    surface = load_boundary(base_input)
    values = np.asarray(contract["center"]) + (2.0 * unit - 1.0) * np.asarray(
        contract["half_width"]
    )
    surface.unfix_all()
    for name, value in zip(contract["names"], values, strict=True):
        surface.set(name, float(value))
    _enforce_radii(
        surface,
        float(contract["major_radius"]),
        float(contract["minor_radius"]),
    )
    name = f"candidate-{candidate_id:08d}"
    target = Path(output)
    target.mkdir(parents=True, exist_ok=False)
    request = {
        "candidate_id": candidate_id,
        "unit_x": unit.tolist(),
        "case": name,
    }
    (target / "request.json").write_text(
        json.dumps(request, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    if surface.is_self_intersecting():
        request.update({"status": "failed", "failure_kind": "self_intersection"})
        return request
    input_path = target / f"input.{name}"
    write_vmec_input(base_input, surface, input_path)
    normalize_vmec_input(input_path)
    request.update(
        {
            "status": "ready",
            "input": input_path.name,
            "input_sha256": file_sha256(input_path),
        }
    )
    return request


def _mode_scale(name: str) -> float:
    match = _MODE.fullmatch(name)
    if match is None:
        raise ValueError(f"unexpected Fourier degree of freedom {name}")
    m = int(match.group(2))
    n = int(match.group(3))
    return 0.05 / (1.0 + m + abs(n))


def _enforce_radii(surface, major_radius: float, minor_radius: float) -> None:
    target_aspect = major_radius / minor_radius

    def residual(value):
        dofs = surface.x.copy()
        dofs[0] = value
        surface.x = dofs
        return surface.aspect_ratio() - target_aspect

    root = optimize.newton(residual, x0=major_radius)
    residual(root)
    surface.x = surface.x * (minor_radius / surface.minor_radius())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_vmec_input(path: Path) -> None:
    lines = [line.rstrip() for line in Path(path).read_text().splitlines()]
    Path(path).write_text("\n".join(lines).rstrip() + "\n")
