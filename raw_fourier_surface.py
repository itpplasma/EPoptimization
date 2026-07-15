"""Non-data-informed low-order Fourier coordinates for optimizer comparisons."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from data_informed_surface import _enforce_radii
from generate_vmec_audit_inputs import file_sha256, normalize_vmec_input

_MODE = re.compile(r"(?:.*:)?(rc|zs)\((\d+),(-?\d+)\)$")


def raw_coordinate_contract(base_input: Path, max_mode: int = 2) -> dict:
    from simsopt.mhd import Vmec

    vmec = Vmec(str(Path(base_input).resolve()), verbose=False)
    surface = vmec.boundary
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
    from simsopt.mhd import Vmec

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
    vmec = Vmec(str(Path(base_input).resolve()), verbose=False)
    surface = vmec.boundary
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
    vmec.write_input(str(input_path))
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
