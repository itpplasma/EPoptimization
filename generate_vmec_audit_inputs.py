#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from data_informed_surface import (
    load_weighted_surface_transform,
    normalized_surface_grid,
    surface_from_unit,
)


def generate(args: argparse.Namespace) -> None:
    from simsopt.mhd import Vmec

    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    base_input = args.base_input.resolve()
    vmec = Vmec(str(base_input), verbose=False)
    base = vmec.boundary
    base_dofs = base.x.copy()
    nfp = base.nfp
    major_radius = base.major_radius()
    minor_radius = base.minor_radius()
    cases = []
    for sidecar in args.transforms:
        transform = load_weighted_surface_transform(sidecar.resolve())
        base.x = base_dofs
        anchor = normalized_surface_grid(base, transform)
        for name, dimension, index, unit in case_specs(
            transform.dimension, args.audit_power, args.seed
        ):
            surface = surface_from_unit(
                transform,
                unit,
                nfp=nfp,
                major_radius=major_radius,
                minor_radius=minor_radius,
                mpol=args.mpol,
                ntor=args.ntor,
                margin=args.margin,
                anchor=anchor,
            )
            if surface.is_self_intersecting():
                raise RuntimeError(f"analytic audit changed for {name}")
            vmec.boundary.x = surface.x
            case = output / relative_case_path(dimension, index, name)
            case.mkdir(parents=True)
            input_path = case / f"input.{name}"
            vmec.write_input(str(input_path))
            cases.append(
                {
                    "case": name,
                    "dimension": dimension,
                    "index": index,
                    "unit": unit.tolist(),
                    "input": str(input_path.relative_to(output)),
                    "input_sha256": file_sha256(input_path),
                    "transform": sidecar.name,
                    "transform_sha256": file_sha256(sidecar),
                }
            )
    manifest = {
        "schema_name": "alpha-loss.vmec-chart-audit",
        "schema_version": 1,
        "base_input_sha256": file_sha256(base_input),
        "sobol_seed": args.seed,
        "sobol_points_per_chart": 2**args.audit_power,
        "quantile_margin": args.margin,
        "mpol": args.mpol,
        "ntor": args.ntor,
        "cases": cases,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def case_specs(
    dimension: int, audit_power: int, seed: int
) -> list[tuple[str, int, int, np.ndarray]]:
    if dimension < 1 or audit_power < 0:
        raise ValueError("dimension and audit power are outside their ranges")
    points = qmc.Sobol(dimension, scramble=True, seed=seed).random_base2(audit_power)
    return [
        (f"d{dimension:02d}_{index:03d}", dimension, index, point)
        for index, point in enumerate(points)
    ]


def relative_case_path(dimension: int, index: int, name: str) -> Path:
    return Path(f"d{dimension:02d}") / f"b{index // 32:02d}" / name


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--base-input", type=Path, required=True)
    root.add_argument("--transforms", type=Path, nargs="+", required=True)
    root.add_argument("--out", type=Path, required=True)
    root.add_argument("--margin", type=float, default=0.1)
    root.add_argument("--mpol", type=int, default=9)
    root.add_argument("--ntor", type=int, default=6)
    root.add_argument("--audit-power", type=int, default=8)
    root.add_argument("--seed", type=int, default=20260713)
    return root


def main() -> None:
    generate(parser().parse_args())


if __name__ == "__main__":
    main()
