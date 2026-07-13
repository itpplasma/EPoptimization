#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import qmc

from data_informed_surface import (
    fit_weighted_surface_transform,
    normalized_surface_grid,
    surface_from_unit,
)

WEIGHTEDPCA_COMMIT = "6c7b83dd6b245f14dbdbb8abeff70f62f75ec8d8"


def fit(args: argparse.Namespace) -> None:
    from simsopt.mhd import Vmec

    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    base = Vmec(str(args.base_input.resolve()), verbose=False).boundary
    source_inventory = _source_inventory(args.source)
    charts = []
    for dimension in args.dimensions:
        transform = fit_weighted_surface_transform(args.source, dimension)
        anchor = normalized_surface_grid(base, transform)
        sidecar = output / f"weighted-pca-d{dimension}.npz"
        transform.save(sidecar)
        audit = _audit(transform, anchor, base, args, dimension)
        charts.append(
            {
                "dimension": dimension,
                "explained_variance_fraction": float(
                    np.sum(transform.explained_variance)
                ),
                "sidecar": sidecar.name,
                "sidecar_sha256": _file_sha256(sidecar),
                **audit,
            }
        )
    manifest = {
        "schema_name": "alpha-loss.data-informed-charts",
        "schema_version": 1,
        "method": "Landreman weighted real-space PCA and inverse marginal quantiles",
        "adaptation": "quantile displacements anchored at the ALPES reference",
        "weightedpca_commit": WEIGHTEDPCA_COMMIT,
        "source": source_inventory,
        "base_input_sha256": _file_sha256(args.base_input),
        "nfp": base.nfp,
        "stellarator_symmetry": base.stellsym,
        "major_radius": base.major_radius(),
        "minor_radius": base.minor_radius(),
        "mpol": args.mpol,
        "ntor": args.ntor,
        "quantile_margin": args.margin,
        "sobol_seed": args.seed,
        "sobol_points": 2**args.audit_power,
        "charts": charts,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _audit(transform, anchor, base, args, dimension) -> dict[str, float | int]:
    center = surface_from_unit(
        transform,
        np.full(dimension, 0.5),
        nfp=base.nfp,
        major_radius=base.major_radius(),
        minor_radius=base.minor_radius(),
        mpol=args.mpol,
        ntor=args.ntor,
        margin=args.margin,
        anchor=anchor,
    )
    error = _surface_error(center, base) / base.minor_radius()
    points = qmc.Sobol(dimension, scramble=True, seed=args.seed).random_base2(
        args.audit_power
    )
    usable = sum(_usable(transform, anchor, base, point, args) for point in points)
    return {
        "reference_max_point_error_over_a": error,
        "analytic_usable": usable,
        "analytic_usable_fraction": usable / len(points),
    }


def _usable(transform, anchor, base, point, args) -> int:
    try:
        surface = surface_from_unit(
            transform,
            point,
            nfp=base.nfp,
            major_radius=base.major_radius(),
            minor_radius=base.minor_radius(),
            mpol=args.mpol,
            ntor=args.ntor,
            margin=args.margin,
            anchor=anchor,
        )
        return int(not surface.is_self_intersecting())
    except (RuntimeError, ValueError):
        return 0


def _surface_error(first, second) -> float:
    from simsopt.geo import SurfaceRZFourier

    grid = SurfaceRZFourier.from_nphi_ntheta(
        nfp=first.nfp,
        mpol=max(first.mpol, second.mpol),
        ntor=max(first.ntor, second.ntor),
        range="half period",
        nphi=64,
        ntheta=128,
    )
    values = []
    for surface in (first, second):
        sampled = SurfaceRZFourier(
            nfp=surface.nfp,
            stellsym=surface.stellsym,
            mpol=surface.mpol,
            ntor=surface.ntor,
            quadpoints_phi=grid.quadpoints_phi,
            quadpoints_theta=grid.quadpoints_theta,
            dofs=surface.dofs,
        )
        values.append(sampled.gamma())
    return float(np.max(np.linalg.norm(values[0] - values[1], axis=2)))


def _source_inventory(path: Path) -> dict[str, object]:
    with h5py.File(path, "r") as archive:
        return {
            "file": path.name,
            "sha256": _file_sha256(path),
            "rows": int(archive["data"].shape[0]),
            "real_space_dimension": int(archive["data"].shape[1]),
            "n_kappel": int(archive["n_Kappel"][()]),
            "n_quasr": int(archive["n_QUASR"][()]),
            "n_constellaration": int(archive["n_constellaration"][()]),
            "weight_sum": float(np.sum(archive["weights"][()])),
        }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--source", type=Path, required=True)
    root.add_argument("--base-input", type=Path, required=True)
    root.add_argument("--out", type=Path, required=True)
    root.add_argument("--dimensions", type=int, nargs="+", default=(8, 12, 16))
    root.add_argument("--margin", type=float, default=0.1)
    root.add_argument("--mpol", type=int, default=9)
    root.add_argument("--ntor", type=int, default=6)
    root.add_argument("--audit-power", type=int, default=8)
    root.add_argument("--seed", type=int, default=20260713)
    return root


def main() -> None:
    fit(parser().parse_args())


if __name__ == "__main__":
    main()
