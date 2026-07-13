#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data_informed_surface import (
    load_weighted_surface_transform,
    normalized_surface_grid,
    surface_from_unit,
)
from generate_vmec_audit_inputs import file_sha256, normalize_vmec_input


def generate(args: argparse.Namespace) -> None:
    from simsopt.mhd import Vmec

    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    transform = load_weighted_surface_transform(args.transform.resolve())
    requests = load_requests(args.requests, transform.dimension)
    vmec = Vmec(str(args.base_input.resolve()), verbose=False)
    base = vmec.boundary
    anchor = normalized_surface_grid(base, transform)
    major_radius = base.major_radius()
    minor_radius = base.minor_radius()
    fixed = (base.nfp, major_radius, minor_radius)
    cases = [
        generate_case(vmec, transform, anchor, request, args, output, fixed)
        for request in requests
    ]
    manifest = {
        "schema_name": "alpha-loss.scbo-candidate-wave",
        "schema_version": 1,
        "base_input_sha256": file_sha256(args.base_input),
        "transform_sha256": file_sha256(args.transform),
        "quantile_margin": args.margin,
        "mpol": args.mpol,
        "ntor": args.ntor,
        "cases": cases,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def generate_case(vmec, transform, anchor, request, args, output, fixed):
    candidate_id = request["candidate_id"]
    name = f"candidate-{candidate_id:08d}"
    unit = np.asarray(request["unit_x"], dtype=float)
    nfp, major_radius, minor_radius = fixed
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
    case = output / name
    case.mkdir()
    record = {"candidate_id": candidate_id, "unit_x": unit.tolist(), "case": name}
    if surface.is_self_intersecting():
        record.update({"status": "failed", "failure_kind": "self_intersection"})
        return record
    vmec.boundary.x = surface.x
    input_path = case / f"input.{name}"
    vmec.write_input(str(input_path))
    normalize_vmec_input(input_path)
    record.update(
        {
            "status": "ready",
            "input": str(input_path.relative_to(output)),
            "input_sha256": file_sha256(input_path),
        }
    )
    return record


def load_requests(path: Path, dimension: int) -> list[dict]:
    document = json.loads(Path(path).read_text())
    rows = document["requests"] if isinstance(document, dict) else document
    ids = [int(row["candidate_id"]) for row in rows]
    if len(set(ids)) != len(ids) or ids != sorted(ids):
        raise ValueError("SCBO request candidate IDs must be unique and ordered")
    requests = []
    for candidate_id, row in zip(ids, rows, strict=True):
        unit = np.asarray(row["unit_x"], dtype=float)
        if (
            unit.shape != (dimension,)
            or not np.isfinite(unit).all()
            or np.any((unit < 0.0) | (unit > 1.0))
        ):
            raise ValueError("SCBO request coordinates differ from the chart")
        requests.append({"candidate_id": candidate_id, "unit_x": unit.tolist()})
    return requests


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--base-input", type=Path, required=True)
    root.add_argument("--transform", type=Path, required=True)
    root.add_argument("--requests", type=Path, required=True)
    root.add_argument("--out", type=Path, required=True)
    root.add_argument("--margin", type=float, default=0.35)
    root.add_argument("--mpol", type=int, default=9)
    root.add_argument("--ntor", type=int, default=6)
    return root


def main() -> None:
    generate(parser().parse_args())


if __name__ == "__main__":
    main()
