#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from spatial_grid import file_sha256


def build_manifest(atlas_root: Path) -> str:
    lines = []
    for manifest_path in sorted(atlas_root.glob("*/design/manifest.json")):
        configuration = manifest_path.parent.parent.name
        manifest = json.loads(manifest_path.read_text())
        wout = manifest_path.parent.parent / "wout.nc"
        if file_sha256(wout) != manifest["wout_sha256"]:
            raise ValueError(f"wout hash mismatch for {configuration}")
        for record in manifest["records"]:
            case = f"{configuration}-{record['name']}"
            values = (
                case,
                f"{configuration}/design/{record['name']}",
                record["design_sha256"],
                record["start_sha256"],
                f"{configuration}/wout.nc",
                manifest["wout_sha256"],
                f"{configuration}/surfaces/{record['name']}",
            )
            lines.append("\t".join(values))
    if not lines:
        raise ValueError("no spatial design manifests found")
    return "\n".join(lines) + "\n"


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--atlas-root", type=Path, required=True)
    root.add_argument("--out", type=Path, required=True)
    return root


if __name__ == "__main__":
    args = parser().parse_args()
    args.out.write_text(build_manifest(args.atlas_root.resolve()))
