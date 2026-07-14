import json
from pathlib import Path

from build_spatial_manifest import build_manifest
from spatial_grid import file_sha256


def test_build_manifest_includes_wout_and_surface_hashes(tmp_path: Path) -> None:
    design = tmp_path / "alpes" / "design"
    surface = design / "s0p30000"
    surface.mkdir(parents=True)
    wout = tmp_path / "alpes" / "wout.nc"
    wout.write_bytes(b"equilibrium")
    (surface / "design.npz").write_bytes(b"design")
    (surface / "start.dat").write_bytes(b"starts")
    manifest = {
        "wout_sha256": file_sha256(wout),
        "records": [
            {
                "name": "s0p30000",
                "design_sha256": file_sha256(surface / "design.npz"),
                "start_sha256": file_sha256(surface / "start.dat"),
            }
        ],
    }
    (design / "manifest.json").write_text(json.dumps(manifest))
    fields = build_manifest(tmp_path).strip().split("\t")
    assert fields[0] == "alpes-s0p30000"
    assert fields[4] == "alpes/wout.nc"
    assert fields[6] == "alpes/surfaces/s0p30000"


def test_build_manifest_includes_separate_refinement_designs(tmp_path: Path) -> None:
    case = tmp_path / "candidate074"
    wout = case / "wout.nc"
    wout.parent.mkdir()
    wout.write_bytes(b"equilibrium")
    for directory, name in (("design", "s0p25000"), ("design_s0p67500", "s0p67500")):
        design = case / directory
        surface = design / name
        surface.mkdir(parents=True)
        (surface / "design.npz").write_bytes(name.encode())
        (surface / "start.dat").write_bytes(b"starts")
        manifest = {
            "wout_sha256": file_sha256(wout),
            "records": [
                {
                    "name": name,
                    "design_sha256": file_sha256(surface / "design.npz"),
                    "start_sha256": file_sha256(surface / "start.dat"),
                }
            ],
        }
        (design / "manifest.json").write_text(json.dumps(manifest))
    lines = build_manifest(tmp_path).splitlines()
    assert len(lines) == 2
    assert lines[1].split("\t")[1] == "candidate074/design_s0p67500/s0p67500"
