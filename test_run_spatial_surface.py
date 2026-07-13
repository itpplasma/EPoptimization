import os
import subprocess
from pathlib import Path


def test_worker_accepts_existing_topology_without_rerun(tmp_path: Path) -> None:
    output = tmp_path / "configuration" / "surfaces" / "s0p80000"
    output.mkdir(parents=True)
    (output / "topology.npz").write_bytes(b"complete")
    (tmp_path / "manifest.tsv").write_text(
        "case\tdesign\tdesign-sha\tstart-sha\twout.nc\twout-sha\t"
        "configuration/surfaces/s0p80000\n"
    )
    environment = {**os.environ, "ATLAS_ROOT": str(tmp_path)}

    completed = subprocess.run(
        ["bash", str(Path(__file__).parent / "run_spatial_surface.sh"), "0"],
        env=environment,
        check=False,
    )

    assert completed.returncode == 0
