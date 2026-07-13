import hashlib
import os
from pathlib import Path
import subprocess


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_direct_worker_rejects_missing_code_dependency(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    code = tmp_path / "code"
    fake_bin = tmp_path / "bin"
    campaign.mkdir()
    code.mkdir()
    fake_bin.mkdir()
    (campaign / "manifest.tsv").write_text("reference\t12345\n")
    wout = tmp_path / "wout.nc"
    simple = tmp_path / "simple.x"
    evaluator = code / "evaluate_threshold_loss.py"
    wout.write_text("wout")
    simple.write_text("simple")
    evaluator.write_text("evaluator")
    marker = tmp_path / "python-called"
    python = fake_bin / "python3"
    python.write_text('#!/bin/sh\ntouch "$PYTHON_CALLED"\n')
    python.chmod(0o755)
    environment = {
        **os.environ,
        "ALLOCATED_CPUS": "1",
        "BARRIER_EVALUATOR_SHA256": "missing",
        "CODE_ROOT": str(code),
        "EVALUATOR_SHA256": _sha256(evaluator),
        "LOSS_THRESHOLD": "0.38",
        "PARTICLES": "1",
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "PYTHON_CALLED": str(marker),
        "REFERENCE_ROOT": str(campaign),
        "SIMPLE_BARRIER_SHA256": "missing",
        "SIMPLE_SHA256": _sha256(simple),
        "SIMPLE_X": str(simple),
        "THRESHOLD_OBJECTIVE_SHA256": "missing",
        "TRACE_TIME": "0.3",
        "WOUT": str(wout),
        "WOUT_SHA256": _sha256(wout),
    }
    result = subprocess.run(
        ["bash", str(Path(__file__).parent / "run_direct_reference.sh"), "0"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert not marker.exists()
