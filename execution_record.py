from __future__ import annotations

import os
import socket
from datetime import datetime, timezone
from pathlib import Path

from simple_direct import file_sha256


def write_execution_record(
    run: Path,
    executable: Path,
    executable_hash: str,
    wout_hash: str,
    seed: int,
) -> None:
    input_hash = file_sha256(run / "simple.in")
    (run / "output_checksums.txt").write_text(
        f"{executable_hash}  {executable}\n"
        f"{wout_hash}  wout.nc\n"
        f"{input_hash}  simple.in\n"
    )
    (run / "execution.txt").write_text(
        f"simple_bin={executable}\n"
        f"host={socket.gethostname()}\n"
        f"slurm_job={os.environ.get('SLURM_JOB_ID', '')}\n"
        f"seed={seed}\n"
        f"completed={datetime.now(timezone.utc).astimezone().isoformat()}\n"
    )
