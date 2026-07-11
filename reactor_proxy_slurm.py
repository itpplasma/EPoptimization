#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import simple_barrier
from reactor_proxy_calibration import write_json


def prepare(args: argparse.Namespace) -> None:
    if args.max_concurrent < 1:
        raise ValueError("max_concurrent must be positive")
    if args.class_particles < 1 or args.direct_particles < 1:
        raise ValueError("particle counts must be positive")
    candidate_root = args.candidates.resolve()
    cases = sorted(path.name for path in candidate_root.iterdir() if path.is_dir())
    if not cases:
        raise ValueError(f"no candidate directories in {candidate_root}")
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    shutil.copytree(candidate_root, output / "candidates")
    code = output / "code"
    code.mkdir()
    source_root = Path(__file__).resolve().parent
    source_hashes = {}
    for name in ("reactor_proxy_calibration.py", "simple_barrier.py"):
        source = source_root / name
        target = code / name
        shutil.copyfile(source, target)
        source_hashes[name] = simple_barrier.file_sha256(target)
    write_json(output / "source_checksums.json", source_hashes)
    remote = Path(args.remote_root)
    rows = [
        "\t".join(
            (
                name,
                str(remote / "candidates" / name),
                str(remote / "results" / name),
            )
        )
        for name in cases
    ]
    (output / "manifest.tsv").write_text("\n".join(rows) + "\n")
    sbatch = f"""#!/bin/bash
#SBATCH --job-name=alpha-calibrate
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=56
#SBATCH --time={args.walltime}
#SBATCH --array=1-{len(rows)}%{args.max_concurrent}
#SBATCH --chdir={remote}
#SBATCH --output={remote}/logs/%A_%a.out
set -euo pipefail
export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}
test "$(sha256sum {remote}/code/reactor_proxy_calibration.py | cut -d' ' -f1)" = "{source_hashes['reactor_proxy_calibration.py']}"
test "$(sha256sum {remote}/code/simple_barrier.py | cut -d' ' -f1)" = "{source_hashes['simple_barrier.py']}"
line=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {remote}/manifest.tsv)
IFS=$'\t' read -r name candidate result <<< "$line"
{args.python} {remote}/code/reactor_proxy_calibration.py evaluate \\
  --candidate "$candidate" \\
  --out "$result" \\
  --simple-executable {args.simple_executable} \\
  --simple-sha256 {args.simple_sha256} \\
  --desc-python {args.python} \\
  --desc-version {args.desc_version} \\
  --desc-source-sha256 {args.desc_source_sha256} \\
  --class-particles {args.class_particles} \\
  --direct-particles {args.direct_particles}
"""
    (output / "run_calibration.sbatch").write_text(sbatch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--python", default="/home/ert/desc-env/bin/python")
    parser.add_argument("--desc-version", required=True)
    parser.add_argument("--desc-source-sha256", required=True)
    parser.add_argument("--simple-executable", required=True)
    parser.add_argument("--simple-sha256", required=True)
    parser.add_argument("--class-particles", type=int, default=1000)
    parser.add_argument("--direct-particles", type=int, default=512)
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument("--walltime", default="12:00:00")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
