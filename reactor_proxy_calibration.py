#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import simple_barrier


SURFACES = (0.3, 0.6)
DESC_SETTINGS = {
    "Y_B": 32,
    "num_transit": 1,
    "num_well": 15,
    "num_quad": 8,
    "num_pitch": 5,
}
GEOMETRY_KEYS = (
    "aspect",
    "mean_iota",
    "vacuum_well",
    "mirror_ratio",
    "max_elongation",
    "qa_residual",
)


def write_json(path: Path, data) -> None:
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def require_finite(values, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"non-finite {label}")
    return array


def free_boundary(vmec, max_mode: int) -> tuple[np.ndarray, list[str]]:
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
    return np.asarray(surface.x, dtype=float).copy(), list(surface.dof_names)


def geometry_metrics(vmec) -> dict[str, float | int]:
    from Alan_objectives import MaxElongationPen, MirrorRatioPen
    from simsopt.mhd import QuasisymmetryRatioResidual

    qs = QuasisymmetryRatioResidual(
        vmec,
        np.arange(0.0, 1.01, 0.1),
        helicity_m=1,
        helicity_n=0,
    )
    metrics = {
        "aspect": float(vmec.aspect()),
        "mean_iota": float(vmec.mean_iota()),
        "vacuum_well": float(vmec.vacuum_well()),
        "mirror_ratio": float(MirrorRatioPen(v=vmec, output_mirror=True)),
        "max_elongation": float(
            MaxElongationPen(vmec=vmec, return_elongation=True)
        ),
        "qa_residual": float(qs.total()),
        "ier_flag": int(vmec.wout.ier_flag),
    }
    require_finite([metrics[key] for key in GEOMETRY_KEYS], "geometry metrics")
    return metrics


def geometry_constraints(
    metadata: dict, base: dict, args: argparse.Namespace
) -> dict:
    require_finite([metadata[key] for key in GEOMETRY_KEYS], "candidate geometry")
    require_finite([base[key] for key in GEOMETRY_KEYS], "base geometry")
    deltas = {
        "aspect_relative": abs(metadata["aspect"] - base["aspect"])
        / abs(base["aspect"]),
        "mean_iota_absolute": abs(metadata["mean_iota"] - base["mean_iota"]),
        "qa_increase": metadata["qa_residual"] - base["qa_residual"],
        "mirror_increase": metadata["mirror_ratio"] - base["mirror_ratio"],
        "elongation_increase": metadata["max_elongation"]
        - base["max_elongation"],
    }
    limits = {
        "aspect_relative": args.max_aspect_relative,
        "mean_iota_absolute": args.max_iota_change,
        "qa_increase": args.max_qa_increase,
        "mirror_increase": args.max_mirror_increase,
        "elongation_increase": args.max_elongation_increase,
    }
    checks = {key: deltas[key] <= limits[key] for key in limits}
    return {
        "feasible": bool(all(checks.values())),
        "checks": checks,
        "deltas": deltas,
        "limits": limits,
    }


def archive_difference(first: Path, second: Path) -> float:
    from scipy.io import netcdf_file

    maximum = 0.0
    with netcdf_file(str(first), "r", mmap=False) as left:
        with netcdf_file(str(second), "r", mmap=False) as right:
            for name in ("rmnc", "zmns", "lmns", "bmnc"):
                a = np.asarray(left.variables[name][:])
                b = np.asarray(right.variables[name][:])
                if a.shape != b.shape:
                    return float("inf")
                maximum = max(maximum, float(np.max(np.abs(a - b))))
    return maximum


def candidate_specs(
    dimension: int, directions: int, amplitude: float, seed: int
) -> list[tuple[str, np.ndarray, int, int]]:
    if dimension <= 0 or directions <= 0 or amplitude <= 0.0:
        raise ValueError("dimension, directions, and amplitude must be positive")
    rng = np.random.default_rng(seed)
    specs = [("base", np.zeros(dimension), -1, 0)]
    for index in range(directions):
        direction = rng.normal(size=dimension)
        direction /= np.linalg.norm(direction)
        specs.extend(
            (
                (f"d{index:02d}_minus", -amplitude * direction, index, -1),
                (f"d{index:02d}_plus", amplitude * direction, index, 1),
            )
        )
    return specs


def generate(args: argparse.Namespace) -> None:
    from simsopt.mhd import Vmec

    output = args.out.resolve()
    input_source = args.input.resolve()
    base_wout = args.base_wout.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="reactor_proxy_vmec_") as tmp:
        work = Path(tmp)
        input_path = work / "input.calibration"
        shutil.copyfile(input_source, input_path)
        previous = Path.cwd()
        os.chdir(work)
        try:
            vmec = Vmec(input_path.name, verbose=False)
            vmec.keep_all_files = True
            initial, names = free_boundary(vmec, args.max_mode)
            vmec.run()
            minor_radius = float(vmec.wout.Aminor_p)
            scale = args.amplitude * minor_radius
            specs = candidate_specs(len(initial), args.directions, scale, args.seed)
            manifest = []
            for name, perturbation, direction, sign in specs:
                vmec.boundary.x = initial + perturbation
                vmec.run()
                if int(vmec.wout.ier_flag) != 0:
                    raise RuntimeError(f"VMEC failed for {name}: ier_flag={vmec.wout.ier_flag}")
                case = output / name
                case.mkdir()
                generated_wout = Path(vmec.output_file)
                target_wout = case / "wout.nc"
                if name == "base":
                    difference = archive_difference(base_wout, generated_wout)
                    if difference > args.archive_tolerance:
                        raise RuntimeError(
                            f"base rerun differs from archive by {difference:.3e}"
                        )
                    shutil.copyfile(base_wout, target_wout)
                else:
                    difference = None
                    shutil.copyfile(generated_wout, target_wout)
                vmec.write_input(str(case / "input.vmec"))
                np.save(case / "perturbation.npy", perturbation)
                metadata = {
                    "case": name,
                    "direction": direction,
                    "sign": sign,
                    "amplitude_over_a": float(
                        np.linalg.norm(perturbation) / minor_radius
                    ),
                    "max_mode": args.max_mode,
                    "generation_seed": args.seed,
                    "dof_names": names,
                    "perturbation": perturbation.tolist(),
                    "archive_rerun_max_difference": difference,
                    "wout_sha256": simple_barrier.file_sha256(target_wout),
                    **geometry_metrics(vmec),
                }
                write_json(case / "metadata.json", metadata)
                manifest.append(metadata)
        finally:
            os.chdir(previous)
    write_json(output / "manifest.json", manifest)


def desc_worker(args: argparse.Namespace) -> None:
    import desc
    from desc.grid import LinearGrid
    from desc.objectives import EffectiveRipple, GammaC
    from desc.vmec import VMECIO

    equilibrium = VMECIO.load(args.wout, profile="iota")
    grid = LinearGrid(
        rho=np.sqrt(SURFACES),
        M=equilibrium.M_grid,
        N=equilibrium.N_grid,
        NFP=equilibrium.NFP,
        sym=True,
    )
    objective_type = {"gamma_c": GammaC, "effective_ripple": EffectiveRipple}[
        args.metric
    ]
    objective = objective_type(
        equilibrium,
        grid=grid,
        use_bounce1d=True,
        **DESC_SETTINGS,
    )
    objective.build(use_jit=False, verbose=0)
    values = np.asarray(objective.compute(equilibrium.params_dict), dtype=float)
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise ValueError(f"unexpected {args.metric} output {values}")
    source = Path(desc.__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    print(
        json.dumps(
            {"values": values.tolist(), "version": desc.__version__, "commit": commit}
        )
    )


def desc_metric(metric: str, wout: Path, python: Path) -> dict:
    completed = subprocess.run(
        [
            str(python),
            str(Path(__file__).resolve()),
            "desc-worker",
            "--metric",
            metric,
            "--wout",
            str(wout),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout.strip())


def evaluate(args: argparse.Namespace) -> None:
    candidate = args.candidate.resolve()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    wout = candidate / "wout.nc"
    metadata = json.loads((candidate / "metadata.json").read_text())
    wout_hash = simple_barrier.file_sha256(wout)
    if wout_hash != metadata["wout_sha256"]:
        raise ValueError(f"candidate equilibrium checksum mismatch in {candidate}")
    base_metadata = json.loads((candidate.parent / "base" / "metadata.json").read_text())
    constraints = geometry_constraints(metadata, base_metadata, args)
    if not constraints["feasible"]:
        write_json(
            output / "result.json",
            {
                "case": candidate.name,
                "status": "rejected_geometry",
                "candidate_wout_sha256": wout_hash,
                "candidate_metadata": metadata,
                "geometry_constraints": constraints,
            },
        )
        return
    binary_hash = simple_barrier.file_sha256(args.simple_executable)
    if binary_hash != args.simple_sha256:
        raise ValueError(
            f"unexpected SIMPLE executable hash {binary_hash}; "
            f"expected {args.simple_sha256}"
        )
    rz_scale, b_scale = simple_barrier.reactor_scale(wout)
    barrier = simple_barrier.barrier_metrics(
        wout,
        s_inner=0.3,
        s_outer=0.6,
        ntestpart=args.class_particles,
        rz_scale=rz_scale,
        b_scale=b_scale,
        facE_al=1.0,
        trace_time=2.0e-2,
        seed=12345,
        classifier="topology",
        simple_executable=args.simple_executable,
        keep_workdir=True,
        timeout_s=args.timeout,
    )
    direct = simple_barrier.direct_loss_metrics(
        wout,
        ntestpart=args.direct_particles,
        expected_simple_sha256=args.simple_sha256,
        simple_executable=args.simple_executable,
        keep_workdir=True,
        timeout_s=args.timeout,
    )
    class_workdir = Path(str(barrier.pop("workdir")))
    direct_workdir = Path(str(direct.pop("workdir")))
    shutil.move(str(class_workdir), output / "classification")
    shutil.move(str(direct_workdir), output / "direct")
    gamma_c = desc_metric("gamma_c", wout, args.desc_python)
    effective_ripple = desc_metric("effective_ripple", wout, args.desc_python)
    if gamma_c["commit"] != effective_ripple["commit"]:
        raise RuntimeError("DESC source changed during candidate evaluation")
    for name, values in (
        ("gamma_c", gamma_c["values"]),
        ("effective_ripple", effective_ripple["values"]),
    ):
        if np.asarray(values).shape != (2,):
            raise ValueError(f"unexpected {name} output shape")
        require_finite(values, name)
    require_finite(
        [
            barrier["barrier_overlap_jpar"],
            barrier["barrier_overlap_topology"],
            direct["prompt_loss"],
            direct["late_loss"],
            direct["total_loss"],
        ],
        "orbit metrics",
    )
    for key in (
        "n_trapped_inner_jpar",
        "n_trapped_outer_jpar",
        "n_trapped_inner_topology",
        "n_trapped_outer_topology",
    ):
        if int(barrier[key]) <= 0:
            raise ValueError(f"invalid trapped-particle count {key}={barrier[key]}")
    if direct["total_count"] != direct["prompt_count"] + direct["late_count"]:
        raise ValueError("direct loss windows do not partition total loss")
    result = {
        "case": candidate.name,
        "status": "accepted",
        "candidate_wout_sha256": wout_hash,
        "simple_sha256": binary_hash,
        "candidate_metadata": metadata,
        "geometry_constraints": constraints,
        "barrier": barrier,
        "direct": direct,
        "gamma_c": dict(zip(("s03", "s06"), gamma_c["values"])),
        "effective_ripple": dict(
            zip(("s03", "s06"), effective_ripple["values"])
        ),
        "desc_settings": DESC_SETTINGS,
        "desc_version": gamma_c["version"],
        "desc_commit": gamma_c["commit"],
        "desc_python": str(args.desc_python),
    }
    write_json(output / "result.json", result)


def prepare_slurm(args: argparse.Namespace) -> None:
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
line=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {remote}/manifest.tsv)
IFS=$'\t' read -r name candidate result <<< "$line"
{args.python} {remote}/code/reactor_proxy_calibration.py evaluate \\
  --candidate "$candidate" \\
  --out "$result" \\
  --simple-executable {args.simple_executable} \\
  --simple-sha256 {args.simple_sha256} \\
  --desc-python {args.python} \\
  --class-particles {args.class_particles} \\
  --direct-particles {args.direct_particles}
"""
    (output / "run_calibration.sbatch").write_text(sbatch)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    generate_parser = commands.add_parser("generate")
    generate_parser.add_argument("--input", type=Path, required=True)
    generate_parser.add_argument("--base-wout", type=Path, required=True)
    generate_parser.add_argument("--out", type=Path, required=True)
    generate_parser.add_argument("--directions", type=int, default=4)
    generate_parser.add_argument("--amplitude", type=float, default=5.0e-3)
    generate_parser.add_argument("--max-mode", type=int, default=2)
    generate_parser.add_argument("--seed", type=int, default=20260711)
    generate_parser.add_argument("--archive-tolerance", type=float, default=1.0e-12)
    generate_parser.set_defaults(function=generate)

    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--candidate", type=Path, required=True)
    evaluate_parser.add_argument("--out", type=Path, required=True)
    evaluate_parser.add_argument("--simple-executable", type=Path, required=True)
    evaluate_parser.add_argument("--simple-sha256", required=True)
    evaluate_parser.add_argument("--desc-python", type=Path, required=True)
    evaluate_parser.add_argument("--class-particles", type=int, default=1000)
    evaluate_parser.add_argument("--direct-particles", type=int, default=512)
    evaluate_parser.add_argument("--timeout", type=float, default=86400.0)
    evaluate_parser.add_argument("--max-aspect-relative", type=float, default=0.02)
    evaluate_parser.add_argument("--max-iota-change", type=float, default=0.02)
    evaluate_parser.add_argument("--max-qa-increase", type=float, default=0.01)
    evaluate_parser.add_argument("--max-mirror-increase", type=float, default=0.02)
    evaluate_parser.add_argument(
        "--max-elongation-increase", type=float, default=0.2
    )
    evaluate_parser.set_defaults(function=evaluate)

    worker_parser = commands.add_parser("desc-worker")
    worker_parser.add_argument("--metric", choices=("gamma_c", "effective_ripple"), required=True)
    worker_parser.add_argument("--wout", type=Path, required=True)
    worker_parser.set_defaults(function=desc_worker)

    slurm_parser = commands.add_parser("prepare-slurm")
    slurm_parser.add_argument("--candidates", type=Path, required=True)
    slurm_parser.add_argument("--out", type=Path, required=True)
    slurm_parser.add_argument("--remote-root", required=True)
    slurm_parser.add_argument("--python", default="/home/ert/desc-env/bin/python")
    slurm_parser.add_argument("--simple-executable", required=True)
    slurm_parser.add_argument("--simple-sha256", required=True)
    slurm_parser.add_argument("--class-particles", type=int, default=1000)
    slurm_parser.add_argument("--direct-particles", type=int, default=512)
    slurm_parser.add_argument("--max-concurrent", type=int, default=1)
    slurm_parser.add_argument("--walltime", default="12:00:00")
    slurm_parser.set_defaults(function=prepare_slurm)
    return root


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
