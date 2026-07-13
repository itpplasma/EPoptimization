from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def runnable_nodes(snapshot: str, required_cpus: int = 48) -> list[str]:
    nodes = []
    for line in snapshot.splitlines():
        if not line.strip():
            continue
        name, state, counts = line.split("|", 2)
        _, idle, _, _ = (int(value) for value in counts.split("/"))
        if state.rstrip("*") in {"idle", "mix"} and idle >= required_cpus:
            nodes.append(name)
    return nodes


def choose_backend(snapshot: str, workers: int = 8, required_cpus: int = 48) -> str:
    return "slurm" if len(runnable_nodes(snapshot, required_cpus)) >= workers else "condor"


def _condor_environment(values: dict[str, str]) -> str:
    items = [f"{key}={value}" for key, value in sorted(values.items())]
    if any('"' in item or "\n" in item for item in items):
        raise ValueError("Condor environment values cannot contain quotes or newlines")
    return " ".join(items)


def condor_submit_text(
    *,
    executable: Path,
    campaign_root: Path,
    wave: str,
    environment: dict[str, str],
    jobs: int,
    cpus: int,
    memory_mb: int,
    max_materialize: int | None = None,
) -> str:
    if jobs <= 0 or cpus <= 0 or memory_mb <= 0:
        raise ValueError("jobs, cpus, and memory must be positive")
    if max_materialize is not None and max_materialize <= 0:
        raise ValueError("max_materialize must be positive")
    logs = campaign_root / wave / "logs"
    merged = {
        **environment,
        "ALLOCATED_CPUS": str(cpus),
        "CAMPAIGN_ROOT": str(campaign_root),
        "WAVE": wave,
    }
    lines = [
        "universe = vanilla",
        f"executable = {executable}",
        "arguments = $(Process)",
        f"initialdir = {campaign_root}",
        f'environment = "{_condor_environment(merged)}"',
        f"request_cpus = {cpus}",
        f"request_memory = {memory_mb}MB",
        'requirements = (OpSys == "LINUX") && (Arch == "X86_64")',
        "should_transfer_files = NO",
        "getenv = False",
        f"output = {logs}/condor_$(Cluster)_$(Process).out",
        f"error = {logs}/condor_$(Cluster)_$(Process).err",
        f"log = {logs}/condor_$(Cluster).log",
        "notification = Never",
    ]
    if max_materialize is not None:
        lines.append(f"max_materialize = {max_materialize}")
    lines.extend([f"queue {jobs}", ""])
    return "\n".join(lines)


def snapshot_aclustercapacity(host: str) -> str:
    command = [
        "ssh",
        host,
        'sinfo -N -h -p compute -o "%N|%t|%C"',
    ]
    return subprocess.run(command, check=True, text=True, capture_output=True).stdout


def write_submit(args: argparse.Namespace) -> None:
    environment = json.loads(args.environment.read_text())
    text = condor_submit_text(
        executable=args.executable.resolve(),
        campaign_root=args.campaign_root.resolve(),
        wave=args.wave,
        environment=environment,
        jobs=args.jobs,
        cpus=args.cpus,
        memory_mb=args.memory_mb,
        max_materialize=args.max_materialize,
    )
    args.out.write_text(text)


def report_choice(args: argparse.Namespace) -> None:
    snapshot = snapshot_aclustercapacity(args.host)
    nodes = runnable_nodes(snapshot, args.required_cpus)
    result = {
        "backend": "slurm" if len(nodes) >= args.workers else "condor",
        "required_cpus": args.required_cpus,
        "runnable_nodes": nodes,
        "snapshot": snapshot.splitlines(),
        "workers": args.workers,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(required=True)
    choose = commands.add_parser("choose")
    choose.add_argument("--host", default="acluster")
    choose.add_argument("--workers", type=int, default=8)
    choose.add_argument("--required-cpus", type=int, default=48)
    choose.set_defaults(action=report_choice)
    submit = commands.add_parser("write-condor")
    submit.add_argument("--executable", type=Path, required=True)
    submit.add_argument("--campaign-root", type=Path, required=True)
    submit.add_argument("--wave", required=True)
    submit.add_argument("--environment", type=Path, required=True)
    submit.add_argument("--jobs", type=int, default=8)
    submit.add_argument("--cpus", type=int, required=True)
    submit.add_argument("--memory-mb", type=int, required=True)
    submit.add_argument("--max-materialize", type=int)
    submit.add_argument("--out", type=Path, required=True)
    submit.set_defaults(action=write_submit)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.action(arguments)
