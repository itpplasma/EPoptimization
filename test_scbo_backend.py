from pathlib import Path

from scbo_backend import (
    choose_backend,
    condor_submit_text,
    runnable_nodes,
    slurm_array_text,
)


SNAPSHOT = """node1|idle|0/64/0/64
node2|mix|16/48/0/64
node3|mix|32/32/0/64
node4|down*|0/64/0/64
"""


def test_capacity_requires_whole_candidate_slots() -> None:
    assert runnable_nodes(SNAPSHOT) == ["node1", "node2"]
    assert choose_backend(SNAPSHOT, workers=2) == "slurm"
    assert choose_backend(SNAPSHOT, workers=3) == "condor"


def test_condor_submit_is_shared_filesystem_and_credential_free() -> None:
    text = condor_submit_text(
        executable=Path("/temp/ert/run_scbo_candidate.sh"),
        campaign_root=Path("/temp/ert/runs/campaign"),
        wave="wave03",
        environment={"CODE_ROOT": "/temp/ert/code", "SIMPLE_X": "/temp/ert/simple.x"},
        jobs=8,
        cpus=4,
        memory_mb=4096,
        max_materialize=8,
    )
    assert "queue 8" in text
    assert "request_cpus = 4" in text
    assert "request_memory = 4096MB" in text
    assert "should_transfer_files = NO" in text
    assert "getenv = False" in text
    assert "CAMPAIGN_ROOT=/temp/ert/runs/campaign" in text
    assert "WAVE=wave03" in text
    assert "max_materialize = 8" in text


def test_slurm_array_caps_workers_and_exports_contract() -> None:
    text = slurm_array_text(
        executable=Path("/home/ert/code/run_scbo_candidate.sh"),
        campaign_root=Path("/home/ert/runs/campaign"),
        wave="wave03",
        environment={"CODE_ROOT": "/home/ert/code", "PARTICLES": "256"},
        jobs=28,
        cpus=48,
        memory_mb=4096,
        time_limit="04:00:00",
        max_concurrent=8,
    )
    assert "#SBATCH --array=0-27%8" in text
    assert "#SBATCH --cpus-per-task=48" in text
    assert "#SBATCH --mem=4096M" in text
    assert "export CAMPAIGN_ROOT=/home/ert/runs/campaign" in text
    assert '/home/ert/code/run_scbo_candidate.sh "$SLURM_ARRAY_TASK_ID"' in text
