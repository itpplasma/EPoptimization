from pathlib import Path

from scbo_backend import choose_backend, condor_submit_text, runnable_nodes


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
    )
    assert "queue 8" in text
    assert "request_cpus = 4" in text
    assert "request_memory = 4096MB" in text
    assert "should_transfer_files = NO" in text
    assert "getenv = False" in text
    assert "CAMPAIGN_ROOT=/temp/ert/runs/campaign" in text
    assert "WAVE=wave03" in text
