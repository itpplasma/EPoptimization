from pathlib import Path

import numpy as np

from evaluate_classifier_proxy import evaluate


def test_evaluate_classifier_proxy_records_surface_and_binary_identity(
    tmp_path: Path,
) -> None:
    topology = np.ones((1, 1, 1, 2, 2), dtype=np.int8)
    topology[..., 0, 0] = 0
    path = tmp_path / "topology.npz"
    np.savez_compressed(
        path,
        topology=topology,
        jpar=topology,
        particle_index=np.arange(4).reshape(topology.shape),
        passing=np.zeros_like(topology, dtype=bool),
        weights=np.ones((1, 2, 2)) / 4.0,
        surface=np.array(0.25),
        simple_sha256=np.array("simple-hash"),
        trace_time=np.array(0.02),
        wout_sha256=np.array("wout-hash"),
    )

    payload = evaluate(path)

    assert payload["surface"] == 0.25
    assert payload["unclassified_fraction"] == 0.25
    assert payload["jpar_nonideal_fraction"] == 0.0
    assert payload["simple_sha256"] == "simple-hash"
    assert payload["trace_time"] == 0.02
    assert payload["wout_sha256"] == "wout-hash"
