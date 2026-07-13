from __future__ import annotations

import argparse
import json

import numpy as np

from build_scbo_response import base_response, successful_response


def test_failure_response_preserves_candidate_identity() -> None:
    request = {"candidate_id": 19, "unit_x": [0.2, 0.8]}

    response = base_response(request, "failed", "equilibrium_failure")

    assert response == {
        "candidate_id": 19,
        "unit_x": [0.2, 0.8],
        "observation": None,
        "status": "failed",
        "failure_kind": "equilibrium_failure",
    }


def test_success_response_uses_paired_total_and_late_changes(tmp_path) -> None:
    reference = tmp_path / "reference.dat"
    result_dir = tmp_path / "result"
    direct = result_dir / "direct"
    direct.mkdir(parents=True)
    np.savetxt(reference, [[1, 0.0005], [2, 0.1], [3, 0.3], [4, 0.3]])
    np.savetxt(
        direct / "times_lost.dat",
        [[1, 0.0005], [2, 0.3], [3, 0.3], [4, 0.3]],
    )
    result = result_dir / "result.json"
    result.write_text(
        json.dumps(
            {
                "status": "ok",
                "wout_sha256": "a" * 64,
                "direct": {"total_loss": 0.5, "late_loss": 0.25},
            }
        )
    )
    args = argparse.Namespace(
        result=result,
        reference_times=reference,
        prompt_time=0.001,
        trace_time=0.3,
        late_target=0.01,
    )

    response = successful_response({"candidate_id": 4, "unit_x": [0.4]}, args)

    assert response["observation"]["value"] == -0.25
    assert response["observation"]["constraints"] == [-0.24]
