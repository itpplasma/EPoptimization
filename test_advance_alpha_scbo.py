from __future__ import annotations

import json

import pytest

from advance_alpha_scbo import pending_responses


def test_pending_responses_selects_exact_active_wave(tmp_path) -> None:
    state = {"pending": [{"candidate_id": 8}, {"candidate_id": 9}]}
    for candidate_id in (7, 8, 9):
        case = tmp_path / str(candidate_id)
        case.mkdir()
        (case / "response.json").write_text(json.dumps({"candidate_id": candidate_id}))

    rows = pending_responses(state, tmp_path)

    assert [row["candidate_id"] for row in rows] == [8, 9]


def test_pending_responses_rejects_incomplete_wave(tmp_path) -> None:
    state = {"pending": [{"candidate_id": 8}, {"candidate_id": 9}]}
    case = tmp_path / "8"
    case.mkdir()
    (case / "response.json").write_text(json.dumps({"candidate_id": 8}))

    with pytest.raises(ValueError, match="every pending"):
        pending_responses(state, tmp_path)
