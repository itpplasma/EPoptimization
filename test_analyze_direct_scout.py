from __future__ import annotations

import numpy as np
import pytest

from analyze_direct_scout import paired_summary


def test_paired_summary_uses_particle_matched_differences() -> None:
    reference = np.array([True, True, False, False])
    candidate = np.array([False, True, True, False])

    result = paired_summary(reference, candidate)

    assert result["change"] == 0.0
    assert result["paired_se"] == pytest.approx(np.std([-1, 0, 1, 0], ddof=1) / 2)


def test_paired_summary_rejects_unmatched_particles() -> None:
    with pytest.raises(ValueError, match="differ"):
        paired_summary(np.array([True]), np.array([True, False]))
