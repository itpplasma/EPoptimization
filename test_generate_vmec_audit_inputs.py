from __future__ import annotations

import numpy as np
import pytest

from generate_vmec_audit_inputs import case_specs, relative_case_path


def test_case_specs_are_deterministic_and_cover_sobol_block() -> None:
    first = case_specs(8, 4, 29)
    second = case_specs(8, 4, 29)

    assert len(first) == 16
    assert [item[:3] for item in first] == [item[:3] for item in second]
    assert [item[0] for item in first] == [f"d08_{index:03d}" for index in range(16)]
    np.testing.assert_array_equal(
        np.stack([item[3] for item in first]),
        np.stack([item[3] for item in second]),
    )
    assert np.all((np.stack([item[3] for item in first]) >= 0.0))
    assert np.all((np.stack([item[3] for item in first]) <= 1.0))


def test_case_specs_reject_invalid_shape() -> None:
    with pytest.raises(ValueError, match="outside"):
        case_specs(0, 2, 1)


def test_relative_case_path_keeps_directories_bounded() -> None:
    assert str(relative_case_path(12, 95, "d12_095")) == "d12/b02/d12_095"
