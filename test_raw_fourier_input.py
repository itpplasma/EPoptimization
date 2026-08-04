from __future__ import annotations

import pytest

from raw_fourier_surface import write_vmec_input

BASE = """&INDATA
NFP = 2
MPOL = 9
NTOR = 6
NS_ARRAY = 16 51
PHIEDGE = 1.234
RBC(   0,   0) =  1.0,    ZBS(   0,   0) =  0.0
RBC(   1,   0) =  0.1,    ZBS(   1,   0) =  0.1
CURTOR = 0.0
/
"""


class FakeSurface:
    def __init__(self, namelist: str) -> None:
        self._namelist = namelist

    def get_nml(self) -> str:
        return self._namelist


NEW = """&INDATA
LASYM = .FALSE.
NFP = 2
RBC(   0,   0) =  2.0,    ZBS(   0,   0) =  0.0
RBC(   1,   0) =  0.5,    ZBS(   1,   0) =  0.5
/
"""


def write(tmp_path, base: str = BASE, namelist: str = NEW):
    base_path = tmp_path / "input.base"
    base_path.write_text(base)
    out = tmp_path / "input.candidate"
    write_vmec_input(base_path, FakeSurface(namelist), out)
    return out.read_text()


def test_base_settings_survive_the_rewrite(tmp_path) -> None:
    text = write(tmp_path)
    for setting in ("MPOL = 9", "NTOR = 6", "NS_ARRAY = 16 51", "PHIEDGE = 1.234",
                    "CURTOR = 0.0"):
        assert setting in text


def test_new_boundary_replaces_the_old_one(tmp_path) -> None:
    text = write(tmp_path)
    assert "RBC(   0,   0) =  2.0" in text
    assert "RBC(   1,   0) =  0.5" in text
    assert "1.0," not in text
    assert text.count("RBC(") == 2
    assert text.count("ZBS(") == 2


def test_boundary_lands_where_the_old_block_was(tmp_path) -> None:
    lines = write(tmp_path).splitlines()
    boundary = [i for i, line in enumerate(lines) if line.startswith("RBC(")]
    assert lines.index("PHIEDGE = 1.234") < min(boundary)
    assert max(boundary) < lines.index("CURTOR = 0.0")
    assert lines[-1] == "/"


def test_a_base_without_a_boundary_block_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError):
        write(tmp_path, base="&INDATA\nNFP = 2\n/\n")


def test_a_surface_without_coefficients_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError):
        write(tmp_path, namelist="&INDATA\nNFP = 2\n/\n")
