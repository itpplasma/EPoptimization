from pathlib import Path

import numpy as np

from raw_fourier_surface import _mode_scale, raw_coordinate_contract


BASE = Path(
    "/home/ert/proj/alpha-loss-optimization-data/optimization/"
    "local_search_20260711/base/candidate/input.vmec"
)


def test_raw_contract_matches_rogerio_low_order_dimension():
    contract = raw_coordinate_contract(BASE)
    assert len(contract["names"]) == 24
    assert contract["max_mode"] == 2
    assert len(contract["center"]) == len(contract["half_width"]) == 24


def test_mode_box_decays_with_harmonic_order():
    assert np.isclose(_mode_scale("rc(0,1)"), 0.025)
    assert _mode_scale("zs(2,2)") < _mode_scale("zs(1,0)")
