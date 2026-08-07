import numpy as np

from simple_direct import DIRECT_NAMELIST, loss_windows


def test_direct_contract_uses_production_resolution_and_symplectic_euler():
    assert "npoiper2 = 256" in DIRECT_NAMELIST
    assert "ns_s = 5" in DIRECT_NAMELIST
    assert "ns_tp = 5" in DIRECT_NAMELIST
    assert "multharm = 5" in DIRECT_NAMELIST
    assert "integmode = 1" in DIRECT_NAMELIST
    assert "fast_class = .False." in DIRECT_NAMELIST


def test_recorded_endpoint_is_censored_as_survival():
    endpoint = 0.09999999999999999
    metrics = loss_windows(
        np.array([0.0005, 0.02, endpoint]),
        prompt_time=0.001,
        final_time=endpoint,
    )
    assert metrics["prompt_count"] == 1
    assert metrics["late_count"] == 1
    assert metrics["total_count"] == 2
