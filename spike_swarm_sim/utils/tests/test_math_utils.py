import numpy as np
import pytest
from spike_swarm_sim.utils import *


@pytest.mark.parametrize("u,v,expected", (
    (0, 30, 30),
    (30, 0, 30),
    (-30, 30, 60),
    (0, 360, 0),
    (400, 90, 50)
))
def test_compute_angle(u, v, expected): 
    result = compute_angle(np.radians(u), np.radians(v))
    assert np.degrees(result) == expected