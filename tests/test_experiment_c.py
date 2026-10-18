import numpy as np
import pandas as pd

from imonet.cube import Cube
from imonet.evaluate import range_windows
from imonet.experiment_c import Period, period_days


def test_range_windows_tile_and_align_to_the_end():
    w = range_windows(10, 10 + 600, 256)
    assert w == [(10, 10), (266, 266), (354, 522)]           # last aligned, scoring only new days
    assert w[-1][0] + 256 == 610


def test_period_days_from_dates():
    days = np.datetime64("2015-12-20") + np.arange(30)
    cube = Cube(r=np.zeros((1, 30, 3), np.float32), sigma=np.zeros((1, 30, 3), np.float32),
                avail=np.ones((1, 30), bool), sta=np.array(["A"]), lat=np.zeros(1), lon=np.zeros(1),
                days=days, split_day=np.zeros(30, np.int8), split_sta=np.zeros(1, np.int8),
                exclude=np.zeros((1, 30), bool))
    p = Period("x", (("2015-12-25", "2015-12-31"), ("2016-01-10", None)))
    assert period_days(cube, p) == [(5, 12), (21, 30)]
    assert pd.Timestamp(cube.days[11]) == pd.Timestamp("2015-12-31")
