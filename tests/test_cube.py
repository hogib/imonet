import numpy as np

from imonet.cube import HELD_OUT, NORMAL, TEST, TRAIN, VALIDATION, Cube


def tiny_cube() -> Cube:
    s, t = 3, 5
    r = np.arange(s * t * 3, dtype=np.float32).reshape(s, t, 3)
    avail = np.ones((s, t), dtype=bool)
    avail[0, 1] = False
    exclude = np.zeros((s, t), dtype=bool)
    exclude[1, 0] = True
    return Cube(
        r=r, sigma=np.ones_like(r), avail=avail,
        sta=np.array(["AAAA", "BBBB", "CCCC"]),
        lat=np.array([35.0, 35.1, 35.2]), lon=np.array([-118.0, -118.1, -118.2]),
        days=np.arange("2019-12-30", "2020-01-04", dtype="datetime64[D]"),
        split_day=np.array([TRAIN, TRAIN, VALIDATION, VALIDATION, TEST], dtype=np.int8),
        split_sta=np.array([NORMAL, NORMAL, HELD_OUT], dtype=np.int8),
        exclude=exclude,
    )


def test_save_and_load_round_trip(tmp_path):
    c = tiny_cube()
    c.save(tmp_path / "c.npz")
    d = Cube.load(tmp_path / "c.npz")
    assert (d.r == c.r).all() and (d.sta == c.sta).all() and (d.days == c.days).all()
    assert d.qstep_amp.shape == (0, 3)
    assert not (tmp_path / "c.tmp.npz").exists()


def test_trainable_cells():
    c = tiny_cube()
    ok = c.trainable()
    assert ok[:, 2:].sum() == 0                  # not training days
    assert not ok[2].any()                       # held-out station
    assert not ok[0, 1] and not ok[1, 0]         # missing, excluded
    assert ok[0, 0] and ok[1, 1]
    assert list(c.stations(HELD_OUT)) == [2]
