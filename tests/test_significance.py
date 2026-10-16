import numpy as np
import pandas as pd
import pytest

from imonet.significance import binomial_two_sided, compare, headline


def write_runs(root, method, shift, seeds=(0, 1), n=60, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.2, 0.4, n)
    for s in seeds:
        d = root / "A" / method / "validation" / f"60km_s{s}"
        d.mkdir(parents=True)
        noise = rng.normal(0, 0.002, (n, 3))
        df = pd.DataFrame({"station": [f"S{i:03d}" for i in range(n)], "cells": 100,
                           **{f"crps_fast_{c}": base + shift + noise[:, k] for k, c in enumerate("enu")}})
        df.to_csv(d / "stations.csv", index=False)


def test_binomial_two_sided():
    assert binomial_two_sided(5, 10) == pytest.approx(1.0)
    assert binomial_two_sided(0, 10) == pytest.approx(2 / 1024)
    assert binomial_two_sided(10, 10) == pytest.approx(2 / 1024)


def test_headline_is_the_table_summary():
    x = np.array([[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]])           # 3 stations, 1 seed, 2 columns
    assert headline(x) == pytest.approx((3.0 + 4.0) / 2)


def test_a_clearly_better_method_is_detected(tmp_path):
    write_runs(tmp_path, "good", 0.0, seed=1)
    write_runs(tmp_path, "bad", 0.01, seed=1)
    r = compare("good", "bad", root=tmp_path, n_boot=500)
    assert r["difference"] == pytest.approx(-0.01, abs=2e-3)
    assert r["ci_high"] < 0 and r["a_better_stations"] == r["compared_stations"] == 60
    assert r["sign_p"] < 1e-10 and r["seeds"] == 2


def test_equal_methods_are_not_separated(tmp_path):
    write_runs(tmp_path, "one", 0.0, seed=2)
    write_runs(tmp_path, "two", 0.0, seed=3)
    r = compare("one", "two", root=tmp_path, n_boot=500)
    assert r["ci_low"] < 0 < r["ci_high"] and r["sign_p"] > 0.01
