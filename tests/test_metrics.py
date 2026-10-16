import numpy as np
import pandas as pd
import pytest

from imonet.metrics import (band_psd, crps, energy_score, event_mask, inside, neighbour_correlations, screen, slow_part,
                            station_noise,
                            station_scores, summarise)

from test_data import synthetic_cube


def test_crps_point_forecast_is_absolute_error():
    y = np.array([1.0, -2.0])
    assert crps(np.array([[0.5, 0.0]]), y) == pytest.approx([0.5, 2.0])


def test_crps_matches_brute_force_fair_estimator():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(7, 5)), rng.normal(size=5)
    m = len(x)
    brute = np.abs(x - y).mean(0) - np.abs(x[:, None] - x[None]).sum((0, 1)) / (2 * m * (m - 1))
    assert crps(x, y) == pytest.approx(brute)


def test_crps_of_a_standard_normal_at_zero():
    x = np.random.default_rng(1).normal(size=(4000, 1))
    assert crps(x, np.zeros(1))[0] == pytest.approx(2 / np.sqrt(2 * np.pi) - 1 / np.sqrt(np.pi), abs=0.01)


def test_energy_score_reduces_to_brute_force():
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=(6, 4, 3)), rng.normal(size=(4, 3))
    m = len(x)
    first = np.linalg.norm(x - y, axis=-1).mean(0)
    pair = np.linalg.norm(x[:, None] - x[None], axis=-1).sum((0, 1))
    assert energy_score(x, y) == pytest.approx(first - pair / (2 * m * (m - 1)))
    assert energy_score(x[:1], y) == pytest.approx(np.linalg.norm(x[0] - y, axis=-1))


def test_inside_central_interval():
    x = np.arange(101, dtype=float)[:, None]
    assert inside(x, np.array([50.0]), 0.5)[0]
    assert not inside(x, np.array([80.0]), 0.5)[0]


def test_slow_part_skips_missing_days():
    x = np.arange(200, dtype=float)
    x[::3] = np.nan
    s = slow_part(x[:, None])[:, 0]
    i = 100
    assert s[i] == pytest.approx(np.nanmedian(x[i - 30:i + 31]))
    assert np.isfinite(s).all()


def test_band_psd_of_white_noise_is_flat():
    rng = np.random.default_rng(3)
    p = np.mean([band_psd(rng.normal(0, 2.0, (4096, 1)))[0] for _ in range(16)], axis=0)
    assert p == pytest.approx(np.full(len(p), 4.0), rel=0.15)
    x = rng.normal(0, 2.0, (4096, 1))
    x[:600] = np.nan
    assert band_psd(x) is None


def test_scores_of_a_calibrated_ensemble():
    rng = np.random.default_rng(4)
    t, m = 1024, 32
    truth = rng.normal(0, 1, (t, 3))
    samples = rng.normal(0, 1, (m, t, 3))
    cells, skip = np.ones(t, dtype=bool), np.zeros(t, dtype=bool)
    row = station_scores(truth, samples, cells, skip, np.ones(3))
    assert row["cover50_total"] == pytest.approx(0.5, abs=0.04)
    assert row["cover90_total"] == pytest.approx(0.9, abs=0.03)
    assert row["spread_skill_e"] == pytest.approx(1.0, abs=0.1)
    assert row["scale_log2_n"] == pytest.approx(0.0, abs=0.1)
    assert abs(row["psd_log10_4_8_u"]) < 0.1


def test_scores_of_predicting_zero():
    rng = np.random.default_rng(5)
    t = 400
    truth = rng.normal(0, 2.0, (t, 3))
    row = station_scores(truth, np.zeros((1, t, 3)), np.ones(t, bool), np.zeros(t, bool), np.full(3, 2.0))
    assert row["nrmse_total_e"] == pytest.approx(1.0, abs=0.1)
    assert row["crps_total_u"] == pytest.approx(np.sqrt(2 / np.pi), abs=0.1)
    assert "cover50_total" not in row and "spread_skill_e" not in row
    assert station_scores(truth, np.zeros((1, t, 3)), np.zeros(t, bool), np.zeros(t, bool), np.ones(3)) is None


def test_neighbour_correlation_of_a_copy_is_one():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(256, 3))
    corr = neighbour_correlations(x, np.stack([x, x]), np.stack([x, rng.normal(size=(256, 3))]),
                                  np.ones(256, bool))
    assert corr[0] == pytest.approx(np.ones((3, 2)))
    assert np.abs(corr[1]).max() < 0.25


def test_event_mask_covers_listed_stations_only():
    cube = synthetic_cube()
    steps = pd.DataFrame({"sta": [cube.sta[2], cube.sta[3]], "kind": ["quake", "quake"],
                          "date": [cube.days[100], cube.days[100]], "mag": [6.5, 5.0]})
    m = event_mask(cube, steps)
    assert m[2, 70:131].all() and not m[2, 69] and not m[2, 131]
    assert not m[3].any()


def test_summary_is_median_and_component_mean():
    per = pd.DataFrame({"station": ["a", "b", "c"], "cells": [1, 1, 1],
                        "nrmse_fast_e": [1.0, 2.0, 3.0], "nrmse_fast_n": [1.0, 1.0, 1.0],
                        "nrmse_fast_u": [3.0, 3.0, 3.0]})
    s = summarise(per)
    assert s["nrmse_fast_e"] == 2.0 and s["nrmse_fast"] == pytest.approx(2.0)


def test_screen_flags_spikes_not_steps():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((400, 2, 3))
    x[200:, 1] += 20.0                                   # a step: kept
    x[100, 0, 1] += 15.0                                 # a spike: replaced
    x[50:60, 0] = np.nan
    out, bad = screen(x)
    assert bad[100, 0, 1] and not bad[195:205, 1].any()
    assert bad.mean() < 0.005
    assert abs(out[100, 0, 1]) < 3 and np.isnan(out[50:60, 0]).all()
    assert np.array_equal(out[~bad], x[~bad], equal_nan=True)


def test_screen_matches_the_cube_rule():
    """Same days as gnssdl's dataset.screen_outliers (61 days, 5 σ, 15 days)."""
    rng = np.random.default_rng(1)
    x = rng.standard_t(3, size=(600, 3))
    x[rng.random(600) < 0.1] = np.nan
    daily = pd.DataFrame(x)
    ref = np.zeros(600, dtype=bool)
    for c in range(3):
        med = daily[c].rolling(61, center=True, min_periods=15).median()
        dev = (daily[c] - med).abs()
        ref |= (dev > 5.0 * 1.4826 * dev.rolling(61, center=True, min_periods=15).median()).to_numpy()
    _, bad = screen(x)
    assert ref.any() and np.array_equal(bad.any(axis=1), ref)


def test_calibration_pooled_and_by_noise():
    """An ensemble with one width for every station: too wide where the
    station is quiet, too narrow where it is noisy, right pooled."""
    noise = np.linspace(0.5, 1.5, 40)
    err = noise                                         # error grows with the station's noise
    spread = np.full(40, np.sqrt(np.mean(noise ** 2)))  # one spread for all, right on average
    per = pd.DataFrame({"station": [f"S{i}" for i in range(40)], "cells": 100, "noise": noise,
                        "cover90_fast": 0.9,
                        **{f"nrmse_total_{c}": err for c in "enu"},
                        **{f"spread_skill_{c}": spread / err for c in "enu"},
                        **{f"crps_fast_{c}": 0.3 * noise for c in "enu"}})
    out = summarise(per)
    assert out["spread_skill_pooled"] == pytest.approx(1.0)
    assert out["spread_skill_q1"] > 1.4 and out["spread_skill_q4"] < 0.9
    assert out["noise_q1"] < out["noise_q4"] and out["cover90_fast_pooled"] == pytest.approx(0.9)
    assert "noise" not in out


def test_station_noise_is_relative_to_a_typical_station():
    cube = synthetic_cube()
    cube.r[0] *= 3.0
    noise = station_noise(cube)
    assert np.nanmedian(noise) == pytest.approx(1.0, rel=0.2)
    assert noise[0] > 2.0 * np.nanmedian(noise)
