import pandas as pd

from imonet.figures import experiment_a as A


def fake_runs() -> pd.DataFrame:
    rows = []
    for m, samples in (("zero", 1), ("idw", 1), ("idw+noise", 32)):
        for dens, seeds in (("full", 1), ("25km", 2), ("60km", 2)):
            for seed in range(seeds):
                r = {"method": m, "density": dens, "seed": seed, "samples": samples,
                     "nrmse_fast": 0.5 + seed / 100, "crps_fast": 0.4, "crps_total": 0.8}
                if samples > 1:
                    r |= {"cover90_fast": 0.88, "spread_skill": 0.9, "scale_log2": 0.1}
                rows.append(r)
    return pd.DataFrame(rows)


def test_scores_by_density_writes_a_pdf(tmp_path):
    path = A.scores_by_density(fake_runs(), tmp_path)
    assert path.exists() and path.stat().st_size > 1000


def test_table_has_a_row_per_method_and_density(tmp_path):
    text = A.table(fake_runs(), tmp_path).read_text()
    assert text.count(r"\\") == 1 + 3 * 3                     # header + rows
    assert "idw + noise" in text and "--" in text            # point forecasts lack coverage
    assert r"\addlinespace" in text
