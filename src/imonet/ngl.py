"""Nevada Geodetic Laboratory files.

`steps.txt` lists, per station, the days its series may be offset:
equipment changes (type 1) and earthquakes (type 2, with the event's
influence radius, the station's distance from it, the magnitude and the
event id).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

STEPS = Path("data/meta/steps.txt")


def read_steps(path: Path | str = STEPS) -> pd.DataFrame:
    """One row per listed step. Columns: sta, date, kind ('equipment' or
    'quake'), info (the change description, or the event id), and for
    quakes radius_km, dist_km, mag."""
    rows = []
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if len(p) < 4:
                continue
            if p[2] == "1":
                rows.append((p[0], p[1], "equipment", " ".join(p[3:]), None, None, None))
            elif p[2] == "2" and len(p) >= 7:
                rows.append((p[0], p[1], "quake", p[6], float(p[3]), float(p[4]), float(p[5])))
    df = pd.DataFrame(rows, columns=["sta", "date", "kind", "info", "radius_km", "dist_km", "mag"])
    df["date"] = pd.to_datetime(df["date"], format="%y%b%d")
    return df
