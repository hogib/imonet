"""Experiment B: planted fault-slip events (docs/00-design.md §6.2).

Events from `imonet.events` (slow slip and afterslip on SCEC CFM faults) are
added to every station's residuals, the held-out targets included. Each
affected target is generated twice with identical noise, once from the
planted network and once from the original, over the 256-day windows of
experiment A that overlap the event. The difference d of the two sample
means is what the method passed on from the event; s is the planted signal
at the target, detrended per window exactly as the truth is (detrending is
linear, so planted truth − original truth = detrend(s)).

Pairing: every (target, window) has its own seed (`item_seed`), so its noise
does not depend on what else is in the batch, and a target's prediction
from the original network is computed once and reused for every event.

Scores per event and target (horizontal components, the scored event days
of each window: from the onset to the end of the event plus HOLD_DAYS,
skipping the days experiment A skips):

- rho: ⟨d, s⟩ / ⟨s, s⟩, the share of the planted signal passed on (also
  pooled over target-events, Σ⟨d, s⟩ / Σ⟨s, s⟩, so that signals the
  per-window detrending has mostly removed count little);
- shape: correlation of d with s;
- t50 error: days between the times d and s reach half their change;
- crps / cover90 of the planted truth by the planted samples (total part),
  against the original truth by the original samples;
- spurious: at held-out targets the event leaves still (< INSIDE_MM) but
  whose neighbours it moves (> SEEN_MM), the peak of |d| as a share of the
  largest neighbour displacement.

An event is kept if at least one held-out target sees it (peak horizontal
displacement > SEEN_MM). Its footprint is the radius of a disc with the
area over which the final horizontal displacement exceeds SEEN_MM.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from imonet.cube import HELD_OUT, Cube, offsets_km
from imonet.data import WINDOW_DAYS, context, day_of_year, detrend, target, target_noise
from imonet.evaluate import RESULTS, Setup, density_name, method_for, network, period_windows
from imonet.progress import progress_line
from imonet import events as ev_mod
from imonet.metrics import crps, inside as inside_interval
from imonet.network import K_NEIGHBOURS

SEEN_MM = 1.0                # a station sees an event above this peak horizontal motion
INSIDE_MM = 0.1              # below this a station is outside the event
HOLD_DAYS = 60               # days scored after the event ends
PER_TEMPLATE = 30
MAX_SPURIOUS = 5             # outside targets scored per event (the nearest)
SEED = 20261017
BATCH = 32


@dataclass
class Event:
    id: int
    placed: ev_mod.PlacedEvent
    t0: int                  # day index of the onset
    end: int                 # last scored day + 1 (onset + duration + HOLD_DAYS)
    peak_h: np.ndarray       # S peak (final) horizontal displacement, mm
    footprint_km: float
    stations: np.ndarray | None = None   # stations moving > INSIDE_MM; placed.green holds only these

    @property
    def template(self) -> str:
        return self.placed.template.name


def footprint_km(placed: ev_mod.PlacedEvent, step_km: float = 4.0, half_km: float = 300.0) -> float:
    """Radius of the disc with the area where the final horizontal
    displacement exceeds SEEN_MM (a grid around the patch)."""
    fault = placed.meta.get("fault")
    if fault is None:
        return float("nan")
    lon0, lat0 = placed.centre
    ax = np.arange(-half_km, half_km + 1e-9, step_km)
    gx, gy = np.meshgrid(ax, ax)
    lat = lat0 + gy.ravel() / 111.2                       # cell areas need not be exact: a radius in km
    lon = lon0 + gx.ravel() / (111.2 * np.cos(np.deg2rad(lat0)))
    area = 0
    for a in range(0, len(lon), 2000):
        u = ev_mod.station_displacement(fault, placed.slip_m, placed.rake, lon[a:a + 2000], lat[a:a + 2000],
                                         select=placed.tri) * 1000
        area += int((np.hypot(u[:, 0], u[:, 1]) > SEEN_MM).sum())
    return float(np.sqrt(area * step_km ** 2 / np.pi))


def plan_events(cube: Cube, setup: Setup, split: int, per_template: int = PER_TEMPLATE, seed: int = SEED,
                length: int = WINDOW_DAYS, templates=None, catalogue=None, max_tries: int = 2000) -> list[Event]:
    """Place `per_template` events of each template on random suitable CFM
    faults, each seen by at least one scored held-out target, with an onset
    in the split such that the event and its hold end inside it."""
    rng = np.random.default_rng(seed)
    templates = templates or ev_mod.load_templates()
    catalogue = catalogue or ev_mod.fault_catalogue()
    days = np.flatnonzero(cube.split_day == split)
    targets = cube.stations(HELD_OUT)
    scored = np.zeros(cube.n_sta, bool)
    scored[targets] = setup.keep[targets] & cube.avail[np.ix_(targets, days)].any(axis=1)
    out = []
    for tp in templates:
        placed = tries = 0
        while placed < per_template:
            tries += 1
            if tries > max_tries:
                raise RuntimeError(f"{tp.name}: only {placed} events seen by a held-out target")
            ev = ev_mod.random_event(tp, catalogue, rng, cube.lon, cube.lat)
            u = ev.final_mm()
            peak = np.hypot(u[:, 0], u[:, 1])
            if not (scored & (peak > SEEN_MM)).any():
                continue
            span = int(np.ceil(ev.duration_days)) + HOLD_DAYS
            if span >= len(days):
                raise ValueError(f"{tp.name}: event lasts longer than the split")
            t0 = int(days[rng.integers(0, len(days) - span)])
            moving = np.flatnonzero(np.linalg.norm(u, axis=1) > INSIDE_MM)
            fp = footprint_km(ev)
            light = replace(ev, green=ev.green[moving].astype(np.float32))   # only what is planted
            out.append(Event(len(out), light, t0, t0 + span, peak, fp, moving))
            placed += 1
    return out


def event_signal(cube: Cube, ev: Event) -> tuple[np.ndarray, np.ndarray]:
    """(stations moving more than INSIDE_MM, their displacement S'×(T−t0)×3
    in mm from the onset), held at its final value after the event."""
    if ev.stations is not None:
        sta, sub = ev.stations, ev.placed
    else:
        final = ev.placed.final_mm()
        sta = np.flatnonzero(np.linalg.norm(final, axis=1) > INSIDE_MM)
        sub = replace(ev.placed, green=ev.placed.green[sta])
    n = cube.n_days - ev.t0
    m = min(n, int(np.ceil(ev.placed.duration_days)) + 1)
    s = np.empty((len(sta), n, 3), dtype=np.float32)
    s[:, :m] = sub.displacement_mm(np.arange(m, dtype=float))
    s[:, m:] = sub.final_mm()[:, None, :]
    return sta, s


def planted(cube: Cube, ev: Event) -> Cube:
    sta, s = event_signal(cube, ev)
    r = cube.r.copy()
    r[sta, ev.t0:] += s
    return replace(cube, r=r)


def _sample_items(method, batch: dict, seeds: np.ndarray) -> np.ndarray:
    """B×m×L×3 samples with each item's randomness from its own seed."""
    from imonet.model.methods import DiffusionMethod
    if isinstance(method, DiffusionMethod):
        return method.sample(batch | {"item_seed": seeds}, np.random.default_rng(0))
    parts = [method.sample({k: v[i:i + 1] for k, v in batch.items()}, np.random.default_rng(int(seeds[i])))
             for i in range(len(seeds))]
    return np.concatenate(parts)


class Predictor:
    """Paired predictions of targets over windows for one method and network;
    the original network's are cached."""

    def __init__(self, method, setup: Setup, net: np.ndarray, length: int = WINDOW_DAYS, k: int = K_NEIGHBOURS,
                 seed: int = 0):
        self.method, self.setup, self.net, self.length, self.k, self.seed = method, setup, net, length, k, seed
        self.cache: dict[tuple[int, int], np.ndarray] = {}

    def item_seed(self, s: int, t0: int) -> int:
        return int(np.random.SeedSequence((self.seed, s, t0)).generate_state(1)[0])

    def predict(self, cube: Cube, jobs: list[tuple[int, int]], use_cache: bool) -> dict:
        """{(target, t0): m×L×3 samples in mm, detrended over the target's
        days}, from `cube`; cached if `use_cache`."""
        todo = [j for j in jobs if not (use_cache and j in self.cache)]
        out = {j: self.cache[j] for j in jobs if use_cache and j in self.cache}
        if todo:
            self.method.prepare(cube, self.setup.scale, self.net, cube.avail)
        for a in range(0, len(todo), BATCH):
            chunk = todo[a:a + BATCH]
            ctxs, oks = [], []
            for s, t0 in chunk:
                ctx = context(cube, self.setup.scale, cube.lat[s], cube.lon[s], t0, self.net[self.net != s],
                              cube.avail, self.length, self.k)
                ctx |= {"target": s, "t0": t0, "doy0": np.float32(day_of_year(cube.days[t0]))}
                ctxs.append(ctx | target_noise(self.setup.noise_log, s))
                oks.append(cube.avail[s, t0:t0 + self.length])
            batch = {key: np.stack([np.asarray(c[key]) for c in ctxs]) for key in ctxs[0]}
            seeds = np.array([self.item_seed(s, t0) for s, t0 in chunk])
            x = _sample_items(self.method, batch, seeds)
            x = detrend(x, np.stack(oks)[:, None], zero_missing=False) * self.setup.scale
            for i, j in enumerate(chunk):
                out[j] = x[i]
                if use_cache:
                    self.cache[j] = x[i]
        return out


def _t50(x: np.ndarray, days: np.ndarray) -> float:
    """Day at which the horizontal magnitude of x (days×2, from its value at
    the first day) first reaches half its final value."""
    m = np.hypot(*(x - x[0]).T)
    if m[-1] <= 0:
        return float("nan")
    return float(days[np.argmax(m >= 0.5 * m[-1])])


def score_event(pred: Predictor, cube: Cube, ev: Event, split: int) -> list[dict]:
    """Rows for the targets that see the event and for up to MAX_SPURIOUS
    outside targets whose neighbours it moves."""
    setup, length = pred.setup, pred.length
    targets = cube.stations(HELD_OUT)
    targets = targets[setup.keep[targets]]
    wins = [(t0, first) for t0, first in period_windows(cube, split, length)
            if t0 < ev.end and t0 + length > ev.t0]
    seen = targets[ev.peak_h[targets] > SEEN_MM]
    still = targets[(ev.peak_h[targets] < INSIDE_MM)]
    moved_net = pred.net[ev.peak_h[pred.net] > SEEN_MM]
    spurious = []
    if len(moved_net):
        lat0, lon0 = cube.lat[moved_net].mean(), cube.lon[moved_net].mean()
        e, n = offsets_km(lat0, lon0, cube.lat[still], cube.lon[still])
        for s in still[np.argsort(np.hypot(e, n))]:
            ctx = context(cube, setup.scale, cube.lat[s], cube.lon[s], wins[0][0], pred.net[pred.net != s],
                          cube.avail, length, pred.k)
            nb = ctx["nbr"][ctx["valid"]]
            if len(nb) and (ev.peak_h[nb] > SEEN_MM).any():
                spurious.append((s, float(ev.peak_h[nb].max())))
            if len(spurious) >= MAX_SPURIOUS:
                break
    jobs = [(int(s), t0) for s in list(seen) + [s for s, _ in spurious] for t0, _ in wins]
    if not jobs:
        return []
    pc = planted(cube, ev)
    base = pred.predict(cube, jobs, use_cache=True)
    plant = pred.predict(pc, jobs, use_cache=False)
    rows = []
    for s in list(seen) + [s for s, _ in spurious]:
        num = den = 0.0
        d_all, s_all, day_all, crps_p, crps_o, cov_p = [], [], [], [], [], []
        for t0, first in wins:
            win = np.arange(t0, t0 + length)
            ok = cube.avail[s, win]
            sl = (win >= max(first, ev.t0)) & (win < ev.end) & ok & ~setup.skip[s, win]
            if sl.sum() < 5:
                continue
            truth_o = target(cube, setup.scale, s, t0, cube.avail, length)["x"] * setup.scale
            truth_p = target(pc, setup.scale, s, t0, pc.avail, length)["x"] * setup.scale
            sig = (truth_p - truth_o)[sl]                                     # detrended planted signal
            d = (plant[(s, t0)].mean(0) - base[(s, t0)].mean(0))[sl]
            num += float((d[:, :2] * sig[:, :2]).sum())
            den += float((sig[:, :2] ** 2).sum())
            d_all.append(d[:, :2]), s_all.append(sig[:, :2]), day_all.append(win[sl])
            norm = setup.station_scale[s]
            crps_p.append(crps(plant[(s, t0)][:, sl] / norm, truth_p[sl] / norm).mean())
            crps_o.append(crps(base[(s, t0)][:, sl] / norm, truth_o[sl] / norm).mean())
            cov_p.append(inside_interval(plant[(s, t0)][:, sl], truth_p[sl], 0.9).mean())
        if not d_all:
            continue
        d_cat, s_cat, days = np.concatenate(d_all), np.concatenate(s_all), np.concatenate(day_all)
        onset_win = next((i for i, dy in enumerate(day_all) if dy[0] <= ev.t0 + 1), 0)
        row = {"event": ev.id, "template": ev.template, "station": str(cube.sta[s]),
               "role": "seen" if ev.peak_h[s] > SEEN_MM else "outside", "peak_mm": float(ev.peak_h[s]),
               "footprint_km": ev.footprint_km, "days": int(len(days)),
               "crps_planted": float(np.mean(crps_p)), "crps_original": float(np.mean(crps_o)),
               "cover90_planted": float(np.mean(cov_p))}
        e, n = offsets_km(ev.placed.centre[1], ev.placed.centre[0], cube.lat[s:s + 1], cube.lon[s:s + 1])
        row["dist_km"] = float(np.hypot(e, n)[0])
        nd = offsets_km(cube.lat[s], cube.lon[s], cube.lat[pred.net], cube.lon[pred.net])
        row["nearest_km"] = float(np.min(np.hypot(*nd)))
        row["net_seeing"] = int((ev.peak_h[pred.net] > SEEN_MM).sum())
        if row["role"] == "seen":
            row["rho"] = num / den if den > 0 else float("nan")
            row["projection"], row["signal_energy"] = num, den                 # for the pooled ρ
            a, b = d_cat.ravel(), s_cat.ravel()
            row["shape"] = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
            dw, sw, dy = d_all[onset_win], s_all[onset_win], day_all[onset_win]   # detrended per window
            row["t50_error_days"] = _t50(dw, dy) - _t50(sw, dy)
            row["signal_rms_mm"] = float(np.sqrt((s_cat ** 2).mean()))
        else:
            nb_peak = dict(spurious)[s]
            row["spurious"] = float(np.hypot(*d_cat.T).max() / nb_peak)
            row["neighbour_peak_mm"] = nb_peak
        rows.append(row)
    return rows


def load_or_plan(cube: Cube, setup: Setup, split: int, split_name: str, per_template: int = PER_TEMPLATE,
                 root: Path = RESULTS, log=print) -> list[Event]:
    """The planned events of a split, from runs/B/events_<split>.pkl, or
    planned (deterministically) and saved there with a readable table."""
    import pickle
    path = root / "B" / f"events_{split_name}_{per_template}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    t = time.time()
    events = plan_events(cube, setup, split, per_template)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(events))
    events_table(events, cube).to_csv(path.with_suffix(".csv"), index=False)
    log(f"{len(events)} events planned in {time.time() - t:.0f} s ({path})")
    return events


def events_table(events: list[Event], cube: Cube) -> pd.DataFrame:
    rows = []
    for ev in events:
        tp = ev.placed.template
        rows.append({"event": ev.id, "template": ev.template, "fault": ev.placed.fault, "rake": ev.placed.rake,
                     "onset": str(cube.days[ev.t0]), "end": str(cube.days[min(ev.end, cube.n_days - 1)]),
                     "lon": ev.placed.centre[0], "lat": ev.placed.centre[1],
                     "footprint_km": ev.footprint_km, "peak_mm": float(ev.peak_h.max()),
                     **tp.drawn_values()})
    return pd.DataFrame(rows)


def run(setup: Setup, method_name: str, split: int, split_name: str, density: float | None, seed: int,
        events: list[Event], n_samples: int = 16, root: Path = RESULTS, model_dir: Path | None = None,
        label: str | None = None, target_noise_mode: str = "unknown", progress=None) -> pd.DataFrame:
    t = time.time()
    method, extra = method_for(setup, method_name, density, n_samples, root, model_dir, target_noise=target_noise_mode)
    net = network(setup.cube, density, seed)
    pred = Predictor(method, setup, net, seed=seed)
    rows = []
    for i, ev in enumerate(events):
        rows += score_event(pred, setup.cube, ev, split)
        if progress:
            progress(progress_line(i + 1, len(events), t, f"events ({ev.template})"))
    df = pd.DataFrame(rows)
    out = root / "B" / (label or method_name) / split_name / f"{density_name(density)}_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "targets.csv", index=False)
    summary = {"method": label or method_name, "kind": method_name, "density": density_name(density), "seed": seed,
               "samples": n_samples, "events": len(events), **extra, **summarise_b(df),
               "seconds": round(time.time() - t, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return df


def summarise_b(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    seen, out = df[df.role == "seen"], df[df.role == "outside"]
    res = {"seen_pairs": int(len(seen)), "outside_pairs": int(len(out))}
    for col in ("rho", "shape", "t50_error_days"):
        if col in seen:
            res[f"{col}_median"] = float(seen[col].median())
    if "signal_energy" in seen and seen.signal_energy.sum() > 0:
        # Pooled over target-events, weighted by the planted signal's energy:
        # weak, mostly detrended signals count little.
        res["rho_pooled"] = float(seen.projection.sum() / seen.signal_energy.sum())
    if len(seen):
        res["crps_planted_median"] = float(seen.crps_planted.median())
        res["crps_original_median"] = float(seen.crps_original.median())
        res["cover90_planted"] = float(seen.cover90_planted.mean())
    if len(out):
        res["spurious_median"] = float(out.spurious.median())
    return res


# ---------------------------------------------------------------------------
# A real event: the 2019 Ridgecrest earthquakes
# ---------------------------------------------------------------------------

RIDGECREST_DAY = np.datetime64("2019-07-05")       # between the M6.4 (07-04) and M7.1 (07-06) shocks
RIDGECREST_LONLAT = (-117.60, 35.77)
OFFSET_GAP, OFFSET_SPAN = 2, 10                    # days skipped either side, days averaged


def offset(x: np.ndarray, ok: np.ndarray, centre: int) -> np.ndarray:
    """Mean over OFFSET_SPAN days after the event minus before (…×L×3, ok
    …×L), skipping OFFSET_GAP days either side of `centre`."""
    after = np.zeros(x.shape[-2], bool)
    after[centre + OFFSET_GAP:centre + OFFSET_GAP + OFFSET_SPAN] = True
    before = np.zeros(x.shape[-2], bool)
    before[centre - OFFSET_GAP - OFFSET_SPAN:centre - OFFSET_GAP] = True
    mean = lambda m: (np.where((ok & m)[..., None], x, 0).sum(-2) / np.maximum((ok & m).sum(-1), 1)[..., None])
    return mean(after) - mean(before)


def ridgecrest(setup: Setup, method_name: str, density: float | None, seed: int, n_samples: int = 16,
               root: Path = RESULTS, model_dir: Path | None = None, label: str | None = None,
               target_noise_mode: str = "unknown", max_km: float = 300.0) -> pd.DataFrame:
    """Held-out stations within `max_km` of the Ridgecrest earthquakes,
    generated from the real network over a 256-day window centred on them:
    the coseismic offset of the samples against the one observed in the
    station's own data (mean of OFFSET_SPAN days after minus before). The
    window lies in the training years, but the held-out stations never
    entered training and the stations near the rupture were excluded from
    it in the second half of 2019. Stations are chosen by distance, not by
    the cube's listed offsets: those are missing for exactly the stations
    near the rupture (their window was excluded when the cube was built)
    and unreliable far from it. Summaries are given for observed horizontal
    offsets of at least 1 and 5 mm."""
    cube, length = setup.cube, WINDOW_DAYS
    t = int(np.flatnonzero(cube.days == RIDGECREST_DAY)[0])
    t0, centre = t - length // 2, length // 2
    listed = np.zeros((cube.n_sta, 3))
    near = (cube.qstep_day >= t - 2) & (cube.qstep_day <= t + 2)
    np.add.at(listed, cube.qstep_sta[near], cube.qstep_amp[near])
    targets = cube.stations(HELD_OUT)
    ok_win = cube.avail[:, t0:t0 + length]
    e_all, n_all = offsets_km(RIDGECREST_LONLAT[1], RIDGECREST_LONLAT[0], cube.lat, cube.lon)
    usable = [s for s in targets if setup.keep[s] and np.hypot(e_all[s], n_all[s]) <= max_km
              and ok_win[s, centre - OFFSET_GAP - OFFSET_SPAN:centre - OFFSET_GAP].sum() >= 5
              and ok_win[s, centre + OFFSET_GAP:centre + OFFSET_GAP + OFFSET_SPAN].sum() >= 5]
    method, extra = method_for(setup, method_name, density, n_samples, root, model_dir, target_noise=target_noise_mode)
    net = network(cube, density, seed)
    pred = Predictor(method, setup, net, seed=seed)
    samples = pred.predict(cube, [(int(s), t0) for s in usable], use_cache=False)
    rows = []
    for s in usable:
        ok = ok_win[s]
        truth = target(cube, setup.scale, s, t0, cube.avail, length)["x"] * setup.scale
        obs = offset(truth, ok, centre)
        gen = offset(samples[(s, t0)], np.broadcast_to(ok, (len(samples[(s, t0)]), length)), centre)   # m×3
        mean = gen.mean(0)
        lo, hi = np.quantile(gen, [0.05, 0.95], axis=0, method="weibull") if len(gen) > 1 else (mean, mean)
        e, n = offsets_km(RIDGECREST_LONLAT[1], RIDGECREST_LONLAT[0], cube.lat[s:s + 1], cube.lon[s:s + 1])
        nd = offsets_km(cube.lat[s], cube.lon[s], cube.lat[net], cube.lon[net])
        rows.append({"station": str(cube.sta[s]), "dist_km": float(np.hypot(e, n)[0]),
                     "nearest_km": float(np.min(np.hypot(*nd))),
                     **{f"listed_{c}": float(listed[s, k]) for k, c in enumerate("enu")},
                     **{f"observed_{c}": float(obs[k]) for k, c in enumerate("enu")},
                     **{f"generated_{c}": float(mean[k]) for k, c in enumerate("enu")},
                     "recovered_h": float((mean[:2] @ obs[:2]) / max(obs[:2] @ obs[:2], 1e-9)),
                     "error_h_mm": float(np.hypot(*(mean[:2] - obs[:2]))),
                     "observed_h_mm": float(np.hypot(*obs[:2])),
                     "inside90_h": float(np.mean((obs[:2] >= lo[:2]) & (obs[:2] <= hi[:2])))})
    df = pd.DataFrame(rows)
    out = root / "B" / (label or method_name) / "ridgecrest" / f"{density_name(density)}_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "targets.csv", index=False)
    summary = {"method": label or method_name, "density": density_name(density), "seed": seed, "targets": len(df),
               **extra}
    for mm in (1.0, 5.0):
        sub = df[df.observed_h_mm >= mm] if len(df) else df
        tag = f"{mm:g}mm"
        summary |= {f"targets_{tag}": int(len(sub)),
                    f"recovered_h_median_{tag}": float(sub.recovered_h.median()) if len(sub) else None,
                    f"error_h_mm_median_{tag}": float(sub.error_h_mm.median()) if len(sub) else None,
                    f"inside90_h_{tag}": float(sub.inside90_h.mean()) if len(sub) else None}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return df
