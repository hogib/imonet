"""Synthetic transients planted in training samples (docs/00-design.md §4.3).

California's training years hold few transients, so a model trained on them
alone learns that neighbours carry only shared noise and passes on little of
a localised signal (experiment B, Ridgecrest). With probability p a training
sample gets one transient, written into the target and every neighbour.

The family is deliberately simpler than experiment B's fault-slip events,
so that B tests generalisation:

- **Footprint**: Gaussian of radius R (its σ), 10–300 km log-uniform,
  centred within 1.5 R of the target (uniform over the disk), so the target
  is inside, at the edge or outside.
- **Pattern**: half *uniform* (one direction everywhere: a random horizontal
  azimuth and a vertical share in ±0.5, amplitude A·exp(−r²/2R²)); half
  *radial* (horizontal displacement away from or toward the centre,
  A·(r/R)·exp(½ − r²/2R²), peak A at r = R; vertical v·A·exp(−r²/2R²), v in
  ±1), so that direction varies over the network as in real deformation.
- **Time**: 70% slow, a raised-cosine rise over 5–120 days (log-uniform),
  30% of them reversed back to zero over 1–3 times the rise; 30% steps (a
  one-day rise). Onset uniform from a quarter-window before the window to
  its end.
- **Amplitude**: slow 1–30 mm log-uniform; steps heavy-tailed (Pareto,
  α = 1.5, from 1 mm), capped at 50 mm.

**Visibility guard.** A transient no neighbour shows cannot be predicted at
the target; training on it would teach the model to add random excursions
everywhere. A transient is kept only if at least 2 neighbours see it: a
range of ≥ 1 mm (norm over E, N, U) over their available days in the
window. Otherwise the sample stays clean. Transients seen by the neighbours
but not at the target are kept: they teach the model what not to copy.

Planting happens after the sample is built: detrending is linear, so adding
the detrended transient to the detrended series equals detrending the
planted series. The draws come after the sample's own, so a sample keeps its
target, window and network whatever p.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RADIUS_KM = (10.0, 300.0)       # Gaussian footprint σ, log-uniform
CENTRE_RADII = 1.5              # centre within this many radii of the target
RADIAL_P = 0.5                  # share of radial patterns
VERTICAL_UNIFORM = 0.5          # |vertical share| of uniform patterns
VERTICAL_RADIAL = 1.0           # |vertical share| of radial patterns
STEP_P = 0.3                    # share of steps (one-day rise)
RISE_DAYS = (5.0, 120.0)        # slow rise, log-uniform
REVERSE_P = 0.3                 # share of slow transients that reverse
REVERSE_FACTOR = (1.0, 3.0)     # reversal duration / rise duration
SLOW_MM = (1.0, 30.0)           # slow amplitude, log-uniform
STEP_ALPHA, STEP_MIN_MM, STEP_MAX_MM = 1.5, 1.0, 50.0
ONSET_BEFORE = 0.25             # earliest onset, in windows before the first day
SEEN_MM = 1.0                   # a neighbour sees the transient: range ≥ this
MIN_SEEN = 2                    # neighbours that must see it


@dataclass
class Transient:
    radius_km: float
    centre_km: tuple[float, float]   # east, north of the target
    radial: bool
    direction: np.ndarray            # 3: uniform: unit horizontal + vertical share; radial: (sense, 0, v)
    amplitude_mm: float
    onset: float                     # days from the window's first day
    rise: float                      # days
    reverse: float                   # days of reversal, 0 if held

    def history(self, length: int) -> np.ndarray:
        """L: the time function, 0 before onset, 1 at full displacement."""
        t = np.arange(length) - self.onset
        f = _rise(t, self.rise)
        if self.reverse:
            f = f - _rise(t - self.rise, self.reverse)
        return f

    def pattern(self, east_km: np.ndarray, north_km: np.ndarray) -> np.ndarray:
        """n×3 mm: final displacement at stations east_km, north_km of the target."""
        de, dn = east_km - self.centre_km[0], north_km - self.centre_km[1]
        r = np.hypot(de, dn) / self.radius_km
        g = np.exp(-0.5 * r ** 2)
        if not self.radial:
            return self.amplitude_mm * g[:, None] * self.direction[None, :]
        sense, _, v = self.direction
        with np.errstate(invalid="ignore", divide="ignore"):
            ue, un = np.where(r > 0, de / (r * self.radius_km), 0.0), np.where(r > 0, dn / (r * self.radius_km), 0.0)
        h = sense * self.amplitude_mm * r * np.exp(0.5) * g
        return np.stack([h * ue, h * un, v * self.amplitude_mm * g], axis=1)

    def signal(self, east_km: np.ndarray, north_km: np.ndarray, length: int) -> np.ndarray:
        """n×L×3 mm."""
        return self.pattern(east_km, north_km)[:, None, :] * self.history(length)[None, :, None]


def _rise(t: np.ndarray, duration: float) -> np.ndarray:
    """Raised cosine from 0 at t = 0 to 1 at t = duration."""
    u = np.clip(t / duration, 0.0, 1.0)
    return 0.5 * (1 - np.cos(np.pi * u))


def draw(rng: np.random.Generator, length: int) -> Transient:
    radius = float(np.exp(rng.uniform(*np.log(RADIUS_KM))))
    rc, az = CENTRE_RADII * radius * np.sqrt(rng.random()), rng.uniform(0, 2 * np.pi)
    radial = bool(rng.random() < RADIAL_P)
    if radial:
        direction = np.array([rng.choice([-1.0, 1.0]), 0.0, rng.uniform(-VERTICAL_RADIAL, VERTICAL_RADIAL)])
    else:
        th = rng.uniform(0, 2 * np.pi)
        direction = np.array([np.cos(th), np.sin(th), rng.uniform(-VERTICAL_UNIFORM, VERTICAL_UNIFORM)])
    step = bool(rng.random() < STEP_P)
    if step:
        amp = min(STEP_MIN_MM * (1 - rng.random()) ** (-1 / STEP_ALPHA), STEP_MAX_MM)
        rise, reverse = 1.0, 0.0
    else:
        amp = float(np.exp(rng.uniform(*np.log(SLOW_MM))))
        rise = float(np.exp(rng.uniform(*np.log(RISE_DAYS))))
        reverse = rise * rng.uniform(*REVERSE_FACTOR) if rng.random() < REVERSE_P else 0.0
    onset = rng.uniform(-ONSET_BEFORE * length, length)
    return Transient(radius, (float(rc * np.cos(az)), float(rc * np.sin(az))), radial, direction,
                     float(amp), float(onset), rise, reverse)


def seen(signal: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """n bool: the range of signal (n×L×3 mm) over each station's days in
    ok (n×L), as a norm over components, is at least SEEN_MM."""
    m = ok[..., None]
    span = np.where(m, signal, -np.inf).max(axis=1) - np.where(m, signal, np.inf).min(axis=1)
    span = np.where(ok.any(axis=1)[:, None], span, 0.0)
    return np.linalg.norm(span, axis=-1) >= SEEN_MM
