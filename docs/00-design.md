# 00 — Design

imonet generates daily GNSS residual series for stations that do not exist
("virtual stations"), from the real stations around them. The goal is to
densify sparse networks: train and validate on California, where the network
is dense enough to hide real stations and check the guesses, and test on
Türkiye, where the network is sparse.

This document is the plan agreed before implementation. Anything that changes
it is recorded here first.

## 1. Why a generative model

A station's residual series is the sum of a part shared with its neighbours
(common-mode error, loading, tectonic transients) and a part that is its own
(monument and site noise, multipath, local effects).

In gnssdl's benchmark the best deterministic neighbour predictor removes only
part of a held-out station's variance: the median normalised error of the fast
part falls from 0.67 (predict zero) to about 0.49. The rest cannot be
predicted from neighbours. A regression model would therefore produce virtual
stations that are too smooth and too quiet.

A conditional diffusion model takes the shared part from the neighbours and
samples the station's own part from a learned distribution. Its samples
should be scored as samples: the mean of many samples against the truth, and
the spread and character of single samples against real stations.

## 2. Problem

**Input**

- A target coordinate (lat, lon).
- The K nearest real stations, each with:
  - its residual series over the window (E, N, U in mm),
  - its availability mask and formal σ,
  - its east and north offset from the target in km.

**Output**

- Samples of the target's residual series over the window, E, N, U in mm.

Only residuals are generated: the trajectory (velocity, seasonal terms, steps)
is out of scope.

No absolute coordinate enters the model, only offsets relative to the target,
so that a model trained in California can be applied in Türkiye.

## 3. Data

### 3.1 California

The gnssdl cube `data/cube/california.npz` is used as it is (`data/` is a
symlink to gnssdl's data directory). See gnssdl `docs/00-benchmark-contract.md`
§1 for how it is made. In short:

- Residuals after a constant + rate + annual + semi-annual + steps fit on
  2008–2019.
- Equipment steps are removed; earthquake offsets are kept.
- Outliers, noisy stations and co-located twins are removed.

It holds 1251 stations × 6844 days (2008-01-01 to 2026-09-26) × 3 components.

| Split | Days | Stations |
| --- | --- | --- |
| Train | `split_day == 0` (2008–2019), not `exclude` | `split_sta == 0` (1078) |
| Validation | `split_day == 1` (2020–2021) | held-out (173) and normal |
| Test | `split_day == 2` (2022 onward) | held-out (173) and normal |

**Held-out stations are removed from training completely.** They are neither
targets nor neighbours in any training sample, and they are never in the
input network when another station is the target. They exist only as
evaluation targets.

The `exclude` cells (the Ridgecrest window) are never used for training.

Decisions use the validation split. The test split is used once, for the
final report.

### 3.2 Türkiye

There is no Türkiye cube yet. `imonet build` makes one, standalone, with the
same steps as the California cube:

1. Download NGL's station table and step catalogue, then the IGS20 daily
   `.tenv3` series of every station in the Türkiye box with at least three
   years of data (about 175 stations).
2. Per station and component, fit constant + rate + annual + semi-annual +
   steps on days up to 2019-12-31. Stations with less than a year of data
   before 2020 use the gnssdl fallback (the first 730 days or 365 epochs) and
   are flagged in every score table, since their fit then overlaps scored days.
3. Remove equipment steps everywhere; keep earthquake offsets.
4. Apply the same outlier screening, station QC and twin removal as the
   California cube.

Türkiye is used for testing only: no training, no tuning, no choice made on
its scores. Scored days are 2020 onward, like California's validation and
test periods, so in both regions the scored residuals lie outside the
trajectory fit.

**Kahramanmaraş.** The 2023-02-06 Mw 7.8 Pazarcık and Mw 7.6 Elbistan
earthquakes are kept out of the main Türkiye scores. From 30 days before
the first mainshock to 365 days after it, no station is scored in the main
tables. That window is the Kahramanmaraş case study instead (§6.3).

## 4. Training

### 4.1 Samples

A training sample is (target station, window, network) (`imonet.data`):

1. Draw a target from the normal stations and a 256-day window lying
   entirely in the training days, in which at least half of the target's
   days are usable. Usable cells are available, not excluded, on normal
   stations (`Cube.trainable`); every such (station, window) pair is equally
   likely.
2. Thin the rest of the training network to a random density (§4.2).
3. Take the K = 16 nearest remaining stations within 300 km of the target,
   counting only stations with at least a quarter of the window usable, so
   no slot goes to a station that is down. Fewer are padded and masked.
4. The target's series over the window is the training target. Its missing
   days are left out of the loss.

Validation samples are drawn the same way from the held-out stations on
validation days, with the normal stations as the network; a fixed set of
them gives the validation loss.

### 4.2 Network thinning

California (median nearest neighbour 9 km) is much denser than Türkiye
(about 60 km). To learn sparse networks from a dense one, 80% of samples:

- thin the network to a median nearest-neighbour distance drawn
  log-uniformly from 10 to 100 km, and
- drop any station closer to the target than a minimum distance drawn
  log-uniformly from 1 to 100 km.

The other 20% use the unthinned network and keep every neighbour, as
full-density evaluation does (§6.1). Without them the model would almost
never see a close neighbour: even the lightest thinning removes close
pairs, yet 13% of held-out California stations have a normal station within
5 km. (The minimum distance was first planned as 10–100 km on every sample;
that left 1% of samples with a neighbour within 5 km. Now it is 3%.)

Thinning is greedy minimum-separation ("disk") thinning: stations are
visited in random order and kept if no kept station lies within a separation
chosen to reach the target spacing (`imonet.network.thin`). Random
subsampling was the first plan, but it keeps California's clustering and
cannot make a network like Türkiye's:

| Network | Stations | Nearest neighbour, km (10th / 50th / 90th percentile) |
| --- | --- | --- |
| Türkiye (box 35.8–42.2 N, 25.6–44.9 E, ≥ 3 years) | 174 | 29 / 60 / 77 |
| California, normal stations | 1078 | 3 / 9 / 23 |
| California, disk-thinned to 60 km (3 seeds) | 110–118 | 53–56 / 60–61 / 72–73 |
| California, randomly thinned to 60 km (3 seeds) | 11–23 | 11–28 / 62–86 / 117–176 |

Disk thinning matches Türkiye's median and upper tail. It has fewer close
pairs than Türkiye (10th percentile 55 km against 29 km); the effect of this
is checked on experiment C rather than tuned for.

The minimum distance stops the model from relying on a nearby station, and
it matches the use case: a virtual station is placed where there is no
station.

Thinnings are precomputed (`ThinningBank`): 64 spacings × 4 seeds, about 9 s
at start-up, since one thinning takes up to 0.1 s.

### 4.3 Planted transients in training

California's training years hold few transients, and the Ridgecrest window
is excluded. A model trained on them alone would learn that neighbours carry
only shared noise and would not pass on signal localised in space.

With probability p (to be tuned, starting at 0.5) a sample gets one planted
transient, written into the target and every station in the network:

- Gaussian spatial footprint, 10–300 km;
- amplitude 1–30 mm, random horizontal direction, random vertical share;
- raised-cosine rise over 5–120 days, held to the window end or reversed;
- a share of them as offsets (rise of one day) with heavy-tailed amplitudes,
  so that the model also learns to pass on coseismic steps.

This family is deliberately simpler than the fault-slip events of
experiment B (§6.2). B then tests generalisation to realistic events rather
than recall of the training family.

### 4.4 Model

A conditional diffusion model with EDM training and sampling (Karras et al.
2022).

- **Scaling.** Each component is divided by a fixed scale: the robust scale
  (1.4826 × MAD) of the trainable cells, 2.12, 2.24 and 5.60 mm for E, N and
  U. The residuals are heavy-tailed because earthquake offsets are kept
  (standard deviations 9.1, 15.3 and 10.3 mm; El Mayor–Cucapah reaches about
  200 mm), so the standard deviation would be set by a few events.
- **Neighbour encoder.** For each neighbour, a convolution over its series,
  availability and σ, then one residual block per U-Net level (with a
  stride-2 convolution between levels), shared by all neighbours. A
  geometry embedding (east and north offset / 100 km, log distance, sines
  and cosines of the offsets at 10, 30, 100 and 300 km) modulates every
  block (scale and shift). The output, normalised once per level, is a
  feature per neighbour, channel and time step at every U-Net resolution.
  It does not depend on the noise level, so sampling computes it once.
- **Denoiser.** A 1D U-Net over time on the noisy target plus the sine and
  cosine of the day of year, with channels 64, 128, 192, 256 at L, L/2, L/4,
  L/8 and two residual blocks per level, modulated by the noise-level
  embedding. After every residual block, and in the bottleneck, each time
  step attends to the K neighbours at the same time step (4 heads). A
  learned null token is always among the keys, so padding and dropped
  conditioning are handled the same way. Temporal self-attention runs in
  the bottleneck and the two coarsest decoder levels. 11.1 M parameters.
- **Conditioning dropout.** All neighbours are dropped from 10% of samples,
  which allows classifier-free guidance at sampling time.
- **EDM.** σ_data = 1 (the detrended, scaled targets have a standard
  deviation of about 0.9, with tails to about 100 from earthquake offsets);
  ln σ ~ N(−1.2, 1.2²) in training; loss weighted by (σ² + σ_data²)/(σ
  σ_data)² and averaged over the target days with data. Sampling: Heun's
  method on the probability-flow ODE, 32 steps from σ = 80 to 0.002 (ρ = 7),
  deterministic given the initial noise.
- **Training.** AdamW, learning rate 2·10⁻⁴ with 2,000 warm-up steps and
  cosine decay, batch 64, gradient clipping at 1, bf16, EMA of the weights
  (decay 0.999), which is the model used for sampling. Every 2,000 steps a
  validation loss on 1,024 fixed samples (held-out stations, validation
  days) at fixed noise levels (σ = 0.02 to 10) with fixed noise, and a
  checkpoint; runs resume from it. Speed on the RTX 3060 Ti: 142 ms per
  step compiled (`torch.compile`), 276 ms without, peak memory 5.7 GiB.
  The cross-attention, one query against 17 keys at every time step and 17
  layers, is memory-bound, which is why compiling halves the time.
- **`regress`** is the same network called with a zero input and noise
  level, trained with squared error on the target days with data.

Sampling draws 32 samples per target and window. All randomness in the
sampler comes from a seeded generator, so a run can be repeated exactly. The
paired runs of experiment B depend on this.

Windows are generated independently. Stitching them into long series is
future work.

## 5. Baselines

All written in this repository:

| Name | What it is |
| --- | --- |
| `zero` | predict zero |
| `idw` | distance-weighted mean of the neighbours (as gnssdl's M1) |
| `idw+noise` | `idw` plus white + flicker noise, with amplitudes fitted to the neighbours' departures from their own `idw` predictions |
| `regress` | the same network as the diffusion model, trained with mean squared error (one output, no noise) |

`idw+noise` is the simplest generative model to beat. `regress` shows what
diffusion adds beyond its own conditional mean.

Details (`imonet.baselines`):

- `idw` weights a neighbour by exp(−d²/2ℓ²)/σ̄², σ̄ its median formal σ over
  the window, over the neighbours with data on each day. ℓ is chosen from
  {10, 25, 50, 100, 200, 500 km, ∞} per density, by the fast nRMSE on
  validation days with 200 normal stations as targets (each predicted
  without itself), so the held-out stations play no part in it.
- `idw+noise` fits, per window and component, P(f) = A + B/f to the
  periodogram of each neighbour's departure from its own `idw` prediction
  (least squares over 1/64–0.5 cycles/day, A, B ≥ 0), takes the median A
  and B over the neighbours, and adds noise with that spectrum, made on
  twice the window length and cut so it does not wrap around.

## 6. Experiments

### 6.1 A — Held-out stations

**Targets.** The 173 held-out stations, predicted from the normal stations.

**Days.** Validation for decisions, test for the final report. The period is
tiled with 256-day windows. The last window is aligned to the period end, and
only days not already scored by an earlier window are scored in it.

**Density.** Every score is reported at four densities:

- the full network;
- thinned to a median nearest-neighbour distance of 25 km;
- thinned to 60 km (Türkiye-matched; **the main California result**);
- thinned to 100 km.

Thinning is disk thinning (§4.2) with fixed seeds; each thinned density is
repeated with five seeds.

**Frequency band.** Every score is reported on the fast part (periods under
about two months, as in gnssdl) and the slow part, and on the total.

**Scores.**

| Kind | Score |
| --- | --- |
| Point | normalised RMSE of the sample mean (gnssdl's measure) |
| Probabilistic | CRPS; energy score over E, N, U; coverage of the 50% and 90% intervals; spread against error |
| Realism | power spectrum of single samples against the real station's, by period band; per-component scale; correlation with neighbours against distance |
| Overall | accuracy of a classifier trained to tell real windows from generated ones (0.5 is indistinguishable) |

How they are computed (`imonet.metrics`, `imonet.evaluate`):

- **Normalisation and summary, as gnssdl** (contract §4). Errors are in
  units of the station's robust training-years scale; every score is
  computed per station (at least 20 scored days) and summarised by the
  median over stations; a component-free name (`nrmse_fast`) is the mean of
  the E, N, U medians. The same stations and days are left out: days within
  ±30 of an M ≥ 6 earthquake NGL lists for the station (fast and slow parts
  only), and stations failing gnssdl's scoring QC (19 of the 173 held-out
  stations on the California cube). Check: `zero` scores a fast nRMSE of
  0.729 on the held-out stations, gnssdl's M0 0.728.
- **Fast and slow parts.** The slow part is the 61-day running median over
  the scored days. nRMSE splits the error of the sample mean, as gnssdl.
  CRPS and coverage on the fast part split the truth and each sample
  separately (a median of differences is not a difference of medians).
- **CRPS and energy score** use the fair ensemble estimators (unbiased for
  the members' distribution). A one-sample method is a point forecast: its
  CRPS is its absolute error, and coverage, spread and realism scores are
  not defined for it.
- **Coverage** uses Weibull plotting positions (the i-th of m sorted members
  at i/(m+1)), under which a draw from the members' distribution is inside
  a 90% interval 90% of the time. Numpy's default positions give about 85%
  for 32 members.
- **Spread against error**: ensemble spread (with the (m+1)/m finite-
  ensemble factor) divided by the RMSE of the sample mean, total part; 1 is
  calibrated.
- **Spectra**: periodograms of 256-day segments of the scored period with at
  least 90% of days (gaps filled linearly, line removed, Hann taper), as
  log10 of sample over truth in bands of 2–4, 4–8, …, 64–128 days; samples
  are cut to the truth's days first. **Scale**: log2 of the robust scale of
  each sample's fast part over the truth's.
- **Neighbour correlation**: correlation of the fast parts of target and
  neighbour over each window, truth and samples, averaged by distance bin
  (0–10, 10–25, 25–50, 50–100, 100–200, 200–300 km).
- **The classifier test is built in step 4**, with the diffusion model.

Results are written to `runs/A/<method>/<split>/<density>_s<seed>/` (not
`data/`, which is shared with gnssdl); `imonet report` prints the table.

### 6.2 B — Planted realistic events (California only, for now)

Events are fault slip on real fault surfaces of the SCEC Community Fault
Model 7.0, ported from gnssdl's event library:

- triangular dislocations in an elastic half-space (`cutde`);
- templates for shallow slow slip and afterslip, with length, depth, slip and
  timing drawn per event.

**Procedure**

1. Place an event on a fault, with a start day in the scored period.
2. Add its displacement to every station's real data, including the
   held-out targets. A target's truth is its real data plus the event.
3. Generate each held-out target twice with the same seeds: once from the
   planted network, once from the original. The difference of the two sample
   means is what the model passed on from the event.

**Scores, per target and event**

| Score | Meaning |
| --- | --- |
| ρ | signal recovered: ⟨d, s⟩ / ⟨s, s⟩, with d the paired difference and s the planted signal at the target (1 = all of it) |
| Shape | correlation of d with s over the event window |
| Timing | error of the onset found in d |
| Spurious | size of d at targets outside the event footprint |

**Main figure.** ρ against event footprint divided by neighbour spacing, at
each density. It shows when densification can recover an event and when it
cannot.

Türkiye events (a Türkiye fault database and simplified planar faults) come
later.

### 6.3 C — Türkiye

The frozen model, with no tuning.

**Main tables.** Every Türkiye station is a target in turn, predicted from
all the others. The scores of §6.1 are computed on 2020 onward without the
Kahramanmaraş window, at Türkiye's own density.

**Kahramanmaraş case study.** In the excluded window, each station near the
rupture is generated from the others. Scores:

- coseismic offsets of the virtual station against the real ones;
- postseismic signal over the year after the mainshocks;
- how both degrade with distance from the rupture and from the nearest
  neighbour.

The offsets reach metres near the rupture, far beyond anything in training,
so this tests the input scaling as much as the model. If samples break down
at these amplitudes, the fallback to compare is scaling each window by its
neighbours' range rather than by a fixed constant. Any change is recorded
here first.

## 7. Code layout

```
src/imonet/
  cube.py        load a cube; distance, azimuth and ENU offsets
  build.py       Türkiye: NGL download, tenv3 parser, trajectory fit, screening
  network.py     thinning; neighbour selection for any coordinate
  data.py        window sampler, torch Dataset
  transients.py  Gaussian (training) and fault-slip (evaluation) events
  model/         neighbour encoder, U-Net denoiser, EDM loss and sampler
  baselines.py   zero, idw, idw+noise, regress
  metrics.py     CRPS, coverage, spectra, classifier test, ρ, spurious signal
  train.py
  evaluate.py
  figures/       figures and tables for the report and the paper (`imonet figures`)
  cli.py
tests/           unit tests on synthetic data
docs/            this document and later design notes
paper/           report.tex (working report, drafted with Claude Code), paper.tex
                 (the paper, written by the authors), references.bib, figures/,
                 Makefile (`make`, `make figures`); Elsevier elsarticle format
```

Dependencies: numpy, pandas, scipy, torch, cutde.

## 8. Order of work

1. `cube`, `network`: load the cube, thinning, neighbour selection.
2. `data`: the training sampler.
3. `baselines`, `metrics`: experiment A for the baselines. This sets the bar
   before any model is trained.
4. `model`, `train`: the diffusion model and `regress`; first training run;
   experiment A.
5. `transients`: training augmentation and the fault-slip events;
   experiment B.
6. `build`: the Türkiye cube; experiment C.

## 9. Open questions

- The probability p and the amplitude range of planted training transients
  (§4.3), tuned on validation.
- Window length: 256 days to start. Longer windows reach more of the slow
  part.
- Whether classifier-free guidance helps, and at what weight.
- How to scale inputs for very large offsets (§6.3).
- Excess variability of the first diffusion model (`diffusion-v1`, validation
  2026-10-16): calibrated in the bulk, but rare large departures (slow drifts
  of up to 10 mm, jumps) make its ensembles 21–26% too wide over the whole
  series and its 99.9th percentile 1.5× the truth's. Candidates, to compare
  on validation: screen samples with the cube's running-median rule (the
  truth was screened, the samples are not); train on residuals without the
  estimated earthquake offsets (`qstep_*`), as gnssdl's learned models do;
  classifier-free guidance with weight > 1.
