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

A training sample is (target station, window, network):

1. Draw a target from the normal stations and a 256-day window of training
   days in which the target has data.
2. Thin the rest of the training network to a random density (§4.2).
3. Take the K = 16 nearest remaining stations within 300 km of the target.
   Fewer are padded and masked.
4. The target's series over the window is the training target. Its missing
   days are left out of the loss.

### 4.2 Network thinning

California (median nearest neighbour 9 km) is much denser than Türkiye
(about 60 km). To learn sparse networks from a dense one, each sample:

- thins the network to a median nearest-neighbour distance drawn
  log-uniformly from 10 to 100 km, and
- drops any station closer to the target than a minimum distance drawn from
  the same range.

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

The second rule stops the model from copying a nearby station, and it matches
the use case: a virtual station is placed where there is no station.

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

- **Scaling.** Each component is divided by a fixed scale from the training
  residuals (about 2.6, 2.7 and 6.4 mm for E, N and U).
- **Neighbour encoder.** A temporal convolution stack shared by all
  neighbours over each neighbour's series, availability mask and σ. A
  geometry embedding is added from east/north offsets, log distance and
  Fourier features of the offset. The output is one token per neighbour and
  time patch.
- **Denoiser.** A 1D U-Net over time on the noisy target. At every
  resolution, each time patch attends to the neighbour tokens of the same
  patch. The bottleneck adds temporal self-attention.
- **Conditioning dropout.** All neighbours are dropped from 10% of samples,
  which allows classifier-free guidance at sampling time.
- **Size.** 5–20 M parameters, bf16, trained on one RTX 3060 Ti (8 GB).

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
| Realism | power spectrum of single samples against the real station's (spectral index, white and flicker amplitudes); per-component scale; correlation with neighbours against distance |
| Overall | accuracy of a classifier trained to tell real windows from generated ones (0.5 is indistinguishable) |

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
  cli.py
tests/           unit tests on synthetic data
docs/            this document and later design notes
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
