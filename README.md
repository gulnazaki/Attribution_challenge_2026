# Extreme Event Attribution Challenge — ELLIS Winter School 2026

## Overview

This challenge focuses on **probabilistic extreme event attribution**: given an observed
extreme temperature event, how much did human-induced climate change increase its
probability? Participants evaluate existing methods, diagnose their statistical
limitations, and develop improved approaches.

The central quantity of interest is the **Probability of Necessity (PN)**:

$$\text{PN} = \max\left(0,\ 1 - \frac{P(X > u \mid \text{counterfactual})}{P(X > u \mid \text{factual})}\right)$$

where $u$ is the event threshold, the *factual* world reflects current forced climate,
and the *counterfactual* world reflects pre-industrial (no anthropogenic forcing)
conditions.

---

## Project structure

```
├── src/
│   ├── data_utils.py      Event detection, extraction, SLP slicing utilities
│   ├── attribution.py     PN kernel + thermodynamic and dynamical adjustment methods
│   ├── analogues.py       Analogue-based attribution methods (KNN variants)
│   ├── visualization.py   QQ plot and time-evolution evaluation figures
│   └── config.py          Shared plot style colours
│
├── 01_extraction.ipynb    Step 1 — extract events from NetCDF, save to data/
├── 02_attribution.ipynb   Step 2 — run methods, save CSV to results/
├── 03_evaluation.ipynb    Step 3 — load CSV, produce figures in figures/
│
├── data/                  Pre-extracted .pkl files (output of 01_extraction)
├── results/               Attribution CSVs (output of 02_attribution, timestamped)
└── figures/               Evaluation figures (output of 03_evaluation, timestamped)
```

---

## Workflow

### Step 1 — Extract data (`01_extraction.ipynb`)

Edit the `CONFIG` block at the top of the notebook (paths, variable names, percentile,
number of members) then **Run All**.

Each ensemble member produces a dict with the following keys:

| Key | Shape | Description |
|---|---|---|
| `member` | str | Member identifier (e.g. `r1i1p1f1`) |
| `times` | `(T,)` | Datetime axis |
| `gmt4_f` | `(T,)` | 4-year rolling-mean factual GMT anomaly |
| `location` | `(n_events, 2)` | Event barycentres `[lat, lon]` |
| `slp_lat` | `(n_lat,)` | SLP latitude axis |
| `slp_lon` | `(n_lon,)` | SLP longitude axis |
| `idx_f` / `idx_c` | `(n_events,)` | Time indices of factual / counterfactual events |
| `f_tas` / `c_tas` | `(T, n_events)` | Area-averaged temperature time series |
| `f_tas_vals` / `c_tas_vals` | `(n_events,)` | Threshold at each event timestep |
| `f_slp` / `c_slp` | `(T, n_lat, n_lon)` | **Global SLP field — 3-D, not pre-sliced** |
| `event_frequency_map` | `(lat, lon)` | Pixel-wise event count |

> **Design choice:** The SLP field is stored as a 3-D array together with coordinate
> axes (`slp_lat`, `slp_lon`). Local sub-domains are **not** pre-computed.
> Use `extract_local_slp()` from `src/data_utils.py` to slice any box you want
> at attribution time, giving full flexibility over box size and location.

Output files are saved to `data/` with the naming convention:
```
extracted_{var}_nmem{N}_start{Y}_p{P}.pkl
```

### Step 2 — Attribution (`02_attribution.ipynb`)

Set `DATA_PATH` and `RESULTS_DIR` at the top of the notebook, then run all cells.
Results are saved to `results/attribution_{YYYYMMDD_HHMM}.csv`.

#### Adding a custom method

A template and registration instruction are provided directly in the notebook.
The minimal pattern is:

```python
# 1. Define your function
def run_my_method(tas_f, slp_f, slp_lat, slp_lon,
                  ev_lat, ev_lon, obs_val, t_range, mth):
    # Build your SLP representation, e.g. a local box:
    slp_local = extract_local_slp(slp_f, slp_lat, slp_lon,
                                   ev_lat, ev_lon, half_width_deg=20.0)
    # Construct counterfactual temperature distribution
    tas_cf = ...
    t0, t1 = t_range
    return compute_pn(tas_f[t0:t1], tas_cf[t0:t1], obs_val, method=mth)

# 2. Register it (add to the dict already defined in the notebook)
ATTRIBUTION_METHODS['pn_my_method'] = lambda ctx: run_my_method(
    ctx['tas_f'], ctx['slp_f'], ctx['slp_lat'], ctx['slp_lon'],
    ctx['ev_lat'], ctx['ev_lon'], ctx['val'], ctx['range'], ctx['mth'])
```

#### Output CSV columns

| Column | Description |
|---|---|
| `member`, `event_id`, `scenario` | Event identifier |
| `time`, `lat`, `lon` | Event metadata |
| `pn_ensemble_{mth}` | Ground-truth PN from the full ensemble |
| `pn_thermo_ML_{mth}` | Thermodynamic adjustment |
| `pn_dyn_adj_global_pca_{mth}` | Global SLP + PCA dynamical adjustment |
| `pn_dyn_adj_local_25_{mth}` | Local 25×25 deg box, raw Ridge |
| `pn_dyn_adj_local_50_pca_{mth}` | Local 50×50 deg box + PCA |
| `pn_analogues_lasso_{mth}` | KNN analogues with Lasso feature selection |
| `pn_{custom}_{mth}` | Any custom method registered in the notebook |

`{mth}` is one of `empirical`, `gaussian`, `gev`.

### Step 3 — Evaluation (`03_evaluation.ipynb`)

Set `RESULTS_DIR` and optionally `RESULTS_PATH` (defaults to the latest CSV in
`results/`). Edit `algo_groups` to match the columns you want to compare, then
**Run All**.

Two figures are produced and saved to `figures/`:
- `qq_analysis_{timestamp}.png` — log-log QQ plot (Type I error control + power curves)
- `time_evolution_{timestamp}.png` — rolling yearly Type I error rate and power

---

## Implemented methods

### `src/attribution.py`

| Function | Description |
|---|---|
| `compute_pn` | PN kernel supporting `empirical`, `gaussian`, and `gev` tail estimators |
| `run_thermo_ml` | Thermodynamic adjustment: removes GMT trend via linear regression |
| `run_dyn_adj_global_pca` | Ridge on global SLP compressed by PCA |
| `run_dyn_adj_local` | Ridge on a raw local SLP box (no PCA) |
| `run_dyn_adj_local_pca` | Ridge on a local SLP box compressed by PCA |

### `src/analogues.py`

| Function | Description |
|---|---|
| `run_analogues` | Standard KNN on SLP features with optional GMT detrending |
| `run_analogues_lasso` | KNN restricted to Lasso-selected SLP dimensions |
| `run_analogues_causal` | KNN restricted to Mutual-Information-selected dimensions |
| `run_analogues_ridge` | Local Ridge bias-correction inside the KNN pool |

### `src/data_utils.py`

| Function | Description |
|---|---|
| `detect_extreme_events` | Connected-component detection of extreme events |
| `extract_event_fast` | Vectorised area-averaged extraction + barycentre |
| `extract_local_slp` | Slice a lat/lon box from the 3-D SLP array at attribution time |
| `get_smoothed_gmt` | Centred rolling mean for monthly GMT series |
| `event_frequency_map` | Pixel-wise event count diagnostic |

---

## Getting started (Google Colab)

```python
from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.insert(0, '/content/drive/MyDrive/ellis-attribution/src')

from attribution import compute_pn, run_thermo_ml, run_dyn_adj_global_pca
from analogues   import run_analogues_lasso
from data_utils  import extract_local_slp
```

Run notebooks in order: `01_extraction` → `02_attribution` → `03_evaluation`.

---

## Research directions

### Type I error control
Do current methods keep the false-positive rate at or below the nominal α across
the full time period? The QQ plot directly tests this: a calibrated method's curve
follows the diagonal. Dynamical adjustment methods trained on factual data may
inadvertently capture part of the forced signal, inflating Type I error. Analogue
methods face non-stationarity: past analogues may no longer represent present
variability as the climate warms.

### Promising improvements
- **Conformal prediction** — distribution-free calibration that guarantees valid
  Type I error control by construction, without tail distribution assumptions.
- **Non-stationary counterfactual** — replace the constant GMT baseline with a
  proper pre-industrial trajectory from detection-and-attribution methods.
- **Better feature selection** — causal discovery (PC algorithm, NOTEARS) for a
  principled Markov Blanket in analogue matching.
- **Bias-corrected analogue pools** — quantile mapping or optimal transport to
  account for the mean-state shift between the past pool and the present event.
- **Multivariate attribution** — compound events (e.g. hot and dry) require
  multivariate PN estimation.

---

## Recommended reading

- Philip et al. (2020) — A protocol for probabilistic extreme event attribution
- Naveau et al. (2023) — Statistical methods for extreme event attribution in climate science
- Deser et al. (2020) — Insights from Earth system model large ensembles
- Mamalakis et al. (2022) — Carefully choose the baseline to compare climate forcing
