# Perturbation stress test of a data-driven dekadal streamflow model

Code and figure scripts for:

> Nasridinov, R. B. (2026) Can tree-based streamflow models respond to climate
> change they never saw? A perturbation stress test in nine Central Asian
> mountain basins. Submitted to *Journal of Water & Climate Change*.

## What this does

A gradient-boosted ensemble that forecasts dekadal streamflow well is driven
with delta-change perturbations of temperature and precipitation, with no
retraining, and its response is compared against an uncalibrated degree-day
water balance forced identically.

The result is a dissociation. The model keeps its forecasting skill across an
observed warming period, but under +2 K of imposed warming it reproduces about
8% of the physically expected seasonal redistribution of runoff, with the wrong
sign in five of nine basins. Permutation importance explains why: lagged
discharge carries a median 44% of predictive reliance and temperature under 7%.

## Repository layout

```
src/core.py        data loading, features, perturbation, degree-day benchmark
src/run_all.py     runs all five experiments and writes results/*.csv
figures/fig1_study_area.py   study-area map (Python, geopandas + contextily)
matlab/jwcc_figures.m        Figures 2 to 6 (MATLAB)
data/              basin_<gauge>.csv, one per basin (see below)
results/           written by run_all.py
```

## Data

Nine gauges over a common window from October 1959 to December 1990
(1,125 dekads each):

| Gauge | River | Mountain system |
|---|---|---|
| 16279 | Chatkal | W. Tien Shan |
| 16290 | Pskem | W. Tien Shan |
| 16300 | Ugam | W. Tien Shan |
| 16936 | Naryn | C. Tien Shan |
| 17202 | Karatag | Hissar-Alay |
| 17211 | Sangardak | Hissar-Alay |
| 17288 | Zeravshan | Hissar-Alay |
| 16176 | Padshaata | Fergana |
| 16202 | Chadak | Kurama |

Each `data/basin_<gauge>.csv` has one row per dekad with columns:

```
date, Q, phase, CODE, T2M, PRCP, SNOW, MELT, SNOWF, PET, SWE_dif,
DDsum, T2M_roll3, PRCP_roll3, MELT_roll3
```

`phase` is the dekad of the water year (1 to 36). Discharge is in m³ s⁻¹.
Forcing variables are basin-averaged ERA5-Land, computed over the ERA5-Land
grid cells intersecting each catchment polygon rather than at a single grid
point, since a point at the outlet misrepresents the snow regime of the
contributing area.

Discharge observations come from the Central Asia Discharge compilation of
Soviet-era hydrometric records. ERA5-Land is freely available from the
[Copernicus Climate Data Store](https://cds.climate.copernicus.eu).

## Reproducing the results

```bash
pip install -r requirements.txt
python src/run_all.py --data data --out results
```

About 15 minutes on a single core. `SEED` is fixed, so the numbers reproduce
exactly. The script prints the headline figures at the end:

```
seasonal shift at T+2 K: model -1.35%, physical +26.37%, ratio -0.080
feature importance (median): lagged Q 44.3%, temperature 6.7%
```

## Figures

Figure 1 (study area) is produced by `figures/fig1_study_area.py`, which needs
the basin polygons in a GeoPackage with a `basins` and a `gauges` layer.

Figures 2 to 6 are produced by `matlab/jwcc_figures.m`, which reads
`results/stress_test.csv` and `results/parameter_sweep.csv`. Run it from the
`matlab/` directory after copying those two files there. Output is vector PDF
plus 600 dpi PNG.

## Relation to the companion study

The same basin set, forcing data and model configuration are used in a
companion study submitted to *Hydrology and Earth System Sciences* (preprint
EGUSPHERE-2026-5328), which asks a different question — the effect of
evaluation protocol on apparent forecast skill — over a different period. The
code for that study is archived separately.

## Licence

MIT. See `LICENSE`.
