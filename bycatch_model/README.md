# EcoCast-style bycatch model

This package fits **one model per species** from purse-seine logbooks plus satellite covariates.
It predicts maps for each species and combines them into **one score per cell**, following the
NOAA EcoCast idea (Hazen et al. 2018): positive means good fishing with little bycatch, negative
means a bycatch hotspot.

It reads the schema in `purse_seine_sim/DATA_SPEC.md`, so the same command runs on simulated or real data.

```bash
python -m bycatch_model train --data sim_output --out model_output   # fit, test on last year, maps
python -m bycatch_model power --out power_output                     # how many sets are needed?
```

## Model

For each species, using the same covariates as the simulator outputs:

| Part | What it predicts | How |
|---|---|---|
| Presence (all species) | P(species caught in a set) | Logistic GAM |
| Amount (targets with ≥30 positives) | E[kg \| caught] | Log-normal GAM |
| Expected catch | P × E[kg \| caught] | Combination of the two |

- **GAM covariates:** cubic P-splines on SST, SST anomaly, log CHL-a, SST front strength, depth,
  distance to coast, seagrass index and wind. Linear terms for set type (FAD or light), light × moon,
  and a seasonal cycle.
- **Smoothing:** one smoothing strength per model, chosen by AIC. The fit uses step-halving so that
  data where presence and absence separate completely cannot make it diverge.
- **Implementation:** plain numpy, with no scikit-learn needed. EcoCast also used boosted regression
  trees, which can be swapped in later behind the same `fit_species` / `predict_presence` functions.

The EcoCast score is `0.40 × revenue − Σ w_b × P(bycatch_b)`. Each layer is scaled to 0–1 inside
the data footprint. Weights and fish prices in `ecocast.py` are **assumed**; agree them with DOF
and the fishers, as EcoCast did with managers.

The **data footprint** is the set of cells within 40 km of a training set. Cells outside it are
extrapolation and are greyed out.

Protected-species layers with fewer than **40 sets with a catch** in the training data are left out
of the score (`MIN_POSITIVES_SCORE`), because the power analysis shows their maps are noise. In the
10-vessel run this raised the score's agreement with the truth from 0.49 to 0.56.

## One-month-ahead forecast

```bash
python -m bycatch_model forecast --data sim_output --out forecast_output
python export_forecast_web.py --forecast forecast_output --data sim_output
```

`bycatch_model/forecast.py` predicts next month's P(caught) per species using only what is known at
the end of the current month: last month's SST, SST anomaly, CHL-a and wind at the cell, the heat
accumulated over the last 3 months (sum of positive SST anomaly, a monthly stand-in for degree heating
weeks), the change in anomaly, static habitat, and the seasonal cycle of the target month (dropped
when there is less than a year of training data, because it cannot be learned from part of a year).

- **Rolling validation:** for each month M, train on every set before M, forecast M, score against the
  sets fished in M (AUC, Brier) and against the true density map (simulations only).
- **Baselines:** *persistence* (catch rate within ~30 km last month) and *climatology* (same calendar
  month in earlier years). The forecast is only worth showing where it beats both.
- **Heat terms** (heat accumulation and anomaly change) are kept for a species only if they lower the
  Brier score by at least 0.5% compared with the same model without them.
- **Upcoming month:** a final model on all data forecasts the month after the last grid, with an
  uncertainty layer from refits on resampled trips (`--bootstrap`, default 10).
- **Response curves:** `response_curves.json` and `plots/response_curves.png` show how the odds of a
  catch change with each heat/climate input.

The web export keeps the 0.05 deg legal-fishing cells (no land, islands or 3 nm zone) and the
**Forecast** tab draws them as a grid, like the EcoCast map. The SST and wind inputs are ~0.25 deg, so
cell-level detail is not more precise than that.

## Real ocean data

`fetch_env_data.py` (repo root) cuts the area of interest out of real data:
NOAA OISST v2.1 daily SST (ERDDAP download, or already-downloaded PSL `sst.day.mean.YYYY.nc` files via
`--oisst-nc`), monthly satellite CHL-a (ERDDAP), ERA5 monthly 10 m wind (`--era5`, needs a CDS key)
and ETOPO 2022 bathymetry (`--bathy`, or a GEBCO file via `--bathy-nc`).
The simulator then runs on those fields (`--sst-npz`, `--chl-npz`, `--wind-npz`, `--bathy-npz`), so the
training data, the maps and the forecast all see the real ocean. Fish behaviour and the logbooks stay simulated.
`real_env/sst_daily.npz` holds the 2024 OISST cut-out. Without the PSL long-term-mean file
(`--oisst-ltm`) the SST anomaly is measured against the simulator's assumed climatology.

### Study area

With `--bathy-npz` the land mask, coastline, depth and distance to coast come from the bathymetry
(only the Gulf of Thailand side: the Andaman Sea in the south-west corner is masked out). The study
area is the band of sea within 50 km of the coast (`RegionConfig.coastal_buffer_km`). Vessels still
fish anywhere and every set is used for training, but prediction maps, skill scores, EcoCast and the
forecast cover only the study area. `python export_study_area_web.py` writes its outline and the
10-70 m isobaths to `study_area_data.js` for the website maps.

With a single year of data, `train` holds out the last 3 months instead of a year and still maps
every month.

## Website

`python export_ecocast_web.py` turns `model_output/maps/*.npz` into `ecocast_data.js`. The
**EcoCast · Bycatch** tab in `index.html` loads that file when the tab is opened. The page has:

- a monthly map of the EcoCast score, revenue, kg per set for each target species, and bycatch
  probability for each protected species;
- a click popup showing every species at a point;
- per-province sea-zone averages and the model-reliability table.

As in EcoCast, the score is rescaled to −1…+1 each month: it shows *relatively* good and bad areas.

## Validation

- **Held-out year:** train on every year except the last, then test on the last.
- **Test AUC:** how well P(catch) ranks the logbook's sets with and without catch.
- **Ceiling:** the AUC you would get from the *true* probabilities. A set's catch is partly luck,
  so even a perfect model cannot reach 1.
- **Map rho:** rank correlation between the predicted map and the true density map (simulations
  only), inside the footprint.

## Power analysis results (10 vessels, 2023–2025 simulated, test year 2025)

Test AUC by number of training sets:

| Species | 200 | 500 | 1,000 | 2,000 | 4,000 | 8,527 | ceiling |
|---|---|---|---|---|---|---|---|
| Indo-Pacific mackerel | 0.75 | 0.77 | 0.78 | 0.79 | 0.79 | 0.79 | 0.84 |
| Sardinella | 0.83 | 0.92 | 0.92 | 0.93 | 0.93 | 0.93 | 0.94 |
| Longtail tuna | 0.75 | 0.77 | 0.79 | 0.80 | 0.80 | 0.80 | 0.83 |
| Requiem sharks (ETP) | 0.70 | 0.71 | 0.72 | 0.74 | 0.74 | 0.74 | 0.77 |
| Devil rays (ETP) | 0.60 | 0.55 | 0.53 | 0.55 | 0.61 | 0.63 | 0.74 |
| Whale shark (ETP) | – | – | – | 0.67 | 0.51 | 0.70 | 0.89 |

- **Common species reach close to the ceiling with about 1,000–2,000 sets,** roughly 2–4
  vessel-years. That works out to about 200–500 sets that caught the species.
- **Devil rays need about 4,000 sets (about 40 sets with a catch).** Below that, their predicted
  maps are no better than noise (map rho 0.26–0.43); at 4,000 sets map rho reaches 0.70.
- **Whale sharks, turtles and Irrawaddy dolphins cannot be modelled from purse-seine logbooks
  alone,** even with 10 vessels over 3 years. Whale sharks reach only 17 sets with a catch, and
  their map is wrong. Leave such layers out of the score, or replace them with other data:
  strandings, tagging, or an expert-drawn hotspot layer.
- **The combined score reaches about 0.5 correlation with the truth.** It captures the broad
  gradients well. It misses the small patches, which the simulator generates as random
  patchiness that no covariate can explain. The remaining mismatch is mostly that fine-scale
  patchiness, plus the weak rare-species layers.

These numbers depend on the assumed species parameters. Repeat the analysis once real data
is available.
