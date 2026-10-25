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
