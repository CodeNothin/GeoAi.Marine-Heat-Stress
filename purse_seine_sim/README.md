# Purse-seine bycatch simulator (southern Gulf of Thailand)

Generates **synthetic purse-seine logbooks** in the same format we will ask real
vessels to send. It covers sets, catch by species with zeros recorded, ETP bycatch, and
AIS/VMS tracks, plus satellite covariates and the hidden "truth". This lets us build and
test an EcoCast-style bycatch model before the real data arrives.

Every biological and behavioural number is an **ASSUMPTION** (see `config.py` and
`species.py`). The simulator tests the pipeline and answers how much data is needed.
It does not describe the real ocean.

## Quick start

Requires Python 3.10+ and numpy. matplotlib is needed only for `--plots`.

```bash
python -m purse_seine_sim --out sim_output --plots          # 4 vessels, 2024-2025 (~20 s)
python -m purse_seine_sim --vessels 10 --start 2023-01-01 --end 2025-12-31 --no-track
python -m purse_seine_sim --config my_config.json           # override any setting
python -m purse_seine_sim --bycatch-avoidance 3             # skippers avoid ETP hotspots
python -m purse_seine_sim --validate path/to/real_logbooks  # check real data against the schema
python -m purse_seine_sim --write-spec DATA_SPEC.md         # regenerate the data spec
python -m unittest discover -s tests -v
```

The run prints how many sets caught each species, and how many sets would be needed
to reach about 200 positive sets, the rule-of-thumb minimum for a species model.

## Output

```
sim_output/
  logbook/              <- what the VESSEL sends (see DATA_SPEC.md)
    vessels.csv  fads.csv  trips.csv
    sets.csv            one row per set, including water hauls  = model training rows
    catch.csv           one row per set x checklist species, zeros included
    etp_interactions.csv one row per protected animal (condition, fate, photo)
    track.csv           GPS pings (true_activity column is simulator-only)
  env/env_at_sets.csv   <- what OUR pipeline adds: satellite SST/CHL (cloud gaps), wind, depth...
  truth/                <- ONLY the simulator knows this. Use it to score models, never to train them
    truth_sets.csv      true density (lambda_*) and catch probability (p_present_*) per set
    truth_catch.csv     catch before logbook errors
    maps/truth_YYYY-MM.npz  monthly true environment + density grids (0.05 deg)
  reference/            species checklist, ASSUMED species parameters, schema.json
  plots/                quick-look PNGs (--plots)
  summary.json  config_used.json
```

Training rows = `logbook/sets.csv` + `logbook/catch.csv` (pivoted per species) +
`env/env_at_sets.csv`, joined on `event_id`.

## What is simulated

| Part | How |
|---|---|
| Geography | Coastline, islands and depth from real ETOPO 2022 bathymetry with `--bathy-npz` (otherwise a hand-digitised coastline and analytic depth). River mouths and seagrass areas are approximate. Depth, distance to coast and the 0-50 km coastal-buffer study area are on a 0.02 deg grid |
| Environment | Daily SST = monthly climatology + interannual AR(1) + the 2024 marine heatwave (assumed) + moving eddies. CHL-a falls offshore, rises at river plumes in the NE monsoon, and drops during heatwaves. Wind follows the monsoon, plus storms. Also SST fronts and moon phase |
| Species | 6 targets (Indo-Pacific mackerel, Indian mackerel, sardinella, scad, longtail tuna, kawakawa), trash fish, and 5 ETP groups (whale shark, mobulids, requiem sharks, green turtle, Irrawaddy dolphin). Each has habitat responses, seasonality, heat-stress avoidance and random patchiness |
| Vessels | Trips of 5–12 days from 5 ports. Free-school, own-FAD and light-luring sets (light luring only on dark-moon nights). Skippers pick spots by noisy knowledge of fish value minus steaming distance. Rules: no sets within 3 nm of the coast, Feb–May closure box (approximate), weather limits |
| Catch | One dominant school species plus mixed species, a net capacity cap, water hauls, juvenile fraction (higher at FADs), and ETP encounters as a Poisson process with group size and condition |
| Logbook errors | Optional kg rounding, mackerel misidentification, ETP under-reporting, and dropped pings (`reporting` config) |
| Real data | `--sst-npz` / `env.sst_override_npz` takes an npz (`time`, `lat`, `lon`, `sst[t,lat,lon]`, optional `sst_anom`); `--chl-npz` and `--wind-npz` do the same for CHL-a (mg/m3) and monthly wind (m/s). `fetch_env_data.py` writes all three from NOAA OISST, satellite CHL-a and ERA5 |

## Known limitations

- The coastline and closure box are approximate; replace them with official DOF/GEBCO data for real use.
- All species parameters are guesses. Fitting real data is the whole point.
- **Irrawaddy dolphins essentially never appear.** They live within ~12 km of the coast,
  while large purse seiners set 20–60 km offshore and are banned inside 3 nm. This is a
  real limitation of using purse-seine data alone: coastal species need other sources,
  such as strandings or small-scale fishery observers.
- Whale sharks and turtles show up only a few times per 1,000 sets. To model them you need
  fleet-scale data or presence-only sources (DMCR strandings, tagging, OBIS).
