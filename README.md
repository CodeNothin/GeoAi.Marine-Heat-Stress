# Marine Heat Stress & Coastal Fishery Risk Intelligence

ระบบ GeoAI ประเมินความเสี่ยงประมงชายฝั่ง · ภาคใต้ฝั่งอ่าวไทย

A prototype website that combines satellite ocean data with Thai Department of Fisheries
statistics. It forecasts where fish will be next month, shows where fishing pressure is high, and
ranks the six southern Gulf of Thailand provinces (ชุมพร, สุราษฎร์ธานี, นครศรีธรรมราช, สงขลา,
ปัตตานี, นราธิวาส) by how closely they should be watched.

**Live site:** https://codenothin.github.io/GeoAi.Marine-Heat-Stress/

## What's on the site

| Tab | What it shows |
|---|---|
| ภาพรวม | Home page: highlights, live stats, fisheries news and alerts |
| พยากรณ์ปลา · Forecast | One-month-ahead fish distribution on a 0.05° grid. It is built from SST and ocean-colour fronts, then picks the best of four models (GLM, GAM, boosted trees, random forest), with SHAP explanations |
| เรือประมง · Fishing effort | Vessel density, fishing days and CPUE per grid cell, with "risk of overfishing" alerts where many boats meet few fish |
| ชุมชน · Forum | Demo forum for fishers and local government. Posts stay in your browser; there is no server |
| Real Data Pipeline | Upload a province-month SST CSV and get a heat-risk map |
| ข้อมูลเชิงลึกรายจังหวัด | Province ranking by a monthly warning score, the data behind each province's place ("ทำไม?"), and policy suggestions |
| แหล่งข้อมูล | Every dataset, news article and method we use, with citations |

## Real vs. simulated data

**Real:**
- Sea surface temperature and its anomaly: NOAA OISST v2.1
- Chlorophyll-a: NOAA VIIRS
- Wind: ERA5
- Depth and coastline: NOAA ETOPO 2022
- Licensed vessels, fishing days and landings: กรมประมง open data

**Simulated:** the purse-seine logbook (catch per set) that the forecast model is trained on. No
public logbook or VMS data exists for Thai purse seiners, so `purse_seine_sim` generates one.
It runs on the real ocean conditions above, but how each species responds to them is assumed.
The forecast therefore tests the pipeline. It is not a finding about real fish, and the site says so.

The province pages use only real data. Province waters are approximate: the sea within 50 km of
shore, split by nearest main port, because there are no official province sea boundaries in the data.

| Dataset | Source | Used for |
|---|---|---|
| SST + anomaly (daily, 0.25°) | [NOAA OISST v2.1](https://doi.org/10.1175/JCLI-D-20-0166.1) via CoastWatch ERDDAP | Forecast fronts, province heat index |
| CHL-a (monthly, 0.05°) | [NOAA CoastWatch VIIRS](https://coastwatch.noaa.gov/erddap) | Forecast colour fronts, province CHL-a drop |
| 10 m wind (monthly) | [ERA5](https://cds.climate.copernicus.eu), Copernicus CDS | Forecast, effort spread |
| Depth, coastline (0 m contour), distance to coast | [ETOPO 2022 60″](https://doi.org/10.25921/fd45-gt74) | Study area, forecast rings, province waters |
| Vessels, fishing days, landings | [กรมประมง strategy-stat](https://catalog.fisheries.go.th/group/strategy-stat) | Effort map, CPUE, province score |
| Logbook (biomass) | `purse_seine_sim/` (simulated) | Training the forecast |
| Basemap | © OpenStreetMap contributors | Map background only |

The full reference list, including method papers, is on the site's แหล่งข้อมูล tab.

## Repository layout

```
index.html                 the whole website (one page, GitHub Pages)
*_data.js                  generated data the site loads (forecast, effort, province, study area)
home_img/, monthly_maps/   images used by the site

fetch_env_data.py          download OISST, VIIRS CHL-a, ERA5 wind, ETOPO -> real_env/*.npz
fetch_dof_stats.py         download DOF statistics -> real_env/dof_stats.json
purse_seine_sim/           logbook simulator (see its README)
bycatch_model/             models; `map1` is the forecast the site uses (see its README)
export_*_web.py            turn model/data outputs into the *_data.js files
web_stamp.py               cache-busting ?v= hashes for the data files in index.html
tests/                     unit tests for the simulator and models
```

## Rebuilding the data

Needs Python 3.10+ and `pip install -r requirements.txt`. ERA5 also needs a free
[CDS API key](https://cds.climate.copernicus.eu/how-to-api).

```bash
# 1. Real data (only missing months are downloaded)
python fetch_env_data.py --start 2022-01 --end 2026-10 --out real_env --era5
python fetch_env_data.py --bathy --out real_env
python fetch_dof_stats.py

# 2. Simulated logbook on top of the real ocean (example settings)
python -m purse_seine_sim --out sim_output --start 2022-01-01 --end 2026-10-31 --no-track \
  --sst-npz real_env/sst_daily.npz --chl-npz real_env/chl_monthly.npz \
  --wind-npz real_env/wind_monthly.npz --bathy-npz real_env/bathy.npz

# 3. Forecast model (MAP 1)
python -m bycatch_model map1 --data sim_output --env real_env --out forecast_output

# 4. Website data files (each re-stamps index.html)
python export_study_area_web.py
python export_forecast_web.py --forecast forecast_output --data sim_output
python export_effort_web.py
python export_province_web.py
```

`sim_output/` and `forecast_output/` are git-ignored, but the `*_data.js` files are committed
because they are what the website reads.

Run the tests with `python -m unittest discover -s tests -v`.

## Viewing and deploying

Open `index.html` through any local web server, for example `python -m http.server`, and go to
http://localhost:8000. Pushing to `main` deploys the site to GitHub Pages (branch `main`, root folder) within about a minute.
Always commit `index.html` together with any regenerated `*_data.js`. GitHub Pages caches files
for 10 minutes, and the `?v=` stamps are what make browsers fetch the new data.

## Limitations

- The forecast is trained on simulated catches, so its maps show what the model learned from the simulator's assumptions.
- Satellite data is about 0.25° (SST) or 0.05° (CHL-a), so single grid cells should not be read too precisely.
- DOF landings are published only up to December 2024. After that, the province score uses satellite data and vessel counts only.
- Province sea areas and the effort map's spread of boats across cells are modelled, not observed (there is no public AIS/VMS data).
- This is a decision-support prototype. It does not replace field surveys or expert judgement.
