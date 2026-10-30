"""Province summaries for the website's "ข้อมูลเชิงลึกรายจังหวัด" page -> province_data.js.

    python export_province_web.py [--env real_env] [--out province_data.js]

Everything here is real data, averaged over each province's coastal waters:
  - SST anomaly (deg C): NOAA OISST v2.1 daily, OISST's own anomaly, averaged per month.
  - heat_3m (deg C x months): positive SST anomaly summed over this month and the two before,
    the same heat term the forecast model uses (bycatch_model/forecast.py), here ending this month.
  - CHL-a (mg m-3): NOAA VIIRS monthly; the anomaly is % vs that calendar month's mean over all
    years on file (2022 onward), because there is no longer satellite climatology in real_env.
  - Wind (m s-1): ERA5 monthly mean 10 m wind speed.
  - Landings (t) and licensed vessels: Department of Fisheries open data (real_env/dof_stats.json).

Province waters are approximate: sea pixels (GEBCO/ETOPO bathymetry below 0 m) within 50 km of the
coast inside the study-area box, each given to the province of the nearest main landing port.
There are no province sea boundaries in the data, so this is a stand-in.

Warning score (0-100) used to rank provinces each month: the mean of up to four parts, each the
percentile of this value among all province-months on file (so 80 = higher than 80% of the record):
  heat   heat_3m                       (more accumulated heat = worse)
  chl    CHL-a anomaly, reversed       (less chlorophyll than usual = worse)
  catch  landings change vs a year ago, reversed (DOF; missing after the last DOF month)
  fleet  licensed vessels per 100 km2 of province waters, that year (latest year on file after it)
Parts with no data that month are left out of the mean. A part >= 75 counts as a warning sign.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import warnings
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.stats import rankdata

from web_stamp import stamp_index

LAT0, LAT1, LON0, LON1 = 6.30, 11.50, 99.00, 102.20    # the website's study-area box
BUFFER_KM = 50.0
MAX_PORT_KM = 160.0
KM_PER_DEG = 111.32

# Main landing ports (same as export_effort_web.py). Prachuap is only used so its waters are not
# handed to Chumphon; it is not shown on the page (only a sliver of it is inside the box).
PORTS = [
    ("chumphon", "ชุมพร", "ชุมพร", 10.35, 99.28),
    ("surat", "สุราษฎร์ธานี", "สุราษฎร์ฯ", 9.25, 99.45),
    ("nakhon", "นครศรีธรรมราช", "นครศรีฯ", 8.36, 100.22),
    ("songkhla", "สงขลา", "สงขลา", 7.22, 100.60),
    ("pattani", "ปัตตานี", "ปัตตานี", 6.90, 101.27),
    ("narathiwat", "นราธิวาส", "นราธิวาส", 6.43, 101.83),
    ("prachuap", "ประจวบคีรีขันธ์", "ประจวบฯ", 11.20, 99.55),
]
SHOWN = [p[0] for p in PORTS if p[0] != "prachuap"]


def month_key(days):
    d = dt.date(1970, 1, 1) + dt.timedelta(days=int(days))
    return f"{d.year:04d}-{d.month:02d}"


def nearest_index(axis, values):
    """Index of the nearest axis point for each value (axis ascending)."""
    i = np.clip(np.searchsorted(axis, values), 1, len(axis) - 1)
    return np.where(np.abs(values - axis[i - 1]) <= np.abs(values - axis[i]), i - 1, i)


def province_zones(bathy):
    lat, lon, elev = bathy["lat"], bathy["lon"], bathy["elevation"]
    sea = elev < 0
    dy = abs(lat[1] - lat[0]) * KM_PER_DEG
    dx = abs(lon[1] - lon[0]) * KM_PER_DEG * np.cos(np.radians(lat.mean()))
    dist_coast = ndimage.distance_transform_edt(sea, sampling=(dy, dx))
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    keep = sea & (dist_coast <= BUFFER_KM) & (LA >= LAT0) & (LA <= LAT1) & (LO >= LON0) & (LO <= LON1)

    d_port = np.stack([np.hypot((LA - plat) * KM_PER_DEG, (LO - plon) * KM_PER_DEG * np.cos(np.radians(plat)))
                       for _, _, _, plat, plon in PORTS])
    owner = d_port.argmin(axis=0)
    keep &= d_port.min(axis=0) <= MAX_PORT_KM
    pixel_km2 = dy * KM_PER_DEG * abs(lon[1] - lon[0]) * np.cos(np.radians(LA))

    zones = {}
    for k, (key, *_rest) in enumerate(PORTS):
        m = keep & (owner == k)
        zones[key] = {"lat": LA[m], "lon": LO[m], "area_km2": float(pixel_km2[m].sum()),
                      "depth_m": float(-elev[m].mean())}
    return zones


def zone_mean(field, f_lat, f_lon, zone):
    """Mean of a (lat, lon) field over a zone's pixels, sampled at the nearest field cell."""
    i = nearest_index(f_lat, zone["lat"])
    j = nearest_index(f_lon, zone["lon"])
    v = field[i, j]
    return float(np.nanmean(v)) if np.isfinite(v).any() else None


def percentiles(values):
    """Percentile (0-100) of each value among the non-missing ones; ties share the mean rank."""
    v = np.array([np.nan if x is None else x for x in values], dtype=float)
    ok = np.isfinite(v)
    out = np.full(v.shape, np.nan)
    if ok.sum() > 1:
        out[ok] = 100.0 * (rankdata(v[ok]) - 1) / (ok.sum() - 1)
    return out


def latest(by_year, year):
    years = [int(y) for y in by_year if int(y) <= year] or [int(y) for y in by_year]
    return by_year[str(max(years))] if years else None


def rnd(x, n):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="real_env")
    ap.add_argument("--out", default="province_data.js")
    args = ap.parse_args()
    env = Path(args.env)

    sst = np.load(env / "sst_daily.npz")
    chl = np.load(env / "chl_monthly.npz")
    wind = np.load(env / "wind_monthly.npz")
    bathy = np.load(env / "bathy.npz")
    dof = json.loads((env / "dof_stats.json").read_text(encoding="utf-8"))
    zones = province_zones(bathy)

    # Monthly mean SST anomaly grids from the daily file.
    sst_months = np.array([month_key(t) for t in sst["time"]])
    months = sorted(set(sst_months) | {month_key(t) for t in chl["time"]})
    warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
    anom_grid = {m: np.nanmean(sst["sst_anom"][sst_months == m], axis=0) for m in set(sst_months)}
    sst_grid = {m: np.nanmean(sst["sst"][sst_months == m], axis=0) for m in set(sst_months)}
    chl_idx = {month_key(t): k for k, t in enumerate(chl["time"])}
    wind_idx = {month_key(t): k for k, t in enumerate(wind["time"])}

    out = {}
    for key in SHOWN:
        z = zones[key]
        s_anom = [zone_mean(anom_grid[m], sst["lat"], sst["lon"], z) if m in anom_grid else None for m in months]
        s_mean = [zone_mean(sst_grid[m], sst["lat"], sst["lon"], z) if m in sst_grid else None for m in months]
        c = [zone_mean(chl["chl"][chl_idx[m]], chl["lat"], chl["lon"], z) if m in chl_idx else None for m in months]
        w = [zone_mean(wind["wind"][wind_idx[m]], wind["lat"], wind["lon"], z) if m in wind_idx else None for m in months]

        heat = []
        for k in range(len(months)):
            last3 = [s_anom[i] for i in range(max(0, k - 2), k + 1)]
            heat.append(sum(max(a, 0.0) for a in last3) if len(last3) == 3 and None not in last3 else None)

        # CHL-a anomaly: % vs the same calendar month's mean over the years on file.
        clim = {}
        for m, v in zip(months, c):
            if v is not None:
                clim.setdefault(m[5:], []).append(v)
        c_anom = [None if v is None else 100.0 * (v / np.mean(clim[m[5:]]) - 1.0) for m, v in zip(months, c)]

        land = dof["landings_t"].get(key, {})
        landings = [land.get(m) for m in months]
        yoy = []
        for m, v in zip(months, landings):
            prev = land.get(f"{int(m[:4]) - 1}{m[4:]}")
            yoy.append(None if v is None or not prev else 100.0 * (v / prev - 1.0))

        _, name, short, plat, plon = next(p for p in PORTS if p[0] == key)
        out[key] = {
            "name": name, "short": short, "port": [plat, plon],
            "area_km2": round(z["area_km2"]), "depth_m": round(z["depth_m"], 1),
            "vessels": dof["vessels"].get(key, {}),
            "sst": [rnd(v, 2) for v in s_mean],
            "sst_anom": [rnd(v, 2) for v in s_anom],
            "heat_3m": [rnd(v, 2) for v in heat],
            "chl": [rnd(v, 3) for v in c],
            "chl_anom": [rnd(v, 1) for v in c_anom],
            "wind": [rnd(v, 2) for v in w],
            "landings_t": [rnd(v, 0) for v in landings],
            "landings_yoy": [rnd(v, 1) for v in yoy],
        }

    # Warning score: percentile of each part over every province-month, then the mean.
    keys, n = SHOWN, len(months)
    def flat(get):
        return [get(out[k], i) for k in keys for i in range(n)]
    density = lambda p, i: (latest(p["vessels"], int(months[i][:4])) or 0) * 100.0 / p["area_km2"]
    raw = {
        "heat": flat(lambda p, i: p["heat_3m"][i]),
        "chl": flat(lambda p, i: None if p["chl_anom"][i] is None else -p["chl_anom"][i]),
        "catch": flat(lambda p, i: None if p["landings_yoy"][i] is None else -p["landings_yoy"][i]),
        "fleet": flat(density),
    }
    pct = {part: percentiles(v) for part, v in raw.items()}
    for a, k in enumerate(keys):
        out[k]["fleet_density"] = [rnd(density(out[k], i), 2) for i in range(n)]
        parts = {part: [rnd(pct[part][a * n + i], 0) for i in range(n)] for part in pct}
        out[k]["parts"] = parts
        out[k]["score"] = [rnd(np.mean(v), 0) if (v := [parts[q][i] for q in parts if parts[q][i] is not None]) else None
                           for i in range(n)]

    last_landing = max((m for k in SHOWN for m in dof["landings_t"].get(k, {})), default=None)
    meta = {
        "generated": dt.date.today().isoformat(),
        "buffer_km": BUFFER_KM,
        "sst_last_day": (dt.date(1970, 1, 1) + dt.timedelta(days=int(sst["time"][-1]))).isoformat(),
        "landings_last_month": last_landing,
        "wind_partial": [str(x) for x in wind["partial"]] if "partial" in wind.files else [],
        "chl_source": str(chl["source"]) if "source" in chl.files else "NOAA VIIRS",
        "bathy_source": str(bathy["source"]) if "source" in bathy.files else "GEBCO/ETOPO",
        "dof_source": dof["meta"]["source"],
    }
    payload = {"meta": meta, "months": months, "provinces": out}
    text = ("/* Generated by export_province_web.py: do not edit by hand. */\n"
            "window.PROVINCE_DATA = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n")
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"Wrote {args.out}: {len(SHOWN)} provinces x {len(months)} months ({months[0]}..{months[-1]})")
    for key in SHOWN:
        print(f"  {key:11s} {out[key]['area_km2']:7d} km2  depth {out[key]['depth_m']:5.1f} m")
    stamp_index(args.out)


if __name__ == "__main__":
    main()
