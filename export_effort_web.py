"""Export the monthly ship-density / fishing-effort map, CPUE and the "risk of overfishing" alert to
effort_data.js for the Fishing effort tab in index.html.

    python fetch_dof_stats.py                     # real DOF vessel, effort and landing statistics
    python -m bycatch_model forecast --data sim_output --out forecast_output
    python export_effort_web.py

Fishing effort (fishing days per 0.05 deg cell per month):
  - How much: REAL. Each port's fleet is its province's licensed Gulf-of-Thailand vessels that year,
    and the days each vessel fishes that month are the Gulf-wide recorded fishing days / licensed
    vessels (Department of Fisheries, real_env/dof_stats.json).
  - Where: MODELLED (no public AIS/VMS for these boats). Each port's fishing days are spread over the
    sea cells by distance from port x depth preference x where the fish were last month x wind, with
    the spawning closure shutting its box; effort it displaces goes elsewhere. Gamma/Poisson noise
    with a fixed seed. Replace this with gridded AIS/VMS fishing hours (e.g. Global Fishing Watch)
    when available.
  - The upcoming month has no DOF effort yet: same month last year's days per vessel, latest fleet,
    no noise.
Fish and CPUE come from the forecast: expected kg per purse-seine set = sum over the target species
of P(catch) x mean kg in a set that caught it (logbook). Predicted CPUE (t of target species per
fishing day, any gear) = that x one factor fitted so predicted catch matches the DOF province landings.
Risk of overfishing: a cell with effort in the top quarter of all cell-months while its forecast
fish is in the bottom quarter of that month's map (High), or effort >= P60 and fish <= P40 (Watch).
"""
from pathlib import Path
import argparse
import base64
import calendar
import csv
import json
from datetime import date

import numpy as np

from export_forecast_web import encode_u8
from web_stamp import stamp_index

FORECAST_DIR = Path("forecast_output")
SIM_DIR = Path("sim_output")
DOF_FILE = Path("real_env/dof_stats.json")
OUT_FILE = Path("effort_data.js")
TARGETS = ["RAST_BRA", "RAST_KAN", "SARD_GIB", "DECA_MAR", "THUN_TON", "EUTH_AFF"]

# Landing ports (approximate, the simulator's plus two neighbours) and the share of the province's
# licensed fleet based there that fishes inside the study area. `area` = study-area zone (index.html key).
PORTS = [
    {"name": "Chumphon", "name_th": "ชุมพร", "lat": 10.35, "lon": 99.28, "province": "chumphon", "share": 1.0},
    {"name": "Surat Thani", "name_th": "สุราษฎร์ธานี", "lat": 9.25, "lon": 99.45, "province": "surat", "share": 1.0},
    {"name": "Pak Phanang", "name_th": "ปากพนัง", "lat": 8.36, "lon": 100.22, "province": "nakhon", "share": 1.0},
    {"name": "Songkhla", "name_th": "สงขลา", "lat": 7.22, "lon": 100.60, "province": "songkhla", "share": 1.0},
    {"name": "Pattani", "name_th": "ปัตตานี", "lat": 6.90, "lon": 101.27, "province": "pattani", "share": 1.0},
    {"name": "Narathiwat", "name_th": "นราธิวาส", "lat": 6.43, "lon": 101.83, "province": "narathiwat", "share": 1.0},
    # Most of Prachuap's fleet works north of 11.5N, outside the study area (assumed share).
    {"name": "Bang Saphan", "name_th": "บางสะพาน", "lat": 11.20, "lon": 99.55, "province": "prachuap", "share": 0.35},
]
AREAS = ["chumphon", "surat", "nakhon", "songkhla", "pattani"]   # Narathiwat waters join Pattani
AREA_TH = {"chumphon": "ชุมพร", "surat": "สุราษฎร์ธานี", "nakhon": "นครศรีธรรมราช", "songkhla": "สงขลา",
           "pattani": "ปัตตานี"}

PORT_RANGE_KM = 45.0       # e-folding distance of a day trip / short trip from port
FAR_RANGE_KM = 150.0       # longer trips
FAR_SHARE = 0.25
DEPTH_PREF_M = (28.0, 16.0)
FISH_FOLLOW = 0.5          # 0 = boats ignore where the fish were, 1 = effort proportional to fish
CLOSED_EFFORT = 0.1        # effort left inside the spawning closure while it is in force
NOISE_SHAPE = 6.0
HIGH_Q, WATCH_Q = (75, 25), (60, 40)    # (effort percentile >=, fish percentile <=)
AREA_LEVEL_SHARE = 0.10                 # share of an area's cells flagged to flag the area


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def km_between(lat, lon, lat0, lon0):
    dy = (lat - lat0) * 111.2
    dx = (lon - lon0) * 111.2 * np.cos(np.radians((lat + lat0) / 2))
    return np.hypot(dx, dy)


def month_shift(m, k):
    y, mm = map(int, m.split("-"))
    i = y * 12 + mm - 1 + k
    return f"{i // 12}-{i % 12 + 1:02d}"


def closure_fraction(closure, month):
    """Share of the month's days inside the closure window (mm-dd, same year)."""
    y, mm = map(int, month.split("-"))
    days = np.arange(np.datetime64(f"{month}-01"), np.datetime64(month_shift(month, 1) + "-01"))
    md = np.array([str(d)[5:] for d in days])
    return float(np.mean((md >= closure["start_mmdd"]) & (md < closure["end_mmdd"])))


def latest(d, year):
    """Value for `year`, or the nearest earlier year available (keys are years as strings)."""
    years = sorted(int(k) for k in d)
    ok = [y for y in years if y <= year] or years[:1]
    return d[str(ok[-1])]


def pct_or_none(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--forecast", default=str(FORECAST_DIR))
    p.add_argument("--data", default=str(SIM_DIR), help="simulator folder: truth maps, config, logbook")
    p.add_argument("--dof", default=str(DOF_FILE))
    p.add_argument("--out", default=str(OUT_FILE))
    p.add_argument("--seed", type=int, default=20261030)
    a = p.parse_args()
    fdir, sdir, out_file = Path(a.forecast), Path(a.data), Path(a.out)
    rng = np.random.default_rng(a.seed)

    meta = json.loads((fdir / "forecast_meta.json").read_text(encoding="utf-8"))
    upcoming = meta.get("upcoming_month")
    months = list(meta["target_months"]) + ([upcoming] if upcoming else [])
    fmaps = {m: np.load(fdir / "maps" / f"forecast_{m}.npz") for m in months}
    first = fmaps[months[0]]
    lat, lon, mask = first["lat"], first["lon"], np.asarray(first["mask"], bool)
    res = float(lat[1] - lat[0])
    cells = np.flatnonzero(mask.ravel())
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    clat, clon = LAT.ravel()[cells], LON.ravel()[cells]
    cell_km2 = (res * 111.2) ** 2 * np.cos(np.radians(clat))

    truth_dir = sdir / "truth" / "maps"
    truth = {}

    def tget(month, key):
        if month not in truth:
            path = truth_dir / f"truth_{month}.npz"
            truth[month] = np.load(path) if path.exists() else None
        t = truth[month]
        return None if t is None else np.asarray(t[key], float).ravel()[cells]

    cfg = json.loads((sdir / "config_used.json").read_text(encoding="utf-8"))
    closures = cfg["regulation"]["closures"]
    sets_per_day = float(cfg["fleet"]["day_sets_mean"])
    dof = json.loads(Path(a.dof).read_text(encoding="utf-8"))

    # Mean kg in a set that caught the species (logbook): turns P(catch) into expected kg per set.
    pos = {c: [] for c in TARGETS}
    for r in read_csv(sdir / "logbook" / "catch.csv"):
        kg = float(r["catch_kg"] or 0)
        if r["species_code"] in pos and kg > 0:
            pos[r["species_code"]].append(kg)
    kg_pos = {c: float(np.mean(v)) for c, v in pos.items()}

    # Static part of where boats fish: distance from each port and depth.
    depth = tget(months[-2] if upcoming else months[-1], "depth")
    depth_w = np.exp(-0.5 * ((depth - DEPTH_PREF_M[0]) / DEPTH_PREF_M[1]) ** 2) * np.clip((depth - 2) / 5, 0, 1)
    port_w = []
    for pt in PORTS:
        d = km_between(clat, clon, pt["lat"], pt["lon"])
        port_w.append((1 - FAR_SHARE) * np.exp(-d / PORT_RANGE_KM) + FAR_SHARE * np.exp(-d / FAR_RANGE_KM))
    port_lat = np.array([pt["lat"] for pt in PORTS[:5]])
    port_lon = np.array([pt["lon"] for pt in PORTS[:5]])
    area_of = np.argmin([km_between(clat, clon, la, lo) for la, lo in zip(port_lat, port_lon)], axis=0)

    gulf_days = dof["gulf_effort_days"]

    def days_per_vessel(month):
        y = int(month[:4])
        if month in gulf_days:
            return gulf_days[month] / latest(dof["gulf_vessels"], y), "dof"
        prev = month_shift(month, -12)
        return gulf_days[prev] / latest(dof["gulf_vessels"], y - 1), "last_year"

    def fleet(pt, year):
        return latest(dof["vessels"][pt["province"]], year) * pt["share"]

    layers = {k: {} for k in ("effort", "density", "fish", "cpue")}
    month_info = {}
    for month in months:
        is_up = month == upcoming
        y, mm = map(int, month.split("-"))
        ndays = calendar.monthrange(y, mm)[1]
        dpv, dpv_src = days_per_vessel(month)

        prev_fish = sum(np.nan_to_num(tget(month_shift(month, -1), f"lambda_{c}")) for c in TARGETS)
        fish_w = (prev_fish / max(prev_fish.mean(), 1e-9)) ** FISH_FOLLOW
        wind = tget(month, "wind")
        if wind is None:   # upcoming month: that calendar month's mean over the earlier years
            past = [tget(f"{yy}-{mm:02d}", "wind") for yy in range(y - 3, y)]
            wind = np.nanmean([w for w in past if w is not None], axis=0)
        wind_w = np.exp(-np.maximum(wind - 4.0, 0) / 2.0)
        closed = np.ones(cells.size)
        for c in closures:
            frac = closure_fraction(c, month)
            inside = (clat >= c["lat_min"]) & (clat <= c["lat_max"]) & (clon >= c["lon_min"]) & (clon <= c["lon_max"])
            closed[inside] *= 1 - (1 - CLOSED_EFFORT) * frac
        suit = depth_w * fish_w * wind_w * closed

        effort = np.zeros(cells.size)
        fleet_days = 0.0
        for pt, w in zip(PORTS, port_w):
            fd = fleet(pt, y) * dpv
            fleet_days += fd
            ws = w * suit
            effort += fd * ws / ws.sum()
        if not is_up:
            effort = rng.poisson(effort * rng.gamma(NOISE_SHAPE, 1 / NOISE_SHAPE, cells.size)).astype(float)

        fp = fmaps[month]
        kg_set = sum(np.asarray(fp[f"p_{c}"], float).ravel()[cells] * kg_pos[c] for c in TARGETS if f"p_{c}" in fp.files)
        layers["effort"][month] = effort
        layers["density"][month] = effort / ndays / cell_km2 * 100     # boats fishing per 100 km2 on an average day
        layers["fish"][month] = kg_set
        month_info[month] = {"days": ndays, "days_per_vessel": round(dpv, 3), "days_source": dpv_src,
                             "fleet_days": round(fleet_days)}

    # CPUE: the forecast's purse-seine kg per set says where fish are, not what an average licensed boat
    # (any gear) lands. Scale it so predicted target-species catch matches the DOF landings over every
    # area-month that has them: CPUE (t per fishing day) = k x expected kg per set.
    pred_kg, real_t = 0.0, 0.0
    for month in months:
        if month_info[month]["days_source"] != "dof":
            continue
        for i, key in enumerate(AREAS):
            provs = [key] + (["narathiwat"] if key == "pattani" else [])
            land = [dof["landings_t"].get(pv, {}).get(month) for pv in provs]
            if all(x is None for x in land):
                continue
            sel = area_of == i
            pred_kg += float((layers["fish"][month][sel] * layers["effort"][month][sel]).sum())
            real_t += sum(x for x in land if x is not None)
    k_cal = real_t / pred_kg
    for month in months:
        layers["cpue"][month] = layers["fish"][month] * k_cal

    # Effort thresholds over every observed cell-month (the upcoming month is judged against them), so
    # busier months flag more. The forecast fish varies mostly with the season, so its threshold is
    # that month's own: the alert marks where boats crowd into the poorest water of the month.
    obs = [m for m in months if m != upcoming]
    all_e = np.concatenate([layers["effort"][m] for m in obs])
    thr = {"high_effort": float(np.percentile(all_e, HIGH_Q[0])), "watch_effort": float(np.percentile(all_e, WATCH_Q[0])),
           "high_fish_q": HIGH_Q[1], "watch_fish_q": WATCH_Q[1], "fish": {}}
    alerts, areas = {}, {k: {} for k in AREAS}
    for month in months:
        e, f, c = layers["effort"][month], layers["fish"][month], layers["cpue"][month]
        hf, wf = np.percentile(f, HIGH_Q[1]), np.percentile(f, WATCH_Q[1])
        thr["fish"][month] = [round(float(hf), 1), round(float(wf), 1)]
        al = np.zeros(cells.size, np.uint8)
        al[(e >= thr["watch_effort"]) & (f <= wf)] = 1
        al[(e >= thr["high_effort"]) & (f <= hf)] = 2
        alerts[month] = al
        y = int(month[:4])
        for i, key in enumerate(AREAS):
            sel = area_of == i
            n = int(sel.sum())
            days = float(e[sel].sum())
            high, watch = int((al[sel] == 2).sum()), int((al[sel] == 1).sum())
            level = 2 if high / n >= AREA_LEVEL_SHARE else 1 if (high + watch) / n >= AREA_LEVEL_SHARE else 0
            hot = np.flatnonzero(sel & (al > 0))
            hot = hot[np.argmax(e[hot])] if hot.size else None
            # Real CPUE: target-species landings / the province's fishing days (DOF). Pattani area adds Narathiwat.
            provs = [key] + (["narathiwat"] if key == "pattani" else [])
            land = [dof["landings_t"].get(pv, {}).get(month) for pv in provs]
            real_days = sum(latest(dof["vessels"][pv], y) for pv in provs) * month_info[month]["days_per_vessel"]
            real_land = sum(x for x in land if x is not None) if any(x is not None for x in land) else None
            real_cpue = real_land / real_days if real_land is not None and month_info[month]["days_source"] == "dof" else None
            land_ly = [dof["landings_t"].get(pv, {}).get(month_shift(month, -12)) for pv in provs]
            ly = sum(x for x in land_ly if x is not None) if any(x is not None for x in land_ly) else None
            areas[key][month] = {
                "licensed_vessels": int(sum(latest(dof["vessels"][pv], y) for pv in provs)),
                "fishing_days": round(days),
                "boats_per_day": round(days / month_info[month]["days"], 1),
                "fish_kg_set": round(float(f[sel].mean()), 1),
                "cpue_t_day": round(float((c[sel] * e[sel]).sum() / days), 3) if days else None,
                "catch_t": round(float((c[sel] * e[sel]).sum())),
                "real_landings_t": None if real_land is None else round(real_land, 1),
                "real_cpue_t_day": pct_or_none(real_cpue),
                "real_landings_yoy": pct_or_none((real_land - ly) / ly) if real_land is not None and ly else None,
                "cells": n, "high": high, "watch": watch, "level": level,
                "hotspot": None if hot is None else [round(float(clat[hot]), 3), round(float(clon[hot]), 3)],
            }

    def pack(name):
        vmax = float(np.percentile(np.concatenate(list(layers[name].values())), 99))
        return {"vmax": vmax, "months": {m: encode_u8(v, vmax) for m, v in layers[name].items()}}

    payload = {
        "meta": {"generated": date.today().isoformat(), "upcoming_month": upcoming,
                 "effort_source": "Department of Fisheries (catalog.fisheries.go.th): licensed vessels per province, "
                                  "monthly Gulf fishing days, province landings",
                 "spatial": "modelled", "kg_per_positive_set": {k: round(v) for k, v in kg_pos.items()},
                 "sets_per_day": sets_per_day, "cpue_per_kg_set": k_cal,
                 "purse_seine_cpue_per_kg_set": sets_per_day / 1000, "seed": a.seed,
                 "ports": [{k: pt[k] for k in ("name", "name_th", "lat", "lon", "province", "share")} for pt in PORTS],
                 "closures": closures},
        "grid": {"lat0": float(lat[0]), "lon0": float(lon[0]), "res": res, "nlat": int(lat.size), "nlon": int(lon.size)},
        "cells": base64.b64encode(cells.astype(np.uint16).tobytes()).decode("ascii"),
        "months": months,
        "month_info": month_info,
        "layers": {k: pack(k) for k in layers},
        "alert": {m: base64.b64encode(v.tobytes()).decode("ascii") for m, v in alerts.items()},
        "thresholds": thr,
        "areas": areas,
        "area_names": AREA_TH,
    }
    js = ("/* Generated by export_effort_web.py: do not edit by hand. */\n"
          "window.EFFORT_DATA = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n")
    out_file.write_text(js, encoding="utf-8")
    stamp_index(out_file)
    print(f"Created: {out_file.resolve()} ({out_file.stat().st_size / 1024:.0f} KB)")
    print(f"CPUE calibration: {k_cal * 1000:.4f} t/day per t/set (purse seine {sets_per_day:.1f} sets/day; "
          f"fitted on {real_t:,.0f} t of DOF landings)")
    print(f"Months: {months[0]} .. {months[-1]} | {cells.size} sea cells | thresholds "
          + f"effort high >= {thr['high_effort']:.1f} days, watch >= {thr['watch_effort']:.1f} days")
    for m in months[-13:]:
        flags = " ".join(f"{k}:{'HWo'[2 - areas[k][m]['level']]}" for k in AREAS)
        print(f"  {m} fleet-days {month_info[m]['fleet_days']:>6} ({month_info[m]['days_source']}) | "
              f"high {int((alerts[m] == 2).sum()):>3} watch {int((alerts[m] == 1).sum()):>3} | {flags}")


if __name__ == "__main__":
    main()
