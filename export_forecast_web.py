"""Export the one-month-ahead species forecast to forecast_data.js for the Forecast tab in index.html.

    python -m bycatch_model forecast --data sim_output --out forecast_output
    python export_forecast_web.py

Like the EcoCast export, only the map's sea cells of the 0.05 deg grid are written (land and
islands are left out), quantised to 0-254 and base64-encoded.
"""
from pathlib import Path
import argparse
import base64
import csv
import json
from datetime import date

import numpy as np

from web_stamp import stamp_index
from purse_seine_sim.species import checklist

FORECAST_DIR = Path("forecast_output")
DATA_DIR = Path("sim_output")
OUT_FILE = Path("forecast_data.js")
# The logbook counts sharks and rays instead of weighing them; ASSUMED mean weight per animal (kg) so they
# can join the catch-weight total. Turtles and dolphins are not fish and are left out of "all the fish".
KG_PER_ANIMAL = {"RHIN_TYP": 2000.0, "MOBU_SPP": 60.0, "CARC_SPP": 20.0}
NOT_FISH = {"CHEL_MYD", "ORCA_BRE"}


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def num(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def catch_shares(data_dir):
    """Each species' share of all fish caught in the study area (logbook catch weight)."""
    kg = {}
    for r in read_csv(Path(data_dir) / "logbook" / "catch.csv"):
        code = r["species_code"]
        if code in NOT_FISH:
            continue
        w = num(r["catch_kg"]) or (num(r["count"]) or 0.0) * KG_PER_ANIMAL.get(code, 0.0)
        kg[code] = kg.get(code, 0.0) + (w or 0.0)
    total = sum(kg.values())
    return {code: w / total for code, w in kg.items()} if total > 0 else {}


def encode_u8(values, vmax):
    q = np.full(values.shape, 255, dtype=np.uint8)
    ok = np.isfinite(values)
    q[ok] = np.clip(np.round(values[ok] / vmax * 254), 0, 254).astype(np.uint8) if vmax > 0 else 0
    return base64.b64encode(q.ravel().tobytes()).decode("ascii")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--forecast", default=str(FORECAST_DIR))
    p.add_argument("--data", default=str(DATA_DIR), help="simulator/real data folder (catch shares, real-SST labels)")
    p.add_argument("--out", default=str(OUT_FILE))
    a = p.parse_args()
    fdir, out_file = Path(a.forecast), Path(a.out)

    meta = json.loads((fdir / "forecast_meta.json").read_text(encoding="utf-8"))
    months = list(meta["target_months"]) + ([meta["upcoming_month"]] if meta.get("upcoming_month") else [])
    maps = {m: np.load(fdir / "maps" / f"forecast_{m}.npz") for m in months}
    first = maps[months[0]]
    lat, lon, mask = first["lat"], first["lon"], first["mask"]
    res = float(lat[1] - lat[0])
    cells = np.flatnonzero(np.asarray(mask, bool).ravel())   # sea cells, row-major index into (nlat, nlon)

    def legal(mp, key):
        return np.asarray(mp[key], float).ravel()[cells]

    names = {r["code"]: r for r in checklist()}
    codes = sorted({k[2:] for mp in maps.values() for k in mp.files if k.startswith("p_")})
    summary = read_csv(fdir / "forecast_summary.csv")
    curves = json.loads((fdir / "response_curves.json").read_text(encoding="utf-8"))["species"]
    shares = catch_shares(a.data)

    species = []
    for code in codes:
        info = names.get(code, {"name_en": code, "name_th": code, "group": ""})
        fc = {m: legal(mp, f"p_{code}") for m, mp in maps.items() if f"p_{code}" in mp.files}
        tr = {m: legal(mp, f"lambda_{code}") for m, mp in maps.items() if f"lambda_{code}" in mp.files}
        sd = {m: legal(mp, f"sd_{code}") for m, mp in maps.items() if f"sd_{code}" in mp.files}
        vmax = float(np.nanpercentile(np.concatenate([v[np.isfinite(v)] for v in fc.values()]), 99))
        tmax = float(np.nanpercentile(np.concatenate([v[np.isfinite(v)] for v in tr.values()]), 99)) if tr else 1.0
        smax = float(np.nanpercentile(np.concatenate([v[np.isfinite(v)] for v in sd.values()]), 99)) if sd else 1.0
        skill = {r["method"]: {k: num(r[k]) for k in ("auc", "brier", "map_rho", "brier_skill_vs_persistence",
                                                     "brier_skill_vs_climatology")}
                 for r in summary if r["species_code"] == code}
        # The maps use the no-heat model when the heat terms did not earn their place; score that one.
        if not meta["heat_terms_kept"].get(code) and "forecast_no_heat" in skill:
            skill["forecast"] = skill["forecast_no_heat"]
        species.append({
            "code": code, "group": info["group"], "name_en": info["name_en"], "name_th": info["name_th"],
            "heat_terms_kept": bool(meta["heat_terms_kept"].get(code)),
            "vmax": vmax, "tmax": tmax, "smax": smax,
            "share": shares.get(code),
            "forecast": {m: encode_u8(v, vmax) for m, v in fc.items()},
            "truth": {m: encode_u8(v, tmax) for m, v in tr.items()},
            "sd": {m: encode_u8(v, smax) for m, v in sd.items()},
            "skill": skill,
            "curves": curves.get(code, {}),
        })

    sim_cfg = {}
    if a.data and (Path(a.data) / "config_used.json").exists():
        sim_cfg = json.loads((Path(a.data) / "config_used.json").read_text(encoding="utf-8"))
    env_cfg, region_cfg = sim_cfg.get("env", {}), sim_cfg.get("region", {})
    payload = {
        "meta": {"generated": date.today().isoformat(), "simulated": meta.get("simulated", True), "real_sst": bool(env_cfg.get("sst_override_npz")),
                 "real_chl": bool(env_cfg.get("chl_override_npz")), "real_wind": bool(env_cfg.get("wind_override_npz")),
                 "real_bathy": bool(region_cfg.get("bathy_npz")),
                 "train_sets": meta["train_sets"], "kernel_km": meta["kernel_km"], "n_boot": meta["n_boot"],
                 "upcoming_month": meta.get("upcoming_month")},
        "grid": {"lat0": float(lat[0]), "lon0": float(lon[0]), "res": res,
                 "nlat": int(lat.size), "nlon": int(lon.size)},
        "cells": base64.b64encode(cells.astype(np.uint16).tobytes()).decode("ascii"),
        "months": months,
        "species": species,
    }
    js = ("/* Generated by export_forecast_web.py: do not edit by hand. */\n"
          "window.FORECAST_DATA = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n")
    out_file.write_text(js, encoding="utf-8")
    stamp_index(out_file)
    print(f"Created: {out_file.resolve()} ({out_file.stat().st_size / 1024:.0f} KB)")
    print(f"Months: {months[0]} .. {months[-1]} | grid {lat.size} x {lon.size} at {res:.2f} deg, {cells.size} sea cells | species: {len(species)}")


if __name__ == "__main__":
    main()
