"""Export the one-month-ahead species forecast to forecast_data.js for the Forecast tab in index.html.

    python -m bycatch_model forecast --data sim_output --out forecast_output
    python export_forecast_web.py

Like the EcoCast export, only the legal fishing cells of the 0.05 deg grid are written (land,
islands, the 3 nm zone and water shallower than 15 m are left out), quantised to 0-254 and base64-encoded.
"""
from pathlib import Path
import argparse
import base64
import csv
import json
from datetime import date

import numpy as np

from purse_seine_sim.species import checklist

FORECAST_DIR = Path("forecast_output")
OUT_FILE = Path("forecast_data.js")


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def num(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def encode_u8(values, vmax):
    q = np.full(values.shape, 255, dtype=np.uint8)
    ok = np.isfinite(values)
    q[ok] = np.clip(np.round(values[ok] / vmax * 254), 0, 254).astype(np.uint8) if vmax > 0 else 0
    return base64.b64encode(q.ravel().tobytes()).decode("ascii")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--forecast", default=str(FORECAST_DIR))
    p.add_argument("--data", help="simulator/real data folder (to label real-SST runs)")
    p.add_argument("--out", default=str(OUT_FILE))
    a = p.parse_args()
    fdir, out_file = Path(a.forecast), Path(a.out)

    meta = json.loads((fdir / "forecast_meta.json").read_text(encoding="utf-8"))
    months = list(meta["target_months"]) + ([meta["upcoming_month"]] if meta.get("upcoming_month") else [])
    maps = {m: np.load(fdir / "maps" / f"forecast_{m}.npz") for m in months}
    first = maps[months[0]]
    lat, lon, mask = first["lat"], first["lon"], first["mask"]
    res = float(lat[1] - lat[0])
    cells = np.flatnonzero(np.asarray(mask, bool).ravel())   # row-major index into (nlat, nlon)

    def legal(mp, key):
        return np.asarray(mp[key], float).ravel()[cells]

    names = {r["code"]: r for r in checklist()}
    codes = sorted({k[2:] for mp in maps.values() for k in mp.files if k.startswith("p_")})
    summary = read_csv(fdir / "forecast_summary.csv")
    curves = json.loads((fdir / "response_curves.json").read_text(encoding="utf-8"))["species"]

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
    print(f"Created: {out_file.resolve()} ({out_file.stat().st_size / 1024:.0f} KB)")
    print(f"Months: {months[0]} .. {months[-1]} | grid {lat.size} x {lon.size} at {res:.2f} deg, {cells.size} legal cells | species: {len(species)}")


if __name__ == "__main__":
    main()
