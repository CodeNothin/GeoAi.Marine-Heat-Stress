"""Export the one-month-ahead species forecast (MAP 1 workflow) to forecast_data.js for the Forecast tab.

    python -m bycatch_model map1 --data sim_output --env real_env --out forecast_output
    python export_forecast_web.py --forecast forecast_output --data sim_output

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


def layer_max(arrays):
    vals = [v[np.isfinite(v)] for v in arrays.values()]
    vals = [v for v in vals if v.size]
    return float(np.nanpercentile(np.concatenate(vals), 99)) if vals else 1.0


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

    def layer(key):
        return {m: legal(mp, key) for m, mp in maps.items() if key in mp.files}

    names = {r["code"]: r for r in checklist()}
    codes = sorted({k[2:] for mp in maps.values() for k in mp.files if k.startswith("p_")})
    cls = read_csv(fdir / "map1_classification.csv")
    reg = {r["species_code"]: {k: num(v) for k, v in r.items() if k != "species_code"}
           for r in read_csv(fdir / "map1_regression.csv")}
    shap = json.loads((fdir / "map1_shap.json").read_text(encoding="utf-8"))
    col = json.loads((fdir / "map1_collinearity.json").read_text(encoding="utf-8"))
    shares = catch_shares(a.data)
    metric_keys = ("auc", "brier", "mae", "rmse", "spearman", "jsd", "taylor_r", "taylor_sd_ratio", "taylor_crmsd",
                   "taylor_skill", "mean_rank", "map_rho", "n")

    species = []
    for code in codes:
        info = names.get(code, {"name_en": code, "name_th": code, "group": ""})
        fc, tr, sd, bm = layer(f"p_{code}"), layer(f"lambda_{code}"), layer(f"sd_{code}"), layer(f"b_{code}")
        # Model spread only for the upcoming month (the one people act on), to keep the file small.
        sd = {m: v for m, v in sd.items() if m == meta.get("upcoming_month")}
        vmax, tmax, smax, bmax = layer_max(fc), layer_max(tr), layer_max(sd), layer_max(bm)
        models = {r["model"]: {k: num(r[k]) for k in metric_keys} for r in cls if r["species_code"] == code}
        species.append({
            "code": code, "group": info["group"], "name_en": info["name_en"], "name_th": info["name_th"],
            "vmax": vmax, "tmax": tmax, "smax": smax, "bmax": bmax,
            "share": shares.get(code),
            "forecast": {m: encode_u8(v, vmax) for m, v in fc.items()},
            "truth": {m: encode_u8(v, tmax) for m, v in tr.items()},
            "sd": {m: encode_u8(v, smax) for m, v in sd.items()},
            "biomass": {m: encode_u8(v, bmax) for m, v in bm.items()},
            "best": meta["best_model"].get(code),
            "models": models,
            "regression": reg.get(code),
            "shap": shap.get(code),
        })

    fronts = {}
    for key in ("sst_front_km", "chl_front_km"):
        lay = layer(key)
        vmax = layer_max(lay)
        fronts[key] = {"vmax": vmax, "months": {m: encode_u8(np.minimum(v, vmax), vmax) for m, v in lay.items()}}

    sim_cfg = {}
    if a.data and (Path(a.data) / "config_used.json").exists():
        sim_cfg = json.loads((Path(a.data) / "config_used.json").read_text(encoding="utf-8"))
    env_cfg, region_cfg = sim_cfg.get("env", {}), sim_cfg.get("region", {})
    payload = {
        "meta": {"generated": date.today().isoformat(), "method": meta.get("method", "map1"),
                 "simulated": meta.get("simulated", True), "real_sst": bool(env_cfg.get("sst_override_npz")),
                 "real_chl": bool(env_cfg.get("chl_override_npz")), "real_wind": bool(env_cfg.get("wind_override_npz")),
                 "real_bathy": bool(region_cfg.get("bathy_npz")),
                 "train_sets": meta["train_sets"], "validation_sets": meta["validation_sets"], "sets": meta["sets"],
                 "train_share": meta["train_share"], "n_zones": meta["n_zones"], "ring_km": meta["ring_km"],
                 "buffer_km": meta["buffer_km"], "fine_deg": meta["fine_deg"], "models": meta["models"],
                 "features": meta["features"], "features_th": meta["features_th"],
                 "collinearity": {"dropped": col["dropped"], "vif_after": col["vif_after"], "max_vif": col["max_vif"]},
                 "upcoming_month": meta.get("upcoming_month")},
        "grid": {"lat0": float(lat[0]), "lon0": float(lon[0]), "res": res,
                 "nlat": int(lat.size), "nlon": int(lon.size)},
        "cells": base64.b64encode(cells.astype(np.uint16).tobytes()).decode("ascii"),
        "months": months,
        "species": species,
        "fronts": fronts,
    }
    js = ("/* Generated by export_forecast_web.py: do not edit by hand. */\n"
          "window.FORECAST_DATA = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n")
    out_file.write_text(js, encoding="utf-8")
    stamp_index(out_file)
    print(f"Created: {out_file.resolve()} ({out_file.stat().st_size / 1024:.0f} KB)")
    print(f"Months: {months[0]} .. {months[-1]} | grid {lat.size} x {lon.size} at {res:.2f} deg, {cells.size} sea cells | species: {len(species)}")


if __name__ == "__main__":
    main()
