"""Export trained bycatch-model maps to ecocast_data.js for the EcoCast page in index.html.

    python -m purse_seine_sim --vessels 10 --start 2023-01-01 --end 2025-12-31 --no-track --out sim_output
    python -m bycatch_model train --data sim_output --out model_output
    python export_ecocast_web.py

Layers are quantised to 0-254 (255 = no data) and base64-encoded per month to keep
the file small. The page loads it only when the EcoCast tab is opened.
"""
from pathlib import Path
import argparse
import base64
import csv
import json
from datetime import date

import numpy as np

from purse_seine_sim.geo import distance_km
from purse_seine_sim.species import TARGETS, checklist

# =========================
# CONFIG
# =========================

MODEL_DIR = Path("model_output")
DATA_DIR = Path("sim_output")
OUT_FILE = Path("ecocast_data.js")

# Same pilot provinces / ports as the main map; each sea cell belongs to its nearest port.
PROVINCE_ZONES = [
    ("chumphon", "ชุมพร", 10.47, 99.25),
    ("surat", "สุราษฎร์ธานี", 9.20, 99.38),
    ("nakhon", "นครศรีธรรมราช", 8.35, 100.20),
    ("songkhla", "สงขลา", 7.23, 100.58),
    ("pattani", "ปัตตานี", 6.87, 101.25),
]

MIN_POSITIVES_SCORE = 40     # keep in sync with bycatch_model.ecocast
RELIABLE_POSITIVES = 200


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def num(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def reliability(train_pos, auc):
    if train_pos is None or train_pos < 5 or auc is None:
        return "none"
    if train_pos >= RELIABLE_POSITIVES and auc >= 0.7:
        return "good"
    if train_pos >= MIN_POSITIVES_SCORE:
        return "limited"
    return "unreliable"


def encode_u8(values, vmin, vmax):
    q = np.full(values.shape, 255, dtype=np.uint8)
    ok = np.isfinite(values)
    if vmax > vmin:
        q[ok] = np.clip(np.round((values[ok] - vmin) / (vmax - vmin) * 254), 0, 254).astype(np.uint8)
    else:
        q[ok] = 0
    return base64.b64encode(q.tobytes()).decode("ascii")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(MODEL_DIR))
    p.add_argument("--data", default=str(DATA_DIR))
    p.add_argument("--out", default=str(OUT_FILE))
    a = p.parse_args()
    model_dir, data_dir, out_file = Path(a.model), Path(a.data), Path(a.out)

    paths = sorted((model_dir / "maps").glob("pred_*.npz"))
    if not paths:
        raise FileNotFoundError(f"ไม่พบ {model_dir}/maps/pred_*.npz: run `python -m bycatch_model train` first")
    months = [p.stem[len("pred_"):] for p in paths]
    maps = [np.load(p) for p in paths]
    first = maps[0]
    lat, lon, mask = first["lat"], first["lon"], first["mask"]
    res = float(lat[1] - lat[0])
    cells = np.flatnonzero(mask.ravel())             # row-major index into (nlat, nlon)
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    clat, clon = LAT.ravel()[cells], LON.ravel()[cells]
    footprint = np.nan_to_num(first["data_footprint"]).ravel()[cells] > 0.5

    zone_d = np.stack([distance_km(clat, clon, zlat, zlon) for _, _, zlat, zlon in PROVINCE_ZONES])
    zone = np.argmin(zone_d, axis=0).astype(np.uint8)

    metrics = {r["species_code"]: r for r in read_csv(model_dir / "metrics.csv")}
    names = {r["code"]: r for r in checklist()}
    prices = {s.code: s.price_thb_kg for s in TARGETS}

    # ---------- layer definitions ----------
    layer_defs = [("score", "ecocast_score", "EcoCast score", "คะแนน EcoCast", "score", "diverging", None)]
    if "true_ecocast_score" in first.files:
        layer_defs.append(("true_score", "true_ecocast_score", "TRUE EcoCast score (simulator)",
                           "คะแนน EcoCast จริง (จากตัวจำลอง)", "score", "diverging", None))
    layer_defs.append(("revenue", "revenue_thb_per_set", "Expected revenue per set", "รายได้คาดหวังต่อการล้อมอวน",
                       "THB/set", "sequential", None))
    species_rows = []
    for code, m in metrics.items():
        info = names.get(code, {"name_en": code, "name_th": code, "sci_name": ""})
        train_pos = num(m.get("train_positives"))
        auc = num(m.get("test_auc"))
        rel = reliability(train_pos, auc)
        key = f"ekg_{code}" if m["group"] == "target" else f"p_{code}"
        has_layer = key in first.files and np.isfinite(first[key]).any()
        species_rows.append({
            "code": code, "group": m["group"], "name_en": info["name_en"], "name_th": info["name_th"],
            "sci_name": info["sci_name"], "price_thb_kg": prices.get(code),
            "train_positives": train_pos, "test_positives": num(m.get("test_positives")),
            "test_auc": auc, "ceiling_auc": num(m.get("ceiling_auc")), "map_rho": num(m.get("map_spearman")),
            "reliability": rel, "has_layer": bool(has_layer),
            "in_score": m["group"] == "target" or (train_pos or 0) >= MIN_POSITIVES_SCORE,
        })
        if has_layer:
            if m["group"] == "target":
                layer_defs.append((code, key, f"{info['name_en']}: expected catch", f"{info['name_th']} · ปริมาณที่คาดว่าจับได้",
                                   "kg/set", "sequential", code))
            else:
                layer_defs.append((code, key, f"{info['name_en']}: bycatch probability", f"{info['name_th']} · โอกาสติดอวน",
                                   "probability/set", "risk", code))

    layers = []
    for key, npz_key, label_en, label_th, unit, cmap, code in layer_defs:
        stack = np.stack([np.asarray(mp[npz_key], float).ravel()[cells] for mp in maps])
        if cmap == "diverging":
            # Like EcoCast: rescale each month to -1..+1 (1st-99th percentile inside the footprint),
            # so the map shows the relatively best and worst areas of that month.
            for i in range(stack.shape[0]):
                ref = stack[i, footprint]
                lo, hi = np.nanpercentile(ref, [1, 99]) if np.isfinite(ref).any() else (0.0, 1.0)
                stack[i] = np.clip(2 * (stack[i] - lo) / max(hi - lo, 1e-12) - 1, -1, 1)
            vmin, vmax = -1.0, 1.0
        else:
            inside = stack[:, footprint]
            finite = inside[np.isfinite(inside)]
            vmin, vmax = 0.0, float(np.percentile(finite, 99)) if finite.size else 1.0
        layers.append({
            "key": key, "code": code, "label_en": label_en, "label_th": label_th, "unit": unit, "cmap": cmap,
            "vmin": vmin, "vmax": vmax,
            "data": {month: encode_u8(stack[i], vmin, vmax) for i, month in enumerate(months)},
        })

    skill_path = model_dir / "ecocast_skill.csv"
    skill = [num(r["ecocast_spearman"]) for r in read_csv(skill_path)] if skill_path.exists() else []
    skill = [s for s in skill if s is not None]
    summary_path = data_dir / "summary.json"
    sim_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    cfg_path = data_dir / "config_used.json"
    env_cfg = json.loads(cfg_path.read_text(encoding="utf-8")).get("env", {}) if cfg_path.exists() else {}
    holdout_path = model_dir / "holdout.json"
    holdout = json.loads(holdout_path.read_text(encoding="utf-8")) if holdout_path.exists() else {}
    train_sets = num(next(iter(metrics.values())).get("train_sets"))
    test_sets = num(next(iter(metrics.values())).get("test_sets"))

    payload = {
        "meta": {
            "generated": date.today().isoformat(),
            "simulated": (data_dir / "truth").exists(),
            "n_vessels": sim_summary.get("n_vessels"),
            "train_sets": train_sets, "test_sets": test_sets,
            "test_year": str(holdout["test_year"]) if holdout.get("test_year") else months[0][:4],
            "test_months": holdout.get("test_months"),
            "real_sst": bool(env_cfg.get("sst_override_npz")),
            "real_chl": bool(env_cfg.get("chl_override_npz")),
            "real_wind": bool(env_cfg.get("wind_override_npz")),
            "ecocast_skill": float(np.mean(skill)) if skill else None,
            "footprint_km": 40,
            "weights": {"revenue": 0.40, "RHIN_TYP": 0.15, "MOBU_SPP": 0.10, "CARC_SPP": 0.15,
                        "CHEL_MYD": 0.15, "ORCA_BRE": 0.05},
            "min_positives_score": MIN_POSITIVES_SCORE,
        },
        "grid": {"lat0": float(lat[0]), "lon0": float(lon[0]), "res": res,
                 "nlat": int(lat.size), "nlon": int(lon.size)},
        "cells": base64.b64encode(cells.astype(np.uint16).tobytes()).decode("ascii"),
        "footprint": base64.b64encode(footprint.astype(np.uint8).tobytes()).decode("ascii"),
        "zone": base64.b64encode(zone.tobytes()).decode("ascii"),
        "zones": [{"key": k, "name_th": n} for k, n, _, _ in PROVINCE_ZONES],
        "months": months,
        "layers": layers,
        "species": species_rows,
    }
    js = ("/* Generated by export_ecocast_web.py: do not edit by hand. */\n"
          "window.ECOCAST_DATA = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n")
    out_file.write_text(js, encoding="utf-8")
    print(f"Created: {out_file.resolve()} ({out_file.stat().st_size / 1024:.0f} KB)")
    print(f"Months: {months[0]} .. {months[-1]} | cells: {cells.size} | layers: {len(layers)}")


if __name__ == "__main__":
    main()
