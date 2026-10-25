"""Load training rows (logbook + env) and prediction grids into model features.

Works the same for simulated and real data as long as it follows
purse_seine_sim/DATA_SPEC.md. Truth columns are loaded only if present
(simulations), and only ever used for scoring.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from purse_seine_sim.environment import moon_illumination, to_unix_days
from purse_seine_sim.species import checklist

SMOOTH_TERMS = ["sst", "sst_anom", "log10chl", "front", "depth", "dist_coast", "seagrass_idx", "wind"]
LINEAR_TERMS = ["is_fad", "is_light", "light_x_moon", "sin_doy", "cos_doy", "sin2_doy", "cos2_doy"]

# Fishing-legal area used for maps (same rules as the simulator defaults).
MIN_DEPTH_M = 15.0
MIN_DIST_COAST_KM = 5.56


def _read(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _num(v):
    return float(v) if v not in ("", None) else np.nan


def add_derived(base: dict, t, set_type):
    """Seasonal and gear terms shared by training rows and prediction grids."""
    t = np.asarray(t, float)
    set_type = np.asarray(set_type)
    fy = 2 * np.pi * ((t / 365.2425) % 1.0)
    f = dict(base)
    f["is_fad"] = (set_type == "fad").astype(float)
    f["is_light"] = (set_type == "light").astype(float)
    f["light_x_moon"] = f["is_light"] * np.asarray(base["moon"], float)
    f["sin_doy"], f["cos_doy"] = np.sin(fy), np.cos(fy)
    f["sin2_doy"], f["cos2_doy"] = np.sin(2 * fy), np.cos(2 * fy)
    return f


@dataclass
class Dataset:
    event_id: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    time: np.ndarray            # unix days
    year: np.ndarray
    set_type: np.ndarray
    features: dict
    presence: dict              # code -> bool array (reported)
    amount: dict                # code -> kg (targets) or count (ETP)
    groups: dict                # code -> target | discard | etp
    truth_p: dict = field(default_factory=dict)

    def __len__(self):
        return self.event_id.size

    def subset(self, idx):
        idx = np.asarray(idx)
        return Dataset(self.event_id[idx], self.lat[idx], self.lon[idx], self.time[idx], self.year[idx], self.set_type[idx],
                       {k: v[idx] for k, v in self.features.items()},
                       {k: v[idx] for k, v in self.presence.items()},
                       {k: v[idx] for k, v in self.amount.items()},
                       self.groups, {k: v[idx] for k, v in self.truth_p.items()})


def load_checklist(root):
    path = Path(root) / "reference" / "species_checklist.csv"
    rows = _read(path) if path.exists() else checklist()
    return {r["code"]: r["group"] for r in rows}


def load_dataset(root) -> Dataset:
    root = Path(root)
    sets = _read(root / "logbook" / "sets.csv")
    env = {r["event_id"]: r for r in _read(root / "env" / "env_at_sets.csv")}
    groups = load_checklist(root)

    ids = np.array([s["event_id"] for s in sets])
    pos = {e: i for i, e in enumerate(ids)}
    t = np.array([s["start_time_utc"].rstrip("Z") for s in sets], dtype="datetime64[s]")
    time = t.astype("int64") / 86400.0
    year = t.astype("datetime64[Y]").astype(int) + 1970
    month = t.astype("datetime64[M]").astype(int) % 12 + 1
    set_type = np.array([s["set_type"] for s in sets])

    e = [env[i] for i in ids]
    chl = np.array([_num(r["sat_chl_mg_m3"]) for r in e])
    log10chl = np.log10(chl)
    # Cloud gaps: fill with that month's median (a real pipeline would use gap-filled CHL).
    for m in range(1, 13):
        sel = month == m
        good = sel & np.isfinite(log10chl)
        fill = np.nanmedian(log10chl[good]) if good.any() else np.nanmedian(log10chl)
        log10chl[sel & ~np.isfinite(log10chl)] = fill
    base = {
        "sst": np.array([_num(r["sat_sst_c"]) for r in e]),
        "sst_anom": np.array([_num(r["sat_sst_anom_c"]) for r in e]),
        "log10chl": log10chl,
        "front": np.array([_num(r["sst_front_c_per_10km"]) for r in e]),
        "depth": np.array([_num(r["depth_m"]) for r in e]),
        "dist_coast": np.array([_num(r["dist_coast_km"]) for r in e]),
        "seagrass_idx": np.array([_num(r["seagrass_idx"]) for r in e]),
        "wind": np.array([_num(r["wind_ms"]) for r in e]),
        "moon": np.array([_num(r["moon_illum"]) for r in e]),
    }
    features = add_derived(base, time, set_type)

    amount = {c: np.zeros(ids.size) for c in groups}
    for r in _read(root / "logbook" / "catch.csv"):
        c = r["species_code"]
        if c in amount:
            amount[c][pos[r["event_id"]]] = _num(r["catch_kg"] or r["count"] or "0")
    presence = {c: a > 0 for c, a in amount.items()}

    truth_p = {}
    tpath = root / "truth" / "truth_sets.csv"
    if tpath.exists():
        rows = _read(tpath)
        order = np.array([pos[r["event_id"]] for r in rows])
        for c in groups:
            col = f"p_present_{c}"
            if col in rows[0]:
                v = np.empty(ids.size)
                v[order] = [float(r[col]) for r in rows]
                truth_p[c] = v
    lat = np.array([_num(r["lat"]) for r in e])
    lon = np.array([_num(r["lon"]) for r in e])
    return Dataset(ids, lat, lon, time, year, set_type, features, presence, amount, groups, truth_p)


@dataclass
class Grid:
    month: str
    lat: np.ndarray
    lon: np.ndarray
    mask: np.ndarray            # 2-D: legal fishing cells
    features: dict              # 1-D over mask cells
    truth_lambda: dict          # code -> 1-D (simulations only)

    @property
    def cell_latlon(self):
        LAT, LON = np.meshgrid(self.lat, self.lon, indexing="ij")
        return LAT[self.mask], LON[self.mask]

    def to_2d(self, values):
        out = np.full(self.mask.shape, np.nan)
        out[self.mask] = values
        return out


def load_grid(npz_path, set_type="free_school") -> Grid:
    """Monthly environment grid (simulator truth map or the same keys from real satellite data)."""
    npz_path = Path(npz_path)
    month = re.search(r"(\d{4}-\d{2})", npz_path.name).group(1)
    d = np.load(npz_path)
    mask = d["ocean"] & (np.nan_to_num(d["depth"]) >= MIN_DEPTH_M) & (np.nan_to_num(d["dist_coast"]) >= MIN_DIST_COAST_KM)
    t = to_unix_days(f"{month}-15T05:00")
    base = {k: np.asarray(d[k], float)[mask] for k in SMOOTH_TERMS}
    n = int(mask.sum())
    base["moon"] = np.full(n, float(moon_illumination(t)))
    features = add_derived(base, np.full(n, t), np.full(n, set_type))
    truth = {k[len("lambda_"):]: np.asarray(d[k], float)[mask] for k in d.files if k.startswith("lambda_")}
    return Grid(month, d["lat"], d["lon"], mask, features, truth)
