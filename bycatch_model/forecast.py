"""One-month-ahead species distribution forecast from lagged heat and climate inputs.

To forecast month M we only use what is known at the end of month M-1:

    sst_l1, sst_anom_l1, log10chl_l1, wind_l1   last month's conditions at the cell
    heat_3m      sum of positive SST anomaly over the last 3 months (deg C x months),
                 a monthly stand-in for degree heating weeks
    anom_trend   sst_anom_l1 - sst_anom_l2 (is the water warming or cooling?)
    depth, dist_coast, seagrass_idx   static habitat
    + set type and the seasonal cycle of month M (both known in advance)

One binomial GAM per species predicts P(caught in a set). Validation rolls forward
month by month: train on every set before M, forecast M, score against the sets
fished in M, and compare with two baselines built from the logbook alone:

    persistence   catch rate near each point last month ("same as this month")
    climatology   catch rate near each point in the same calendar month of earlier years

The heat terms are kept for a species only if they lower the forecast's Brier score by
at least 0.5% compared with the same model without them.

Monthly environment grids are read from <data>/env/maps/env_YYYY-MM.npz (real satellite
or ERA5 data, same keys as below) or, for simulations, <data>/truth/maps/truth_YYYY-MM.npz.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

from purse_seine_sim.environment import to_unix_days
from purse_seine_sim.geo import distance_km

from .data import MIN_DEPTH_M, MIN_DIST_COAST_KM, add_derived, load_dataset
from .ecocast import FOOTPRINT_KM, MIN_POSITIVES_SCORE, auc, footprint, spearman
from .gam import GAM

GRID_KEYS = ["sst", "sst_anom", "log10chl", "wind", "depth", "dist_coast", "seagrass_idx"]
HEAT_TERMS = ["heat_3m", "anom_trend"]
BASE_TERMS = ["sst_l1", "sst_anom_l1", "log10chl_l1", "wind_l1", "depth", "dist_coast", "seagrass_idx"]
LINEAR_TERMS = ["is_fad", "is_light", "light_x_moon", "sin_doy", "cos_doy", "sin2_doy", "cos2_doy"]
CURVE_TERMS = ["sst_anom_l1", "heat_3m", "anom_trend", "sst_l1", "log10chl_l1", "wind_l1"]

KERNEL_KM = 30.0            # bandwidth of the persistence / climatology baselines
KERNEL_PRIOR = 2.0          # pseudo-sets pulling sparse areas towards the overall rate
MIN_TEST_SETS = 30          # a month with fewer sets is too thin to score a forecast on
MIN_HEAT_GAIN = 0.005       # heat terms are kept only if they cut the Brier score by at least 0.5%


def month_str(m) -> str:
    return str(np.datetime64(m, "M"))


def add_months(m, k) -> str:
    return month_str(np.datetime64(m, "M") + np.timedelta64(int(k), "M"))


# =========================
# MONTHLY ENVIRONMENT ARCHIVE
# =========================

class EnvArchive:
    """All monthly environment grids of a data folder, on one regular lat/lon grid."""

    def __init__(self, data_dir):
        data_dir = Path(data_dir)
        paths = sorted((data_dir / "env" / "maps").glob("env_*.npz"))
        if not paths:
            paths = sorted((data_dir / "truth" / "maps").glob("truth_*.npz"))
        if not paths:
            raise FileNotFoundError(f"no monthly grids in {data_dir}/env/maps or {data_dir}/truth/maps")
        self.grids, self.truth = {}, {}
        for p in paths:
            month = re.search(r"(\d{4}-\d{2})", p.name).group(1)
            d = np.load(p)
            self.grids[month] = {k: np.asarray(d[k], float) for k in GRID_KEYS}
            self.truth[month] = {k[len("lambda_"):]: np.asarray(d[k], float) for k in d.files if k.startswith("lambda_")}
            if not hasattr(self, "lat"):
                self.lat, self.lon, self.ocean = np.asarray(d["lat"], float), np.asarray(d["lon"], float), d["ocean"]
        self.months = sorted(self.grids)
        self.res = float(self.lat[1] - self.lat[0])
        g = self.grids[self.months[0]]
        self.mask = self.ocean & (np.nan_to_num(g["depth"]) >= MIN_DEPTH_M) & (np.nan_to_num(g["dist_coast"]) >= MIN_DIST_COAST_KM)
        LAT, LON = np.meshgrid(self.lat, self.lon, indexing="ij")
        self.cell_lat, self.cell_lon = LAT[self.mask], LON[self.mask]

    def has_lags(self, month) -> bool:
        return all(add_months(month, -k) in self.grids for k in (1, 2, 3))

    def index(self, lat, lon):
        i = np.clip(np.round((np.asarray(lat) - self.lat[0]) / self.res).astype(int), 0, self.lat.size - 1)
        j = np.clip(np.round((np.asarray(lon) - self.lon[0]) / self.res).astype(int), 0, self.lon.size - 1)
        return i, j

    def lag_features(self, month, i, j) -> dict:
        """Lagged inputs for forecasting `month` at grid cells (i, j)."""
        g1, g2, g3 = (self.grids[add_months(month, -k)] for k in (1, 2, 3))
        a1, a2, a3 = g1["sst_anom"][i, j], g2["sst_anom"][i, j], g3["sst_anom"][i, j]
        return {
            "sst_l1": g1["sst"][i, j], "sst_anom_l1": a1, "log10chl_l1": g1["log10chl"][i, j],
            "wind_l1": g1["wind"][i, j],
            "heat_3m": np.maximum(a1, 0) + np.maximum(a2, 0) + np.maximum(a3, 0),
            "anom_trend": a1 - a2,
            "depth": g1["depth"][i, j], "dist_coast": g1["dist_coast"][i, j], "seagrass_idx": g1["seagrass_idx"][i, j],
        }

    def grid_features(self, month) -> dict:
        """Forecast inputs for every legal cell of `month` (free-school set, no light)."""
        i, j = np.nonzero(self.mask)
        F = self.lag_features(month, i, j)
        n = i.size
        F["moon"] = np.zeros(n)
        return add_derived(F, np.full(n, to_unix_days(f"{month}-15T05:00")), np.full(n, "free_school"))

    def to_2d(self, values):
        out = np.full(self.mask.shape, np.nan)
        out[self.mask] = values
        return out


def set_features(ds, archive):
    """Lagged inputs for every logbook set; `valid` is False where a lag month is missing."""
    months = np.array([month_str(np.datetime64(int(t * 86400), "s")) for t in ds.time])
    i, j = archive.index(ds.lat, ds.lon)
    F = {k: np.full(len(ds), np.nan) for k in BASE_TERMS + HEAT_TERMS}
    for m in np.unique(months):
        if not archive.has_lags(m):
            continue
        sel = months == m
        for k, v in archive.lag_features(m, i[sel], j[sel]).items():
            F[k][sel] = v
    for k in LINEAR_TERMS:
        F[k] = ds.features[k]
    valid = np.all([np.isfinite(F[k]) for k in BASE_TERMS + HEAT_TERMS], axis=0)
    return F, months, valid


# =========================
# MODELS AND BASELINES
# =========================

SEASON_TERMS = ["sin_doy", "cos_doy", "sin2_doy", "cos2_doy"]


def fit_forecast(F, y, heat=True, seasonal=True):
    """seasonal=False drops the seasonal cycle, which cannot be learned from less than a year of data."""
    terms = BASE_TERMS + (HEAT_TERMS if heat else [])
    linear = LINEAR_TERMS if seasonal else [t for t in LINEAR_TERMS if t not in SEASON_TERMS]
    return GAM(terms, linear, "binomial").fit(F, y)


def covers_full_year(months):
    return len({m[-2:] for m in np.asarray(months).tolist()}) == 12


def kernel_rate(src_lat, src_lon, y, q_lat, q_lon, bw_km=KERNEL_KM, prior=KERNEL_PRIOR):
    """Smoothed catch rate around each query point (Gaussian kernel, shrunk to the overall rate)."""
    y = np.asarray(y, float)
    if y.size == 0:
        return np.full(np.size(q_lat), np.nan)
    p0 = (y.sum() + 0.5) / (y.size + 1.0)
    out = np.empty(np.size(q_lat))
    for s in range(0, out.size, 500):
        d = distance_km(np.asarray(q_lat)[s:s + 500, None], np.asarray(q_lon)[s:s + 500, None],
                        np.asarray(src_lat)[None, :], np.asarray(src_lon)[None, :])
        w = np.exp(-0.5 * (d / bw_km) ** 2)
        out[s:s + 500] = (w @ y + prior * p0) / (w.sum(axis=1) + prior)
    return out


def brier(p, y):
    ok = np.isfinite(p)
    return float(np.mean((p[ok] - np.asarray(y, float)[ok]) ** 2)) if ok.any() else np.nan


def _rows_subset(F, idx):
    return {k: v[idx] for k, v in F.items()}


# =========================
# ROLLING-ORIGIN VALIDATION
# =========================

METHODS = ["forecast", "forecast_no_heat", "persistence", "climatology"]
METRIC_COLS = ["month", "species_code", "method", "train_positives", "test_sets", "test_positives",
               "auc", "brier", "map_rho"]
SUMMARY_COLS = ["species_code", "group", "months", "method", "auc", "brier", "map_rho",
                "brier_skill_vs_persistence", "brier_skill_vs_climatology", "heat_terms_kept"]


def eligible_codes(ds, y_mask):
    return [c for c, g in ds.groups.items() if g in ("target", "etp")
            and int(ds.presence[c][y_mask].sum()) >= MIN_POSITIVES_SCORE]


def rolling_validation(ds, archive, F, months, valid, target_months, verbose=True):
    rows, maps = [], {}
    for M in target_months:
        train = np.flatnonzero(valid & (months < M))
        test = np.flatnonzero(valid & (months == M))
        prev = np.flatnonzero(months == add_months(M, -1))
        clim = np.flatnonzero((months < M) & (np.char.endswith(months.astype(str), M[-3:])))
        if train.size == 0 or test.size == 0 or not archive.has_lags(M):
            continue
        Fg = archive.grid_features(M)
        fp = footprint(archive.cell_lat, archive.cell_lon, ds.lat[train], ds.lon[train])
        truth = archive.truth.get(M, {})
        maps[M] = {"footprint": fp}
        Ftr, Fte = _rows_subset(F, train), _rows_subset(F, test)
        codes = eligible_codes(ds, np.isin(np.arange(len(ds)), train))
        seasonal = covers_full_year(months[train])
        for code in codes:
            y_tr, y_te = ds.presence[code][train], ds.presence[code][test]
            preds, grids = {}, {}
            for method, heat in (("forecast", True), ("forecast_no_heat", False)):
                m = fit_forecast(Ftr, y_tr, heat, seasonal)
                preds[method], grids[method] = m.predict(Fte), m.predict(Fg)
            for method, src in (("persistence", prev), ("climatology", clim)):
                y_src = ds.presence[code][src]
                preds[method] = kernel_rate(ds.lat[src], ds.lon[src], y_src, ds.lat[test], ds.lon[test])
                grids[method] = kernel_rate(ds.lat[src], ds.lon[src], y_src, archive.cell_lat, archive.cell_lon)
            t = truth.get(code)
            if t is not None:
                t = t[archive.mask]
            for method in METHODS:
                rows.append({
                    "month": M, "species_code": code, "method": method, "train_positives": int(y_tr.sum()),
                    "test_sets": int(test.size), "test_positives": int(y_te.sum()),
                    "auc": auc(preds[method], y_te), "brier": brier(preds[method], y_te),
                    "map_rho": spearman(grids[method][fp], t[fp]) if t is not None else np.nan})
            maps[M][code] = {"forecast": grids["forecast"], "forecast_no_heat": grids["forecast_no_heat"],
                             "truth": t}
        if verbose:
            f = [r for r in rows if r["month"] == M and r["method"] == "forecast"]
            print(f"  {M}: {test.size:4d} sets, {len(f)} species, mean AUC {np.nanmean([r['auc'] for r in f]):.3f}")
    return rows, maps


def _nanmean(values):
    v = np.asarray(values, float)
    return float(v[np.isfinite(v)].mean()) if np.isfinite(v).any() else np.nan


def summarise(rows, groups):
    out = []
    for code in sorted({r["species_code"] for r in rows}):
        mean = {}
        for method in METHODS:
            sel = [r for r in rows if r["species_code"] == code and r["method"] == method]
            mean[method] = {k: _nanmean([r[k] for r in sel]) for k in ("auc", "brier", "map_rho")}
            mean[method]["months"] = len(sel)
        keep_heat = mean["forecast"]["brier"] < (1 - MIN_HEAT_GAIN) * mean["forecast_no_heat"]["brier"]
        for method in METHODS:
            b = mean[method]["brier"]
            out.append({"species_code": code, "group": groups[code], "months": mean[method]["months"], "method": method,
                        "auc": mean[method]["auc"], "brier": b, "map_rho": mean[method]["map_rho"],
                        "brier_skill_vs_persistence": 1 - b / mean["persistence"]["brier"],
                        "brier_skill_vs_climatology": 1 - b / mean["climatology"]["brier"],
                        "heat_terms_kept": keep_heat if method.startswith("forecast") else ""})
    return out


# =========================
# RESPONSE CURVES AND THE UPCOMING MONTH
# =========================

def response_curves(model, F, n=25):
    """Odds multiplier exp(f(x)) for each heat/climate term, over the central 98% of the data."""
    curves = {}
    for term in CURVE_TERMS:
        if term not in model.smooth_terms:
            continue
        lo, hi = np.nanpercentile(F[term], [1, 99])
        x = np.linspace(lo, hi, n)
        curves[term] = {"x": x.round(4).tolist(), "odds": np.exp(model.partial(term, x)).round(4).tolist()}
    return curves


def bootstrap_forecast(ds, archive, F, valid, code, heat, month, n_boot, rng, seasonal=True):
    """Mean and spread of P(caught) over refits on trips resampled with replacement."""
    idx = np.flatnonzero(valid)
    trips = np.array([e.rsplit("-", 1)[0] for e in ds.event_id[idx]])
    uniq = np.unique(trips)
    by_trip = {t: idx[trips == t] for t in uniq}
    Fg = archive.grid_features(month)
    draws = []
    for _ in range(n_boot):
        pick = np.concatenate([by_trip[t] for t in rng.choice(uniq, uniq.size)])
        y = ds.presence[code][pick]
        if y.sum() < 5:
            continue
        draws.append(fit_forecast(_rows_subset(F, pick), y, heat, seasonal).predict(Fg))
    draws = np.array(draws)
    return draws.mean(axis=0), draws.std(axis=0)


# =========================
# DRIVER
# =========================

def _fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, float):
        return "" if not np.isfinite(v) else f"{v:.4g}"
    return v


def _write(path, rows, cols):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: _fmt(r.get(c)) for c in cols})


def run_forecast(data_dir, out_dir, start_month=None, n_boot=10, seed=0, plots=True, verbose=True):
    data_dir, out = Path(data_dir), Path(out_dir)
    (out / "maps").mkdir(parents=True, exist_ok=True)
    ds = load_dataset(data_dir)
    archive = EnvArchive(data_dir)
    F, months, valid = set_features(ds, archive)

    data_months = sorted(set(months.tolist()))
    if start_month is None:
        years = sorted({m[:4] for m in data_months})
        start_month = f"{years[1] if len(years) > 1 else years[0]}-01"
    # Only months with an environment grid (and so a map to score) and enough sets to test on.
    target_months = [m for m in data_months if m >= start_month and m in archive.grids
                     and int((months == m).sum()) >= MIN_TEST_SETS]
    if verbose:
        print(f"Rolling one-month-ahead forecast: {target_months[0]} .. {target_months[-1]} "
              f"({len(target_months)} months, {int(valid.sum())} sets with full lags)")
    rows, maps = rolling_validation(ds, archive, F, months, valid, target_months, verbose)
    summary = summarise(rows, ds.groups)
    _write(out / "forecast_metrics.csv", rows, METRIC_COLS)
    _write(out / "forecast_summary.csv", summary, SUMMARY_COLS)
    keep_heat = {r["species_code"]: bool(r["heat_terms_kept"]) for r in summary if r["method"] == "forecast"}

    for M, layers in maps.items():
        arrays = {"data_footprint": archive.to_2d(layers["footprint"].astype(float))}
        for code, v in layers.items():
            if code == "footprint":
                continue
            arrays[f"p_{code}"] = archive.to_2d(v["forecast"] if keep_heat[code] else v["forecast_no_heat"])
            if v["truth"] is not None:
                arrays[f"lambda_{code}"] = archive.to_2d(v["truth"])
        np.savez_compressed(out / "maps" / f"forecast_{M}.npz", lat=archive.lat, lon=archive.lon,
                            mask=archive.mask, **arrays)

    # The upcoming month: train on everything, forecast the month after the last grid.
    upcoming = add_months(archive.months[-1], 1)
    codes = sorted(keep_heat)
    curves, arrays = {}, {}
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(valid)
    Fall = _rows_subset(F, idx)
    seasonal = covers_full_year(months[idx])
    if archive.has_lags(upcoming):
        Fg = archive.grid_features(upcoming)
        arrays["data_footprint"] = archive.to_2d(
            footprint(archive.cell_lat, archive.cell_lon, ds.lat[idx], ds.lon[idx]).astype(float))
        for code in codes:
            model = fit_forecast(Fall, ds.presence[code][idx], keep_heat[code], seasonal)
            curves[code] = response_curves(model, Fall)
            arrays[f"p_{code}"] = archive.to_2d(model.predict(Fg))
            if n_boot > 1:
                _, sd = bootstrap_forecast(ds, archive, F, valid, code, keep_heat[code], upcoming, n_boot, rng, seasonal)
                arrays[f"sd_{code}"] = archive.to_2d(sd)
        np.savez_compressed(out / "maps" / f"forecast_{upcoming}.npz", lat=archive.lat, lon=archive.lon,
                            mask=archive.mask, **arrays)
    with open(out / "response_curves.json", "w", encoding="utf-8") as fh:
        json.dump({"terms": CURVE_TERMS, "species": curves}, fh)
    with open(out / "forecast_meta.json", "w", encoding="utf-8") as fh:
        json.dump({"target_months": sorted(maps), "upcoming_month": upcoming if arrays else None,
                   "train_sets": int(valid.sum()), "n_boot": n_boot, "kernel_km": KERNEL_KM,
                   "footprint_km": FOOTPRINT_KM, "heat_terms_kept": keep_heat,
                   "simulated": (data_dir / "truth").exists()}, fh)

    if plots and curves:
        try:
            from .plots import plot_response_curves
            plot_response_curves(out, curves)
        except ImportError:
            pass

    if verbose:
        print(f"\n{'species':<10} {'method':<17} {'AUC':>6} {'Brier':>7} {'map rho':>8} {'BSS clim':>9}")
        for r in summary:
            print(f"{r['species_code']:<10} {r['method']:<17} {_fmt(r['auc']) or '-':>6} {_fmt(r['brier']) or '-':>7} "
                  f"{_fmt(r['map_rho']) or '-':>8} {_fmt(r['brier_skill_vs_climatology']) or '-':>9}")
        kept = [c for c, k in keep_heat.items() if k]
        print(f"\nHeat terms kept (Brier at least {MIN_HEAT_GAIN:.1%} lower than without): {', '.join(kept) or 'none'}")
        if arrays:
            print(f"Upcoming month forecast: {upcoming}")
        print(f"Output written to {out.resolve()}")
    return summary
