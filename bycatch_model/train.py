"""Train per-species models, score them on a held-out year, write predicted maps."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .data import load_dataset, load_grid
from .ecocast import (DEFAULT_WEIGHTS, auc, ecocast_score, fit_all, footprint, predict_presence,
                      predicted_layers, spearman, true_layers)

METRIC_COLS = ["species_code", "group", "train_sets", "train_positives", "test_sets", "test_positives",
               "test_auc", "ceiling_auc", "map_spearman", "map_spearman_all", "edf", "smoothing_lambda", "note"]


HOLDOUT_MONTHS = 3      # with only one year of data, the last 3 months are held out instead


def split_by_year(ds, test_year=None):
    """Held-out year; with a single year, the last HOLDOUT_MONTHS months. Returns train, test, test months."""
    years = sorted(set(ds.year.tolist()))
    if len(years) < 2:
        ym = (ds.time // 1).astype("int64").astype("datetime64[D]").astype("datetime64[M]")
        months = sorted(set(ym.tolist()))
        if len(months) <= HOLDOUT_MONTHS:
            raise ValueError("need more than three months of data for a held-out test")
        test = np.isin(ym, months[-HOLDOUT_MONTHS:])
        label = [str(np.datetime64(m, "M")) for m in months[-HOLDOUT_MONTHS:]]
        return ds.subset(np.flatnonzero(~test)), ds.subset(np.flatnonzero(test)), label
    test_year = test_year or years[-1]
    return ds.subset(np.flatnonzero(ds.year != test_year)), ds.subset(np.flatnonzero(ds.year == test_year)), test_year


def load_test_grids(data_dir, test_year, months=None):
    folder = Path(data_dir) / "truth" / "maps"
    if isinstance(test_year, list):            # single-year data: the held-out months
        paths = [folder / f"truth_{m}.npz" for m in test_year if (folder / f"truth_{m}.npz").exists()]
    else:
        paths = sorted(folder.glob(f"truth_{test_year}-*.npz")) if folder.exists() else []
    if months:
        paths = [p for p in paths if int(p.stem[-2:]) in months]
    return [load_grid(p) for p in paths]


def grid_footprints(grids, train):
    return [footprint(*g.cell_latlon, train.lat, train.lon) for g in grids]


def evaluate_fits(fits, test, grids, footprints):
    rows = []
    for code, fit in fits.items():
        y = test.presence[code]
        p = predict_presence(fit, test.features)
        ceiling = auc(test.truth_p[code], y) if code in test.truth_p else np.nan
        maps, maps_all = [], []
        if fit.presence is not None:
            for g, fp in zip(grids, footprints):
                if code in g.truth_lambda:
                    pred = predict_presence(fit, g.features)
                    maps.append(spearman(pred[fp], g.truth_lambda[code][fp]))
                    maps_all.append(spearman(pred, g.truth_lambda[code]))
        note = "" if fit.presence is not None else f"not fitted: {fit.n_pos} positives in training"
        rows.append({
            "species_code": code, "group": fit.group, "train_sets": fit.n_train, "train_positives": fit.n_pos,
            "test_sets": int(y.size), "test_positives": int(y.sum()),
            "test_auc": auc(p, y), "ceiling_auc": ceiling,
            "map_spearman": float(np.mean(maps)) if maps else np.nan,
            "map_spearman_all": float(np.mean(maps_all)) if maps_all else np.nan,
            "edf": fit.presence.edf if fit.presence else np.nan,
            "smoothing_lambda": fit.presence.lam if fit.presence else np.nan, "note": note})
    return rows


def ecocast_skill(fits, grids, footprints, weights=DEFAULT_WEIGHTS):
    vals = []
    for g, fp in zip(grids, footprints):
        if not g.truth_lambda:
            continue
        layers, _ = predicted_layers(fits, g.features)
        pred, used = ecocast_score(layers, weights, fp)
        true, _ = ecocast_score(true_layers(g.truth_lambda), {k: weights[k] for k in used}, fp)
        vals.append(spearman(pred[fp], true[fp]))
    return float(np.mean(vals)) if vals else np.nan


def _fmt(v):
    if isinstance(v, float):
        return "" if not np.isfinite(v) else f"{v:.4g}"
    return v


def write_rows(path, rows, cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: _fmt(r.get(c)) for c in cols})


def train_and_evaluate(data_dir, out_dir, test_year=None, plot_months=(4, 10), weights=DEFAULT_WEIGHTS,
                       verbose=True):
    data_dir, out = Path(data_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(data_dir)
    train, test, test_year = split_by_year(ds, test_year)
    fits = fit_all(train)
    grids = load_test_grids(data_dir, test_year)
    footprints = grid_footprints(grids, train)

    rows = evaluate_fits(fits, test, grids, footprints)
    write_rows(out / "metrics.csv", rows, METRIC_COLS)
    held_out = None
    if isinstance(test_year, list):
        # Single year: score on the held-out months, but map every month of the year.
        held_out = set(test_year)
        grids = load_test_grids(data_dir, test_year[0][:4])
        footprints = grid_footprints(grids, train)
    with open(out / "holdout.json", "w", encoding="utf-8") as fh:
        json.dump({"test_year": None if held_out else int(test_year),
                   "test_months": sorted(held_out) if held_out else None}, fh)
    models = {c: {"presence": f.presence.to_dict() if f.presence else None,
                  "amount": f.amount.to_dict() if f.amount else None,
                  "group": f.group, "train_positives": f.n_pos} for c, f in fits.items()}
    with open(out / "models.json", "w", encoding="utf-8") as fh:
        json.dump(models, fh)

    skill = []
    (out / "maps").mkdir(exist_ok=True)
    for g, fp in zip(grids, footprints):
        layers, preds = predicted_layers(fits, g.features)
        score, used = ecocast_score(layers, weights, fp)
        arrays = {k: g.to_2d(v) for k, v in preds.items()}
        arrays["ecocast_score"] = g.to_2d(score)
        arrays["revenue_thb_per_set"] = g.to_2d(layers["revenue"])
        arrays["data_footprint"] = g.to_2d(fp.astype(float))
        if g.truth_lambda:
            true, _ = ecocast_score(true_layers(g.truth_lambda), {k: weights[k] for k in used}, fp)
            arrays["true_ecocast_score"] = g.to_2d(true)
            if held_out is None or g.month in held_out:
                skill.append({"month": g.month, "ecocast_spearman": spearman(score[fp], true[fp]),
                              "layers_used": "+".join(used)})
        np.savez_compressed(out / "maps" / f"pred_{g.month}.npz", lat=g.lat, lon=g.lon, mask=g.mask, **arrays)
    write_rows(out / "ecocast_skill.csv", skill, ["month", "ecocast_spearman", "layers_used"])

    from .plots import plot_ecocast_month, plot_species_maps
    for g, fp in zip(grids, footprints):
        if int(g.month[-2:]) in plot_months:
            plot_ecocast_month(out, g, fits, fp, weights)
            plot_species_maps(out, g, fits, fp)

    if verbose:
        print(f"Trained on {len(train)} sets ({', '.join(map(str, sorted(set(train.year.tolist()))))}), "
              f"tested on {len(test)} sets ({test_year if not isinstance(test_year, list) else ', '.join(test_year)})\n")
        if footprints:
            print(f"map rho = rank correlation of predicted vs true density inside the data footprint "
                  f"({np.mean([f.mean() for f in footprints]) * 100:.0f}% of legal cells); 'all' = whole area\n")
        print(f"{'species':<10} {'group':<7} {'train +':>8} {'test +':>7} {'test AUC':>9} {'ceiling':>8} "
              f"{'map rho':>8} {'all':>7}")
        for r in rows:
            print(f"{r['species_code']:<10} {r['group']:<7} {r['train_positives']:>8} {r['test_positives']:>7} "
                  f"{_fmt(r['test_auc']) or '-':>9} {_fmt(r['ceiling_auc']) or '-':>8} "
                  f"{_fmt(r['map_spearman']) or '-':>8} {_fmt(r['map_spearman_all']) or '-':>7}")
        if skill:
            print(f"\nEcoCast score vs truth inside footprint (Spearman, mean over {len(skill)} months): "
                  f"{np.mean([s['ecocast_spearman'] for s in skill]):.3f}")
        print(f"\nOutput written to {out.resolve()}")
    return rows, fits
