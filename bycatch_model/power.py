"""Power analysis: how many sets are needed before each species model is useful?

Train on random subsets of N sets from the training years, score on the held-out
year against the logbook (AUC) and against the simulator's true density maps.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import load_dataset
from .ecocast import DEFAULT_WEIGHTS, fit_all
from .train import ecocast_skill, evaluate_fits, grid_footprints, load_test_grids, split_by_year, write_rows

RAW_COLS = ["n_sets", "rep", "species_code", "group", "train_positives", "test_auc", "ceiling_auc", "map_spearman"]
SUMMARY_COLS = ["n_sets", "species_code", "group", "train_positives", "test_auc", "test_auc_sd",
                "ceiling_auc", "map_spearman", "map_spearman_sd", "reps_fitted", "reps"]


def power_analysis(data_dir, out_dir, sizes=(200, 500, 1000, 2000, 4000, "all"), reps=3, seed=0,
                   test_year=None, map_months=(1, 4, 7, 10), verbose=True):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(data_dir)
    train, test, test_year = split_by_year(ds, test_year)
    grids = load_test_grids(data_dir, test_year, months=map_months)
    rng = np.random.default_rng(seed)

    raw, eco = [], []
    for size in sizes:
        n = len(train) if size == "all" else min(int(size), len(train))
        n_reps = 1 if n == len(train) else reps
        for rep in range(n_reps):
            sub = train.subset(np.sort(rng.choice(len(train), n, replace=False)))
            fits = fit_all(sub)
            footprints = grid_footprints(grids, sub)
            for r in evaluate_fits(fits, test, grids, footprints):
                raw.append({"n_sets": n, "rep": rep, **{k: r[k] for k in RAW_COLS if k in r}})
            eco.append({"n_sets": n, "rep": rep, "ecocast_spearman": ecocast_skill(fits, grids, footprints, DEFAULT_WEIGHTS)})
            if verbose:
                print(f"  n={n:>6} rep={rep}: done")
        if n == len(train) and size != "all":
            break

    summary = []
    for n in sorted({r["n_sets"] for r in raw}):
        for code in dict.fromkeys(r["species_code"] for r in raw):
            rs = [r for r in raw if r["n_sets"] == n and r["species_code"] == code]
            a = np.array([r["test_auc"] for r in rs], float)
            m = np.array([r["map_spearman"] for r in rs], float)
            summary.append({
                "n_sets": n, "species_code": code, "group": rs[0]["group"],
                "train_positives": float(np.mean([r["train_positives"] for r in rs])),
                "test_auc": np.nanmean(a) if np.isfinite(a).any() else np.nan,
                "test_auc_sd": np.nanstd(a) if np.isfinite(a).sum() > 1 else np.nan,
                "ceiling_auc": float(rs[0]["ceiling_auc"]),
                "map_spearman": np.nanmean(m) if np.isfinite(m).any() else np.nan,
                "map_spearman_sd": np.nanstd(m) if np.isfinite(m).sum() > 1 else np.nan,
                "reps_fitted": int(np.isfinite(a).sum()), "reps": len(rs)})

    write_rows(out / "power_raw.csv", raw, RAW_COLS)
    write_rows(out / "power_summary.csv", summary, SUMMARY_COLS)
    write_rows(out / "power_ecocast.csv", eco, ["n_sets", "rep", "ecocast_spearman"])

    from .plots import plot_power
    plot_power(out, summary, eco)

    if verbose:
        sizes_done = sorted({r["n_sets"] for r in summary})
        codes = list(dict.fromkeys(r["species_code"] for r in summary))
        print(f"\nHeld-out year {test_year}: test AUC by training size (ceiling = AUC of the true probabilities)")
        print(f"{'species':<10}" + "".join(f"{n:>8}" for n in sizes_done) + f"{'ceiling':>9}")
        for code in codes:
            cells = []
            for n in sizes_done:
                r = next(x for x in summary if x["n_sets"] == n and x["species_code"] == code)
                cells.append(f"{r['test_auc']:>8.3f}" if np.isfinite(r["test_auc"]) else f"{'-':>8}")
            ceil = next(x for x in summary if x["species_code"] == code)["ceiling_auc"]
            print(f"{code:<10}" + "".join(cells) + (f"{ceil:>9.3f}" if np.isfinite(ceil) else f"{'-':>9}"))
        print(f"\n{'EcoCast map vs truth (Spearman)':<34}" + "".join(
            f"{np.nanmean([e['ecocast_spearman'] for e in eco if e['n_sets'] == n]):>8.3f}" for n in sizes_done))
        print(f"\nOutput written to {out.resolve()}")
    return summary, eco
