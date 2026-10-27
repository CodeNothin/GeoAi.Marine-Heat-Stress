"""Figures for the trained models and the power analysis."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from purse_seine_sim.geo import is_land_analytic

from .ecocast import DEFAULT_WEIGHTS, ecocast_score, predict_presence, predicted_layers, true_layers


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _panel(ax, g, values, title, cmap="viridis", vmin=None, vmax=None, fp=None):
    LON, LAT = np.meshgrid(g.lon, g.lat)
    land = np.where(is_land_analytic(LAT, LON), 1.0, np.nan)
    ax.pcolormesh(g.lon, g.lat, land, cmap="Greys", vmin=0, vmax=2.5, shading="auto")
    values = np.asarray(values, float)
    if fp is not None:
        ax.pcolormesh(g.lon, g.lat, g.to_2d(np.where(fp, np.nan, 1.0)), cmap="Greys", vmin=0, vmax=1.6, shading="auto")
        values = np.where(fp, values, np.nan)
        if vmin is None and np.isfinite(values).any():
            vmin, vmax = np.nanpercentile(values, [1, 99])
    im = ax.pcolormesh(g.lon, g.lat, g.to_2d(values), cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)
    return im


def plot_ecocast_month(out, g, fits, fp, weights=DEFAULT_WEIGHTS):
    plt = _plt()
    layers, _ = predicted_layers(fits, g.features)
    score, used = ecocast_score(layers, weights, fp)
    lim = max(float(np.nanpercentile(np.abs(score[fp]), 99)), 1e-6) if fp.any() else 1.0
    panels = [(score, "Predicted EcoCast score\n(+ fish good, - bycatch risk)", "RdBu", -lim, lim)]
    if g.truth_lambda:
        true, _ = ecocast_score(true_layers(g.truth_lambda), {k: weights[k] for k in used}, fp)
        tlim = max(float(np.nanpercentile(np.abs(true[fp]), 99)), 1e-6) if fp.any() else 1.0
        panels.append((true, "TRUE EcoCast score (simulator)", "RdBu", -tlim, tlim))
    panels.append((layers["revenue"], "Predicted revenue per set (THB)", "viridis", None, None))
    risk = sum(weights[k] * np.nan_to_num(layers[k]) / max(np.nanpercentile(layers[k][fp], 99), 1e-12)
               for k in used if k != "revenue")
    panels.append((risk, "Predicted weighted bycatch risk", "magma", None, None))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 5.2))
    for ax, (v, title, cmap, vmin, vmax) in zip(axes, panels):
        fig.colorbar(_panel(ax, g, v, title, cmap, vmin, vmax, fp), ax=ax, shrink=0.6).ax.tick_params(labelsize=6)
    skipped = [k for k in weights if k not in used]
    fig.suptitle(f"EcoCast-style map, {g.month} (free-school set; dark grey = no training data within 40 km)"
                 + (f"; layers without a model: {', '.join(skipped)}" if skipped else ""), fontsize=10)
    fig.tight_layout()
    fig.savefig(Path(out) / f"ecocast_{g.month}.png", dpi=130)
    plt.close(fig)


def plot_species_maps(out, g, fits, fp):
    plt = _plt()
    codes = [c for c, f in fits.items() if f.presence is not None and c in g.truth_lambda]
    if not codes:
        return
    fig, axes = plt.subplots(2, len(codes), figsize=(2.6 * len(codes), 7.2))
    axes = np.atleast_2d(axes).reshape(2, -1)
    for j, code in enumerate(codes):
        f = fits[code]
        _panel(axes[0, j], g, predict_presence(f, g.features), f"{code} predicted P(catch)\n({f.n_pos} train positives)", fp=fp)
        _panel(axes[1, j], g, g.truth_lambda[code], f"{code} TRUE density", fp=fp)
    fig.suptitle(f"Predicted vs true distribution, {g.month}", fontsize=10)
    fig.tight_layout()
    fig.savefig(Path(out) / f"species_pred_vs_true_{g.month}.png", dpi=120)
    plt.close(fig)


def plot_power(out, summary, eco):
    plt = _plt()
    codes = list(dict.fromkeys(r["species_code"] for r in summary))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    cmap = plt.get_cmap("tab20")
    for i, code in enumerate(codes):
        rs = sorted((r for r in summary if r["species_code"] == code), key=lambda r: r["n_sets"])
        n = np.array([r["n_sets"] for r in rs])
        style = "-" if rs[0]["group"] == "target" else "--"
        for ax, key in ((axes[0], "test_auc"), (axes[1], "map_spearman")):
            v = np.array([r[key] for r in rs], float)
            if np.isfinite(v).any():
                ax.plot(n, v, style, marker="o", ms=3, color=cmap(i % 20), label=code)
    ns = sorted({e["n_sets"] for e in eco})
    axes[2].plot(ns, [np.nanmean([e["ecocast_spearman"] for e in eco if e["n_sets"] == n]) for n in ns],
                 "-o", color="#c0392b")
    titles = ["Held-out-year AUC (logbook presence)", "Map correlation with true density",
              "EcoCast score correlation with truth"]
    for ax, title in zip(axes, titles):
        ax.set_xscale("log")
        ax.set_xlabel("training sets")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].axhline(0.5, color="k", lw=0.6, ls=":")
    axes[0].legend(fontsize=7, ncol=2, title="solid = target, dashed = ETP", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(Path(out) / "power_curves.png", dpi=130)
    plt.close(fig)


CURVE_LABELS = {"sst_anom_l1": "SST anomaly last month (°C)", "heat_3m": "Heat stress, last 3 months (°C·months)",
                "anom_trend": "SST anomaly change (°C)", "sst_l1": "SST last month (°C)",
                "log10chl_l1": "log10 CHL-a last month", "wind_l1": "Wind last month (m/s)"}


def plot_response_curves(out, curves):
    """One panel per heat/climate term; one line per species (odds multiplier, log scale)."""
    plt = _plt()
    terms = [t for t in CURVE_LABELS if any(t in c for c in curves.values())]
    cols = 3
    rows = math.ceil(len(terms) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for ax, term in zip(axes.ravel(), terms):
        for code, c in curves.items():
            if term in c:
                ax.plot(c[term]["x"], c[term]["odds"], lw=1.2, label=code)
        ax.axhline(1.0, color="#999", lw=0.6, ls="--")
        ax.set_yscale("log")
        ax.set_xlabel(CURVE_LABELS[term], fontsize=8)
        ax.set_ylabel("odds of catch (x)", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(terms):]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    (Path(out) / "plots").mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(out) / "plots" / "response_curves.png", dpi=130)
    plt.close(fig)
