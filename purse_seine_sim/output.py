"""Streaming CSV writers, truth maps, quick-look plots and the run summary."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from . import schema
from .environment import to_unix_days, unix_days_to_iso
from .geo import bearing_deg, distance_km


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "1" if v else "0"
    if isinstance(v, (float, np.floating)):
        return "" if not math.isfinite(v) else repr(float(v))
    return v


class Recorder:
    """Writes every table row-by-row so long multi-vessel runs stay light on memory."""

    def __init__(self, out_dir, cfg, species_codes, rng):
        self.out = Path(out_dir)
        self.cfg = cfg
        self.rng = rng
        self.species_codes = list(species_codes)
        self._files, self._writers, self._cols = {}, {}, {}
        tables = {t: schema.columns(t) for t in schema.TABLES}
        if not cfg.write_track:
            tables.pop("logbook/track.csv")
        tables["truth/truth_sets.csv"] = schema.truth_set_columns(self.species_codes)
        tables["truth/truth_catch.csv"] = schema.TRUTH_CATCH
        for table, cols in tables.items():
            path = self.out / table
            path.parent.mkdir(parents=True, exist_ok=True)
            f = open(path, "w", encoding="utf-8", newline="")
            w = csv.writer(f)
            w.writerow(cols)
            self._files[table], self._writers[table], self._cols[table] = f, w, cols

        # Kept for the summary and plots (small: one entry per set).
        self.set_lat, self.set_lon, self.set_type = [], [], []
        self.true_present = {c: [] for c in self.species_codes}
        self.reported_present = {c: [] for c in self.species_codes}
        self.n_trips = 0
        self.n_pings = 0
        self.example_segments = []

    def write(self, table, row: dict):
        self._writers[table].writerow([_fmt(row.get(c)) for c in self._cols[table]])

    def note_set(self, lat, lon, set_type, true_present: dict, reported_present: dict):
        self.set_lat.append(lat)
        self.set_lon.append(lon)
        self.set_type.append(set_type)
        for c in self.species_codes:
            self.true_present[c].append(bool(true_present.get(c, False)))
            self.reported_present[c].append(bool(reported_present.get(c, False)))

    def track(self, vessel_id, trip_id, segments):
        if len(self.example_segments) < 3 and vessel_id.endswith("01"):
            self.example_segments.append(list(segments))
        if not self.cfg.write_track:
            return
        dp = self.cfg.track_ping_min / 1440.0
        drop = self.cfg.reporting.track_drop_prob
        w = self._writers["logbook/track.csv"]
        rng = self.rng
        for t0, t1, la0, lo0, la1, lo1, activity in segments:
            if t1 <= t0:
                continue
            ts = np.arange(np.ceil(t0 / dp), np.ceil(t1 / dp)) * dp
            ts = ts[(ts >= t0) & (ts < t1)]
            if ts.size == 0:
                continue
            f = (ts - t0) / (t1 - t0)
            lat = la0 + f * (la1 - la0)
            lon = lo0 + f * (lo1 - lo0)
            d = float(distance_km(la0, lo0, la1, lo1))
            sog = d / ((t1 - t0) * 24.0) / 1.852
            speed = np.clip(sog + rng.normal(0.0, 0.15, ts.size), 0.0, None)
            course = (np.full(ts.size, float(bearing_deg(la0, lo0, la1, lo1))) if d > 0.05
                      else rng.uniform(0.0, 360.0, ts.size))
            keep = rng.random(ts.size) >= drop
            iso = unix_days_to_iso(ts)
            for k in np.flatnonzero(keep):
                w.writerow([vessel_id, trip_id, iso[k], round(float(lat[k]), 5), round(float(lon[k]), 5),
                            round(float(speed[k]), 2), round(float(course[k]), 1), activity])
            self.n_pings += int(keep.sum())

    def close(self):
        for f in self._files.values():
            f.close()

    # ---------- summary ----------

    def summary(self, species_groups: dict, n_vessels: int):
        n = len(self.set_lat)
        rows = []
        for c in self.species_codes:
            pos = int(np.sum(self.true_present[c]))
            rep = int(np.sum(self.reported_present[c]))
            rate = pos / n if n else 0.0
            rows.append({
                "species_code": c, "group": species_groups.get(c, ""),
                "positive_sets": pos, "reported_positive_sets": rep,
                "positive_rate": round(rate, 5),
                "sets_needed_for_200_positives": (math.ceil(200 / rate) if rate > 0 else None),
            })
        types = {t: self.set_type.count(t) for t in schema.SET_TYPES}
        return {"n_vessels": n_vessels, "n_trips": self.n_trips, "n_sets": n, "n_track_pings": self.n_pings,
                "sets_by_type": types, "species": rows}


def print_summary(summary, cfg):
    print(f"Simulated {summary['n_vessels']} vessels, {cfg.start} -> {cfg.end}: "
          f"{summary['n_trips']} trips, {summary['n_sets']} sets, {summary['n_track_pings']} track pings")
    print("Sets by type: " + ", ".join(f"{k}={v}" for k, v in summary["sets_by_type"].items()))
    print()
    print(f"{'species':<10} {'group':<8} {'positive sets':>13} {'rate':>8} {'sets for 200 positives':>23}")
    for r in summary["species"]:
        need = r["sets_needed_for_200_positives"]
        enough = "enough" if r["positive_sets"] >= 200 else "NOT enough"
        print(f"{r['species_code']:<10} {r['group']:<8} {r['positive_sets']:>13} {100 * r['positive_rate']:>7.2f}% "
              f"{(need if need else '-'):>23}   {enough}")


# =========================
# TRUTH MAPS
# =========================

def months_between(start, end):
    m0, m1 = np.datetime64(start[:7], "M"), np.datetime64(end[:7], "M")
    return [str(m) for m in np.arange(m0, m1 + 1)]


def write_truth_maps(out_dir, cfg, geo, env, models):
    """Monthly (15th, 12:00 local) truth grids: environment + lambda for every species."""
    res = cfg.region.map_res_deg
    r = cfg.region
    lats = np.arange(r.lat_min + res / 2, r.lat_max, res)
    lons = np.arange(r.lon_min + res / 2, r.lon_max, res)
    LAT, LON = np.meshgrid(lats, lons, indexing="ij")
    ocean = geo.is_ocean(LAT, LON)
    folder = Path(out_dir) / "truth" / "maps"
    folder.mkdir(parents=True, exist_ok=True)
    mean_lam = {m.sp.code: np.zeros(LAT.shape) for m in models}
    months = months_between(cfg.start, cfg.end)
    for month in months:
        t = to_unix_days(f"{month}-15T05:00")
        s = env.sample(LAT[ocean], LON[ocean], t)
        fields = {}
        for key in ("sst", "sst_anom", "log10chl", "front", "wind", "depth", "dist_coast", "seagrass_idx"):
            g = np.full(LAT.shape, np.nan, dtype=np.float32)
            g[ocean] = s[key]
            fields[key] = g
        for m in models:
            g = np.full(LAT.shape, np.nan, dtype=np.float32)
            g[ocean] = m.density(s)
            fields[f"lambda_{m.sp.code}"] = g
            mean_lam[m.sp.code] += np.nan_to_num(g) / len(months)
        np.savez_compressed(folder / f"truth_{month}.npz", lat=lats, lon=lons, ocean=ocean, **fields)
    for k in mean_lam:
        mean_lam[k][~ocean] = np.nan
    return lats, lons, mean_lam


# =========================
# QUICK-LOOK PLOTS
# =========================

ACTIVITY_COLORS = {"steaming": "#4a6fa5", "searching": "#e0a030", "drifting": "#999999",
                   "light_luring": "#9b59b6", "setting": "#c0392b"}


def make_plots(out_dir, cfg, geo, rec, lats, lons, mean_lam, species_groups):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = Path(out_dir) / "plots"
    folder.mkdir(parents=True, exist_ok=True)
    land = np.where(geo.land, 1.0, np.nan)
    extent = [cfg.region.lon_min, cfg.region.lon_max, cfg.region.lat_min, cfg.region.lat_max]
    set_lat, set_lon = np.array(rec.set_lat), np.array(rec.set_lon)

    def base(ax):
        ax.imshow(land, origin="lower", extent=extent, cmap="Greys", vmin=0, vmax=2.5, interpolation="nearest")
        for c in cfg.regulation.closures:
            ax.plot([c["lon_min"], c["lon_max"], c["lon_max"], c["lon_min"], c["lon_min"]],
                    [c["lat_min"], c["lat_min"], c["lat_max"], c["lat_max"], c["lat_min"]],
                    ls="--", lw=0.8, color="#333")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)

    codes = list(mean_lam)
    ncol = 4
    nrow = math.ceil(len(codes) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 5.6 * nrow))
    for ax, code in zip(axes.ravel(), codes):
        base(ax)
        im = ax.pcolormesh(lons, lats, mean_lam[code], cmap="viridis", shading="auto")
        pres = np.array(rec.true_present[code], bool)
        if set_lat.size:
            ax.scatter(set_lon[~pres], set_lat[~pres], s=1.5, c="white", alpha=0.35, lw=0)
            ax.scatter(set_lon[pres], set_lat[pres], s=5, c="#ff3b30", lw=0)
        ax.set_title(f"{code} ({species_groups[code]}): {pres.sum()}/{pres.size} sets positive", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.6).ax.tick_params(labelsize=6)
    for ax in axes.ravel()[len(codes):]:
        ax.axis("off")
    fig.suptitle("True mean density index (lambda) with simulated sets: red = caught, white = zero", fontsize=11)
    fig.tight_layout()
    fig.savefig(folder / "quicklook_species.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 8))
    base(ax)
    for segments in rec.example_segments:
        for t0, t1, la0, lo0, la1, lo1, act in segments:
            ax.plot([lo0, lo1], [la0, la1], color=ACTIVITY_COLORS.get(act, "k"), lw=0.9)
    for act, col in ACTIVITY_COLORS.items():
        ax.plot([], [], color=col, label=act)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("First trips of vessel 01 (track segments by activity)", fontsize=9)
    fig.tight_layout()
    fig.savefig(folder / "quicklook_tracks.png", dpi=130)
    plt.close(fig)


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path, rows, cols):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: _fmt(r.get(c)) for c in cols})
