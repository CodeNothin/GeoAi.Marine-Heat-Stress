"""MAP 1: one-month-ahead fish distribution from satellite fronts, zonal statistics and a model comparison.

    python -m bycatch_model map1 --data sim_output --env real_env --out forecast_output

The steps follow the project's MAP 1 workflow diagram:

  Chl-a, SST, ERA5 wind ─┐                                  ┌─ Bathymetry, coastline
  1  CRS harmonizing      (WGS84 lat/lon, ascending axes, deg C / mg m-3 / m s-1 / m)
  2  Raster clipping      sub-process: take only the study area (sea within 50 km of the coast)
                          -> split it into a multi-buffer mesh: 10 km coast-distance rings
                             (0-10 ... 40-50 km) x the 0.05 deg map cells
  3  Raster resampling    bilinear, onto one 0.01 deg (~1.1 km) working grid
  4  Gradient estimation  Sobel |grad| of SST, log10 CHL-a and wind; bathymetric slope
  5  Foreground/background pixels, adaptive threshold
                          Otsu threshold of log |grad| in overlapping ~35 km windows; a pixel is
                          foreground where most of its windows call it so (only windows whose
                          gradients split cleanly into two classes count), and only in the month's
                          top 10% of |grad|
  6  Mesoscale thermal and ocean-colour fronts
                          foreground pixels in connected fronts of at least MIN_FRONT_PX pixels
  7  Minimum distance to thermal and ocean-colour fronts   (Euclidean distance transform, km)
  8  Zonal statistics     per mesh zone (map cell x ring): means, SD, min distance, front share
  Biomass data            logbook sets (catch per set), placed in their mesh zone
  9  Collinearity assessment: variance inflation factors; the highest is dropped until all <= MAX_VIF
 10  X proxy matrix       zone statistics of the PREVIOUS month for every set (one month ahead),
                          split 70% train / 30% validation by zone-month (sets of one zone-month
                          share their X, so they stay on one side)
 11  Classification (caught or not): GLM, GAM, boosted decision trees, random forest
       -> Taylor diagram statistics, MAE, RMSE, Spearman correlation, Jensen-Shannon divergence on the
          validation zone-months -> best classification model (mean rank over the metrics)
     Regression (kg per set): random forest -> 5-fold cross-validation -> accuracy on the 30%
 12  SHAP interpretation of the best classifier and of the regression forest
 13  Visualization: maps per month (best model refitted on all sets), written in the same format as
     bycatch_model.forecast so export_forecast_web.py and the website read them.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

from .data import load_dataset
from .ecocast import FOOTPRINT_KM, MIN_POSITIVES_SCORE, auc, footprint
from .gam import GAM

# Working grid: 0.01 deg pixels, 5 x 5 per 0.05 deg map cell, over the website's study-area box.
LAT0, LAT1, LON0, LON1 = 6.30, 11.50, 99.00, 102.20
FINE = 0.01
CELL = 0.05
PAD = 0.5                    # extra margin for coast distance (Malaysian / Cambodian coast outside the box)
BUFFER_KM = 50.0
RING_KM = 10.0               # multi-buffer mesh: rings of this width
MIN_SEA_SHARE = 1 / 3        # a map cell needs this share of sea pixels (else island / land)
WINDOW_PX, STRIDE_PX = 32, 16
MIN_BIMODAL = 0.75           # between-class / total variance for a window's threshold to count
FRONT_FLOOR_Q = 90           # and a front must also be in the month's top 10% of |grad| over the sea
MIN_FRONT_PX = 12            # mesoscale: connected fronts shorter than ~12 km are noise
MAX_VIF = 5.0
TRAIN_SHARE = 0.7
SEED = 42
MODELS = ["GLM", "GAM", "BRT", "RF"]
MODEL_NAMES = {"GLM": "GLM (logistic)", "GAM": "GAM", "BRT": "Boosted decision tree", "RF": "Random forest classification"}

# X proxy matrix candidates (zonal statistics of the previous month + static seabed/coast).
FEATURES = ["sst_mean", "sst_sd", "sst_anom_mean", "sst_grad_mean", "sst_front_km", "sst_front_share",
            "chl_mean", "chl_grad_mean", "chl_front_km", "chl_front_share",
            "wind_mean", "wind_grad_mean", "depth_mean", "slope_mean", "dist_coast_mean", "ring"]
FEATURE_TH = {
    "sst_mean": "SST เฉลี่ย (°C)", "sst_sd": "ความแปรปรวน SST ในโซน (°C)", "sst_anom_mean": "ค่าผิดปกติ SST (°C)",
    "sst_grad_mean": "ความชัน SST (°C/กม.)", "sst_front_km": "ระยะถึงแนวหน้าน้ำอุณหภูมิ (กม.)",
    "sst_front_share": "สัดส่วนพื้นที่แนวหน้าน้ำอุณหภูมิ", "chl_mean": "log10 CHL-a",
    "chl_grad_mean": "ความชัน CHL-a (log10/กม.)", "chl_front_km": "ระยะถึงแนวหน้าน้ำสีน้ำ (กม.)",
    "chl_front_share": "สัดส่วนพื้นที่แนวหน้าน้ำสีน้ำ", "wind_mean": "ลม (ม./วินาที)",
    "wind_grad_mean": "ความชันลม (ม./วินาที/กม.)", "depth_mean": "ความลึก (ม.)", "slope_mean": "ความชันพื้นทะเล (ม./กม.)",
    "dist_coast_mean": "ระยะจากฝั่ง (กม.)", "ring": "วงระยะจากฝั่ง (0=0-10 กม.)"}


def month_str(m) -> str:
    return str(np.datetime64(m, "M"))


def add_months(m, k) -> str:
    return month_str(np.datetime64(m, "M") + np.timedelta64(int(k), "M"))


def _days_to_month(t):
    return np.array([month_str(np.datetime64(int(round(x * 86400)), "s")) for x in np.atleast_1d(t)])


# =========================
# 1  CRS HARMONIZING
# =========================

def harmonize(lat, lon, values, kind):
    """Bring a raster to WGS84 lat/lon with ascending axes, longitudes in -180..180 and standard units.

    All sources here are already geographic WGS84 (EPSG:4326) grids, so harmonizing means fixing axis
    order and direction, the longitude convention and the units, not reprojecting."""
    lat, lon, v = np.asarray(lat, float), np.asarray(lon, float), np.asarray(values, float)
    lon = np.where(lon > 180, lon - 360, lon)
    if lat[0] > lat[-1]:
        lat, v = lat[::-1], v[..., ::-1, :]
    order = np.argsort(lon)
    lon, v = lon[order], v[..., order]
    if kind == "sst" and np.nanmedian(v) > 200:          # Kelvin -> deg C
        v = v - 273.15
    if kind == "chl":
        v = np.where(v > 0, v, np.nan)                   # mg m-3, no zero/negative chlorophyll
    if kind == "elevation":                              # m, positive up
        v = np.where(np.isfinite(v), v, np.nan)
    return {"lat": lat, "lon": lon, "values": v, "crs": "EPSG:4326", "kind": kind}


def load_sources(env_dir):
    """Monthly SST (+ anomaly), CHL-a and wind cubes and the bathymetry, harmonized."""
    env = Path(env_dir)
    s = np.load(env / "sst_daily.npz")
    months_daily = _days_to_month(s["time"])
    months = sorted(set(months_daily.tolist()))
    sst = np.array([np.nanmean(s["sst"][months_daily == m], axis=0) for m in months])
    anom = np.array([np.nanmean(s["sst_anom"][months_daily == m], axis=0) for m in months])
    out = {"sst": dict(harmonize(s["lat"], s["lon"], sst, "sst"), months=months),
           "sst_anom": dict(harmonize(s["lat"], s["lon"], anom, "sst_anom"), months=months)}
    c = np.load(env / "chl_monthly.npz")
    out["chl"] = dict(harmonize(c["lat"], c["lon"], c["chl"], "chl"), months=_days_to_month(c["time"]).tolist())
    w = np.load(env / "wind_monthly.npz")
    out["wind"] = dict(harmonize(w["lat"], w["lon"], w["wind"], "wind"), months=_days_to_month(w["time"]).tolist())
    b = np.load(env / "bathy.npz")
    out["bathy"] = harmonize(b["lat"], b["lon"], b["elevation"], "elevation")
    return out


# =========================
# 3  RASTER RESAMPLING  (used by 2 and 4)
# =========================

def fill_nearest(v):
    """Fill NaN (land, cloud) with the nearest valid value so interpolation does not bleed NaN into the sea."""
    bad = ~np.isfinite(v)
    if not bad.any() or bad.all():
        return v
    idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return v[tuple(idx)]


def resample(src, lat_t, lon_t, fill=True):
    """Bilinear resampling of a 2-D (lat, lon) raster onto the target axes."""
    v = fill_nearest(src["values_2d"]) if fill else src["values_2d"]
    f = RegularGridInterpolator((src["lat"], src["lon"]), v, method="linear", bounds_error=False, fill_value=None)
    LA, LO = np.meshgrid(lat_t, lon_t, indexing="ij")
    return f(np.stack([LA.ravel(), LO.ravel()], axis=1)).reshape(LA.shape)


def _layer(src, k=None):
    return {"lat": src["lat"], "lon": src["lon"], "values_2d": src["values"] if k is None else src["values"][k]}


# =========================
# 2  RASTER CLIPPING -> STUDY AREA -> MULTI-BUFFER MESH
# =========================

class Mesh:
    """Working grid, study area and the multi-buffer mesh zones (map cell x coast-distance ring)."""

    def __init__(self, bathy, study_cells=None):
        n_pad = int(round(PAD / FINE))
        self.lat = LAT0 + FINE / 2 + FINE * np.arange(int(round((LAT1 - LAT0) / FINE)))
        self.lon = LON0 + FINE / 2 + FINE * np.arange(int(round((LON1 - LON0) / FINE)))
        lat_p = np.concatenate([self.lat[0] - FINE * np.arange(n_pad, 0, -1), self.lat, self.lat[-1] + FINE * np.arange(1, n_pad + 1)])
        lon_p = np.concatenate([self.lon[0] - FINE * np.arange(n_pad, 0, -1), self.lon, self.lon[-1] + FINE * np.arange(1, n_pad + 1)])
        # Clip + resample the bathymetry (right branch), then coastline = the 0 m contour of it.
        elev_p = resample(_layer(bathy), lat_p, lon_p, fill=True)
        land_p = elev_p >= 0
        self.km_y = FINE * 111.2
        self.km_x = FINE * 111.2 * np.cos(np.radians((LAT0 + LAT1) / 2))
        dist_p = ndimage.distance_transform_edt(~land_p, sampling=(self.km_y, self.km_x))
        crop = (slice(n_pad, n_pad + self.lat.size), slice(n_pad, n_pad + self.lon.size))
        elev = elev_p[crop]
        self.land = land_p[crop]
        self.dist_coast = dist_p[crop]
        self.depth = np.where(self.land, np.nan, -elev)
        # 4 (right branch): gradient estimation = bathymetric slope.
        gy, gx = np.gradient(np.where(land_p, 0.0, -elev_p), self.km_y, self.km_x)
        self.slope = np.hypot(gx, gy)[crop]

        # Sub-process: take only the study area = sea within BUFFER_KM of the coast (and the website's buffer).
        sea = ~self.land & (self.dist_coast <= BUFFER_KM)
        self.ci = (np.arange(self.lat.size) // 5)[:, None] * np.ones(self.lon.size, int)
        self.cj = np.ones(self.lat.size, int)[:, None] * (np.arange(self.lon.size) // 5)[None, :]
        self.nlat_c, self.nlon_c = self.lat.size // 5, self.lon.size // 5
        if study_cells is not None:
            sea &= study_cells[self.ci, self.cj]
        share = np.zeros((self.nlat_c, self.nlon_c))
        np.add.at(share, (self.ci, self.cj), sea.astype(float))
        share /= 25.0
        self.cell_mask = share >= MIN_SEA_SHARE
        self.study = sea & self.cell_mask[self.ci, self.cj]
        # Split into the multi-buffer mesh: ring = 10 km band of distance to the coast.
        self.ring = np.minimum((self.dist_coast // RING_KM).astype(int), int(BUFFER_KM // RING_KM) - 1)
        cell_id = self.ci * self.nlon_c + self.cj
        zone_key = np.where(self.study, cell_id * 10 + self.ring, -1)
        keys, inv = np.unique(zone_key[self.study], return_inverse=True)
        self.zone = np.full(self.study.shape, -1)
        self.zone[self.study] = inv
        self.n_zones = keys.size
        self.zone_cell = keys // 10                      # flat map-cell index of each zone
        self.zone_ring = keys % 10
        self.zone_px = np.bincount(inv, minlength=self.n_zones)
        self.lat_c = LAT0 + CELL / 2 + CELL * np.arange(self.nlat_c)
        self.lon_c = LON0 + CELL / 2 + CELL * np.arange(self.nlon_c)
        LA, LO = np.meshgrid(self.lat, self.lon, indexing="ij")
        self.zone_lat = self.zone_mean(LA)
        self.zone_lon = self.zone_mean(LO)
        self.static = {"depth_mean": self.zone_mean(self.depth), "slope_mean": self.zone_mean(self.slope),
                       "dist_coast_mean": self.zone_mean(self.dist_coast), "ring": self.zone_ring.astype(float)}

    # 8  zonal statistics helpers
    def zone_mean(self, v):
        z = self.zone[self.study]
        x = np.asarray(v, float)[self.study]
        ok = np.isfinite(x)
        s = np.bincount(z[ok], x[ok], self.n_zones)
        n = np.bincount(z[ok], minlength=self.n_zones)
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)

    def zone_sd(self, v):
        m = self.zone_mean(v)
        m2 = self.zone_mean(np.asarray(v, float) ** 2)
        return np.sqrt(np.maximum(m2 - m ** 2, 0))

    def zone_min(self, v):
        out = np.full(self.n_zones, np.inf)
        np.minimum.at(out, self.zone[self.study], np.asarray(v, float)[self.study])
        return out

    def pixel_index(self, lat, lon):
        i = np.floor((np.asarray(lat) - LAT0) / FINE).astype(int)
        j = np.floor((np.asarray(lon) - LON0) / FINE).astype(int)
        ok = (i >= 0) & (i < self.lat.size) & (j >= 0) & (j < self.lon.size)
        return np.clip(i, 0, self.lat.size - 1), np.clip(j, 0, self.lon.size - 1), ok

    def zone_of(self, lat, lon):
        """Mesh zone of each point; points off the study area snap to the nearest study pixel."""
        i, j, ok = self.pixel_index(lat, lon)
        z = np.where(ok, self.zone[i, j], -1)
        if (z < 0).any():
            if not hasattr(self, "_near"):
                self._near = ndimage.distance_transform_edt(~self.study, return_distances=False, return_indices=True)
            bad = np.flatnonzero((z < 0) & ok)
            ni, nj = self._near[0][i[bad], j[bad]], self._near[1][i[bad], j[bad]]
            z[bad] = self.zone[ni, nj]
        return z

    def to_cells(self, zone_values):
        """Area-weighted mean over the zones of each map cell -> (nlat_c, nlon_c) map, NaN off the study area."""
        v = np.asarray(zone_values, float)
        ok = np.isfinite(v)
        s = np.bincount(self.zone_cell[ok], v[ok] * self.zone_px[ok], self.nlat_c * self.nlon_c)
        w = np.bincount(self.zone_cell[ok], self.zone_px[ok], self.nlat_c * self.nlon_c)
        out = np.where(w > 0, s / np.maximum(w, 1), np.nan).reshape(self.nlat_c, self.nlon_c)
        out[~self.cell_mask] = np.nan
        return out


# =========================
# 4-7  GRADIENTS, ADAPTIVE THRESHOLD, FRONTS, DISTANCE
# =========================

def gradient(v, mesh):
    """|grad| per km with the Sobel operator (3x3 smoothing + central difference)."""
    gx = ndimage.sobel(v, axis=1) / (8 * mesh.km_x)
    gy = ndimage.sobel(v, axis=0) / (8 * mesh.km_y)
    return np.hypot(gx, gy)


def otsu(x, bins=64):
    """Otsu threshold and the share of variance between the two classes (1 = cleanly bimodal)."""
    x = x[np.isfinite(x)]
    if x.size < 20 or np.ptp(x) <= 0:
        return np.nan, 0.0
    hist, edges = np.histogram(x, bins=bins)
    mids = (edges[:-1] + edges[1:]) / 2
    w = hist / hist.sum()
    w0 = np.cumsum(w)
    m0 = np.cumsum(w * mids)
    mt = m0[-1]
    w1 = 1 - w0
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mt * w0 - m0) ** 2 / (w0 * w1)
    between[~np.isfinite(between)] = 0
    k = int(np.argmax(between))
    total = float(np.sum(w * (mids - mt) ** 2))
    return float(edges[k + 1]), float(between[k] / total) if total > 0 else 0.0


def adaptive_foreground(grad, valid):
    """5  Foreground (front) / background pixels by Otsu thresholds in overlapping windows."""
    grad = np.log(np.maximum(grad, 1e-9))
    votes = np.zeros(grad.shape)
    seen = np.zeros(grad.shape)
    floor = np.nanpercentile(grad[valid], FRONT_FLOOR_Q) if valid.any() else np.inf
    H, W = grad.shape
    for r in range(0, max(H - WINDOW_PX, 0) + STRIDE_PX, STRIDE_PX):
        for c in range(0, max(W - WINDOW_PX, 0) + STRIDE_PX, STRIDE_PX):
            sl = (slice(r, r + WINDOW_PX), slice(c, c + WINDOW_PX))
            ok = valid[sl]
            if ok.sum() < 0.25 * WINDOW_PX ** 2:
                continue
            t, sep = otsu(grad[sl][ok])
            if not np.isfinite(t) or sep < MIN_BIMODAL:
                continue
            seen[sl] += ok
            votes[sl] += ok & (grad[sl] > max(t, floor))
    return valid & (seen > 0) & (votes / np.maximum(seen, 1) > 0.5)


def mesoscale_fronts(fg):
    """6  Keep connected front pixels (8-neighbour) forming fronts of at least MIN_FRONT_PX pixels."""
    lab, n = ndimage.label(fg, structure=np.ones((3, 3)))
    if n == 0:
        return fg
    size = np.bincount(lab.ravel())
    keep = size >= MIN_FRONT_PX
    keep[0] = False
    return keep[lab]


def front_distance(front, mesh):
    """7  Minimum distance (km) from each pixel to the nearest front pixel."""
    if not front.any():
        return np.full(front.shape, np.nan)
    return ndimage.distance_transform_edt(~front, sampling=(mesh.km_y, mesh.km_x))


# =========================
# 8  ZONAL STATISTICS PER MONTH
# =========================

def month_fields(src, month, mesh):
    """Resampled fields of one month on the working grid (None if a source lacks the month)."""
    out = {}
    for key in ("sst", "sst_anom", "chl", "wind"):
        s = src[key]
        if month not in s["months"]:
            return None
        v = resample(_layer(s, s["months"].index(month)), mesh.lat, mesh.lon)
        out[key] = np.log10(np.maximum(v, 1e-3)) if key == "chl" else v
    return out


def zonal_stats(src, month, mesh):
    """Fronts and zonal statistics of one month. Returns (zone stats dict, pixel layers for maps)."""
    f = month_fields(src, month, mesh)
    if f is None:
        return None, None
    sea = ~mesh.land
    st = dict(mesh.static)
    px = {}
    for key, name in (("sst", "sst"), ("chl", "chl")):
        g = gradient(f[key], mesh)
        g[~sea] = np.nan
        front = mesoscale_fronts(adaptive_foreground(g, sea & np.isfinite(g)))
        dist = front_distance(front, mesh)
        st[f"{name}_mean"] = mesh.zone_mean(f[key])
        st[f"{name}_grad_mean"] = mesh.zone_mean(g)
        st[f"{name}_front_km"] = mesh.zone_min(dist) if np.isfinite(dist).any() else np.full(mesh.n_zones, np.nan)
        st[f"{name}_front_share"] = mesh.zone_mean(front.astype(float))
        px[f"{name}_front"] = front
        px[f"{name}_front_km"] = dist
    st["sst_sd"] = mesh.zone_sd(f["sst"])
    st["sst_anom_mean"] = mesh.zone_mean(f["sst_anom"])
    st["wind_mean"] = mesh.zone_mean(f["wind"])
    gw = gradient(f["wind"], mesh)
    st["wind_grad_mean"] = mesh.zone_mean(gw)
    return st, px


# =========================
# 9  COLLINEARITY ASSESSMENT
# =========================

def vif(X):
    """Variance inflation factor of each column (regress it on all the others)."""
    Z = (X - X.mean(axis=0)) / np.where(X.std(axis=0) > 0, X.std(axis=0), 1)
    out = []
    for k in range(Z.shape[1]):
        others = np.delete(Z, k, axis=1)
        A = np.column_stack([np.ones(len(Z)), others])
        beta, *_ = np.linalg.lstsq(A, Z[:, k], rcond=None)
        r2 = 1 - np.sum((Z[:, k] - A @ beta) ** 2) / max(np.sum((Z[:, k] - Z[:, k].mean()) ** 2), 1e-12)
        out.append(1 / max(1 - r2, 1e-9))
    return np.array(out)


def collinearity(X, names):
    names = list(names)
    v0 = vif(X)
    keep = list(range(len(names)))
    dropped = []
    while True:
        v = vif(X[:, keep])
        k = int(np.argmax(v))
        if v[k] <= MAX_VIF:
            break
        dropped.append({"feature": names[keep[k]], "vif": float(v[k])})
        keep.pop(k)
    rho = spearmanr(X).statistic if X.shape[1] > 1 else np.ones((1, 1))
    return {"vif_before": {n: float(x) for n, x in zip(names, v0)},
            "vif_after": {names[i]: float(x) for i, x in zip(keep, vif(X[:, keep]))},
            "dropped": dropped, "kept": [names[i] for i in keep], "max_vif": MAX_VIF,
            "spearman": {"names": names, "matrix": np.round(rho, 3).tolist()}}


# =========================
# 11  MODELS
# =========================

class GamClassifier:
    """The package's P-spline GAM behind a scikit-learn-like interface (ring enters linearly)."""

    def __init__(self, names):
        self.names = list(names)

    def fit(self, X, y):
        smooth = [n for n in self.names if n != "ring"]
        linear = [n for n in self.names if n == "ring"]
        self.m = GAM(smooth, linear, "binomial").fit(self._d(X), np.asarray(y, float))
        return self

    def _d(self, X):
        return {n: X[:, k] for k, n in enumerate(self.names)}

    def predict_proba(self, X):
        p = self.m.predict(self._d(X))
        return np.column_stack([1 - p, p])


def make_classifier(kind, names):
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "GLM":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    if kind == "GAM":
        return GamClassifier(names)
    if kind == "BRT":
        return GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8,
                                          min_samples_leaf=20, random_state=SEED)
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=20, max_features="sqrt", n_jobs=-1,
                                  random_state=SEED)


def make_regressor():
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=10, max_features=0.5, n_jobs=-1,
                                 random_state=SEED)


# =========================
# ACCURACY: TAYLOR, MAE, RMSE, SPEARMAN, JSD
# =========================

def agreement(pred, obs, bins=10):
    pred, obs = np.asarray(pred, float), np.asarray(obs, float)
    ok = np.isfinite(pred) & np.isfinite(obs)
    pred, obs = pred[ok], obs[ok]
    if pred.size < 5:
        return {}
    r = float(np.corrcoef(pred, obs)[0, 1]) if pred.std() > 0 and obs.std() > 0 else np.nan
    s = float(pred.std() / obs.std()) if obs.std() > 0 else np.nan
    crmsd = float(np.sqrt(max(1 + s ** 2 - 2 * s * r, 0))) if np.isfinite(r) and np.isfinite(s) else np.nan
    lo = min(pred.min(), obs.min())
    hi = max(pred.max(), obs.max())
    edges = np.linspace(lo, hi + 1e-12, bins + 1)
    hp = np.histogram(pred, edges)[0] + 1e-9
    ho = np.histogram(obs, edges)[0] + 1e-9
    return {"mae": float(np.mean(np.abs(pred - obs))), "rmse": float(np.sqrt(np.mean((pred - obs) ** 2))),
            "spearman": float(spearmanr(pred, obs).statistic), "jsd": float(jensenshannon(hp, ho, base=2) ** 2),
            "taylor_r": r, "taylor_sd_ratio": s, "taylor_crmsd": crmsd,
            # Taylor (2001) skill score with R0 = 1
            "taylor_skill": float(4 * (1 + r) / ((s + 1 / s) ** 2 * 2)) if np.isfinite(r) and s and np.isfinite(s) else np.nan,
            "n": int(pred.size)}


def by_zone_month(key, values, min_sets=2):
    """Mean of `values` per zone-month key, for keys with at least min_sets sets."""
    uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    s = np.bincount(inv, values)
    keep = cnt >= min_sets
    return (s / cnt)[keep], uniq[keep]


# =========================
# 12  SHAP
# =========================

def shap_summary(model, kind, X, names, rng, n=600, n_bins=15):
    """Mean |SHAP| per feature and a binned dependence curve for each (value -> mean SHAP)."""
    import shap
    rows = X[rng.choice(len(X), min(n, len(X)), replace=False)]
    if kind in ("BRT", "RF", "RFR"):
        sv = shap.TreeExplainer(model).shap_values(rows)
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        unit = {"BRT": "log-odds", "RF": "probability", "RFR": "kg/set"}[kind]
    else:
        bg = X[rng.choice(len(X), min(100, len(X)), replace=False)]
        f = (lambda Z: model.predict_proba(Z)[:, 1])
        rows = rows[:200]
        sv = shap.PermutationExplainer(f, bg)(rows, silent=True).values
        unit = "probability"
    imp = np.abs(sv).mean(axis=0)
    dep = {}
    for k, nme in enumerate(names):
        x = rows[:, k]
        qs = np.unique(np.nanpercentile(x, np.linspace(0, 100, n_bins + 1)))
        if qs.size < 3:
            continue
        b = np.clip(np.digitize(x, qs[1:-1]), 0, qs.size - 2)
        xs, ys = [], []
        for i in range(qs.size - 1):
            sel = b == i
            if sel.any():
                xs.append(float(np.mean(x[sel])))
                ys.append(float(np.mean(sv[sel, k])))
        dep[nme] = {"x": np.round(xs, 4).tolist(), "shap": np.round(ys, 5).tolist()}
    order = np.argsort(-imp)
    return {"unit": unit, "importance": [{"feature": names[i], "mean_abs_shap": float(imp[i])} for i in order],
            "dependence": dep}


# =========================
# DRIVER
# =========================

def _fmt(v):
    if isinstance(v, float):
        return "" if not np.isfinite(v) else f"{v:.4g}"
    return v


def _write(path, rows, cols):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: _fmt(r.get(c)) for c in cols})


def _truth_maps(data_dir, month):
    p = Path(data_dir) / "truth" / "maps" / f"truth_{month}.npz"
    if not p.exists():
        return {}, None
    d = np.load(p)
    return ({k[len("lambda_"):]: np.asarray(d[k], float) for k in d.files if k.startswith("lambda_")},
            d["study_area"] & d["ocean"] if "study_area" in d.files else None)


def run_map1(data_dir, env_dir, out_dir, verbose=True):
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit

    data_dir, out = Path(data_dir), Path(out_dir)
    (out / "maps").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    log = print if verbose else (lambda *a, **k: None)

    # 1 CRS harmonizing (all sources), 2-3 clipping / study area / mesh / resampling of the static layers.
    src = load_sources(env_dir)
    first_truth = sorted((data_dir / "truth" / "maps").glob("truth_*.npz"))
    study_cells = None
    if first_truth:
        d = np.load(first_truth[0])
        if "study_area" in d.files:
            study_cells = np.asarray(d["study_area"], bool)
    mesh = Mesh(src["bathy"], study_cells)
    log(f"Mesh: {int(mesh.study.sum()):,} pixels of {FINE} deg in {mesh.n_zones:,} zones "
        f"({int(mesh.cell_mask.sum())} map cells x {int(BUFFER_KM // RING_KM)} rings)")

    # Biomass data: logbook sets.
    ds = load_dataset(data_dir)
    set_month = _days_to_month(ds.time)
    set_zone = mesh.zone_of(ds.lat, ds.lon)
    env_months = sorted(set(src["sst"]["months"]) & set(src["chl"]["months"]) & set(src["wind"]["months"]))
    first = min(set_month)
    upcoming = add_months(env_months[-1], 1)
    target_months = [m for m in sorted(set(set_month.tolist())) if add_months(m, -1) in env_months] + [upcoming]

    # 4-8 per environment month (each feeds the forecast of the month after).
    stats, fronts = {}, {}
    for m in sorted({add_months(t, -1) for t in target_months}):
        st, px = zonal_stats(src, m, mesh)
        if st is None:
            continue
        stats[m] = st
        fronts[m] = {k: mesh.to_cells(mesh.zone_min(v) if k.endswith("_km") else mesh.zone_mean(v.astype(float)))
                     for k, v in px.items()}
    log(f"Zonal statistics for {len(stats)} months ({min(stats)} .. {max(stats)})")

    # 10 X proxy matrix: previous month's zone statistics for every set.
    rows = np.array([i for i in range(len(ds)) if add_months(set_month[i], -1) in stats and set_zone[i] >= 0])
    X_all = np.full((rows.size, len(FEATURES)), np.nan)
    for k, i in enumerate(rows):
        st = stats[add_months(set_month[i], -1)]
        X_all[k] = [st[f][set_zone[i]] for f in FEATURES]
    ok = np.all(np.isfinite(X_all), axis=1)
    rows, X_all = rows[ok], X_all[ok]
    zm_key = np.array([f"{set_month[i]}|{set_zone[i]}" for i in rows])

    # 9 collinearity assessment on the training rows' candidate features.
    split = GroupShuffleSplit(n_splits=1, train_size=TRAIN_SHARE, random_state=SEED)
    tr, va = next(split.split(X_all, groups=zm_key))
    col = collinearity(X_all[tr], FEATURES)
    names = col["kept"]
    fidx = [FEATURES.index(n) for n in names]
    X = X_all[:, fidx]
    dropped_txt = ", ".join("%s (VIF %.1f)" % (d["feature"], d["vif"]) for d in col["dropped"]) or "none"
    log(f"Collinearity: dropped {dropped_txt}; {len(names)} features kept")
    log(f"X proxy matrix: {rows.size:,} sets x {len(names)} features | train {tr.size:,} / validation {va.size:,} sets "
        f"({np.unique(zm_key[tr]).size:,} / {np.unique(zm_key[va]).size:,} zone-months)")

    codes = [c for c, g in ds.groups.items() if g in ("target", "etp") and int(ds.presence[c][rows].sum()) >= MIN_POSITIVES_SCORE]
    skill_rows, reg_rows, shap_out, best = [], [], {}, {}
    final_cls, final_reg, all_cls = {}, {}, {}
    for code in codes:
        y = ds.presence[code][rows].astype(int)
        preds = {}
        for kind in MODELS:
            m = make_classifier(kind, names).fit(X[tr], y[tr])
            preds[kind] = m.predict_proba(X[va])[:, 1]
        # Validation on zone-months with >= 2 validation sets: observed catch rate vs mean predicted P.
        stats_by = {}
        for kind in MODELS:
            p_zm, keys = by_zone_month(zm_key[va], preds[kind])
            o_zm, _ = by_zone_month(zm_key[va], y[va].astype(float))
            stats_by[kind] = {**agreement(p_zm, o_zm), "auc": auc(preds[kind], y[va]), "brier": float(np.mean((preds[kind] - y[va]) ** 2))}
        # Best model: mean rank over MAE, RMSE, -Spearman, JSD, -Taylor skill.
        crit = {"mae": 1, "rmse": 1, "spearman": -1, "jsd": 1, "taylor_skill": -1}
        rank = {k: 0.0 for k in MODELS}
        for c, sign in crit.items():
            vals = np.array([sign * stats_by[k].get(c, np.nan) for k in MODELS])
            vals = np.where(np.isfinite(vals), vals, np.inf)
            for r, k in enumerate(np.array(MODELS)[np.argsort(vals)]):
                rank[k] += (r + 1) / len(crit)
        best[code] = min(MODELS, key=lambda k: rank[k])
        for kind in MODELS:
            skill_rows.append({"species_code": code, "group": ds.groups[code], "model": kind, **stats_by[kind],
                               "mean_rank": rank[kind], "best": kind == best[code],
                               "train_positives": int(y[tr].sum()), "validation_sets": int(va.size)})
        # Refit every classifier on all sets for the maps (best = map, spread of the four = uncertainty).
        all_cls[code] = {kind: make_classifier(kind, names).fit(X, y) for kind in MODELS}
        final_cls[code] = all_cls[code][best[code]]
        shap_cls = shap_summary(make_classifier(best[code], names).fit(X[tr], y[tr]), best[code], X[va], names, rng)

        # Regression: random forest on kg per set (zeros included), so a leaf's mean is the expected catch.
        # Only species weighed in kg; protected species are counted, not weighed.
        reg = None
        if ds.groups[code] == "target":
            yk = ds.amount[code][rows].astype(float)
            r2 = lambda o, q: float(1 - np.sum((o - q) ** 2) / max(np.sum((o - o.mean()) ** 2), 1e-12))
            cv = []
            for f_tr, f_te in GroupKFold(n_splits=5).split(X[tr], groups=zm_key[tr]):
                r = make_regressor().fit(X[tr][f_tr], yk[tr][f_tr])
                pr = r.predict(X[tr][f_te])
                cv.append({"r2": r2(yk[tr][f_te], pr), "rmse": float(np.sqrt(np.mean((pr - yk[tr][f_te]) ** 2)))})
            r = make_regressor().fit(X[tr], yk[tr])
            pv = r.predict(X[va])
            p_zm, _ = by_zone_month(zm_key[va], pv)
            o_zm, _ = by_zone_month(zm_key[va], yk[va])
            acc = agreement(p_zm, o_zm)
            reg = {"species_code": code, "cv_r2": float(np.mean([c["r2"] for c in cv])),
                   "cv_rmse_kg": float(np.mean([c["rmse"] for c in cv])),
                   "val_r2": r2(yk[va], pv),
                   "val_mae_kg": acc.get("mae"), "val_rmse_kg": acc.get("rmse"), "val_spearman": acc.get("spearman"),
                   "val_jsd": acc.get("jsd"), "val_taylor_r": acc.get("taylor_r"),
                   "val_taylor_sd_ratio": acc.get("taylor_sd_ratio")}
            reg_rows.append(reg)
            final_reg[code] = make_regressor().fit(X, yk)
            shap_reg = shap_summary(r, "RFR", X[va], names, rng)
        shap_out[code] = {"classifier": best[code], "classification": shap_cls,
                          "regression": shap_reg if reg else None}
        s = stats_by[best[code]]
        log(f"  {code:<9} best {best[code]:<3} | AUC {s['auc']:.3f} MAE {s.get('mae', np.nan):.3f} RMSE {s.get('rmse', np.nan):.3f} "
            f"rho {s.get('spearman', np.nan):.2f} JSD {s.get('jsd', np.nan):.3f}"
            + (f" | RF regression CV R2 {reg['cv_r2']:.2f}" if reg else ""))

    # 13 Maps: every target month from the previous month's zone statistics.
    Xz_names = names
    written = []
    for M in target_months:
        prev = add_months(M, -1)
        if prev not in stats:
            continue
        Z = np.column_stack([stats[prev][n] for n in Xz_names])
        okz = np.all(np.isfinite(Z), axis=1)
        arrays = {}
        sel = rows[set_month[rows] < M]
        fp = footprint(mesh.lat_c[:, None].repeat(mesh.nlon_c, 1)[mesh.cell_mask],
                       mesh.lon_c[None, :].repeat(mesh.nlat_c, 0)[mesh.cell_mask],
                       ds.lat[sel if sel.size else rows], ds.lon[sel if sel.size else rows])
        fpm = np.full(mesh.cell_mask.shape, np.nan)
        fpm[mesh.cell_mask] = fp
        arrays["data_footprint"] = fpm
        truth, _ = _truth_maps(data_dir, M)
        for code in codes:
            zp = np.full(mesh.n_zones, np.nan)
            zp[okz] = final_cls[code].predict_proba(Z[okz])[:, 1]
            arrays[f"p_{code}"] = mesh.to_cells(zp)
            spread = np.full(mesh.n_zones, np.nan)
            spread[okz] = np.std([all_cls[code][k].predict_proba(Z[okz])[:, 1] for k in MODELS], axis=0)
            arrays[f"sd_{code}"] = mesh.to_cells(spread)
            if code in final_reg:
                zb = np.full(mesh.n_zones, np.nan)
                zb[okz] = final_reg[code].predict(Z[okz])
                arrays[f"b_{code}"] = mesh.to_cells(zb)
            if code in truth and truth[code].shape == mesh.cell_mask.shape:
                t = truth[code].copy()
                t[~mesh.cell_mask] = np.nan
                arrays[f"lambda_{code}"] = t
        for k, v in fronts[prev].items():
            arrays[k] = v
        np.savez_compressed(out / "maps" / f"forecast_{M}.npz", lat=mesh.lat_c, lon=mesh.lon_c, mask=mesh.cell_mask, **arrays)
        written.append(M)

    # Map vs simulator truth (simulations only): Spearman over the data footprint, per species and month.
    map_rho = {}
    for code in codes:
        vals = []
        for M in written:
            d = np.load(out / "maps" / f"forecast_{M}.npz")
            if f"lambda_{code}" not in d.files:
                continue
            ok = mesh.cell_mask & np.isfinite(d[f"p_{code}"]) & np.isfinite(d[f"lambda_{code}"]) & (d["data_footprint"] > 0)
            if ok.sum() > 10:
                vals.append(spearmanr(d[f"p_{code}"][ok], d[f"lambda_{code}"][ok]).statistic)
        map_rho[code] = float(np.mean(vals)) if vals else np.nan
    for r in skill_rows:
        r["map_rho"] = map_rho[r["species_code"]] if r["best"] else np.nan

    cols = ["species_code", "group", "model", "best", "mean_rank", "mae", "rmse", "spearman", "jsd", "taylor_r",
            "taylor_sd_ratio", "taylor_crmsd", "taylor_skill", "auc", "brier", "n", "map_rho", "train_positives",
            "validation_sets"]
    _write(out / "map1_classification.csv", skill_rows, cols)
    _write(out / "map1_regression.csv", reg_rows, ["species_code", "cv_r2", "cv_rmse_kg", "val_r2", "val_mae_kg",
                                                  "val_rmse_kg", "val_spearman", "val_jsd", "val_taylor_r",
                                                  "val_taylor_sd_ratio"])
    with open(out / "map1_collinearity.json", "w", encoding="utf-8") as fh:
        json.dump(col, fh)
    with open(out / "map1_shap.json", "w", encoding="utf-8") as fh:
        json.dump(shap_out, fh)
    meta = {"method": "map1", "target_months": [m for m in written if m != upcoming],
            "upcoming_month": upcoming if upcoming in written else None,
            "train_sets": int(tr.size), "validation_sets": int(va.size), "sets": int(rows.size),
            "train_share": TRAIN_SHARE, "features": names, "features_th": FEATURE_TH, "candidate_features": FEATURES,
            "best_model": best, "models": MODEL_NAMES, "n_zones": int(mesh.n_zones), "ring_km": RING_KM,
            "buffer_km": BUFFER_KM, "fine_deg": FINE, "max_vif": MAX_VIF, "footprint_km": FOOTPRINT_KM,
            "first_logbook_month": first, "simulated": (data_dir / "truth").exists()}
    with open(out / "forecast_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False)
    log(f"Maps: {written[0]} .. {written[-1]} ({len(written)} months, upcoming {upcoming}) -> {out.resolve()}")
    return meta
