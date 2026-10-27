"""Synthetic ocean environment: SST, CHL-a, wind, fronts and moon, evaluated on demand.

Fields are analytic functions of (lat, lon, time), so no daily grids are stored:
    value = climatology(season, distance to coast) + anomalies (interannual,
            marine heatwaves, moving mesoscale eddies) + noise.

Time is handled everywhere as float "unix days" (days since 1970-01-01 UTC).
"""
from __future__ import annotations

import numpy as np

from .fields import RandomField
from .geo import KM_PER_DEG_LAT, RIVER_MOUTHS, distance_km, km_per_deg_lon, project_km

# ASSUMED monthly climatologies for the southern Gulf of Thailand (Jan..Dec).
SST_CLIM_MONTHLY_C = [27.4, 27.7, 28.8, 30.0, 30.2, 29.6, 29.3, 29.2, 29.1, 29.0, 28.5, 27.7]
WIND_CLIM_MONTHLY_MS = [6.5, 5.5, 4.0, 3.2, 3.8, 4.8, 5.0, 5.2, 4.5, 3.8, 6.0, 7.2]
STORM_MONTH_WEIGHTS = [3, 1, 0.5, 0.3, 0.5, 1, 1, 1, 1, 2, 4, 4]   # NE monsoon storms Oct-Jan

SYNODIC_DAYS = 29.530588853
REF_NEW_MOON_UNIX_DAYS = 10962.7597     # 2000-01-06 18:14 UTC
LOCAL_UTC_OFFSET_H = 7.0


# =========================
# TIME HELPERS
# =========================

def to_unix_days(value) -> float:
    """'2024-01-01', '2024-01-01T05:00' or numpy datetime64 -> float unix days."""
    return float(np.datetime64(value, "s").astype("int64")) / 86400.0


def unix_days_to_iso(days):
    secs = np.round(np.asarray(days, float) * 86400.0).astype("int64")
    return np.char.add(np.datetime_as_string(secs.astype("datetime64[s]"), unit="s"), "Z")


def month_of(t):
    """Calendar month (1-12) of unix-day times."""
    secs = np.floor(np.asarray(t, float) * 86400.0).astype("int64")
    return secs.astype("datetime64[s]").astype("datetime64[M]").astype(int) % 12 + 1


def month_day(t):
    d = np.datetime64(int(np.floor(t * 86400.0)), "s").astype("datetime64[D]")
    month = int(d.astype("datetime64[M]").astype(int) % 12 + 1)
    day = int((d - d.astype("datetime64[M]")).astype(int) + 1)
    return month, day


def year_of(t) -> int:
    return int(np.datetime64(int(np.floor(t * 86400.0)), "s").astype("datetime64[Y]").astype(int) + 1970)


def frac_year(t):
    """Fraction of the year elapsed (0 = Jan 1); accurate to < 1 day."""
    return (np.asarray(t, float) / 365.2425) % 1.0


def monthly_interp(table, t):
    """Periodic interpolation of a 12-value monthly table (values at mid-month)."""
    m = frac_year(t) * 12.0 - 0.5
    i0 = np.floor(m).astype(int) % 12
    w = m - np.floor(m)
    table = np.asarray(table, float)
    return (1.0 - w) * table[i0] + w * table[(i0 + 1) % 12]


def moon_illumination(t):
    phase = ((np.asarray(t, float) - REF_NEW_MOON_UNIX_DAYS) / SYNODIC_DAYS) % 1.0
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * phase))


def local_hour(t):
    return ((np.asarray(t, float) + LOCAL_UTC_OFFSET_H / 24.0) % 1.0) * 24.0


def local_time(local_day: int, hour: float) -> float:
    """Unix days for `hour` (local time) on local day number `local_day`."""
    return local_day + (hour - LOCAL_UTC_OFFSET_H) / 24.0


def local_day_of(t) -> int:
    return int(np.floor(t + LOCAL_UTC_OFFSET_H / 24.0))


# =========================
# OPTIONAL REAL SST
# =========================

class GriddedField:
    """Bilinear lookup in a real gridded cube (e.g. NOAA OISST, CHL-a or ERA5 wind).

    npz keys: time (unix days or datetime64), lat (ascending), lon (ascending) and
    `key`[time, lat, lon]. Returns NaN outside the cube or on land so the analytic
    field is used as fallback.
    """

    def __init__(self, path, key="sst"):
        data = np.load(path)
        time = data["time"]
        if np.issubdtype(time.dtype, np.datetime64):
            time = time.astype("datetime64[s]").astype("int64") / 86400.0
        self.time = np.asarray(time, float)
        self.lat = np.asarray(data["lat"], float)
        self.lon = np.asarray(data["lon"], float)
        self.values = np.asarray(data[key], float)
        self.extra = {k: np.asarray(data[k], float) for k in data.files
                      if k not in ("time", "lat", "lon", key) and data[k].shape == self.values.shape}
        self.max_gap = max(float(np.median(np.diff(self.time))), 1.0) if len(self.time) > 1 else 1.0

    def __call__(self, lat, lon, t, key=None):
        cube = self.values if key is None else self.extra.get(key)
        lat, lon, t = (np.asarray(v, float) for v in (lat, lon, t))
        out = np.full(lat.shape, np.nan)
        if cube is None:
            return out
        # Nearest time step.
        if len(self.time) == 1:
            k = np.zeros(t.shape, int)
        else:
            hi = np.clip(np.searchsorted(self.time, t), 1, len(self.time) - 1)
            k = np.where(np.abs(t - self.time[hi - 1]) <= np.abs(self.time[hi] - t), hi - 1, hi)
        fi = np.interp(lat, self.lat, np.arange(len(self.lat)), left=np.nan, right=np.nan)
        fj = np.interp(lon, self.lon, np.arange(len(self.lon)), left=np.nan, right=np.nan)
        pad = self.max_gap
        ok = np.isfinite(fi) & np.isfinite(fj) & (t >= self.time[0] - pad) & (t <= self.time[-1] + pad)
        if ok.any():
            i0 = np.clip(np.floor(fi[ok]).astype(int), 0, len(self.lat) - 2)
            j0 = np.clip(np.floor(fj[ok]).astype(int), 0, len(self.lon) - 2)
            wi, wj = fi[ok] - i0, fj[ok] - j0
            kk = k[ok]
            c = cube
            out[ok] = ((1 - wi) * (1 - wj) * c[kk, i0, j0] + wi * (1 - wj) * c[kk, i0 + 1, j0]
                       + (1 - wi) * wj * c[kk, i0, j0 + 1] + wi * wj * c[kk, i0 + 1, j0 + 1])
        return out


class GriddedSST(GriddedField):
    """Real SST cube; an optional `sst_anom` array of the same shape is used as the observed anomaly."""

    def __init__(self, path):
        super().__init__(path, "sst")


# =========================
# ENVIRONMENT
# =========================

class Environment:
    def __init__(self, cfg, geo, rng):
        self.cfg = cfg
        self.geo = geo
        ec = cfg.env
        t0 = to_unix_days(cfg.start) - 120.0
        t1 = to_unix_days(cfg.end) + 120.0

        self.sst_meso = RandomField(rng, n_modes=24, length_km=120.0, sd=0.35, time_days=25.0)
        self.chl_meso = RandomField(rng, n_modes=24, length_km=80.0, sd=0.18, time_days=15.0)
        self.wind_field = RandomField(rng, n_modes=12, length_km=300.0, sd=1.2, time_days=3.0)

        # Interannual SST anomaly: monthly AR(1).
        self._ia_t = np.arange(t0, t1 + 31.0, 30.44)
        rho = 0.85
        vals = np.empty(self._ia_t.size)
        vals[0] = rng.normal(0.0, ec.interannual_sd_c)
        for i in range(1, vals.size):
            vals[i] = rho * vals[i - 1] + np.sqrt(1 - rho ** 2) * rng.normal(0.0, ec.interannual_sd_c)
        self._ia_v = vals

        self._mhw = [(to_unix_days(e["start"]), to_unix_days(e["peak"]), to_unix_days(e["end"]),
                      float(e["intensity_c"])) for e in ec.mhw_events]

        storms = []
        w = np.array(STORM_MONTH_WEIGHTS, float)
        for year in range(year_of(t0), year_of(t1) + 1):
            for _ in range(rng.poisson(ec.storms_per_year)):
                month = rng.choice(12, p=w / w.sum())
                tc = to_unix_days(f"{year}-01-01") + (month + rng.random()) * 30.44
                storms.append((tc, rng.uniform(1.0, 2.5), rng.uniform(4.0, 8.0)))
        self._storms = np.array(storms, float).reshape(-1, 3)

        self.sst_override = GriddedSST(ec.sst_override_npz) if ec.sst_override_npz else None
        self.chl_override = GriddedField(ec.chl_override_npz, "chl") if ec.chl_override_npz else None
        self.wind_override = GriddedField(ec.wind_override_npz, "wind") if ec.wind_override_npz else None

    # ---------- components ----------

    def _interannual(self, t):
        return np.interp(t, self._ia_t, self._ia_v)

    def _mhw_intensity(self, t):
        out = np.zeros(np.shape(t))
        for start, peak, end, amp in self._mhw:
            rise = np.clip((t - start) / max(peak - start, 1e-6), 0, 1)
            fall = np.clip((end - t) / max(end - peak, 1e-6), 0, 1)
            out += amp * np.where(t <= peak, rise, fall)
        return out

    def _storm_wind(self, t):
        if self._storms.size == 0:
            return np.zeros(np.shape(t))
        tc, dur, extra = self._storms.T
        dt = np.asarray(t, float)[..., None] - tc
        return (extra * np.exp(-(dt / dur) ** 2)).sum(axis=-1)

    def _sst(self, lat, lon, t, dist, x, y):
        base = monthly_interp(SST_CLIM_MONTHLY_C, t)
        # Shallow nearshore water heats more in the hot season and cools more in the NE monsoon.
        clim = base + 0.5 * np.exp(-dist / 30.0) * (base - 28.9) / 1.4 + 0.06 * (9.0 - lat)
        anom = (self._interannual(t) + self._mhw_intensity(t) * (0.7 + 0.6 * np.exp(-dist / 40.0))
                + self.sst_meso(x, y, t))
        sst = clim + anom
        if self.sst_override is not None:
            obs = self.sst_override(lat, lon, t)
            sst = np.where(np.isfinite(obs), obs, sst)
            # Prefer the observed anomaly (e.g. OISST vs its own 1971-2000 climatology) when supplied.
            obs_anom = self.sst_override(lat, lon, t, key="sst_anom")
            anom = np.where(np.isfinite(obs_anom), obs_anom, sst - clim)
        return sst, clim, anom

    def _log10chl(self, lat, lon, t, dist, x, y, sst_anom):
        fy = frac_year(t)
        runoff = 0.5 + 0.5 * np.cos(2 * np.pi * (fy - 335.0 / 365.25))      # rainy season peak ~Dec
        plume = np.zeros(np.shape(lat))
        for _, rlat, rlon, strength in RIVER_MOUTHS:
            d = distance_km(lat, lon, rlat, rlon)
            plume += 0.6 * strength * runoff * np.exp(-(d / 45.0) ** 2)
        out = (-0.55 + 0.9 * np.exp(-dist / 30.0) + plume
               + 0.12 * np.cos(2 * np.pi * (fy - 15.0 / 365.25))          # monsoon mixing peak ~Jan
               - 0.12 * sst_anom                                            # heatwaves suppress productivity
               + self.chl_meso(x, y, t))
        if self.chl_override is not None:
            obs = self.chl_override(lat, lon, t)
            with np.errstate(divide="ignore", invalid="ignore"):
                obs = np.where(obs > 0, np.log10(obs), np.nan)
            out = np.where(np.isfinite(obs), obs, out)
        return out

    def _wind(self, t, dist, x, y, lat=None, lon=None):
        offshore = 1.0 - np.exp(-dist / 60.0)
        w = (monthly_interp(WIND_CLIM_MONTHLY_MS, t) * (0.85 + 0.3 * offshore)
             + self.wind_field(x, y, t) + self._storm_wind(t) * (0.8 + 0.4 * offshore))
        if self.wind_override is not None and lat is not None:
            # Monthly reanalysis wind sets the level; the analytic storms and gusts keep the day-to-day spread.
            obs = self.wind_override(lat, lon, t)
            anomaly = w - monthly_interp(WIND_CLIM_MONTHLY_MS, t) * (0.85 + 0.3 * offshore)
            w = np.where(np.isfinite(obs), obs + anomaly, w)
        return np.clip(w, 0.3, None)

    # ---------- public ----------

    def wind(self, lat, lon, t):
        dist = self.geo.lookup("dist_coast_km", lat, lon)
        x, y = project_km(lat, lon)
        return float(np.asarray(self._wind(t, dist, x, y, np.atleast_1d(lat), np.atleast_1d(lon))).ravel()[0])

    def sample(self, lat, lon, t) -> dict:
        lat = np.atleast_1d(np.asarray(lat, float))
        lon = np.atleast_1d(np.asarray(lon, float))
        t = np.broadcast_to(np.asarray(t, float), lat.shape).astype(float)
        geo = self.geo
        dist = geo.lookup("dist_coast_km", lat, lon)
        x, y = project_km(lat, lon)
        sst, clim, anom = self._sst(lat, lon, t, dist, x, y)

        # SST front strength: central difference over +-5 km, in deg C per 10 km.
        e = 5.0
        dlat = e / KM_PER_DEG_LAT
        dlon = e / km_per_deg_lon(lat)
        lat4 = np.concatenate([lat + dlat, lat - dlat, lat, lat])
        lon4 = np.concatenate([lon, lon, lon + dlon, lon - dlon])
        t4 = np.tile(t, 4)
        x4, y4 = project_km(lat4, lon4)
        s4 = self._sst(lat4, lon4, t4, geo.lookup("dist_coast_km", lat4, lon4), x4, y4)[0].reshape(4, -1)
        front = np.hypot(s4[0] - s4[1], s4[2] - s4[3]) / (2 * e) * 10.0

        log10chl = self._log10chl(lat, lon, t, dist, x, y, anom)
        return {
            "lat": lat, "lon": lon, "t": t, "x_km": x, "y_km": y,
            "dist_coast": dist,
            "depth": geo.lookup("depth_m", lat, lon),
            "seagrass_idx": geo.lookup("seagrass_idx", lat, lon),
            "sst": sst, "sst_clim": clim, "sst_anom": anom, "front": front,
            "log10chl": log10chl, "chl": 10.0 ** log10chl,
            "wind": self._wind(t, dist, x, y, lat, lon),
            "moon": moon_illumination(t),
        }


def satellite_view(sample: dict, rng, env_cfg) -> dict:
    """What a satellite/reanalysis matching pipeline would report (noise + cloud gaps)."""
    n = sample["lat"].size
    cloud_p = np.asarray(env_cfg.cloud_prob_by_month, float)[month_of(sample["t"]) - 1]
    sat_sst = sample["sst"] + rng.normal(0.0, env_cfg.sst_obs_noise_c, n)
    chl = 10.0 ** (sample["log10chl"] + rng.normal(0.0, env_cfg.chl_obs_noise_log10, n))
    chl = np.where(rng.random(n) < cloud_p, np.nan, chl)
    return {
        "sat_sst_c": sat_sst,
        "sat_sst_anom_c": sat_sst - sample["sst_clim"],
        "sat_chl_mg_m3": chl,
        "sst_front_c_per_10km": sample["front"],
        "wind_ms": np.clip(sample["wind"] + rng.normal(0.0, env_cfg.wind_obs_noise_ms, n), 0.0, None),
        "moon_illum": sample["moon"],
        "depth_m": np.clip(sample["depth"] + rng.normal(0.0, env_cfg.depth_obs_noise_m, n), 0.5, None),
        "dist_coast_km": sample["dist_coast"],
        "seagrass_idx": sample["seagrass_idx"],
    }
