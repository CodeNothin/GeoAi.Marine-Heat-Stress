"""Geography of the southern Gulf of Thailand.

With `region.bathy_npz` (ETOPO 2022 / GEBCO, from fetch_env_data.py --bathy) the land
mask, coastline, distance to coast and depth all come from real bathymetry. Without
it the coastline is a hand-digitised polyline (~5-10 km accuracy) and the depth is
analytic: good enough for a synthetic test bed, NOT for real analysis.

`study_area` is the coastal buffer: ocean within `region.coastal_buffer_km` of the coast.
"""
from __future__ import annotations

import numpy as np

from .fields import RandomField

KM_PER_DEG_LAT = 110.57
_PROJ_LAT0, _PROJ_LON0, _PROJ_COS = 6.0, 99.0, np.cos(np.radians(9.0))

# (lat, lon) mainland coast, north -> south. Land is west of this line.
COASTLINE = [
    (11.50, 99.62), (11.20, 99.52), (10.85, 99.40), (10.50, 99.20),
    (10.25, 99.15), (9.95, 99.12), (9.65, 99.20), (9.40, 99.25),
    (9.25, 99.38),                      # Bandon Bay / Tapi river mouth
    (9.12, 99.85),                      # Khanom
    (8.90, 99.95), (8.60, 100.00), (8.50, 100.20), (8.30, 100.25),
    (8.00, 100.33), (7.70, 100.42), (7.40, 100.50),
    (7.20, 100.60),                     # Songkhla
    (7.00, 100.85), (6.92, 101.20),     # Pattani Bay
    (6.90, 101.30), (6.70, 101.55),
    (6.45, 101.80),                     # Narathiwat
    (6.25, 102.05), (6.10, 102.25),
]

ISLANDS = [  # name, lat, lon, radius_deg
    ("Ko Samui", 9.50, 100.00, 0.11),
    ("Ko Phangan", 9.75, 100.03, 0.08),
    ("Ko Tao", 10.09, 99.84, 0.03),
]

RIVER_MOUTHS = [  # name, lat, lon, relative plume strength
    ("Chumphon", 10.35, 99.25, 0.4),
    ("Tapi (Bandon Bay)", 9.18, 99.42, 1.0),
    ("Pak Phanang", 8.40, 100.22, 0.5),
    ("Songkhla Lake outlet", 7.22, 100.60, 0.6),
    ("Pattani", 6.90, 101.25, 0.5),
]

HABITAT_SITES = [  # approx. seagrass / reef areas (turtle habitat): name, lat, lon, radius_km
    ("Chumphon islands", 10.45, 99.35, 12.0),
    ("Bandon Bay", 9.30, 99.55, 12.0),
    ("Samui-Phangan", 9.60, 100.10, 15.0),
    ("Sichon-Khanom", 8.95, 100.00, 10.0),
    ("Pattani Bay", 6.95, 101.25, 10.0),
]

_COAST = np.array(COASTLINE)
_COAST_LAT_ASC = _COAST[::-1, 0]
_COAST_LON_ASC = _COAST[::-1, 1]


# =========================
# DISTANCE HELPERS
# =========================

def km_per_deg_lon(lat):
    return 111.32 * np.cos(np.radians(lat))


def distance_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.asarray(v, float) for v in (lat1, lon1, lat2, lon2))
    dy = (lat2 - lat1) * KM_PER_DEG_LAT
    dx = (lon2 - lon1) * km_per_deg_lon(0.5 * (lat1 + lat2))
    return np.hypot(dx, dy)


def bearing_deg(lat1, lon1, lat2, lon2):
    dy = (np.asarray(lat2) - lat1) * KM_PER_DEG_LAT
    dx = (np.asarray(lon2) - lon1) * km_per_deg_lon(0.5 * (np.asarray(lat1) + lat2))
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def offset_point(lat, lon, dist_km, bearing):
    b = np.radians(bearing)
    return (lat + dist_km * np.cos(b) / KM_PER_DEG_LAT,
            lon + dist_km * np.sin(b) / km_per_deg_lon(lat))


def project_km(lat, lon):
    """Local flat projection (km) used by the random fields."""
    x = (np.asarray(lon, float) - _PROJ_LON0) * 111.32 * _PROJ_COS
    y = (np.asarray(lat, float) - _PROJ_LAT0) * KM_PER_DEG_LAT
    return x, y


def coast_lon_at(lat):
    return np.interp(lat, _COAST_LAT_ASC, _COAST_LON_ASC)


def is_land_analytic(lat, lon):
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    land = lon <= coast_lon_at(lat)
    for _, ilat, ilon, r in ISLANDS:
        land |= (lat - ilat) ** 2 + ((lon - ilon) * np.cos(np.radians(ilat))) ** 2 <= r ** 2
    return land


def _coast_cells(land, LAT, LON):
    """Centres of land cells that touch the sea (4-neighbour)."""
    padded = np.pad(land, 1, constant_values=True)
    ocean_neighbour = (~padded[:-2, 1:-1]) | (~padded[2:, 1:-1]) | (~padded[1:-1, :-2]) | (~padded[1:-1, 2:])
    coast = land & ocean_neighbour
    return LAT[coast], LON[coast]


def _flood(mask, seed):
    """Cells of `mask` 4-connected to the `seed` cells."""
    out = seed & mask
    while True:
        grown = out.copy()
        grown[1:] |= out[:-1]
        grown[:-1] |= out[1:]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        grown &= mask
        if (grown == out).all():
            return out
        out = grown


def _gulf(wet):
    """Wet cells connected to the region's east edge, where the Gulf of Thailand opens.

    Drops the Andaman Sea in the south-west corner and inland water below sea level.
    """
    seed = np.zeros_like(wet)
    seed[:, -1] = True
    return _flood(wet, seed)


def _largest_component(mask):
    best, left = np.zeros_like(mask), mask.copy()
    while left.any():
        seed = np.zeros_like(mask)
        seed.flat[np.flatnonzero(left)[0]] = True
        part = _flood(left, seed)
        if part.sum() > best.sum():
            best = part
        left &= ~part
    return best


# =========================
# GRIDDED STATIC FIELDS
# =========================

class Geography:
    """Land mask, distance to coast, depth, coastal-buffer study area and habitat index on a regular grid."""

    def __init__(self, region, rng):
        self.region = region
        self.res = res = region.grid_res_deg
        self.lats = np.arange(region.lat_min + res / 2, region.lat_max, res)
        self.lons = np.arange(region.lon_min + res / 2, region.lon_max, res)
        self.LAT, self.LON = np.meshgrid(self.lats, self.lons, indexing="ij")
        if region.bathy_npz:
            self._from_bathymetry(region.bathy_npz)
        else:
            self.land = is_land_analytic(self.LAT, self.LON)
            self.dist_coast_km = self._distance_to_coast(*_coast_cells(self.land, self.LAT, self.LON))
            self.depth_m = self._depth(rng)
        # One connected band along our coast: no stray pieces near coasts just outside the region.
        self.study_area = _largest_component(~self.land & (self.dist_coast_km <= region.coastal_buffer_km))
        self.seagrass_idx = self._habitat()

    def _from_bathymetry(self, path):
        b = np.load(path)
        blat, blon, elev = b["lat"], b["lon"], np.nan_to_num(b["elevation"].astype(float), nan=1.0)
        i = np.clip(np.searchsorted(blat, self.lats), 1, blat.size - 1)
        i -= (self.lats - blat[i - 1]) < (blat[i] - self.lats)        # nearest neighbour
        j = np.clip(np.searchsorted(blon, self.lons), 1, blon.size - 1)
        j -= (self.lons - blon[j - 1]) < (blon[j] - self.lons)
        e = elev[np.ix_(i, j)]
        self.land = ~_gulf(e < 0)
        # Coast cells from the full (padded) bathymetry, so coast just outside the region counts too.
        BLAT, BLON = np.meshgrid(blat, blon, indexing="ij")
        self.dist_coast_km = self._distance_to_coast(*_coast_cells(elev >= 0, BLAT, BLON))
        depth = np.clip(-e, 1.5, None)
        depth[self.land] = np.nan
        self.depth_m = depth

    def _distance_to_coast(self, clat, clon):
        land = self.land
        olat, olon = self.LAT[~land], self.LON[~land]
        out = np.empty(olat.size)
        chunk = 2000
        for s in range(0, olat.size, chunk):
            d = distance_km(olat[s:s + chunk, None], olon[s:s + chunk, None], clat[None, :], clon[None, :])
            out[s:s + chunk] = d.min(axis=1)
        dist = np.zeros(land.shape)
        # Coast lies between cell centres: subtract half a cell.
        dist[~land] = np.maximum(out - 0.5 * self.res * KM_PER_DEG_LAT, 0.0)
        return dist

    def _depth(self, rng):
        """Analytic depth (no bathymetry file): a shelf deepening away from the coast."""
        x, y = project_km(self.LAT, self.LON)
        bumps = RandomField(rng, n_modes=20, length_km=60.0, sd=2.5)(x, y)
        dist = self.dist_coast_km
        depth = 82.0 * (1.0 - np.exp(-dist / 55.0)) + bumps * (1.0 - np.exp(-dist / 20.0))
        depth = np.clip(depth, 1.5, None)
        depth[self.land] = np.nan
        return depth

    def _habitat(self):
        idx = np.zeros(self.LAT.shape)
        for _, hlat, hlon, r in HABITAT_SITES:
            d = distance_km(self.LAT, self.LON, hlat, hlon)
            idx = np.maximum(idx, np.exp(-(d / r) ** 2))
        idx[self.land] = 0.0
        return idx

    # ---------- lookups ----------

    def _ij(self, lat, lon):
        i = np.floor((np.asarray(lat, float) - self.region.lat_min) / self.res).astype(int)
        j = np.floor((np.asarray(lon, float) - self.region.lon_min) / self.res).astype(int)
        return np.clip(i, 0, len(self.lats) - 1), np.clip(j, 0, len(self.lons) - 1)

    def in_domain(self, lat, lon):
        r = self.region
        lat = np.asarray(lat, float)
        lon = np.asarray(lon, float)
        return (lat >= r.lat_min) & (lat < r.lat_max) & (lon >= r.lon_min) & (lon < r.lon_max)

    def is_ocean(self, lat, lon):
        i, j = self._ij(lat, lon)
        return self.in_domain(lat, lon) & ~self.land[i, j]

    def lookup(self, name, lat, lon):
        i, j = self._ij(lat, lon)
        return getattr(self, name)[i, j]

    def nearest_ocean(self, lat, lon, min_dist_km=0.0):
        mask = ~self.land & (self.dist_coast_km >= min_dist_km)
        d = distance_km(lat, lon, self.LAT[mask], self.LON[mask])
        k = int(np.argmin(d))
        return float(self.LAT[mask][k]), float(self.LON[mask][k])

    # ---------- routing ----------

    def segment_clear(self, lat0, lon0, lat1, lon1, step_km=1.0, end_tolerance_km=1.5):
        d = float(distance_km(lat0, lon0, lat1, lon1))
        if d <= 2 * end_tolerance_km:
            return True
        s = np.arange(end_tolerance_km, d - end_tolerance_km, step_km) / d
        return bool(np.all(self.is_ocean(lat0 + s * (lat1 - lat0), lon0 + s * (lon1 - lon0))))

    def route(self, lat0, lon0, lat1, lon1, _depth=0):
        """Waypoints from A to B that avoid land by pushing the midpoint seaward (east)."""
        if self.segment_clear(lat0, lon0, lat1, lon1):
            return [(lat0, lon0), (lat1, lon1)]
        mlat, mlon = 0.5 * (lat0 + lat1), 0.5 * (lon0 + lon1)
        fallback = None
        for shift in np.arange(0.05, 1.01, 0.05):
            c = (mlat, mlon + shift)
            if not self.is_ocean(*c):
                continue
            if fallback is None:
                fallback = c
            if self.segment_clear(lat0, lon0, *c) and self.segment_clear(*c, lat1, lon1):
                return [(lat0, lon0), c, (lat1, lon1)]
        if fallback is None or _depth >= 3:
            return [(lat0, lon0), (lat1, lon1)]
        first = self.route(lat0, lon0, *fallback, _depth=_depth + 1)
        second = self.route(*fallback, lat1, lon1, _depth=_depth + 1)
        return first[:-1] + second
