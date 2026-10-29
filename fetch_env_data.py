"""Download real ocean data for the area of interest and save it as cubes the simulator and models read.

    python fetch_env_data.py --start 2023-01 --end 2025-12 --out real_env          # NOAA SST + CHL-a
    python fetch_env_data.py --start 2023-01 --end 2025-12 --out real_env --era5   # + ERA5 wind
    python fetch_env_data.py --start 2023-01 --end 2025-12 --out real_env --psl --era5   # SST from NOAA PSL files
    python fetch_env_data.py --oisst-nc sst.day.mean.2024.nc --oisst-ltm sst.day.mean.ltm.1991-2020.nc \
        --no-chl --out real_env                     # already-downloaded NOAA PSL OISST files (needs netCDF4)
    python fetch_env_data.py --bathy --out real_env                     # bathymetry only (ETOPO 2022)
    python fetch_env_data.py --bathy-nc gebco_2024_subset.nc --out real_env   # or a GEBCO download

Sources:
  - SST and SST anomaly: NOAA OISST v2.1 daily, 0.25 deg, via NOAA CoastWatch ERDDAP (no account;
    tries a mirror if the main server does not answer). The anomaly is OISST's own, relative to
    its 1971-2000 climatology. If ERDDAP is unreachable, --psl downloads the yearly NOAA PSL files
    (about 470 MB each, plus the 1991-2020 daily mean for the anomaly) and cuts them instead.
  - CHL-a: monthly satellite chlorophyll via the same ERDDAP server (no account). Tries the
    datasets in CHL_DATASETS in order; if none answers, CHL-a is skipped and the simulator keeps
    its analytic CHL-a.
  - Wind (optional, --era5): ERA5 monthly mean 10 m wind speed from the Copernicus Climate Data
    Store. Needs a free CDS account, the key in ~/.cdsapirc, and `pip install cdsapi xarray netcdf4`.
  - Bathymetry (--bathy): NCEI ETOPO 2022, 60 arc-second (about 1.8 km), via the NOAA PIFSC
    OceanWatch ERDDAP (no account). It also gives the coastline (elevation >= 0 is land) that sets
    the 0-50 km coastal buffer study area. --bathy-nc reads a GEBCO 2024 subset downloaded from
    https://download.gebco.net instead (netCDF, variable `elevation`).

Output (all npz with time in unix days, lat/lon ascending, values[time, lat, lon]):
  real_env/sst_daily.npz     sst, sst_anom (deg C)
  real_env/chl_monthly.npz   chl (mg/m3)
  real_env/wind_monthly.npz  wind (m/s)
  real_env/bathy.npz         elevation (m, positive up; no time axis)

Then run the simulator on them, e.g.
  python -m purse_seine_sim --vessels 10 --start 2023-01-01 --end 2025-12-31 --no-track --out sim_real \\
      --sst-npz real_env/sst_daily.npz --chl-npz real_env/chl_monthly.npz --wind-npz real_env/wind_monthly.npz
"""
from __future__ import annotations

import argparse
import csv
import io
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from purse_seine_sim.config import RegionConfig

ERDDAP_SERVERS = [
    "https://coastwatch.pfeg.noaa.gov/erddap/griddap",
    "https://upwell.pfeg.noaa.gov/erddap/griddap",          # mirror with the same datasets
]
PSL_URL = "https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres"
PSL_LTM = "sst.day.mean.ltm.1991-2020.nc"
OISST_DATASET = "ncdcOisst21Agg_LonPM180"
# (dataset id, variable, has an altitude dimension) tried in order.
CHL_DATASETS = [
    ("erdMH1chlamday", "chlorophyll", False),       # MODIS Aqua, 4 km, monthly
    ("nesdisVHNSQchlaMonthly", "chlor_a", True),     # VIIRS SNPP, 4 km, monthly
]
PAD_DEG = 0.25      # fetch a little beyond the region so interpolation works at the edges
BATHY_SERVERS = ["https://oceanwatch.pifsc.noaa.gov/erddap/griddap"]
BATHY_DATASET = "ETOPO_2022_v1_60s"
BATHY_PAD_DEG = 0.5     # wider, so coast just outside the region still counts for distance to coast


# =========================
# HELPERS
# =========================

def months_between(start, end):
    m0, m1 = np.datetime64(start, "M"), np.datetime64(end, "M")
    return [str(m) for m in np.arange(m0, m1 + np.timedelta64(1, "M"))]


def unix_days(iso):
    return float(np.datetime64(iso.rstrip("Z"), "s").astype("int64")) / 86400.0


def fetch(url, retries=4, timeout=120):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code in (400, 404):
                raise                                   # bad request: retrying will not help
            err = e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            err = e
        time.sleep(2 ** (attempt + 1))
    raise err


def erddap(query):
    """griddap request, trying each server in ERDDAP_SERVERS until one answers."""
    err = None
    for server in ERDDAP_SERVERS:
        try:
            return fetch(f"{server}/{query}", retries=2, timeout=60)
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            print(f"  {server.split('/')[2]} did not answer ({e}); trying the next server")
            err = e
    raise ConnectionError("no NOAA ERDDAP server answered. Try --psl (downloads the yearly NOAA PSL files) "
                          "or --oisst-nc with files downloaded in a browser") from err


def download(url, path):
    """Stream a large file to disk (skipped if it is already there)."""
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        print(f"  {path.name}: already downloaded")
        return path
    tmp = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {path.name}: {done / total:5.1%} of {total / 1e6:.0f} MB", end="", flush=True)
    print()
    if total and done != total:
        raise IOError(f"{path.name}: download stopped at {done} of {total} bytes; run again to retry")
    # On Windows an antivirus scan can hold the new file for a few seconds: retry the rename.
    for attempt in range(30):
        try:
            tmp.replace(path)
            return path
        except PermissionError:
            time.sleep(2)
    raise PermissionError(f"{tmp} is still locked by another program; rename it to {path.name} and run again")


def parse_griddap_csv(text, var):
    """ERDDAP griddap .csv (header row + units row) -> times, lats, lons, cube[time, lat, lon]."""
    rows = list(csv.reader(io.StringIO(text)))
    header, body = rows[0], rows[2:]
    it, ila, ilo, iv = header.index("time"), header.index("latitude"), header.index("longitude"), header.index(var)
    t = np.array([unix_days(r[it]) for r in body])
    la = np.array([float(r[ila]) for r in body])
    lo = np.array([float(r[ilo]) for r in body])
    v = np.array([float(r[iv]) if r[iv] not in ("", "NaN") else np.nan for r in body])
    times, lats, lons = np.unique(t), np.unique(la), np.unique(lo)
    cube = np.full((times.size, lats.size, lons.size), np.nan)
    cube[np.searchsorted(times, t), np.searchsorted(lats, la), np.searchsorted(lons, lo)] = v
    return times, lats, lons, cube


def bin_to_grid(lats, lons, cube, res):
    """Average a fine cube (e.g. 4 km CHL-a) onto a coarser regular grid of cell size `res`."""
    bi = np.round((lats - lats.min()) / res).astype(int)
    bj = np.round((lons - lons.min()) / res).astype(int)
    out = np.full((cube.shape[0], bi.max() + 1, bj.max() + 1), np.nan)
    for i in range(out.shape[1]):
        rows = cube[:, bi == i]
        for j in range(out.shape[2]):
            block = rows[:, :, bj == j].reshape(cube.shape[0], -1)
            if block.size:
                with np.errstate(all="ignore"):
                    out[:, i, j] = np.nanmean(block, axis=1)
    return lats.min() + res * np.arange(out.shape[1]), lons.min() + res * np.arange(out.shape[2]), out


# =========================
# SOURCES
# =========================

def fetch_oisst(region, months, verbose=True):
    """Daily OISST SST and anomaly, one request per month."""
    lat0, lat1 = region.lat_min - PAD_DEG, region.lat_max + PAD_DEG
    lon0, lon1 = region.lon_min - PAD_DEG, region.lon_max + PAD_DEG
    parts = []
    for m in months:
        last = str((np.datetime64(m, "M") + np.timedelta64(1, "M")).astype("datetime64[D]") - np.timedelta64(1, "D"))
        box = f"[({m}-01T12:00:00Z):1:({last}T12:00:00Z)][(0.0)][({lat0}):1:({lat1})][({lon0}):1:({lon1})]"
        text = erddap(f"{OISST_DATASET}.csv?sst{box},anom{box}")
        t, la, lo, sst = parse_griddap_csv(text, "sst")
        _, _, _, anom = parse_griddap_csv(text, "anom")
        parts.append((t, sst, anom))
        if verbose:
            print(f"  OISST {m}: {t.size} days, {la.size} x {lo.size} cells")
    t = np.concatenate([p[0] for p in parts])
    return {"time": t, "lat": la, "lon": lo,
            "sst": np.concatenate([p[1] for p in parts]), "sst_anom": np.concatenate([p[2] for p in parts])}


def read_oisst_nc(paths, region, ltm_path=None, verbose=True):
    """Cut the area of interest out of NOAA PSL OISST files (sst.day.mean.YYYY.nc, 0-360 longitude).

    With the PSL daily long-term mean file (sst.day.mean.ltm.1991-2020.nc) the anomaly is
    SST minus that day's 1991-2020 mean; without it, no anomaly is saved and the simulator
    falls back to SST minus its own assumed climatology.
    """
    import netCDF4

    def window(ds):
        lat, lon = ds["lat"][:].astype(float), ds["lon"][:].astype(float)
        i = np.flatnonzero((lat >= region.lat_min - PAD_DEG) & (lat <= region.lat_max + PAD_DEG))
        j = np.flatnonzero((lon >= region.lon_min - PAD_DEG) & (lon <= region.lon_max + PAD_DEG))
        return lat[i], lon[j], slice(i[0], i[-1] + 1), slice(j[0], j[-1] + 1)

    times, cubes = [], []
    for path in sorted(paths):
        with netCDF4.Dataset(path) as ds:
            lat, lon, si, sj = window(ds)
            t = netCDF4.num2date(ds["time"][:], ds["time"].units, only_use_cftime_datetimes=False)
            times.append(np.array([unix_days(d.isoformat()) for d in t]))
            cubes.append(np.ma.filled(ds["sst"][:, si, sj].astype(float), np.nan))
        if verbose:
            print(f"  {Path(path).name}: {times[-1].size} days, {lat.size} x {lon.size} cells")
    t, sst = np.concatenate(times), np.concatenate(cubes)
    order = np.argsort(t)
    out = {"time": t[order], "lat": lat, "lon": lon, "sst": sst[order]}
    if ltm_path:
        with netCDF4.Dataset(ltm_path) as ds:
            _, _, si, sj = window(ds)
            ltm = np.ma.filled(ds["sst"][:, si, sj].astype(float), np.nan)       # [day of year, lat, lon]
        doy = (out["time"].astype("int64").astype("datetime64[D]")
               - out["time"].astype("int64").astype("datetime64[D]").astype("datetime64[Y]")).astype(int)
        out["sst_anom"] = out["sst"] - ltm[np.minimum(doy, ltm.shape[0] - 1)]
    return out


def fetch_chl(region, months, res=0.05, verbose=True):
    """Monthly CHL-a, averaged onto a `res` grid. Returns None if no dataset answers."""
    t0, t1 = f"{months[0]}-01T00:00:00Z", f"{months[-1]}-28T00:00:00Z"
    lon0, lon1 = region.lon_min - PAD_DEG, region.lon_max + PAD_DEG
    for dataset, var, has_alt in CHL_DATASETS:
        alt = "[(0.0)]" if has_alt else ""
        # Some datasets store latitude north-to-south; ERDDAP wants the range in storage order.
        for lat0, lat1 in ((region.lat_max + PAD_DEG, region.lat_min - PAD_DEG),
                           (region.lat_min - PAD_DEG, region.lat_max + PAD_DEG)):
            query = f"{dataset}.csv?{var}[({t0}):1:({t1})]{alt}[({lat0}):1:({lat1})][({lon0}):1:({lon1})]"
            try:
                text = erddap(query)
            except urllib.error.HTTPError:
                continue
            except ConnectionError:
                print("  CHL-a: ERDDAP unreachable; skipping (the simulator keeps its analytic CHL-a)")
                return None
            t, la, lo, chl = parse_griddap_csv(text, var)
            la, lo, chl = bin_to_grid(la, lo, chl, res)
            if verbose:
                print(f"  CHL-a from {dataset}: {t.size} months, {la.size} x {lo.size} cells")
            return {"time": t, "lat": la, "lon": lo, "chl": chl, "source": np.array(dataset)}
    print("  CHL-a: no ERDDAP dataset answered; skipping (the simulator keeps its analytic CHL-a)")
    return None


def fetch_bathy(region, verbose=True):
    """ETOPO 2022 elevation (m, positive up) around the region."""
    lat0, lat1 = region.lat_min - BATHY_PAD_DEG, region.lat_max + BATHY_PAD_DEG
    lon0, lon1 = region.lon_min - BATHY_PAD_DEG, region.lon_max + BATHY_PAD_DEG
    query = f"{BATHY_DATASET}.csv?z[({lat0}):1:({lat1})][({lon0}):1:({lon1})]"
    err = None
    for server in BATHY_SERVERS:
        try:
            text = fetch(f"{server}/{query}", retries=2, timeout=180)
            break
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            print(f"  {server.split('/')[2]} did not answer ({e})")
            err = e
    else:
        raise ConnectionError("no bathymetry server answered. Download a GEBCO subset from "
                              "https://download.gebco.net and pass it with --bathy-nc") from err
    rows = list(csv.reader(io.StringIO(text)))[2:]
    la = np.array([float(r[0]) for r in rows])
    lo = np.array([float(r[1]) for r in rows])
    z = np.array([float(r[2]) if r[2] not in ("", "NaN") else np.nan for r in rows])
    lats, lons = np.unique(la), np.unique(lo)
    elev = np.full((lats.size, lons.size), np.nan)
    elev[np.searchsorted(lats, la), np.searchsorted(lons, lo)] = z
    if verbose:
        print(f"  Bathymetry from {BATHY_DATASET}: {lats.size} x {lons.size} cells")
    return {"lat": lats, "lon": lons, "elevation": elev.astype(np.float32), "source": np.array(BATHY_DATASET)}


def read_bathy_nc(path, region, verbose=True):
    """Cut the region out of a GEBCO (or any lat/lon `elevation`) netCDF file."""
    import netCDF4

    with netCDF4.Dataset(path) as ds:
        lat, lon = ds["lat"][:].astype(float), ds["lon"][:].astype(float)
        i = np.flatnonzero((lat >= region.lat_min - BATHY_PAD_DEG) & (lat <= region.lat_max + BATHY_PAD_DEG))
        j = np.flatnonzero((lon >= region.lon_min - BATHY_PAD_DEG) & (lon <= region.lon_max + BATHY_PAD_DEG))
        elev = np.ma.filled(ds["elevation"][i[0]:i[-1] + 1, j[0]:j[-1] + 1].astype(float), np.nan)
    lat, lon = lat[i], lon[j]
    if lat[0] > lat[-1]:
        lat, elev = lat[::-1], elev[::-1]
    if verbose:
        print(f"  Bathymetry from {Path(path).name}: {lat.size} x {lon.size} cells")
    return {"lat": lat, "lon": lon, "elevation": elev.astype(np.float32), "source": np.array(Path(path).name)}


def fetch_era5_wind(region, months, out_dir, verbose=True):
    """ERA5 monthly mean 10 m wind speed (needs cdsapi + a CDS key, and xarray + netcdf4 to read it)."""
    import cdsapi
    import xarray as xr

    years = sorted({m[:4] for m in months})
    target = Path(out_dir) / "era5_wind_monthly.nc"
    cdsapi.Client().retrieve("reanalysis-era5-single-levels-monthly-means", {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": ["10m_wind_speed"],
        "year": years,
        "month": [f"{i:02d}" for i in range(1, 13)],
        "time": ["00:00"],
        "area": [region.lat_max + PAD_DEG, region.lon_min - PAD_DEG, region.lat_min - PAD_DEG, region.lon_max + PAD_DEG],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }, str(target))
    ds = xr.open_dataset(target)
    tname = "valid_time" if "valid_time" in ds.coords else "time"
    ds = ds.sortby("latitude").sortby("longitude").sortby(tname)
    times = ds[tname].values.astype("datetime64[s]").astype("int64") / 86400.0 + 14.0   # mid-month
    keep = np.isin(ds[tname].values.astype("datetime64[M]").astype(str), months)
    wind = np.asarray(ds["si10"].values, float)[keep]
    if verbose:
        print(f"  ERA5 wind: {int(keep.sum())} months, {ds.latitude.size} x {ds.longitude.size} cells")
    return {"time": times[keep], "lat": ds.latitude.values.astype(float), "lon": ds.longitude.values.astype(float),
            "wind": wind}


# =========================
# MAIN
# =========================

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--start", help="first month, YYYY-MM")
    p.add_argument("--end", help="last month, YYYY-MM")
    p.add_argument("--out", default="real_env")
    p.add_argument("--era5", action="store_true", help="also fetch ERA5 wind (needs a CDS key)")
    p.add_argument("--no-chl", action="store_true")
    p.add_argument("--oisst-nc", nargs="+", help="read these NOAA PSL OISST files instead of downloading SST")
    p.add_argument("--oisst-ltm", help="PSL daily long-term mean file, for the SST anomaly")
    p.add_argument("--psl", action="store_true",
                   help="download the yearly NOAA PSL OISST files (+ 1991-2020 mean) instead of using ERDDAP")
    p.add_argument("--bathy", action="store_true", help="also fetch ETOPO 2022 bathymetry")
    p.add_argument("--bathy-nc", help="read bathymetry from this GEBCO netCDF file instead of downloading it")
    a = p.parse_args(argv)
    bathy_only = (a.bathy or a.bathy_nc) and not (a.oisst_nc or a.start or a.end)
    if not bathy_only and not a.oisst_nc and not (a.start and a.end):
        p.error("--start and --end are required unless --oisst-nc or only --bathy is given")
    if a.psl and not (a.start and a.end):
        p.error("--psl needs --start and --end")

    region = RegionConfig()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.bathy or a.bathy_nc:
        bathy = read_bathy_nc(a.bathy_nc, region) if a.bathy_nc else fetch_bathy(region)
        np.savez_compressed(out / "bathy.npz", **bathy)
        if bathy_only:
            print(f"Saved to {out.resolve()}")
            return
    if a.psl:
        first = np.datetime64(a.start, "M") - np.timedelta64(1, "M")
        years = range(int(str(first)[:4]), int(a.end[:4]) + 1)
        (out / "nc").mkdir(exist_ok=True)
        a.oisst_nc = [download(f"{PSL_URL}/sst.day.mean.{y}.nc", out / "nc" / f"sst.day.mean.{y}.nc") for y in years]
        a.oisst_ltm = a.oisst_ltm or download(f"{PSL_URL}/{PSL_LTM}", out / "nc" / PSL_LTM)
    if a.oisst_nc:
        sst = read_oisst_nc(a.oisst_nc, region, a.oisst_ltm)
        span = sst["time"][[0, -1]].astype("int64").astype("datetime64[D]").astype("datetime64[M]")
        months = months_between(str(span[0]), str(span[1]))
    else:
        # One extra month before the start so the first month's lagged forecast inputs exist.
        months = months_between(str(np.datetime64(a.start, "M") - np.timedelta64(1, "M")), a.end)
        sst = fetch_oisst(region, months)
    print(f"Area {region.lat_min}-{region.lat_max} N, {region.lon_min}-{region.lon_max} E, {months[0]} .. {months[-1]}")

    np.savez_compressed(out / "sst_daily.npz", **sst)
    if not a.no_chl:
        chl = fetch_chl(region, months, res=region.map_res_deg)
        if chl is not None:
            np.savez_compressed(out / "chl_monthly.npz", **chl)
    if a.era5:
        np.savez_compressed(out / "wind_monthly.npz", **fetch_era5_wind(region, months, out))
    print(f"Saved to {out.resolve()}")


if __name__ == "__main__":
    main()
