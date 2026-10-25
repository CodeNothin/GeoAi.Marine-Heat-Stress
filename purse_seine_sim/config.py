"""Simulation configuration.

Every number in this file is an ASSUMPTION chosen to be plausible for large
commercial purse seiners in the southern Gulf of Thailand. Replace them with
real values once the vessel operator / DOF data arrives.

Override any field with a JSON file (nested keys, only what you want to change):

    python -m purse_seine_sim --config my_config.json

    {"fleet": {"n_vessels": 10}, "reporting": {"etp_underreport_prob": 0.3}}
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field


def _list(*values):
    return field(default_factory=lambda: list(values))


# =========================
# REGION
# =========================

@dataclass
class RegionConfig:
    lat_min: float = 6.3
    lat_max: float = 11.5
    lon_min: float = 99.0
    lon_max: float = 102.2
    grid_res_deg: float = 0.02      # static fields (land mask, depth, distance to coast)
    map_res_deg: float = 0.05       # exported truth maps / prediction grid


# =========================
# REGULATION
# =========================

def _default_closures():
    return [
        {
            # ASSUMED approximation of the Feb 15 - May 15 spawning closure off
            # Prachuap / Chumphon / Surat Thani. Replace with official DOF coordinates.
            "name": "Gulf spawning closure (approx. box)",
            "lat_min": 9.3, "lat_max": 11.5,
            "lon_min": 99.0, "lon_max": 100.3,
            "start_mmdd": "02-15", "end_mmdd": "05-15",
        }
    ]


@dataclass
class RegulationConfig:
    min_dist_coast_km: float = 5.56     # 3 nm coastal zone reserved for small-scale fishing
    min_set_depth_m: float = 15.0
    closures: list = field(default_factory=_default_closures)
    compliance_prob: float = 1.0        # probability a vessel respects closures


# =========================
# FLEET / SKIPPER BEHAVIOUR
# =========================

def _default_ports():
    # Approximate positions; snapped to the nearest ocean cell at runtime.
    return [
        {"name": "Chumphon", "lat": 10.35, "lon": 99.28},
        {"name": "Surat Thani", "lat": 9.25, "lon": 99.45},
        {"name": "Pak Phanang", "lat": 8.36, "lon": 100.22},
        {"name": "Songkhla", "lat": 7.22, "lon": 100.60},
        {"name": "Pattani", "lat": 6.90, "lon": 101.27},
    ]


@dataclass
class FleetConfig:
    n_vessels: int = 4
    ports: list = field(default_factory=_default_ports)
    gross_tonnage_range: list = _list(60.0, 150.0)
    length_m_range: list = _list(20.0, 32.0)
    hold_capacity_t_range: list = _list(40.0, 90.0)
    net_capacity_t: float = 40.0
    net_length_m_range: list = _list(600.0, 1000.0)
    net_depth_m_range: list = _list(60.0, 100.0)
    light_fraction: float = 0.7             # share of vessels equipped for light luring

    trip_days_range: list = _list(5, 12)
    port_days_range: list = _list(1, 3)
    steam_speed_kn: float = 9.0
    search_speed_kn: float = 6.0
    max_departure_wind_ms: float = 9.0
    max_fishing_wind_ms: float = 10.0

    day_sets_mean: float = 1.6              # Poisson mean of daytime sets per day
    max_day_sets: int = 3
    light_max_moon: float = 0.55            # light luring only when moon illumination below this
    light_hours_range: list = _list(2.0, 5.0)
    set_duration_h: list = _list(2.2, 0.5)  # mean, sd
    search_hours_mean: float = 1.2

    search_radius_km: float = 60.0
    n_candidates: int = 150
    skipper_noise_sd: float = 0.5           # log-scale error in the skipper's knowledge of fish density
    choice_temperature: float = 0.35
    distance_cost_km: float = 80.0          # e-folding distance penalty when choosing a spot
    explore_prob: float = 0.05              # random legal spot instead of the best one
    bycatch_avoidance: float = 0.0          # >0 = skipper avoids predicted ETP hotspots (EcoCast scenario)

    fads_per_vessel_range: list = _list(5, 12)
    fad_max_dist_port_km: float = 120.0
    fad_depth_range_m: list = _list(25.0, 65.0)
    fad_set_prob: float = 0.35              # share of daytime sets made on the vessel's own FADs
    hold_fill_return: float = 0.9


# =========================
# ENVIRONMENT
# =========================

def _default_mhw():
    return [
        {
            # ASSUMED, loosely modelled on the record April-May 2024 Gulf heatwave.
            "name": "2024 spring marine heatwave",
            "start": "2024-03-25", "peak": "2024-05-05", "end": "2024-06-20",
            "intensity_c": 1.4,
        }
    ]


@dataclass
class EnvConfig:
    mhw_events: list = field(default_factory=_default_mhw)
    interannual_sd_c: float = 0.35
    storms_per_year: float = 5.0
    sst_obs_noise_c: float = 0.25
    chl_obs_noise_log10: float = 0.10
    cloud_prob_by_month: list = _list(0.55, 0.35, 0.20, 0.20, 0.30, 0.45,
                                      0.45, 0.45, 0.45, 0.50, 0.65, 0.65)
    wind_obs_noise_ms: float = 0.8
    depth_obs_noise_m: float = 2.0
    # Optional real SST: .npz with arrays time (unix days or datetime64), lat, lon, sst[time, lat, lon]
    sst_override_npz: str | None = None


# =========================
# CATCH PROCESS
# =========================

@dataclass
class CatchConfig:
    base_water_haul_prob: float = 0.10      # set made but the school escaped
    mix_q: float = 1.2                      # how readily non-dominant species mix into a school
    juvenile_discard_frac: float = 0.3
    trash_discard_frac: float = 0.4


# =========================
# REPORTING ERRORS (logbook realism)
# =========================

@dataclass
class ReportingConfig:
    etp_underreport_prob: float = 0.0       # whole ETP interaction left out of the logbook
    mackerel_misid_prob: float = 0.0        # RAST_BRA / RAST_KAN swapped
    kg_rounding: float = 10.0
    track_drop_prob: float = 0.05           # missing AIS/VMS pings


# =========================
# TOP LEVEL
# =========================

@dataclass
class SimConfig:
    start: str = "2024-01-01"
    end: str = "2025-12-31"
    seed: int = 42
    track_ping_min: float = 10.0
    write_track: bool = True
    region: RegionConfig = field(default_factory=RegionConfig)
    regulation: RegulationConfig = field(default_factory=RegulationConfig)
    fleet: FleetConfig = field(default_factory=FleetConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    catch: CatchConfig = field(default_factory=CatchConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SimConfig":
        cfg = cls()
        _merge(cfg, d)
        return cfg

    @classmethod
    def from_json(cls, path) -> "SimConfig":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def _merge(obj, d: dict, prefix=""):
    for key, value in d.items():
        if not hasattr(obj, key):
            raise KeyError(f"unknown config key: {prefix}{key}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _merge(current, value, prefix=f"{prefix}{key}.")
        else:
            setattr(obj, key, value)
