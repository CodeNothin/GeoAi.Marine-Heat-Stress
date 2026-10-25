"""Virtual species: the "true" habitat responses the ML model should later recover.

All parameters are ASSUMED (literature-inspired, not fitted). Each species gets
a relative density index (lambda, roughly 0-1):

    lambda = prod(responses to covariates) * season * heat-stress penalty * patchiness

Covariates available (see Environment.sample): sst, sst_anom, log10chl, front,
depth, dist_coast, seagrass_idx.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .environment import frac_year
from .fields import RandomField

SET_TYPES = ("free_school", "fad", "light")


@dataclass
class Species:
    code: str
    name_en: str
    name_th: str
    sci_name: str
    group: str                          # "target" | "etp" (endangered, threatened, protected)
    responses: tuple                    # (covariate, "gauss"|"logistic", a, b)
    season_peak_doy: float = 0.0
    season_amp: float = 0.0
    heat_threshold_c: float = 99.0      # SST anomaly above which the species moves away
    heat_rate: float = 0.0
    patch_sd: float = 0.5               # unexplained spatial patchiness (log scale)
    patch_scale_km: float = 60.0
    patch_time_days: float = 20.0
    gear: dict = field(default_factory=lambda: {"free_school": 1.0, "fad": 1.0, "light": 1.0})
    # target species
    price_thb_kg: float = 0.0
    school_median_kg: float = 0.0
    school_sigma: float = 0.9
    juvenile: tuple | None = None       # (base_logit, fad_effect, light_effect, nursery_peak_doy, nursery_amp)
    # ETP species
    base_rate: float = 0.0              # expected encounters per set at lambda = 1, gear = 1
    group_mean: float = 1.0
    condition_probs: tuple = (1.0, 0.0, 0.0)  # alive, injured, dead


TARGETS = [
    Species("RAST_BRA", "Indo-Pacific mackerel", "ปลาทู", "Rastrelliger brachysoma", "target",
            responses=(("sst", "gauss", 29.0, 1.6), ("log10chl", "gauss", 0.1, 0.45), ("depth", "gauss", 30.0, 15.0)),
            season_peak_doy=75, season_amp=0.4, heat_threshold_c=1.0, heat_rate=0.7, patch_sd=0.6,
            gear={"free_school": 1.0, "fad": 1.3, "light": 1.2},
            price_thb_kg=40, school_median_kg=3000, school_sigma=0.9,
            juvenile=(-1.5, 0.8, 0.5, 150, 1.0)),
    Species("RAST_KAN", "Indian mackerel", "ปลาลัง", "Rastrelliger kanagurta", "target",
            responses=(("sst", "gauss", 29.2, 1.8), ("log10chl", "gauss", -0.1, 0.5), ("depth", "gauss", 45.0, 18.0)),
            season_peak_doy=300, season_amp=0.25, heat_threshold_c=1.2, heat_rate=0.5,
            gear={"free_school": 1.0, "fad": 1.2, "light": 1.1},
            price_thb_kg=35, school_median_kg=2500, school_sigma=0.9,
            juvenile=(-1.8, 0.7, 0.4, 200, 0.8)),
    Species("SARD_GIB", "Goldstripe sardinella", "ปลาหลังเขียว", "Sardinella gibbosa", "target",
            responses=(("sst", "gauss", 28.8, 1.8), ("log10chl", "gauss", 0.4, 0.45), ("depth", "gauss", 20.0, 10.0)),
            season_peak_doy=330, season_amp=0.3, heat_threshold_c=1.5, heat_rate=0.4,
            gear={"free_school": 0.9, "fad": 0.8, "light": 2.0},
            price_thb_kg=12, school_median_kg=4000, school_sigma=1.0),
    Species("DECA_MAR", "Japanese scad", "ปลาทูแขก", "Decapterus maruadsi", "target",
            responses=(("sst", "gauss", 28.6, 1.5), ("log10chl", "gauss", -0.35, 0.4), ("depth", "gauss", 60.0, 15.0)),
            season_peak_doy=20, season_amp=0.3, heat_threshold_c=0.8, heat_rate=0.8,
            gear={"free_school": 0.8, "fad": 1.0, "light": 1.8},
            price_thb_kg=25, school_median_kg=3000, school_sigma=0.9),
    Species("THUN_TON", "Longtail tuna", "ปลาโอดำ", "Thunnus tonggol", "target",
            responses=(("sst", "gauss", 29.6, 1.4), ("log10chl", "gauss", -0.3, 0.5),
                       ("depth", "logistic", 35.0, 6.0), ("front", "logistic", 0.15, 0.05)),
            season_peak_doy=240, season_amp=0.3, heat_threshold_c=2.0, heat_rate=0.3,
            gear={"free_school": 1.0, "fad": 2.5, "light": 0.3},
            price_thb_kg=65, school_median_kg=1500, school_sigma=1.0,
            juvenile=(-1.0, 1.2, 0.0, 120, 0.6)),
    Species("EUTH_AFF", "Kawakawa", "ปลาโอลาย", "Euthynnus affinis", "target",
            responses=(("sst", "gauss", 29.4, 1.5), ("log10chl", "gauss", -0.2, 0.5),
                       ("depth", "gauss", 50.0, 20.0), ("front", "logistic", 0.15, 0.05)),
            season_peak_doy=180, season_amp=0.3, heat_threshold_c=2.0, heat_rate=0.3,
            gear={"free_school": 1.0, "fad": 2.2, "light": 0.4},
            price_thb_kg=50, school_median_kg=1800, school_sigma=1.0,
            juvenile=(-1.2, 1.0, 0.0, 130, 0.6)),
]

ETP = [
    Species("RHIN_TYP", "Whale shark", "ฉลามวาฬ", "Rhincodon typus", "etp",
            responses=(("sst", "gauss", 29.5, 1.5), ("log10chl", "logistic", 0.0, 0.2),
                       ("front", "logistic", 0.15, 0.05), ("depth", "logistic", 25.0, 5.0)),
            season_peak_doy=100, season_amp=0.5, patch_sd=0.7, patch_scale_km=80.0,
            gear={"free_school": 1.5, "fad": 1.2, "light": 0.3},
            base_rate=0.02, group_mean=1.0, condition_probs=(0.85, 0.10, 0.05)),
    Species("MOBU_SPP", "Devil rays (mobulids)", "กระเบนราหู", "Mobula spp.", "etp",
            responses=(("sst", "gauss", 29.0, 2.0), ("log10chl", "logistic", -0.2, 0.25),
                       ("front", "logistic", 0.12, 0.05)),
            season_peak_doy=90, season_amp=0.3,
            gear={"free_school": 1.2, "fad": 1.0, "light": 0.8},
            base_rate=0.06, group_mean=1.5, condition_probs=(0.5, 0.2, 0.3)),
    Species("CARC_SPP", "Requiem sharks (silky / blacktip)", "ฉลาม (วงศ์ Carcharhinidae)", "Carcharhinidae", "etp",
            responses=(("depth", "logistic", 30.0, 8.0), ("front", "logistic", 0.12, 0.05),
                       ("sst", "gauss", 29.3, 2.0)),
            gear={"free_school": 1.0, "fad": 3.0, "light": 0.6},
            base_rate=0.3, group_mean=2.0, condition_probs=(0.3, 0.2, 0.5)),
    Species("CHEL_MYD", "Green turtle", "เต่าตนุ", "Chelonia mydas", "etp",
            responses=(("seagrass_idx", "logistic", 0.25, 0.08), ("depth", "logistic", 30.0, -8.0)),
            patch_sd=0.4,
            gear={"free_school": 0.6, "fad": 3.0, "light": 0.5},
            base_rate=0.08, group_mean=1.0, condition_probs=(0.8, 0.1, 0.1)),
    Species("ORCA_BRE", "Irrawaddy dolphin", "โลมาอิรวดี", "Orcaella brevirostris", "etp",
            responses=(("dist_coast", "logistic", 12.0, -3.0), ("log10chl", "logistic", 0.3, 0.2)),
            patch_sd=0.4,
            gear={"free_school": 1.0, "fad": 0.3, "light": 0.5},
            base_rate=0.05, group_mean=2.0, condition_probs=(0.7, 0.1, 0.2)),
]

# Mixed low-value catch, not a habitat model: derived from the set's target catch.
TRASH = {"code": "TRASH_MIX", "name_en": "Mixed trash fish (fishmeal grade)", "name_th": "ปลาเป็ด",
         "sci_name": "mixed", "group": "discard"}

ALL_SPECIES = TARGETS + ETP


def checklist():
    """Species every logbook set must report, zeros included."""
    rows = [{"code": s.code, "group": s.group, "name_en": s.name_en, "name_th": s.name_th,
             "sci_name": s.sci_name, "unit": "kg" if s.group == "target" else "count"} for s in TARGETS]
    rows.append({"code": TRASH["code"], "group": TRASH["group"], "name_en": TRASH["name_en"],
                 "name_th": TRASH["name_th"], "sci_name": TRASH["sci_name"], "unit": "kg"})
    rows += [{"code": s.code, "group": s.group, "name_en": s.name_en, "name_th": s.name_th,
              "sci_name": s.sci_name, "unit": "count"} for s in ETP]
    return rows


def species_params():
    return [asdict(s) for s in ALL_SPECIES]


def response(kind, x, a, b):
    x = np.asarray(x, float)
    if kind == "gauss":
        return np.exp(-0.5 * ((x - a) / b) ** 2)
    if kind == "logistic":
        return 1.0 / (1.0 + np.exp(-(x - a) / b))
    raise ValueError(f"unknown response kind: {kind}")


def gear_multiplier(sp: Species, set_type, moon):
    g = sp.gear.get(set_type, 1.0)
    if set_type == "light":
        g = g * (1.0 - 0.75 * np.asarray(moon, float))   # light luring works poorly near full moon
    return g


class SpeciesModel:
    def __init__(self, sp: Species, rng):
        self.sp = sp
        self.patch = RandomField(rng, n_modes=16, length_km=sp.patch_scale_km, sd=sp.patch_sd,
                                 time_days=sp.patch_time_days) if sp.patch_sd > 0 else None

    def density(self, env: dict):
        sp = self.sp
        lam = np.ones(env["lat"].shape)
        for cov, kind, a, b in sp.responses:
            lam = lam * response(kind, env[cov], a, b)
        if sp.season_amp:
            fy = frac_year(env["t"])
            lam = lam * (1 + sp.season_amp * np.cos(2 * np.pi * (fy - sp.season_peak_doy / 365.25))) / (1 + sp.season_amp)
        if sp.heat_rate:
            lam = lam * np.exp(-sp.heat_rate * np.maximum(0.0, env["sst_anom"] - sp.heat_threshold_c))
        if self.patch is not None:
            lam = lam * np.exp(self.patch(env["x_km"], env["y_km"], env["t"]) - 0.5 * sp.patch_sd ** 2)
        return lam


def build_species_models(rng):
    return [SpeciesModel(sp, rng) for sp in ALL_SPECIES]
