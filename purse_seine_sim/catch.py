"""Catch process of one purse-seine set, plus logbook reporting errors.

A set encircles one school: a dominant species is drawn in proportion to
catchability x density, other targets mix in with probability rising with their
density, and ETP animals are encountered as a Poisson process.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .environment import frac_year
from .species import TRASH, gear_multiplier

CONDITIONS = ("alive", "injured", "dead")
FATES = {"alive": "released_alive", "injured": "released_injured", "dead": "dead_discarded"}


@dataclass
class SetCatch:
    water_haul: bool
    p_water_haul: float
    kg: dict            # target species + TRASH_MIX
    juvenile_kg: dict
    discard_kg: dict
    counts: dict        # ETP species
    individuals: list   # (code, condition)
    p_present: dict     # true probability each checklist species is caught

    @property
    def total_kg(self):
        return float(sum(self.kg.values()))

    @property
    def discard_total_kg(self):
        return float(sum(self.discard_kg.values()))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def simulate_catch(rng, targets, etps, lam, set_type, env_pt, vessel, cfg) -> SetCatch:
    cc = cfg.catch
    moon, wind, dist = env_pt["moon"], env_pt["wind"], env_pt["dist_coast"]
    fy = float(frac_year(env_pt["t"]))
    light = set_type == "light"
    fad = set_type == "fad"

    p_water = float(np.clip(cc.base_water_haul_prob + 0.03 * max(0.0, wind - 6.0) + (0.04 if light else 0.0), 0.0, 0.6))
    water = bool(rng.random() < p_water)

    # ---------- target species ----------
    w = np.array([float(gear_multiplier(m.sp, set_type, moon)) * lam[m.sp.code] for m in targets])
    pdom = w / w.sum() if w.sum() > 0 else np.full(len(w), 1.0 / len(w))
    pmix = 1.0 - np.exp(-cc.mix_q * w)
    kg = np.zeros(len(targets))
    if not water:
        d = rng.choice(len(targets), p=pdom)
        sp = targets[d].sp
        kg[d] = sp.school_median_kg * max(lam[sp.code], 1e-6) ** 0.7 * np.exp(rng.normal(0.0, sp.school_sigma))
        for i in range(len(targets)):
            if i != d and rng.random() < pmix[i]:
                kg[i] = kg[d] * rng.beta(1.2, 6.0)
        total = kg.sum()
        if total > vessel.net_capacity_kg:
            kg *= vessel.net_capacity_kg / total

    out_kg, juv, disc, p_present = {}, {}, {}, {}
    for i, m in enumerate(targets):
        sp = m.sp
        out_kg[sp.code] = float(kg[i])
        j = 0.0
        if sp.juvenile is not None and kg[i] > 0:
            base, fad_eff, light_eff, nursery_doy, nursery_amp = sp.juvenile
            logit = (base + fad_eff * fad + light_eff * light
                     + nursery_amp * np.cos(2 * np.pi * (fy - nursery_doy / 365.25))
                     + 0.8 * np.exp(-dist / 25.0) + rng.normal(0.0, 0.5))
            j = float(kg[i] * _sigmoid(logit))
        juv[sp.code] = j
        disc[sp.code] = j * cc.juvenile_discard_frac
        p_present[sp.code] = float((1 - p_water) * (pdom[i] + (1 - pdom[i]) * pmix[i]))

    target_total = float(kg.sum())
    trash = 0.0
    if target_total > 0:
        trash = target_total * float(_sigmoid(-2.3 + 0.9 * light + 0.8 * np.exp(-dist / 20.0) + rng.normal(0.0, 0.6)))
    code = TRASH["code"]
    out_kg[code], juv[code], disc[code] = trash, 0.0, trash * cc.trash_discard_frac
    p_present[code] = 1.0 - p_water

    # ---------- ETP bycatch ----------
    counts, individuals = {}, []
    for m in etps:
        sp = m.sp
        r = sp.base_rate * lam[sp.code] * float(gear_multiplier(sp, set_type, moon))
        r_eff = r * (0.3 if water else 1.0)
        p_present[sp.code] = float(p_water * (1 - np.exp(-0.3 * r)) + (1 - p_water) * (1 - np.exp(-r)))
        n = 0
        if rng.random() < 1.0 - np.exp(-r_eff):
            n = 1 + int(rng.poisson(max(sp.group_mean - 1.0, 0.0)))
            for cond in rng.choice(CONDITIONS, size=n, p=sp.condition_probs):
                individuals.append((sp.code, str(cond)))
        counts[sp.code] = n

    return SetCatch(water, p_water, out_kg, juv, disc, counts, individuals, p_present)


def report(c: SetCatch, rng, rcfg):
    """Apply logbook errors: kg rounding, mackerel mis-identification, ETP under-reporting."""
    step = rcfg.kg_rounding

    def rnd(x):
        if x <= 0:
            return 0.0
        if step <= 0:
            return round(x, 1)
        return max(step, round(x / step) * step)     # never round a real catch down to zero

    kg = {k: rnd(v) for k, v in c.kg.items()}
    juv = {k: min(rnd(v), kg[k]) for k, v in c.juvenile_kg.items()}
    disc = {k: min(rnd(v), kg[k]) for k, v in c.discard_kg.items()}
    if rng.random() < rcfg.mackerel_misid_prob and "RAST_BRA" in kg and "RAST_KAN" in kg:
        for d in (kg, juv, disc):
            d["RAST_BRA"], d["RAST_KAN"] = d["RAST_KAN"], d["RAST_BRA"]

    counts = dict(c.counts)
    individuals = list(c.individuals)
    if individuals and rng.random() < rcfg.etp_underreport_prob:
        counts = {k: 0 for k in counts}
        individuals = []
    return kg, juv, disc, counts, individuals
