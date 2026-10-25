"""EcoCast-style workflow: per-species models -> predicted maps -> one weighted score.

For each species, the model gives
  P(caught in a set)                      binomial GAM, all species
  E[kg | caught]                          log-normal GAM, target species (the "density" part)
  expected catch per set = P * E[kg | caught]

EcoCast score per cell (Hazen et al. 2018 idea, our weights):
  score = w_target * normalised expected revenue - sum_b w_b * normalised P(bycatch_b)
Positive = good fishing with little bycatch, negative = bycatch hotspot.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from purse_seine_sim.species import TARGETS

from .data import LINEAR_TERMS, SMOOTH_TERMS
from .gam import GAM

PRICES_THB_KG = {s.code: s.price_thb_kg for s in TARGETS}     # ASSUMED; replace with market prices

# ASSUMED weights: 40% on revenue, 60% spread over ETP risk (EcoCast let managers choose these).
DEFAULT_WEIGHTS = {"revenue": 0.40, "RHIN_TYP": 0.15, "MOBU_SPP": 0.10, "CARC_SPP": 0.15,
                   "CHEL_MYD": 0.15, "ORCA_BRE": 0.05}

MIN_POSITIVES = 5           # below this a species model is not fitted
MIN_POSITIVES_KG = 30
FOOTPRINT_KM = 40.0         # cells farther than this from any training set = extrapolation


@dataclass
class SpeciesFit:
    code: str
    group: str
    n_train: int
    n_pos: int
    presence: GAM | None
    amount: GAM | None


def fit_species(ds, code, group):
    y = ds.presence[code].astype(float)
    n_pos = int(y.sum())
    if n_pos < MIN_POSITIVES or n_pos == y.size:
        return SpeciesFit(code, group, y.size, n_pos, None, None)
    pres = GAM(SMOOTH_TERMS, LINEAR_TERMS, "binomial").fit(ds.features, y)
    amount = None
    if group == "target" and n_pos >= MIN_POSITIVES_KG:
        sel = ds.presence[code]
        F = {k: v[sel] for k, v in ds.features.items()}
        amount = GAM(SMOOTH_TERMS, LINEAR_TERMS, "gaussian").fit(F, np.log(ds.amount[code][sel]))
    return SpeciesFit(code, group, y.size, n_pos, pres, amount)


def fit_all(ds, codes=None):
    codes = codes or [c for c, g in ds.groups.items() if g in ("target", "etp")]
    return {c: fit_species(ds, c, ds.groups[c]) for c in codes}


def predict_presence(fit: SpeciesFit, F):
    n = len(next(iter(F.values())))
    if fit.presence is None:
        return np.full(n, np.nan)
    return fit.presence.predict(F)


def predict_expected_kg(fit: SpeciesFit, F):
    p = predict_presence(fit, F)
    if fit.amount is None:
        return np.full(p.shape, np.nan)
    return p * np.exp(fit.amount.predict(F) + 0.5 * fit.amount.sigma2)


def footprint(cell_lat, cell_lon, set_lat, set_lon, max_km=FOOTPRINT_KM):
    """True where a cell is within max_km of at least one training set."""
    from purse_seine_sim.geo import distance_km
    inside = np.zeros(cell_lat.size, bool)
    for s in range(0, cell_lat.size, 500):
        d = distance_km(cell_lat[s:s + 500, None], cell_lon[s:s + 500, None], set_lat[None, :], set_lon[None, :])
        inside[s:s + 500] = d.min(axis=1) <= max_km
    return inside


def _norm(x, ref=None):
    x = np.nan_to_num(x)
    pool = x[ref] if ref is not None else x
    top = np.percentile(pool, 99) if pool.size else 0.0
    return np.clip(x / top, 0, 1) if top > 0 else np.zeros_like(x)


def ecocast_score(layers: dict, weights=DEFAULT_WEIGHTS, ref=None):
    """layers: 'revenue' plus one P(bycatch) layer per ETP code. Missing layers are skipped.
    ref: boolean cell mask used to normalise each layer (the data footprint)."""
    score = None
    used = []
    for key, w in weights.items():
        layer = layers.get(key)
        if layer is None or not np.isfinite(layer).any():
            continue
        contrib = (1 if key == "revenue" else -1) * w * _norm(layer, ref)
        score = contrib if score is None else score + contrib
        used.append(key)
    return score, used


def predicted_layers(fits, F):
    layers = {"revenue": np.zeros(len(next(iter(F.values()))))}
    out = {}
    for code, fit in fits.items():
        p = predict_presence(fit, F)
        out[f"p_{code}"] = p
        if fit.group == "target":
            ekg = predict_expected_kg(fit, F)
            out[f"ekg_{code}"] = ekg
            layers["revenue"] += np.nan_to_num(ekg) * PRICES_THB_KG.get(code, 0.0)
        elif fit.group == "etp":
            layers[code] = p
    return layers, out


def true_layers(truth_lambda):
    """Same score from the simulator's true densities (only for scoring the model)."""
    from purse_seine_sim.species import ALL_SPECIES
    by_code = {s.code: s for s in ALL_SPECIES}
    layers = {"revenue": np.zeros(len(next(iter(truth_lambda.values()))))}
    for code, lam in truth_lambda.items():
        sp = by_code.get(code)
        if sp is None:
            continue
        if sp.group == "target":
            layers["revenue"] += sp.price_thb_kg * sp.school_median_kg * lam ** 0.7 * (1 - np.exp(-lam))
        else:
            layers[code] = 1 - np.exp(-sp.base_rate * lam)
    return layers


# =========================
# METRICS
# =========================

def auc(score, y):
    y = np.asarray(y, bool)
    ok = np.isfinite(score)
    score, y = np.asarray(score)[ok], y[ok]
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(score.size)
    ranks[order] = np.arange(1, score.size + 1)
    # average ranks for ties
    s_sorted = score[order]
    _, start, counts = np.unique(s_sorted, return_index=True, return_counts=True)
    for st, c in zip(start, counts):
        if c > 1:
            ranks[order[st:st + c]] = st + (c + 1) / 2
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    ra = np.argsort(np.argsort(a[ok]))
    rb = np.argsort(np.argsort(b[ok]))
    return float(np.corrcoef(ra, rb)[0, 1])
