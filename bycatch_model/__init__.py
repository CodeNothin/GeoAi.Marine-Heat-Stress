"""EcoCast-style bycatch model: per-species GAMs on logbook + satellite data, combined into one map."""
from .data import load_dataset, load_grid
from .ecocast import ecocast_score, fit_all
from .gam import GAM

__all__ = ["GAM", "load_dataset", "load_grid", "fit_all", "ecocast_score"]
