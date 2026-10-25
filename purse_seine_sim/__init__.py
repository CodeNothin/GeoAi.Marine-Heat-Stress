"""Synthetic purse-seine logbook simulator for the southern Gulf of Thailand.

Generates fishing events, catch (zeros included), ETP bycatch, vessel tracks and
satellite covariates in the same schema real vessel data should follow, plus the
hidden "truth" needed to test an EcoCast-style bycatch model.
"""
from .config import SimConfig
from .run import run_simulation

__all__ = ["SimConfig", "run_simulation"]
