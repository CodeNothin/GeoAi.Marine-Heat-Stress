"""Top-level entry point: build the world, simulate the fleet, write everything."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import schema
from .environment import Environment
from .fleet import FleetSimulator
from .geo import Geography
from .output import Recorder, make_plots, print_summary, write_csv, write_json, write_truth_maps
from .species import ALL_SPECIES, TRASH, build_species_models, checklist, species_params


def checklist_codes():
    return [r["code"] for r in checklist()]


def run_simulation(cfg, out_dir, write_maps=True, plots=False, verbose=True):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    r_geo, r_env, r_species, r_fleet, r_rec = np.random.default_rng(cfg.seed).spawn(5)

    geo = Geography(cfg.region, r_geo)
    env = Environment(cfg, geo, r_env)
    models = build_species_models(r_species)
    codes = checklist_codes()
    groups = {s.code: s.group for s in ALL_SPECIES} | {TRASH["code"]: TRASH["group"]}

    rec = Recorder(out, cfg, codes, r_rec)
    try:
        sim = FleetSimulator(cfg, geo, env, models, rec, r_fleet)
        vessels = sim.make_fleet()
        sim.run(vessels)
    finally:
        rec.close()

    write_csv(out / "reference" / "species_checklist.csv", checklist(),
              ["code", "group", "unit", "name_en", "name_th", "sci_name"])
    write_json(out / "reference" / "species_params_ASSUMED.json", species_params())
    write_json(out / "reference" / "schema.json", schema.as_json())
    write_json(out / "config_used.json", cfg.to_dict())

    if write_maps or plots:
        lats, lons, mean_lam = write_truth_maps(out, cfg, geo, env, models)
        if plots:
            make_plots(out, cfg, geo, rec, lats, lons, mean_lam, groups)

    summary = rec.summary(groups, len(vessels))
    write_json(out / "summary.json", summary)
    if verbose:
        print_summary(summary, cfg)
        print(f"\nOutput written to {out.resolve()}")
    return summary
