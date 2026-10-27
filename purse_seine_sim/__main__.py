"""CLI: python -m purse_seine_sim --out sim_output --plots"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import schema
from .config import SimConfig
from .run import checklist_codes, run_simulation
from .species import checklist


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m purse_seine_sim",
                                description="Simulate purse-seine logbooks, bycatch and tracks (southern Gulf of Thailand).")
    p.add_argument("--out", default="sim_output", help="output folder (default: sim_output)")
    p.add_argument("--config", help="JSON file overriding any SimConfig field")
    p.add_argument("--vessels", type=int, help="number of vessels")
    p.add_argument("--start", help="first day, YYYY-MM-DD")
    p.add_argument("--end", help="last day, YYYY-MM-DD")
    p.add_argument("--seed", type=int)
    p.add_argument("--ping-min", type=float, help="track ping interval in minutes")
    p.add_argument("--no-track", action="store_true", help="skip track.csv (much smaller output)")
    p.add_argument("--no-maps", action="store_true", help="skip monthly truth maps")
    p.add_argument("--plots", action="store_true", help="write quick-look PNGs (needs matplotlib)")
    p.add_argument("--sst-npz", help="real daily SST cube (from fetch_env_data.py) instead of the analytic SST")
    p.add_argument("--chl-npz", help="real CHL-a cube instead of the analytic CHL-a")
    p.add_argument("--wind-npz", help="real monthly wind cube (ERA5) instead of the analytic wind level")
    p.add_argument("--bycatch-avoidance", type=float, help="skipper avoids ETP hotspots (0 = status quo)")
    p.add_argument("--write-spec", metavar="PATH", help="write the data specification markdown and exit")
    p.add_argument("--validate", metavar="DIR", help="validate a folder of real or simulated data and exit")
    a = p.parse_args(argv)

    if a.write_spec:
        Path(a.write_spec).write_text(schema.spec_markdown(checklist()), encoding="utf-8")
        print(f"Wrote {a.write_spec}")
        return 0
    if a.validate:
        errors = schema.validate_dir(a.validate, checklist_codes())
        for e in errors:
            print(e)
        print("valid" if not errors else f"{len(errors)} problem(s)")
        return 0 if not errors else 1

    cfg = SimConfig.from_json(a.config) if a.config else SimConfig()
    if a.vessels is not None:
        cfg.fleet.n_vessels = a.vessels
    if a.start:
        cfg.start = a.start
    if a.end:
        cfg.end = a.end
    if a.seed is not None:
        cfg.seed = a.seed
    if a.ping_min is not None:
        cfg.track_ping_min = a.ping_min
    if a.no_track:
        cfg.write_track = False
    if a.sst_npz:
        cfg.env.sst_override_npz = a.sst_npz
    if a.chl_npz:
        cfg.env.chl_override_npz = a.chl_npz
    if a.wind_npz:
        cfg.env.wind_override_npz = a.wind_npz
    if a.bycatch_avoidance is not None:
        cfg.fleet.bycatch_avoidance = a.bycatch_avoidance
    run_simulation(cfg, a.out, write_maps=not a.no_maps, plots=a.plots)
    return 0


if __name__ == "__main__":
    sys.exit(main())
