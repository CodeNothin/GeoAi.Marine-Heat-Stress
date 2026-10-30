"""CLI:
    python -m bycatch_model train  --data sim_output --out model_output
    python -m bycatch_model power  --out power_output            (simulates its own data)
    python -m bycatch_model forecast --data sim_output --out forecast_output
    python -m bycatch_model map1 --data sim_output --env real_env --out forecast_output   (MAP 1 workflow)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m bycatch_model")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="fit per-species models, score on the last year, write maps")
    t.add_argument("--data", required=True, help="folder in the logbook/env schema (simulated or real)")
    t.add_argument("--out", default="model_output")
    t.add_argument("--test-year", type=int)

    f = sub.add_parser("forecast", help="one-month-ahead species forecast from lagged heat/climate inputs")
    f.add_argument("--data", required=True, help="folder in the logbook/env schema with monthly grids")
    f.add_argument("--out", default="forecast_output")
    f.add_argument("--start-month", help="first month to forecast in the rolling test (default: Jan of the 2nd year)")
    f.add_argument("--bootstrap", type=int, default=10, help="refits for the upcoming month's uncertainty")

    m1 = sub.add_parser("map1", help="MAP 1: fronts + multi-buffer mesh + GLM/GAM/BRT/RF comparison + SHAP")
    m1.add_argument("--data", required=True, help="logbook folder (biomass data) in the logbook/env schema")
    m1.add_argument("--env", default="real_env", help="folder with sst_daily / chl_monthly / wind_monthly / bathy npz")
    m1.add_argument("--out", default="forecast_output")

    w = sub.add_parser("power", help="how many sets are needed? (learning curves)")
    w.add_argument("--data", help="existing simulated folder; omitted = simulate one")
    w.add_argument("--out", default="power_output")
    w.add_argument("--vessels", type=int, default=10)
    w.add_argument("--start", default="2023-01-01")
    w.add_argument("--end", default="2025-12-31")
    w.add_argument("--seed", type=int, default=42)
    w.add_argument("--sizes", default="200,500,1000,2000,4000,all")
    w.add_argument("--reps", type=int, default=3)
    a = p.parse_args(argv)

    if a.cmd == "train":
        from .train import train_and_evaluate
        train_and_evaluate(a.data, a.out, test_year=a.test_year)
        return 0

    if a.cmd == "forecast":
        from .forecast import run_forecast
        run_forecast(a.data, a.out, start_month=a.start_month, n_boot=a.bootstrap)
        return 0

    if a.cmd == "map1":
        from .map1 import run_map1
        run_map1(a.data, a.env, a.out)
        return 0

    from .power import power_analysis
    data = a.data
    if data is None:
        from purse_seine_sim import SimConfig, run_simulation
        data = str(Path(a.out) / "data")
        cfg = SimConfig.from_dict({"start": a.start, "end": a.end, "seed": a.seed, "write_track": False,
                                   "fleet": {"n_vessels": a.vessels}})
        print(f"Simulating {a.vessels} vessels {a.start} -> {a.end} into {data} ...")
        run_simulation(cfg, data, write_maps=True, plots=False, verbose=False)
    sizes = [s if s == "all" else int(s) for s in a.sizes.split(",")]
    power_analysis(data, a.out, sizes=sizes, reps=a.reps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
