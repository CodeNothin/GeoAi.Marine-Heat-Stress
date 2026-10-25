"""CLI:
    python -m bycatch_model train  --data sim_output --out model_output
    python -m bycatch_model power  --out power_output            (simulates its own data)
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
