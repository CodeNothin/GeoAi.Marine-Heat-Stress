"""Run with:  python -m unittest discover -s tests -v"""
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from purse_seine_sim import SimConfig, run_simulation
from purse_seine_sim.environment import Environment, GriddedSST, moon_illumination, to_unix_days
from purse_seine_sim.geo import Geography
from purse_seine_sim.run import checklist_codes
from purse_seine_sim.schema import validate_dir


def small_config(**fleet):
    cfg = SimConfig.from_dict({"start": "2024-02-01", "end": "2024-04-30", "seed": 7,
                               "fleet": {"n_vessels": 2, **fleet}})
    return cfg


def read(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


class SimulationRunTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name) / "run"
        cls.cfg = small_config()
        cls.summary = run_simulation(cls.cfg, cls.out, write_maps=False, verbose=False)
        cls.sets = read(cls.out / "logbook/sets.csv")
        cls.catch = read(cls.out / "logbook/catch.csv")
        cls.env = read(cls.out / "env/env_at_sets.csv")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_output_is_valid(self):
        self.assertEqual(validate_dir(self.out, checklist_codes()), [])

    def test_produces_sets(self):
        self.assertGreater(len(self.sets), 50)
        self.assertEqual(self.summary["n_sets"], len(self.sets))

    def test_every_set_reports_every_checklist_species(self):
        self.assertEqual(len(self.catch), len(self.sets) * len(checklist_codes()))

    def test_sets_are_legal(self):
        reg = self.cfg.regulation
        for e in self.env:
            self.assertGreaterEqual(float(e["dist_coast_km"]), reg.min_dist_coast_km)
        closure = reg.closures[0]
        for s in self.sets:
            lat, lon = float(s["start_lat"]), float(s["start_lon"])
            md = s["start_time_utc"][5:10]
            inside = (closure["lat_min"] <= lat <= closure["lat_max"]) and (closure["lon_min"] <= lon <= closure["lon_max"])
            active = closure["start_mmdd"] <= md <= closure["end_mmdd"]
            self.assertFalse(inside and active, f"set {s['event_id']} inside active closure")

    def test_water_hauls_have_no_catch(self):
        for s in self.sets:
            if s["water_haul"] == "1":
                self.assertEqual(float(s["total_catch_kg"]), 0.0)

    def test_deterministic(self):
        out2 = Path(self.tmp.name) / "run2"
        run_simulation(small_config(), out2, write_maps=False, verbose=False)
        self.assertEqual((self.out / "logbook/catch.csv").read_bytes(), (out2 / "logbook/catch.csv").read_bytes())


class BehaviourTest(unittest.TestCase):
    def test_presence_follows_true_density(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = small_config(n_vessels=3)
            cfg.write_track = False
            run_simulation(cfg, tmp, write_maps=False, verbose=False)
            truth = {r["event_id"]: float(r["lambda_THUN_TON"]) for r in read(Path(tmp) / "truth/truth_sets.csv")}
            rows = [r for r in read(Path(tmp) / "logbook/catch.csv") if r["species_code"] == "THUN_TON"]
            lam_pos = [truth[r["event_id"]] for r in rows if float(r["catch_kg"]) > 0]
            lam_neg = [truth[r["event_id"]] for r in rows if float(r["catch_kg"]) == 0]
            self.assertGreater(np.mean(lam_pos), 1.3 * np.mean(lam_neg))

    def test_bycatch_avoidance_reduces_shark_bycatch(self):
        rates = []
        for avoidance in (0.0, 4.0):
            with tempfile.TemporaryDirectory() as tmp:
                cfg = small_config(n_vessels=3, bycatch_avoidance=avoidance)
                cfg.write_track = False
                s = run_simulation(cfg, tmp, write_maps=False, verbose=False)
                rates.append(next(r["positive_rate"] for r in s["species"] if r["species_code"] == "CARC_SPP"))
        self.assertLess(rates[1], rates[0])


class EnvironmentTest(unittest.TestCase):
    def test_moon_phases(self):
        self.assertGreater(moon_illumination(to_unix_days("2024-01-25T17:54")), 0.99)
        self.assertLess(moon_illumination(to_unix_days("2024-01-11T11:57")), 0.01)

    def test_hot_season_warmer_than_monsoon(self):
        cfg = SimConfig()
        geo = Geography(cfg.region, np.random.default_rng(0))
        env = Environment(cfg, geo, np.random.default_rng(1))
        lat, lon = np.full(50, 8.0), np.linspace(100.8, 101.8, 50)
        jan = env.sample(lat, lon, to_unix_days("2025-01-15")).get("sst").mean()
        may = env.sample(lat, lon, to_unix_days("2025-05-01")).get("sst").mean()
        self.assertGreater(may, jan + 1.0)

    def test_sst_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sst.npz"
            time = np.arange(np.datetime64("2024-01-01"), np.datetime64("2024-01-11"))
            lat, lon = np.arange(6.0, 12.01, 0.25), np.arange(99.0, 102.51, 0.25)
            np.savez(path, time=time, lat=lat, lon=lon, sst=np.full((time.size, lat.size, lon.size), 31.5))
            cfg = SimConfig.from_dict({"env": {"sst_override_npz": str(path)}})
            geo = Geography(cfg.region, np.random.default_rng(0))
            env = Environment(cfg, geo, np.random.default_rng(1))
            inside = env.sample([8.0], [101.0], to_unix_days("2024-01-05"))["sst"][0]
            outside = env.sample([8.0], [101.0], to_unix_days("2024-06-05"))["sst"][0]
            self.assertAlmostEqual(inside, 31.5, places=6)
            self.assertNotAlmostEqual(outside, 31.5, places=2)

    def test_unknown_config_key_rejected(self):
        with self.assertRaises(KeyError):
            SimConfig.from_dict({"fleet": {"n_vesels": 3}})


if __name__ == "__main__":
    unittest.main()
