"""Run with:  python -m unittest discover -s tests -v"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bycatch_model.data import load_dataset, load_grid
from bycatch_model.ecocast import auc, ecocast_score, spearman
from bycatch_model.forecast import EnvArchive, kernel_rate, run_forecast
from bycatch_model.gam import GAM, pspline_basis
from bycatch_model.train import train_and_evaluate
from purse_seine_sim import SimConfig, run_simulation


class GamTest(unittest.TestCase):
    def test_basis_partition_of_unity(self):
        B = pspline_basis(np.linspace(-2, 5, 200), -2, 5)
        self.assertEqual(B.shape, (200, 8))
        self.assertTrue(np.allclose(B.sum(axis=1), 1.0))

    def test_recovers_nonlinear_logistic_curve(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(-3, 3, 5000)
        true_p = 1 / (1 + np.exp(-(2 * np.sin(x) - 0.5)))
        m = GAM(["x"], family="binomial").fit({"x": x}, rng.random(x.size) < true_p)
        g = np.linspace(-2.5, 2.5, 11)
        self.assertLess(np.max(np.abs(m.predict({"x": g}) - 1 / (1 + np.exp(-(2 * np.sin(g) - 0.5))))), 0.08)

    def test_separable_data_does_not_diverge(self):
        x = np.linspace(0, 1, 400)
        m = GAM(["x"], family="binomial").fit({"x": x}, x > 0.8)
        p = m.predict({"x": np.array([0.1, 0.95])})
        self.assertTrue(np.all(np.isfinite(p)))
        self.assertLess(p[0], 0.1)
        self.assertGreater(p[1], 0.9)

    def test_gaussian_and_roundtrip(self):
        rng = np.random.default_rng(1)
        x = rng.uniform(0, 10, 2000)
        y = np.log1p(x) + rng.normal(0, 0.1, x.size)
        m = GAM(["x"], family="gaussian").fit({"x": x}, y)
        m2 = GAM.from_dict(m.to_dict())
        g = {"x": np.array([1.0, 5.0, 9.0])}
        self.assertTrue(np.allclose(m.predict(g), np.log1p(g["x"]), atol=0.05))
        self.assertTrue(np.allclose(m.predict(g), m2.predict(g)))


class MetricTest(unittest.TestCase):
    def test_auc(self):
        self.assertEqual(auc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1], bool)), 1.0)
        self.assertEqual(auc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([0, 1, 0, 1], bool)), 0.5)

    def test_spearman(self):
        self.assertAlmostEqual(spearman(np.arange(10.0), np.arange(10.0) ** 3), 1.0)

    def test_ecocast_signs(self):
        layers = {"revenue": np.array([0.0, 1.0, 1.0]), "CARC_SPP": np.array([0.0, 0.0, 1.0])}
        score, used = ecocast_score(layers, {"revenue": 0.5, "CARC_SPP": 0.5})
        self.assertEqual(used, ["revenue", "CARC_SPP"])
        self.assertEqual(int(np.argmax(score)), 1)     # good fishing, no bycatch
        self.assertLess(score[2], score[1])


class ForecastTest(unittest.TestCase):
    def test_kernel_rate_is_local_and_shrunk(self):
        lat = np.array([7.0, 7.0, 9.0, 9.0])
        lon = np.array([100.0, 100.0, 100.0, 100.0])
        y = np.array([1, 1, 0, 0])
        p = kernel_rate(lat, lon, y, np.array([7.0, 9.0, 11.0]), np.array([100.0, 100.0, 100.0]))
        self.assertGreater(p[0], 0.6)
        self.assertLess(p[1], 0.4)
        self.assertAlmostEqual(p[2], 0.5, places=2)     # far from every set: overall rate

    def test_rolling_forecast_on_simulated_logbooks(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, out = Path(tmp) / "data", Path(tmp) / "fc"
            cfg = SimConfig.from_dict({"start": "2024-01-01", "end": "2025-03-31", "seed": 5, "write_track": False,
                                       "fleet": {"n_vessels": 4}})
            run_simulation(cfg, data, write_maps=True, verbose=False)
            archive = EnvArchive(data)
            self.assertFalse(archive.has_lags("2024-03"))
            self.assertTrue(archive.has_lags("2024-04"))
            summary = run_forecast(data, out, start_month="2025-01", n_boot=2, plots=False, verbose=False)
            fc = {r["species_code"]: r for r in summary if r["method"] == "forecast"}
            self.assertGreater(fc["RAST_BRA"]["auc"], 0.6)
            self.assertGreater(fc["RAST_BRA"]["map_rho"], 0.5)
            self.assertTrue((out / "maps" / "forecast_2025-02.npz").exists())
            upcoming = np.load(out / "maps" / "forecast_2025-04.npz")      # the month after the last grid
            self.assertIn("p_RAST_BRA", upcoming.files)
            self.assertIn("sd_RAST_BRA", upcoming.files)
            self.assertTrue((out / "response_curves.json").exists())


class EndToEndTest(unittest.TestCase):
    def test_train_on_simulated_logbooks(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, out = Path(tmp) / "data", Path(tmp) / "model"
            cfg = SimConfig.from_dict({"start": "2024-01-01", "end": "2025-03-31", "seed": 3, "write_track": False,
                                       "fleet": {"n_vessels": 3}})
            run_simulation(cfg, data, write_maps=True, verbose=False)
            ds = load_dataset(data)
            self.assertEqual(len(ds), len(ds.presence["RAST_BRA"]))
            grid = load_grid(next((data / "truth" / "maps").glob("truth_2025-*.npz")))
            self.assertEqual(int(grid.mask.sum()), grid.features["sst"].size)

            rows, fits = train_and_evaluate(data, out, plot_months=(), verbose=False)
            by_code = {r["species_code"]: r for r in rows}
            self.assertGreater(by_code["RAST_BRA"]["test_auc"], 0.65)
            self.assertGreater(by_code["RAST_BRA"]["map_spearman"], 0.5)
            self.assertTrue((out / "metrics.csv").exists())
            self.assertTrue(list((out / "maps").glob("pred_2025-*.npz")))

    def test_single_year_holds_out_last_months(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, out = Path(tmp) / "data", Path(tmp) / "model"
            cfg = SimConfig.from_dict({"start": "2024-01-01", "end": "2024-08-31", "seed": 4, "write_track": False,
                                       "fleet": {"n_vessels": 3}})
            run_simulation(cfg, data, write_maps=True, verbose=False)
            train_and_evaluate(data, out, plot_months=(), verbose=False)
            holdout = json.loads((out / "holdout.json").read_text())
            self.assertEqual(holdout["test_months"], ["2024-06", "2024-07", "2024-08"])
            self.assertEqual(len(list((out / "maps").glob("pred_2024-*.npz"))), 8)     # every month is mapped


if __name__ == "__main__":
    unittest.main()
