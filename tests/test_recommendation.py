"""orchestrator/recommendation.py — hand-computable cases (no torch, no network).

    python3 -m unittest tests/test_recommendation.py      (from the repo root)
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "orchestrator"))
import recommendation as rec  # noqa: E402


def pt(batch, p90, rps, **extra):
    return {"batch": batch, "status": "ok", "predicted_p90_ms": p90, "predicted_median_ms": p90,
            "predicted_throughput_rps": rps, **extra}


# A: cheap, slow-ish. B: twice the price, faster and higher capacity.
CURVES = {
    "a.xlarge": [pt(1, 100, 10), pt(8, 200, 40), pt(16, 300, 50), pt(32, 900, 60)],
    "b.xlarge": [pt(1, 50, 20), pt(8, 80, 100), pt(16, 120, 130), pt(32, 200, 160)],
}
CANDS = [
    {"instance_type": "a.xlarge", "price_usd_per_hour": 1.0},
    {"instance_type": "b.xlarge", "price_usd_per_hour": 2.0},
]
SLO = {"max_latency_ms": 350, "min_throughput_rps": 30}


def row(result, name):
    return next(r for r in result["rows"] if r["instance_type"] == name)


class GridTest(unittest.TestCase):
    def test_grid_is_capped_by_user_batch_and_includes_it(self):
        self.assertEqual(rec.batch_grid(1), [1])
        self.assertEqual(rec.batch_grid(32), [1, 8, 16, 32])
        self.assertEqual(rec.batch_grid(20), [1, 8, 16, 20])
        self.assertEqual(rec.batch_grid(500), [1, 8, 16, 32, 64, 128, 500])
        with self.assertRaises(ValueError):
            rec.batch_grid(0)


class RecommendTest(unittest.TestCase):
    def test_capacity_operating_batch_and_cost(self):
        r = rec.recommend(CANDS, CURVES, SLO, None)
        a = row(r, "a.xlarge")
        # usable for A at 350 ms: b1, b8, b16 (b32 is 900 ms). capacity 50 -> takes up to 40 rps at 80%
        self.assertEqual((a["status"], a["capacity_rps"], a["capacity_batch"], a["max_demand_rps"]), ("ok", 50, 16, 40))
        # demand 30 -> smallest batch with rps >= 30 is b8 (40 rps, 200 ms)
        self.assertEqual((a["operating_batch"], a["latency_ms"]), (8, 200))
        self.assertAlmostEqual(a["utilization"], 30 / 50)
        self.assertEqual(a["verdict"], "edge")                     # 2 x 200 = 400 > 350
        self.assertEqual((a["cost_usd_per_hour"], a["cost_usd_per_month"]), (1.0, 720.0))
        self.assertNotIn("units", a)
        b = row(r, "b.xlarge")
        self.assertEqual((b["operating_batch"], b["latency_ms"], b["verdict"]), (8, 80, "pass"))
        self.assertEqual((r["status"], r["cheapest"]), ("ok", "a.xlarge"))

    def test_throughput_slo_is_a_per_instance_condition(self):
        r = rec.recommend(CANDS, CURVES, {"max_latency_ms": 350, "min_throughput_rps": 100}, None)
        a = row(r, "a.xlarge")
        self.assertEqual((a["status"], a["max_demand_rps"], a["capacity_batch"]), ("throughput_short", 40, 16))
        self.assertEqual(row(r, "b.xlarge")["status"], "ok")          # 160 x 0.8 = 128 >= 100
        self.assertEqual(r["cheapest"], "b.xlarge")

    def test_no_single_instance_is_enough(self):
        r = rec.recommend(CANDS, CURVES, {"max_latency_ms": 350, "min_throughput_rps": 200}, None)
        self.assertEqual((r["status"], r["cheapest"], r["budget_plan"]), ("throughput_short", None, None))

    def test_target_utilization(self):
        slo = {"max_latency_ms": 350, "min_throughput_rps": 50}
        self.assertEqual(row(rec.recommend(CANDS, CURVES, slo, None), "a.xlarge")["status"], "throughput_short")
        a = row(rec.recommend(CANDS, CURVES, slo, None, target_utilization=1.0), "a.xlarge")
        self.assertEqual((a["status"], a["operating_batch"]), ("ok", 16))
        with self.assertRaises(ValueError):
            rec.recommend(CANDS, CURVES, SLO, None, target_utilization=1.2)

    def test_utilization_never_exceeds_target(self):
        for demand in (1, 7, 30, 41, 99, 128, 333):
            r = rec.recommend(CANDS, CURVES, {"max_latency_ms": 350, "min_throughput_rps": demand}, None)
            for x in r["rows"]:
                if x["status"] == "ok":
                    self.assertLessEqual(x["utilization"], 0.8 + 1e-9)

    def test_max_batch_caps_capacity(self):
        r = rec.recommend(CANDS, CURVES, SLO, None, max_batch=1)
        self.assertEqual((row(r, "a.xlarge")["capacity_rps"], row(r, "a.xlarge")["status"]), (10, "throughput_short"))
        self.assertEqual(r["status"], "throughput_short")              # B: 20 x 0.8 = 16 < 30

    def test_slo_impossible_reports_fastest_point(self):
        r = rec.recommend(CANDS, CURVES, {"max_latency_ms": 60, "min_throughput_rps": 5}, None)
        a = row(r, "a.xlarge")
        self.assertEqual((a["status"], a["min_latency_ms"], a["min_latency_batch"]), ("slo_impossible", 100, 1))
        self.assertEqual(r["cheapest"], "b.xlarge")
        r = rec.recommend(CANDS, CURVES, {"max_latency_ms": 10, "min_throughput_rps": 5}, None)
        self.assertEqual((r["status"], r["cheapest"]), ("slo_impossible", None))

    def test_budget_upper_bound(self):
        r = rec.recommend(CANDS, CURVES, SLO, 1.5)
        self.assertTrue(row(r, "b.xlarge")["over_budget"])
        self.assertEqual(r["cheapest"], "a.xlarge")
        self.assertTrue(r["budget_plan"]["same_as_cheapest"])          # A is the only one in budget
        r = rec.recommend(CANDS, CURVES, SLO, 0.5)
        self.assertEqual((r["status"], r["cheapest"], r["budget_plan"]), ("over_budget", "a.xlarge", None))

    def test_budget_plan_is_the_fastest_instance_in_budget(self):
        bp = rec.recommend(CANDS, CURVES, SLO, 4.0)["budget_plan"]
        self.assertEqual((bp["instance_type"], bp["operating_batch"], bp["latency_ms"]), ("b.xlarge", 8, 80))
        self.assertFalse(bp["same_as_cheapest"])
        self.assertIsNone(rec.recommend(CANDS, CURVES, SLO, None)["budget_plan"])

    def test_missing_price_is_not_zero(self):
        cands = [{"instance_type": "a.xlarge", "price_usd_per_hour": None}, CANDS[1]]
        r = rec.recommend(cands, CURVES, SLO, None)
        self.assertEqual(row(r, "a.xlarge")["status"], "no_price")
        self.assertEqual(r["cheapest"], "b.xlarge")

    def test_failed_predictions_and_no_slo(self):
        curves = {"a.xlarge": [{"batch": 1, "status": "error", "error": "boom"}], "b.xlarge": CURVES["b.xlarge"]}
        r = rec.recommend(CANDS, curves, SLO, None)
        self.assertEqual((row(r, "a.xlarge")["status"], row(r, "a.xlarge")["reason"]), ("no_prediction", "boom"))
        self.assertEqual(rec.recommend(CANDS, CURVES, None, None)["status"], "no_slo")

    def test_similar_within_pct(self):
        cands = [{"instance_type": "a.xlarge", "price_usd_per_hour": 1.0},
                 {"instance_type": "c.xlarge", "price_usd_per_hour": 1.05},
                 {"instance_type": "b.xlarge", "price_usd_per_hour": 2.0}]
        curves = dict(CURVES, **{"c.xlarge": CURVES["a.xlarge"]})
        r = rec.recommend(cands, curves, SLO, None)
        self.assertEqual((r["cheapest"], r["similar"]), ("a.xlarge", ["c.xlarge"]))

    def test_rows_order(self):
        curves = dict(CURVES, **{"d.xlarge": [{"batch": 1, "status": "error"}]})
        cands = CANDS + [{"instance_type": "d.xlarge", "price_usd_per_hour": 0.5}]
        r = rec.recommend(cands, curves, {"max_latency_ms": 350, "min_throughput_rps": 100}, 1.5)
        # B ok but over budget, A throughput_short, D no prediction
        self.assertEqual([x["instance_type"] for x in r["rows"]], ["b.xlarge", "a.xlarge", "d.xlarge"])
        self.assertEqual(r["status"], "over_budget")


class LatencyLimitTest(unittest.TestCase):
    def test_per_token_times_generated_tokens(self):
        r = rec.resolve_latency_limit({"latency_per_token_ms": 50, "min_throughput_rps": 1}, 256)
        self.assertEqual((r["max_latency_ms"], r["basis"], r["tokens"]), (12800, "per_token", 256))

    def test_per_request_and_none(self):
        self.assertEqual(rec.resolve_latency_limit({"max_latency_ms": 300}, 256)["max_latency_ms"], 300)
        self.assertEqual(rec.resolve_latency_limit({"min_throughput_rps": 1}, 256)["basis"], "none")
        self.assertIsNone(rec.resolve_latency_limit(None, None)["max_latency_ms"])
        # per-token limit on a workload with no generated tokens reads per request
        self.assertEqual(rec.resolve_latency_limit({"latency_per_token_ms": 80}, None)["basis"], "per_request")

    def test_presets_are_the_documented_values(self):
        self.assertEqual(rec.LATENCY_PRESETS_MS_PER_TOKEN, {"realtime": 50.0, "standard": 100.0})
        self.assertEqual((rec.TARGET_UTILIZATION, rec.TARGET_UTILIZATION_MAX), (0.8, 0.9))

    def test_no_latency_limit_uses_full_curve(self):
        r = rec.recommend(CANDS, CURVES, {"max_latency_ms": None, "min_throughput_rps": 45}, None)
        a = row(r, "a.xlarge")
        # b32 (900 ms, 60 rps) is usable now: capacity 60 x 0.8 = 48 >= 45; b16 (50 rps) covers 45
        self.assertEqual((a["status"], a["capacity_rps"], a["operating_batch"], a["verdict"]), ("ok", 60, 16, "pass"))
        self.assertIsNone(r["policy"]["max_latency_ms"])


if __name__ == "__main__":
    unittest.main()
