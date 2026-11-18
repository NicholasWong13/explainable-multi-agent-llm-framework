"""Run:  python -m unittest discover -s tests   (from the practical folder)"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from core import decide, EU, load_regions, make_scenarios, rtt_proxy_ms, USER_SITES
from faith import (citation_valid, claim_support, constraint_report_correct, decisive_factor_choice,
                   decisive_factor_correct, neutralise, sound)
from ladder import Result, _parse
import router

R = load_regions("data/regions.csv")


class Truth(unittest.TestCase):
    def test_residency_is_hard(self):
        # Every feasible region in an EU-residency scenario must be in the full EU country set from core.EU,
        # not a hardcoded subset (bug fix: the old test only checked 9 of 27 EU members).
        for s in make_scenarios(R, 60):
            if s.residency == "EU":
                self.assertTrue(all(R[x]["country"] in EU for x in s.feasible))

    def test_best_is_acceptable_and_feasible(self):
        for s in make_scenarios(R, 100):
            self.assertIn(s.best, s.acceptable)
            self.assertIn(s.best, s.feasible)

    def test_latency_cap_respected(self):
        for s in make_scenarios(R, 100):
            if s.max_rtt_ms is not None:
                self.assertTrue(all(rtt_proxy_ms(USER_SITES[s.city], R[x]) <= s.max_rtt_ms for x in s.feasible))

    def test_deterministic(self):
        a = [s.best for s in make_scenarios(R, 50)]
        b = [s.best for s in make_scenarios(R, 50)]
        self.assertEqual(a, b)

    def test_carbon_only_picks_cleanest(self):
        feas, best, _, _ = decide(R, "London", "ANY", None, 1.0, 0.0)
        self.assertEqual(best, min(R, key=lambda k: R[k]["grid"]))


class Checks(unittest.TestCase):
    def res(self, region, **ex):
        return Result(region, ex, parse_ok=True)

    def test_claim_support(self):
        r = self.res("europe-north2", claims=[{"region": "europe-north2", "field": "country", "value": "SE"},
                                              {"region": "europe-north2", "field": "grid_gco2_kwh", "value": 999}])
        self.assertEqual(claim_support(r, R), 0.5)

    def test_citation_valid(self):
        self.assertTrue(citation_valid(self.res("us-east1", cites=["us-east1"], claims=[]), R))
        self.assertFalse(citation_valid(self.res("us-east1", cites=["nope"], claims=[]), R))
        self.assertFalse(citation_valid(self.res("us-east1", cites=["us-west1"], claims=[]), R))
        self.assertIsNone(citation_valid(self.res("us-east1"), R))

    def test_neutralise_recomputes_truth(self):
        s = next(x for x in make_scenarios(R, 100) if x.residency != "ANY")
        n = neutralise(s, "residency", R)
        self.assertEqual(n.residency, "ANY")
        self.assertGreaterEqual(len(n.feasible), len(s.feasible))

    def test_decisive_factor(self):
        s = next(x for x in make_scenarios(R, 100) if x.residency != "ANY" and x.binding)
        r = self.res(s.best, decisive_factor="residency")
        self.assertTrue(decisive_factor_correct(r, s, R))

    def test_constraint_report(self):
        s = make_scenarios(R, 10)[0]
        self.assertTrue(constraint_report_correct(self.res(s.best, constraints_met=True), s))
        self.assertIsNone(constraint_report_correct(self.res(s.best), s))

    def test_parse_takes_first_object(self):
        reg, d, ok = _parse('{"region": "a"} {"region": "b"}')
        self.assertTrue(ok)
        self.assertEqual(reg, "a")
        self.assertFalse(_parse("no json here")[2])


class Router(unittest.TestCase):
    def test_learns_a_simple_rule(self):
        X = [[c / 10, 1, 1, 1, .5, .5, 0, 2, 0] for c in range(10)] * 6
        y = [1 if x[0] < 0.5 else 0 for x in X]
        m = router.train(X, y)
        self.assertGreater(router.p_wrong(m, X[0]), router.p_wrong(m, X[9]))


class Explainer(unittest.TestCase):
    def test_explainer_cannot_change_the_decision(self):
        from ladder import r3_selective, r5_explained
        from llm import SimulatedLLM

        class M(SimulatedLLM):
            def complete(self, prompt, scenario=None, mode="rag", **kw):
                return super().complete(prompt, scenario=scenario, mode=mode, **kw)

        scen = make_scenarios(R, n=30)
        for s in scen:
            a = r3_selective(M(R, seed=3), R, s)
            b = r5_explained(M(R, seed=3), R, s)
            self.assertEqual(a.region, b.region)
            self.assertEqual(b.calls, a.calls + 1)
            if b.region:
                self.assertEqual(b.explanation["claims"][0]["region"], b.region)


class FaithV2(unittest.TestCase):
    def test_f2_choice_rejects_far_region_for_latency(self):
        # The pilot case: users in Sao Paulo, model picks Zurich "because of latency". F2 must fail.
        s = next(x for x in make_scenarios(R, 100) if x.city == "Sao Paulo" and "europe-west6" in x.feasible)
        r = Result("europe-west6", {"decisive_factor": "latency"}, parse_ok=True)
        self.assertFalse(decisive_factor_choice(r, s, R))
        self.assertFalse(decisive_factor_correct(r, s, R))

    def test_f2_choice_accepts_closest_region_for_latency(self):
        s = make_scenarios(R, 10)[0]
        closest = min(s.feasible, key=lambda k: rtt_proxy_ms(USER_SITES[s.city], R[k]))
        self.assertTrue(decisive_factor_choice(Result(closest, {"decisive_factor": "latency"}), s, R))

    def test_sound_needs_acceptable_decision(self):
        self.assertFalse(sound(False, 1.0, True, True, True))
        self.assertTrue(sound(True, 1.0, True, None, True))
        self.assertFalse(sound(True, 0.5, True, True, True))

    def test_neutralise_keeps_weight_ratio(self):
        s = make_scenarios(R, 10)[0]
        n = neutralise(s, "price", R)
        self.assertAlmostEqual(n.w_carbon / (n.w_carbon + n.w_latency), s.w_carbon / (s.w_carbon + s.w_latency))
        n = neutralise(s, "carbon", R)
        self.assertEqual(n.w_carbon, 0.0)
        self.assertAlmostEqual(n.w_latency + n.w_price, 1.0)


class Lara(unittest.TestCase):
    """Checks against arXiv:2605.16297: higher scores = more human involvement, L1 is [1.0, 2.0]."""

    def test_low_score_is_l1(self):
        from lara_assessment import LaraScore
        s = LaraScore(1, 1, 1, 1, 1)
        self.assertAlmostEqual(s.weighted_score, 1.0)
        self.assertEqual(s.level, "L1")

    def test_high_score_is_l4(self):
        from lara_assessment import LaraScore
        self.assertEqual(LaraScore(5, 5, 5, 5, 5).level, "L4")

    def test_weights(self):
        from lara_assessment import LaraScore
        self.assertAlmostEqual(LaraScore(1, 1, 1, 5, 1).weighted_score, (4 + 7.5) / 5.5)

    def test_floor_rule(self):
        from lara_assessment import LaraScore
        s = LaraScore(1, 1, 1, 4, 1)  # weighted mean 1.82 -> L1, but D4 = 4
        self.assertEqual(s.mean_level, "L1")
        self.assertEqual(s.level, "L3")

    def test_boundary_upgrade(self):
        from lara_assessment import LaraScore
        s = LaraScore(2, 2, 2, 2, 1.9)  # 1.98, within 0.15 of the L1/L2 threshold
        self.assertEqual(s.level, "L2")

    def test_cloud_region_level(self):
        from lara_assessment import CLOUD_REGION_LARA
        self.assertEqual(CLOUD_REGION_LARA.mean_level, "L2")
        self.assertEqual(CLOUD_REGION_LARA.level, "L3")


class DualProcess(unittest.TestCase):
    class FakeJev:
        def __init__(self, pick, p):
            self.pick, self.p = pick, p

        def choice(self, state, candidates, scenario=None):
            return (self.pick if self.pick in candidates else None), self.p, {"in": 10, "out": 2}

    def llm(self):
        from llm import SimulatedLLM
        return SimulatedLLM(R, accuracy=1.0, claim_error=0.0, seed=1)

    def test_confident_system1_is_kept_and_explained(self):
        from jev_router import r6_dual_process
        s = make_scenarios(R, 10)[0]
        r = r6_dual_process(self.llm(), R, s, tau=0.65, jev=self.FakeJev(s.best, 0.9))
        self.assertFalse(r.escalated)
        self.assertEqual(r.region, s.best)
        self.assertEqual(r.calls, 2)  # System 1 + explanation agent
        self.assertIn("decisive_factor", r.explanation)
        self.assertIsNotNone(r.flagged)

    def test_low_confidence_escalates(self):
        from jev_router import r6_dual_process
        s = make_scenarios(R, 10)[0]
        r = r6_dual_process(self.llm(), R, s, tau=0.65, jev=self.FakeJev(s.best, 0.3))
        self.assertTrue(r.escalated)
        self.assertEqual(r.calls, 1 + 4 + 1)  # System 1 + R2 (3 agents + synthesiser) + explainer

    def test_infeasible_pick_is_not_a_candidate(self):
        from jev_router import r6_dual_process
        s = next(x for x in make_scenarios(R, 30) if len(x.feasible) < len(R))
        bad = next(k for k in R if k not in set(s.feasible))
        r = r6_dual_process(self.llm(), R, s, tau=0.65, jev=self.FakeJev(bad, 0.99))
        self.assertTrue(r.escalated)

    def test_egress_flags_wrong_claims(self):
        from jev_router import egress_check
        s = make_scenarios(R, 10)[0]
        r = Result(s.best, {"decisive_factor": "carbon", "cites": [s.best], "constraints_met": True,
                            "claims": [{"region": s.best, "field": "grid_gco2_kwh", "value": 9999}]})
        flagged, d = egress_check(r, s, R)
        self.assertTrue(flagged)
        self.assertEqual(d["f1"], 0.0)


class Energy(unittest.TestCase):
    def test_meter_without_codecarbon_or_with_it_reports_method(self):
        from energy import codecarbon_meter
        with codecarbon_meter() as m:
            pass
        self.assertIn("cpu_method", m)
        self.assertIn("cpu_measured", m)


if __name__ == "__main__":
    unittest.main()
