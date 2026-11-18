"""LLM interface. Real models implement `complete(prompt, **kw) -> (text, usage)`.

SimulatedLLM is a TEST DOUBLE that peeks at the ground truth to produce answers with a chosen error rate.
It exists only to check that the pipeline and metrics work end to end. Nothing produced with it is a result.
"""
import json
import random

from core import USER_SITES


class SimulatedLLM:
    def __init__(self, regions, accuracy=0.6, claim_error=0.2, seed=0):
        self.regions, self.acc, self.cerr = regions, accuracy, claim_error
        self.rng = random.Random(seed)

    def complete(self, prompt, scenario=None, **kw):
        s, rng = scenario, self.rng
        right = rng.random() < self.acc
        if kw.get("fixed_region"):
            pick = kw["fixed_region"]
            right = pick in s.acceptable
        elif right:
            pick = rng.choice(s.acceptable)
        else:
            pool = s.feasible if rng.random() < 0.7 else list(self.regions)  # 30% ignore the constraints
            pick = rng.choice(pool)
        r = self.regions[pick]
        claims = [
            {"region": pick, "field": "grid_gco2_kwh", "value": r["grid"]},
            {"region": pick, "field": "country", "value": r["country"]},
        ]
        for c in claims:
            if c["field"] == "grid_gco2_kwh" and rng.random() < self.cerr:
                c["value"] = round(c["value"] * rng.choice([0.6, 1.4]), 1)
        conf = min(0.99, max(0.2, rng.gauss(0.85 if right else 0.55, 0.1)))
        factor = "carbon" if s.w_carbon >= s.w_latency else "latency"
        out = {"region": pick, "confidence": round(conf, 2), "decisive_factor": factor,
               "constraints_met": pick in set(s.feasible), "cites": [pick],
               "explanation": f"Chose {pick} because of the stated {factor} priority and the constraints.", "claims": claims}
        text = json.dumps(out)
        usage = {"in": int(len(prompt.split()) * 1.3), "out": int(len(text.split()) * 1.3)}
        return text, usage
