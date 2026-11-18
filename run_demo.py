"""Pipeline check with a SIMULATED model. The numbers are not results; they show the harness works.
Run:  python run_demo.py   (from this folder, after build_regions.py)

R4 is skipped: it needs a trained router (router.json), which requires real-model outputs.
Run train_router.py after a real R1 run to enable it."""
from collections import defaultdict

from core import load_regions, make_scenarios
from faith import score_row
from ladder import RUNGS
from lara_assessment import CLOUD_REGION_LARA
from llm import SimulatedLLM

regions = load_regions("data/regions.csv")
scen = make_scenarios(regions, n=100)

print(CLOUD_REGION_LARA.report())
print()
print(f"{len(regions)} regions, {len(scen)} scenarios "
      f"({sum(s.difficulty == 'hard' for s in scen)} hard, "
      f"{sum(s.binding for s in scen)} binding)")
chance = sum(len(s.acceptable) / len(s.feasible) for s in scen) / len(scen)
print(f"chance level (random feasible region): accept = {chance:.2f}\n")

# Simulated accuracy per model call mode (test double only - not a result)
ACC = {"single": 0.45, "rag": 0.60, "agent": 0.70, "synth": 0.72, "explain": 0.72, "choice": 0.60}


class ModeLLM(SimulatedLLM):
    def complete(self, prompt, scenario=None, mode="rag", **kw):
        self.acc = ACC.get(mode, self.acc)
        return super().complete(prompt, scenario=scenario, mode=mode, **{k: v for k, v in kw.items() if k == "fixed_region"})


SKIP = {"R4"}  # needs a trained router; cannot run in demo mode

stats = {}
for name, fn in RUNGS.items():
    if name.split()[0] in SKIP:
        print(f"  skipping {name} (needs trained router - run train_router.py first)")
        continue
    llm = ModeLLM(regions, seed=1)
    m = defaultdict(list)
    for s in scen:
        r = fn(llm, regions, s)
        sc = score_row(r, s, regions)
        m["exact"].append(r.region == s.best)
        m["accept"].append(r.region in s.acceptable)
        m["violate"].append(r.region not in set(s.feasible))
        for k in ("claim_support", "decisive_ok", "cite_ok", "sound"):
            if sc[k] is not None:
                m[k].append(sc[k])
        if r.flagged is not None:
            m["flagged"].append(r.flagged)
        m["tokens"].append(r.tok_in + r.tok_out)
        m["calls"].append(r.calls)
        m["escalated"].append(r.escalated)
    stats[name] = m

avg = lambda v: sum(v) / len(v) if v else float("nan")
print()
print(f"{'rung':32} {'exact':>6} {'accept':>7} {'SOUND':>6} {'violate':>8} {'claimOK':>8} {'F2':>5} {'citeOK':>7} "
      f"{'flag':>5} {'tokens':>7} {'calls':>6} {'escal':>6}")
print("-" * 112)
for n, m in stats.items():
    print(f"{n:32} {avg(m['exact']):6.2f} {avg(m['accept']):7.2f} {avg(m['sound']):6.2f} {avg(m['violate']):8.2f} "
          f"{avg(m['claim_support']):8.2f} {avg(m['decisive_ok']):5.2f} {avg(m['cite_ok']):7.2f} {avg(m['flagged']):5.2f} "
          f"{avg(m['tokens']):7.0f} {avg(m['calls']):6.1f} {avg(m['escalated']):6.2f}")

print()
print("NOTE: These numbers come from a simulated test double that peeks at the ground truth.")
print("      They confirm the pipeline runs correctly, but are NOT model results.")
print()
print("Next steps:")
print("  1. Run a real model:  python run_real.py --model llama3.2:3b --n 20 --no-json-mode")
print("  2. See statistics:    python stats.py results_real.jsonl")
print("  3. Train the router:  python train_router.py results_real.jsonl")
print("  4. Build annotation:  python make_annotation.py results_real.jsonl 100")
