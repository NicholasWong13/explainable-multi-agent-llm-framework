"""Statistics for the ladder results: confidence intervals and paired tests.
    python stats.py results_a.jsonl [results_b.jsonl ...]
Reads the per-scenario rows written by run_real.py. Rows are paired by (model, run, scenario).
  - mean and 95% bootstrap confidence interval for each rung and metric
  - exact McNemar test between every pair of rungs on 'accept' and 'exact' (paired binary outcomes)
  - Wilcoxon signed-rank test on tokens per scenario (if scipy is installed), plus a paired bootstrap interval
  - effect size for the paired differences (difference in proportions; Cohen's dz for tokens)
No scipy is needed for the main results."""
import json
import math
import random
import sys
from collections import defaultdict
from itertools import combinations

rows = []
for p in sys.argv[1:] or ["results_real.jsonl"]:
    rows += [json.loads(l) for l in open(p, encoding="utf-8")]
if not rows:
    sys.exit("no rows")
rng = random.Random(0)
SHOW = ["accept", "exact", "sound", "claim_support", "decisive_ok", "decisive_ref", "decisive_choice", "cite_ok",
        "ctr_ok", "f3", "flagged"]


def boot_ci(v, B=2000):
    if not v:
        return float("nan"), float("nan"), float("nan")
    m = sum(v) / len(v)
    bs = sorted(sum(rng.choice(v) for _ in v) / len(v) for _ in range(B))
    return m, bs[int(0.025 * B)], bs[int(0.975 * B) - 1]


def mcnemar_exact(b, c):
    """Two-sided exact McNemar from the discordant counts: b = A right and B wrong, c = A wrong and B right."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


by = defaultdict(dict)  # (model) -> rung -> {(run, scenario): row}
for r in rows:
    by[r["model"]].setdefault(r["rung"], {})[(r["run"], r["scenario"])] = r

for model, rungs in by.items():
    print(f"\n=== {model} ===")
    print("metric: mean [95% bootstrap CI] n   (SOUND = acceptable decision AND all explanation checks pass;\n"
          "         F1 claim_support alone is a copying check and must not be read as faithfulness)")
    for rung, d in rungs.items():
        print(f"\n{rung}  (n={len(d)})")
        for m in SHOW:
            v = [float(x[m]) for x in d.values() if x.get(m) is not None]
            if not v:
                continue
            mu, lo, hi = boot_ci(v)
            print(f"   {m:16} {mu:5.2f} [{lo:4.2f},{hi:4.2f}] n={len(v)}")
    allrows = [x for d in rungs.values() for x in d.values()]
    seen = {}
    for x in allrows:
        if x.get("n_feasible"):
            seen[(x["run"], x["scenario"])] = x["n_acceptable"] / x["n_feasible"]
    if seen:
        print(f"\nchance level (random feasible region) on these scenarios: accept = {sum(seen.values()) / len(seen):.2f}"
              f"  (n={len(seen)}). A rung is only useful if it is clearly above this.")
    else:
        print("\n(no n_feasible/n_acceptable fields: run rescore.py on old result files to get the chance level)")
    print("\npaired comparisons (same scenarios only)")
    for (ra, da), (rb, db) in combinations(rungs.items(), 2):
        keys = sorted(set(da) & set(db))
        if len(keys) < 5:
            continue
        for m in ("accept", "exact", "sound"):
            if any(da[k].get(m) is None or db[k].get(m) is None for k in keys):
                continue
            b = sum(1 for k in keys if da[k][m] and not db[k][m])
            c = sum(1 for k in keys if not da[k][m] and db[k][m])
            diff = (b - c) / len(keys)
            print(f"  {m:6} {ra[:2]} vs {rb[:2]}: {ra[:2]} better on {b}, {rb[:2]} better on {c}, difference in rate {diff:+.2f}, "
                  f"McNemar exact p = {mcnemar_exact(b, c):.3f}  (n={len(keys)})")
        tk = [(da[k]["tok_in"] + da[k]["tok_out"]) - (db[k]["tok_in"] + db[k]["tok_out"]) for k in keys]
        mu, lo, hi = boot_ci(tk)
        sd = (sum((x - sum(tk) / len(tk)) ** 2 for x in tk) / max(len(tk) - 1, 1)) ** 0.5
        extra = ""
        try:
            from scipy.stats import wilcoxon
            if any(tk):
                extra = f", Wilcoxon p = {wilcoxon(tk).pvalue:.3g}"
        except Exception:
            pass
        print(f"  tokens {ra[:2]} minus {rb[:2]}: mean {mu:+.0f} [{lo:+.0f},{hi:+.0f}] (dz = {mu / sd if sd else float('nan'):+.2f}){extra}")
print("\nNote: with many pairwise tests, say which comparisons were planned in advance and correct the rest "
      "(for example Holm). The planned ones are R3 vs R2 and R3 vs R1 on 'accept' and tokens, "
      "and R5 vs R3 and R6 vs R5 on 'sound'.")
