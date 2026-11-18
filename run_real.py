"""Run the baseline ladder with a REAL model (v4).

Local model with Ollama (after `ollama pull <model>` and with Ollama running):
    python run_real.py --model llama3.1:8b --n 10
Hosted API (key kept in an environment variable, never in the code):
    set MY_KEY=...                       (Windows)   |   export MY_KEY=...   (Mac/Linux)
    python run_real.py --base-url https://<provider>/v1 --model <model name> --key-env MY_KEY --n 10

Start with --n 10 to check it works and see the cost. Raw per-scenario rows go to --out (default results_real.jsonl);
stats.py turns them into confidence intervals and paired tests.
  --f3 N              also run the say-do check (F3) on the first N scenarios of each rung
  --start K           skip the first K scenarios (use 0..K-1 to train the router, K.. to test it)
  --router router.json  enables rung R4 (learned router, see router.py)
  --codecarbon        measure energy per rung; add --measure-idle 30 to measure idle power before and after
"""
import argparse
import json
import os
import time
from collections import defaultdict

import ladder
from core import load_regions, make_scenarios
from energy import codecarbon_meter
from faith import say_do, score_row
from ladder import RUNGS
from llm_real import OpenAICompatLLM

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", default="http://localhost:11434/v1")
ap.add_argument("--model", required=True)
ap.add_argument("--key-env", default=None, help="name of the environment variable holding the API key")
ap.add_argument("--n", type=int, default=10, help="number of scenarios")
ap.add_argument("--start", type=int, default=0, help="index of the first scenario to run")
ap.add_argument("--pool", type=int, default=300, help="size of the scenario set to draw from")
ap.add_argument("--timeout", type=int, default=900, help="seconds to wait for each model call")
ap.add_argument("--max-tokens", type=int, default=500, help="cap on answer length")
ap.add_argument("--temperature", type=float, default=0.0)
ap.add_argument("--run-id", default="1", help="label for repeated runs (use with --temperature > 0)")
ap.add_argument("--rungs", default="R0,R1,R2,R3", help="which baselines to run, e.g. R0,R1 (R4 needs --router)")
ap.add_argument("--no-json-mode", action="store_true", help="turn off JSON mode if the server rejects it")
ap.add_argument("--out", default="results_real.jsonl")
ap.add_argument("--f3", type=int, default=0, help="run the say-do check on the first N scenarios per rung")
ap.add_argument("--router", default=None, help="router.json from train_router.py")
ap.add_argument("--router-thr", type=float, default=0.5)
ap.add_argument("--r6-tau", type=float, default=0.65, help="R6 escalation threshold (article: 0.65)")
ap.add_argument("--r6-no-prefilter", action="store_true", help="R6 ablation: System 1 sees every region")
ap.add_argument("--codecarbon", action="store_true", help="measure energy per rung (local models only)")
ap.add_argument("--measure-idle", type=int, default=0, help="seconds of idle measurement before and after the run")
ap.add_argument("--idle-cpu-watts", type=float, default=None, help="use a known idle CPU power instead of measuring it")
ap.add_argument("--country", default="SGP", help="ISO code for CodeCarbon (skips the location lookup)")
a = ap.parse_args()

regions = load_regions("data/regions.csv")
scen = make_scenarios(regions, n=a.pool)[a.start: a.start + a.n]
llm = OpenAICompatLLM(a.base_url, a.model, os.environ.get(a.key_env) if a.key_env else None, temperature=a.temperature,
                      timeout=a.timeout, max_tokens=a.max_tokens, json_mode=not a.no_json_mode)
if a.router:
    import router
    ladder.ROUTER["model"], ladder.ROUTER["thr"] = router.load(a.router), a.router_thr
import jev_router
jev_router.CONF.update(tau=a.r6_tau, prefilter=not a.r6_no_prefilter)
want = a.rungs.split(",")
if "R4" in want and not a.router:
    raise SystemExit("R4 needs --router router.json (train it first with train_router.py)")
avg = lambda v: sum(v) / len(v) if v else float("nan")
mean_or_none = lambda v: (sum(v) / len(v)) if v else None


def idle_power():
    with codecarbon_meter(country=a.country) as m:
        time.sleep(a.measure_idle)
    return (m["cpu_kwh"] or 0) * 3.6e6 / a.measure_idle


def run_rung(name, fn, out):
    rows, fails = [], 0
    for i, s in enumerate(scen, 1):
        t0 = time.time()
        try:
            r = fn(llm, regions, s)
        except RuntimeError as e:
            fails += 1
            print(f"  {name} {i}/{len(scen)} FAILED ({e})", flush=True)
            continue
        dt = time.time() - t0
        print(f"  {name} {i}/{len(scen)} done in {dt:.0f}s", flush=True)
        valid = r.parse_ok and r.region in regions
        row = dict(model=a.model, run=a.run_id, rung=name, scenario=s.sid, difficulty=s.difficulty, binding=s.binding,
                   truth=s.best, chosen=r.region, exact=r.region == s.best, accept=r.region in s.acceptable,
                   parse_fail=not r.parse_ok, invalid_id=r.parse_ok and r.region not in regions,
                   violate=(r.region not in set(s.feasible)) if valid else None,
                   **score_row(r, s, regions), f3=None, flagged=r.flagged,
                   tok_in=r.tok_in, tok_out=r.tok_out, calls=r.calls, escalated=r.escalated, secs=dt, explanation=r.explanation)
        if name.startswith("R6"):
            row["r6_tau"], row["r6_prefilter"] = a.r6_tau, not a.r6_no_prefilter
        if name.startswith("R1") or name.startswith("R3") or name.startswith("R4"):
            import router as _r
            row["feat"] = _r.features(s, r, regions) if name.startswith("R1") else None
        rows.append((s, r, row))
    return rows, fails


results, energy, failed = {}, {}, {}
idle_before = idle_power() if (a.codecarbon and a.measure_idle) else None
with open(a.out, "a", encoding="utf-8") as out:
    for name, fn in RUNGS.items():
        if name.split()[0] not in want:
            continue
        if a.codecarbon:
            t0 = time.time()
            with codecarbon_meter(country=a.country) as meter:
                rows, fails = run_rung(name, fn, out)
            energy[name] = dict(cpu_kwh=meter.get("cpu_kwh") or 0, ram_kwh=meter.get("ram_kwh") or 0,
                                gpu_kwh=meter.get("gpu_kwh") or 0, total_kwh=meter.get("kwh") or 0,
                                cpu_method=meter.get("cpu_method"), cpu_measured=meter.get("cpu_measured"),
                                secs=time.time() - t0, n=len(rows))
            for _, _, row in rows:
                row["energy_cpu_method"] = meter.get("cpu_method")
        else:
            rows, fails = run_rung(name, fn, out)
        failed[name] = fails
        if a.f3:  # say-do check, outside the energy meter so it does not inflate the rung's energy
            for s, r, row in rows[: a.f3]:
                ok, ti, to = say_do(fn, llm, regions, s, r)
                row["f3"] = ok
        for s, r, row in rows:
            out.write(json.dumps(row) + "\n")
        results[name] = [row for _, _, row in rows]
        print(f"finished {name}", flush=True)
idle_after = idle_power() if (a.codecarbon and a.measure_idle) else None

print(f"\n{a.model}, {len(scen)} scenarios (start {a.start}), run {a.run_id}\n")
print(f"chance level (random feasible region): accept = {avg([len(s.acceptable) / len(s.feasible) for s in scen]):.2f}\n")
hdr = (f"{'rung':24} {'exact':>6} {'accept':>7} {'SOUND':>6} {'unparsed':>9} {'badID':>6} {'violate':>8} {'claimOK':>8} "
       f"{'F2':>5} {'F2ref':>6} {'F2cho':>6} {'citeOK':>7} {'ctrOK':>6} {'F3':>5} {'flag':>5}")
print(hdr)
col = lambda rows, k: [r[k] for r in rows if r.get(k) is not None]
for n, rows in results.items():
    print(f"{n:24} {avg(col(rows,'exact')):6.2f} {avg(col(rows,'accept')):7.2f} {avg(col(rows,'sound')):6.2f} "
          f"{avg(col(rows,'parse_fail')):9.2f} {avg(col(rows,'invalid_id')):6.2f} {avg(col(rows,'violate')):8.2f} "
          f"{avg(col(rows,'claim_support')):8.2f} {avg(col(rows,'decisive_ok')):5.2f} {avg(col(rows,'decisive_ref')):6.2f} "
          f"{avg(col(rows,'decisive_choice')):6.2f} {avg(col(rows,'cite_ok')):7.2f} {avg(col(rows,'ctr_ok')):6.2f} "
          f"{avg(col(rows,'f3')):5.2f} {avg(col(rows,'flagged')):5.2f}")
print(f"\n{'rung':24} {'tokens':>7} {'calls':>6} {'escal':>6} {'secs':>6} | {'accept easy':>11} {'accept hard':>11}")
for n, rows in results.items():
    e = [r for r in rows if r["difficulty"] == "easy"]
    h = [r for r in rows if r["difficulty"] == "hard"]
    print(f"{n:24} {avg([r['tok_in']+r['tok_out'] for r in rows]):7.0f} {avg(col(rows,'calls')):6.1f} "
          f"{avg(col(rows,'escalated')):6.2f} {avg(col(rows,'secs')):6.1f} | {avg(col(e,'accept')):11.2f} {avg(col(h,'accept')):11.2f}")
if failed and any(failed.values()):
    print(f"\nSKIPPED because of failed calls (not counted above): { {k: v for k, v in failed.items() if v} }")
if energy:
    idle_w = a.idle_cpu_watts
    if idle_before is not None:
        idle_w = (idle_before + idle_after) / 2
        print(f"\nidle CPU power: {idle_before:.2f} W before, {idle_after:.2f} W after, using {idle_w:.2f} W")
    print(f"energy per rung (CodeCarbon; RAM is always a constant-power estimate):")
    print(f"{'rung':24} {'cpu Wh':>8} {'net cpu Wh':>11} {'net Wh/scen':>12} {'gpu Wh':>8} {'ram Wh':>8} {'secs':>6}  cpu method")
    for n, d in energy.items():
        net = d["cpu_kwh"] * 1000 - (idle_w * d["secs"] / 3600 if idle_w is not None else 0)
        how = f"{d['cpu_method']} ({'hardware counter' if d['cpu_measured'] else 'ESTIMATE, not a measurement'})"
        print(f"{n:24} {d['cpu_kwh']*1000:8.3f} {net:11.3f} {net/max(d['n'],1):12.4f} {d['gpu_kwh']*1000:8.3f} "
              f"{d['ram_kwh']*1000:8.3f} {d['secs']:6.0f}  {how}")
    if any(not d["cpu_measured"] for d in energy.values()):
        print("  WARNING: CodeCarbon did not read a CPU energy counter; the CPU figures are estimates from load and TDP.")
    if idle_w is None:
        print("  (no idle figure given: net equals total. Use --measure-idle 30 or --idle-cpu-watts)")
print(f"\nraw rows: {a.out}")
