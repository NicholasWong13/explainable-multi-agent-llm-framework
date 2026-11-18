"""Recompute the automated checks on an existing results file with the CURRENT definitions in faith.py.
    python rescore.py results_real.jsonl [--pool 300]      -> writes results_real.rescored.jsonl
Uses the stored chosen region and explanation; no model calls. The scenario set is rebuilt with the same seed, so
--pool must match the --pool used by run_real.py (default 300). Rows whose stored reference answer does not match the
rebuilt scenario are skipped and reported (that means the data or the pool changed)."""
import argparse
import json
from pathlib import Path

from core import load_regions, make_scenarios
from faith import score_row
from ladder import Result

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--pool", type=int, default=300)
a = ap.parse_args()

regions = load_regions("data/regions.csv")
scen = {s.sid: s for s in make_scenarios(regions, n=a.pool)}
src = Path(a.path)
dst = src.with_suffix(".rescored.jsonl")
done = skipped = 0
with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as out:
    for line in f:
        row = json.loads(line)
        s = scen.get(row["scenario"])
        if s is None or s.best != row["truth"]:
            skipped += 1
            continue
        r = Result(row["chosen"], row.get("explanation") or {}, parse_ok=not row.get("parse_fail", False))
        row.update(score_row(r, s, regions))
        out.write(json.dumps(row) + "\n")
        done += 1
print(f"rescored {done} rows -> {dst}" + (f"; skipped {skipped} rows (scenario mismatch)" if skipped else ""))
