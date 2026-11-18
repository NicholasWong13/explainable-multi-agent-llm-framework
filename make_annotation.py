"""Build the human annotation sheet from a results file: a stratified sample, blank label columns for two annotators,
and the automatic checks beside it so kappa.py can compare them afterwards.
    python make_annotation.py results_real.jsonl 100
Stratified by rung and difficulty. Annotators must NOT see the auto_* columns while labelling:
send them 'annotation_for_annotators.csv', keep 'annotation_sheet.csv' for yourself."""
import csv
import json
import random
import sys
from collections import defaultdict

src = sys.argv[1] if len(sys.argv) > 1 else "results_real.jsonl"
total = int(sys.argv[2]) if len(sys.argv) > 2 else 100
rows = [json.loads(l) for l in open(src, encoding="utf-8")]
groups = defaultdict(list)
for r in rows:
    groups[(r["rung"], r["difficulty"])].append(r)
rng = random.Random(5)
per = max(1, total // max(len(groups), 1))
pick = []
for g in groups.values():
    rng.shuffle(g)
    pick += g[:per]
rng.shuffle(pick)
L = ["A1_claims_supported(0/1)", "A1_reflects_decision(0/1)", "A2_claims_supported(0/1)", "A2_reflects_decision(0/1)"]
head = ["id", "scenario", "chosen_region", "explanation", "claims", "cites"]
with open("annotation_sheet.csv", "w", newline="", encoding="utf-8") as f, \
     open("annotation_for_annotators.csv", "w", newline="", encoding="utf-8") as g:
    w, v = csv.writer(f), csv.writer(g)
    w.writerow(head + ["rung", "difficulty", "auto_F1_ok", "auto_F2_ok", "auto_C1_ok"] + L)
    v.writerow(head + L)
    for i, r in enumerate(pick, 1):
        ex = r["explanation"] or {}
        base = [i, r["scenario"], r["chosen"], ex.get("explanation"), json.dumps(ex.get("claims")), json.dumps(ex.get("cites"))]
        f1 = "" if r["claim_support"] is None else int(r["claim_support"] == 1.0)
        w.writerow(base + [r["rung"], r["difficulty"], f1, int(bool(r["decisive_ok"])),
                           "" if r["cite_ok"] is None else int(bool(r["cite_ok"])), "", "", "", ""])
        v.writerow(base + ["", "", "", ""])
print(f"wrote {len(pick)} rows to annotation_sheet.csv (yours) and annotation_for_annotators.csv (send this one)")
