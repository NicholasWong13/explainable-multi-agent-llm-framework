"""Inter-annotator agreement and agreement of the automatic checks with the human labels.
    python kappa.py annotation_sheet.csv
Needs the four label columns filled with 0 or 1 by two annotators:
  A1_claims_supported, A1_reflects_decision, A2_claims_supported, A2_reflects_decision
The sheet also carries the automatic F1/F2 results (auto_F1_ok, auto_F2_ok, auto_C1_ok), so this prints:
  - Cohen's kappa and raw agreement between annotators, for each of the two questions
  - agreement of F1 with the human 'claims supported' label, and of F2 with 'reflects decision' (human = both agree)"""
import csv
import sys


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((sum(1 for x in a if x == v) / n) * (sum(1 for y in b if y == v) / n) for v in (0, 1))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan"), po


rows = list(csv.DictReader(open(sys.argv[1] if len(sys.argv) > 1 else "annotation_sheet.csv", encoding="utf-8")))
get = lambda r, k: r[k].strip()
done = [r for r in rows if all(get(r, k) in ("0", "1") for k in
        ("A1_claims_supported(0/1)", "A1_reflects_decision(0/1)", "A2_claims_supported(0/1)", "A2_reflects_decision(0/1)"))]
print(f"{len(done)} of {len(rows)} rows fully labelled")
if len(done) < 10:
    sys.exit("label more rows first")
toint = lambda rs, k: [int(r[k]) for r in rs]
for q in ("claims_supported", "reflects_decision"):
    k, po = kappa(toint(done, f"A1_{q}(0/1)"), toint(done, f"A2_{q}(0/1)"))
    print(f"{q:18} Cohen's kappa {k:.2f}, raw agreement {po:.2f}")
agree = lambda r, q: r[f"A1_{q}(0/1)"] == r[f"A2_{q}(0/1)"]
for auto, q in (("auto_F1_ok", "claims_supported"), ("auto_F2_ok", "reflects_decision")):
    sub = [r for r in done if agree(r, q) and r[auto] in ("0", "1")]
    if sub:
        hit = sum(r[auto] == r[f"A1_{q}(0/1)"] for r in sub) / len(sub)
        print(f"{auto} vs human label ({q}), on {len(sub)} rows where both annotators agree: agreement {hit:.2f}")
