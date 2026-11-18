"""Train the R4 router from a TRAINING run of R1 (the fast path).
    python run_real.py --model M --rungs R1 --start 0 --n 60 --out results_train.jsonl
    python train_router.py results_train.jsonl            (writes router.json and prints training accuracy)
Then evaluate on scenarios the router has never seen:
    python run_real.py --model M --rungs R1,R2,R3,R4 --start 60 --n 40 --router router.json
"""
import json
import sys

import router

path = sys.argv[1] if len(sys.argv) > 1 else "results_train.jsonl"
X, y = [], []
for line in open(path, encoding="utf-8"):
    r = json.loads(line)
    if r["rung"].startswith("R1") and r.get("feat") is not None:
        X.append(r["feat"])
        y.append(0 if r["accept"] else 1)  # 1 = the fast path was wrong
if len(X) < 20 or len(set(y)) < 2:
    sys.exit(f"Need at least 20 R1 rows with both right and wrong answers (found {len(X)} rows, {sum(y)} wrong).")
m = router.train(X, y)
router.save(m)
pred = [router.p_wrong(m, x) > 0.5 for x in X]
acc = sum(p == bool(t) for p, t in zip(pred, y)) / len(y)
print(f"trained on {len(X)} rows, {sum(y)} wrong answers; training accuracy {acc:.2f} (base rate of wrong {sum(y)/len(y):.2f})")
for n, w in sorted(zip(m["names"], m["w"]), key=lambda t: -abs(t[1])):
    print(f"  {n:16} {w:+.2f}")
print("wrote router.json")
