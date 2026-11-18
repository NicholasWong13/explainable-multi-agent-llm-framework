"""Learned escalation router (rung R4).

R3 escalates on a fixed rule (parse failure, invalid ID, constraint violation, confidence < 0.75). A small model's own
confidence is a weak signal, so R4 learns WHEN the fast path is likely to be wrong from cheap features that are known
before any extra model call:
  confidence stated by the fast path, parse ok, chosen region is in the feasible set (a deterministic rule check),
  citation valid, weight on carbon, weight on latency, has latency cap, log(number of feasible regions), residency ANY.
It is a logistic regression in pure Python (no extra packages), trained on a TRAINING split of scenarios and evaluated
on a DIFFERENT test split. Label = the fast-path answer was not in the acceptable set.

Use:  python train_router.py results_train.jsonl  ->  router.json     then   run_real.py --router router.json
"""
import json
import math

from faith import citation_valid

NAMES = ["confidence", "parse_ok", "in_feasible", "cite_ok", "w_carbon", "w_latency", "has_cap", "log_n_feasible", "res_any"]


def features(s, fast, regions):
    try:
        conf = float(fast.explanation.get("confidence", 0))
    except (TypeError, ValueError):
        conf = 0.0
    cv = citation_valid(fast, regions)
    return [min(max(conf, 0.0), 1.0), 1.0 if fast.parse_ok else 0.0,
            1.0 if fast.region in set(s.feasible) else 0.0, 1.0 if cv else 0.0,
            s.w_carbon, s.w_latency, 1.0 if s.max_rtt_ms is not None else 0.0,
            math.log(max(len(s.feasible), 1)), 1.0 if s.residency == "ANY" else 0.0]


def _sigmoid(z):
    return 1 / (1 + math.exp(-max(min(z, 30), -30)))


def train(X, y, epochs=400, lr=0.3, l2=0.01):
    n, d = len(X), len(X[0])
    mean = [sum(r[j] for r in X) / n for j in range(d)]
    std = [(sum((r[j] - mean[j]) ** 2 for r in X) / n) ** 0.5 or 1.0 for j in range(d)]
    Z = [[(r[j] - mean[j]) / std[j] for j in range(d)] for r in X]
    w, b = [0.0] * d, 0.0
    for _ in range(epochs):
        gw, gb = [l2 * v for v in w], 0.0
        for z, t in zip(Z, y):
            e = _sigmoid(sum(a * c for a, c in zip(w, z)) + b) - t
            for j in range(d):
                gw[j] += e * z[j] / n
            gb += e / n
        w = [a - lr * g for a, g in zip(w, gw)]
        b -= lr * gb
    return {"names": NAMES, "mean": mean, "std": std, "w": w, "b": b}


def p_wrong(model, x):
    z = [(v - m) / s for v, m, s in zip(x, model["mean"], model["std"])]
    return _sigmoid(sum(a * c for a, c in zip(model["w"], z)) + model["b"])


def save(model, path="router.json"):
    json.dump(model, open(path, "w"), indent=1)


def load(path="router.json"):
    return json.load(open(path))
