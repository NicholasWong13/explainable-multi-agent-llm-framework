"""Regions, scenarios and computable ground truth for cloud-region selection."""
import csv
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

EU = set("AT BE BG HR CY CZ DK EE FI FR DE GR HU IE IT LV LT LU MT NL PL PT RO SK SI ES SE".split())
RESIDENCY = {
    "EU": EU, "UK": {"GB"}, "US": {"US"}, "CA": {"CA"},
    "APAC": {"JP", "KR", "TW", "HK", "SG", "IN", "ID", "AU", "TH"},
    "ANY": None,
}
USER_SITES = {  # city -> (lat, lon)
    "Singapore": (1.35, 103.82), "London": (51.51, -0.13), "Frankfurt": (50.11, 8.68),
    "New York": (40.71, -74.01), "Mumbai": (19.08, 72.88), "Tokyo": (35.68, 139.69),
    "Sydney": (-33.87, 151.21), "Sao Paulo": (-23.55, -46.63), "Toronto": (43.65, -79.38),
}
WEIGHTS = {  # (carbon, latency, price) priorities
    "carbon-first": (0.8, 0.2, 0.0), "carbon-leaning": (0.65, 0.35, 0.0), "balanced": (0.5, 0.5, 0.0),
    "latency-leaning": (0.35, 0.65, 0.0), "latency-first": (0.2, 0.8, 0.0),
}
PRICE_WEIGHTS = {  # added automatically only when data/prices.csv exists
    "cost-aware": (0.3, 0.2, 0.5), "three-way": (0.34, 0.33, 0.33),
}
LATENCY = {}  # (city, region) -> MEASURED rtt in ms; filled from data/latency.csv when that file exists
TIE_EPS = 0.05  # regions within this score of the best count as acceptable


def load_regions(path="data/regions.csv"):
    """Reads regions.csv. If prices.csv (region,price) or latency.csv (city,region,rtt_ms) sit next to it,
    they are used: price joins the score and measured latency replaces the distance proxy."""
    p = Path(path)
    with open(p, newline="", encoding="utf-8") as f:
        out = {}
        for r in csv.DictReader(f):
            out[r["region"]] = dict(
                region=r["region"], location=r["location"], country=r["country"],
                lat=float(r["lat"]), lon=float(r["lon"]),
                cfe=float(r["cfe"]) if r["cfe"] else None, grid=float(r["grid_gco2_kwh"]), price=None)
    pf = p.with_name("prices.csv")
    if pf.exists():
        with open(pf, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["region"] in out and r["price"].strip():
                    out[r["region"]]["price"] = float(r["price"])
    lf = p.with_name("latency.csv")
    LATENCY.clear()
    if lf.exists():
        with open(lf, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["rtt_ms"].strip():
                    LATENCY[(r["city"], r["region"])] = float(r["rtt_ms"])
    return out


def has_prices(regions):
    return all(r.get("price") is not None for r in regions.values())


def get_rtt(city, region):
    """Measured RTT if available, otherwise the distance proxy (a lower bound)."""
    k = (city, region["region"])
    return LATENCY[k] if k in LATENCY else rtt_proxy_ms(USER_SITES[city], region)


def haversine_km(a, b):
    (la1, lo1), (la2, lo2) = a, b
    p1, p2 = math.radians(la1), math.radians(la2)
    dphi, dl = p2 - p1, math.radians(lo2 - lo1)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))


def rtt_proxy_ms(user_xy, region):
    """Lower-bound RTT proxy: great-circle distance at ~2/3 c in fibre (~1 ms RTT per 100 km).
    Replace with measured inter-region latency data before reporting results."""
    return haversine_km(user_xy, (region["lat"], region["lon"])) / 100.0


@dataclass
class Scenario:
    sid: str
    city: str
    residency: str
    max_rtt_ms: float | None
    weights_name: str
    w_carbon: float
    w_latency: float
    w_price: float = 0.0
    feasible: list = field(default_factory=list)
    best: str = ""
    acceptable: list = field(default_factory=list)
    gap: float = 0.0
    difficulty: str = ""
    binding: bool = False  # the unconstrained best region is ruled out by residency or the latency cap

    def text(self):
        rule = {"ANY": "no data-residency restriction", "EU": "data must stay in an EU member state",
                "UK": "data must stay in the United Kingdom", "US": "data must stay in the United States",
                "CA": "data must stay in Canada", "APAC": "data must stay in the listed Asia-Pacific countries"}[self.residency]
        rtt = "no hard latency limit" if self.max_rtt_ms is None else f"round-trip latency from users must be at most {self.max_rtt_ms:.0f} ms"
        lat, lon = USER_SITES[self.city]
        pr = f", {round(self.w_price*100)}% on low price" if self.w_price else ""
        return (f"A company serves users in {self.city} (lat {lat}, lon {lon}) and must choose ONE cloud region for a new "
                f"workload. Constraints: {rule}; {rtt}. Priorities: {round(self.w_carbon*100)}% weight on low grid carbon "
                f"intensity, {round(self.w_latency*100)}% on low latency{pr}.")


def decide(regions, city, residency, max_rtt, wc, wl, drop=None, wp=0.0):
    """Ground truth. Hard constraints: residency, latency cap. Then min-max normalised weighted score.
    `drop` = 'carbon'|'latency'|'residency'|'price' neutralises one factor (counterfactual checks, F2)."""
    allowed = RESIDENCY[residency] if drop != "residency" else None
    cap = None if drop == "latency" else max_rtt
    feas = []
    for r in regions.values():
        if allowed is not None and r["country"] not in allowed:
            continue
        rtt = get_rtt(city, r)
        if cap is not None and rtt > cap:
            continue
        feas.append((r["region"], r["grid"], rtt, r.get("price")))
    if not feas:
        return [], None, [], 0.0
    if wp > 0 and any(f[3] is None for f in feas):
        raise ValueError("price weight > 0 but some regions have no price")
    def norm(vals):
        lo, hi = min(vals), max(vals)
        return [0.0 if hi == lo else (v - lo) / (hi - lo) for v in vals]
    nc, nl = norm([f[1] for f in feas]), norm([f[2] for f in feas])
    npz = norm([f[3] for f in feas]) if wp > 0 else [0.0] * len(feas)
    wc_ = 0.0 if drop == "carbon" else wc
    wl_ = 0.0 if drop == "latency" else wl
    wp_ = 0.0 if drop == "price" else wp
    scores = sorted(((wc_ * c + wl_ * l + wp_ * p, f[0]) for f, c, l, p in zip(feas, nc, nl, npz)))
    best = scores[0][1]
    acc = [n for sc, n in scores if sc <= scores[0][0] + TIE_EPS]
    gap = (scores[1][0] - scores[0][0]) if len(scores) > 1 else 1.0
    return [f[0] for f in feas], best, acc, gap


def classify(regions, city, residency, max_rtt, wc, wl, wp, feas, best):
    """Difficulty and constraint-binding flag.
    easy    = the reference answer is also the best region on a single factor (carbon alone or latency alone)
              among the feasible regions, so the model can get it by sorting on one column.
    hard    = the reference answer is a COMPROMISE that is best on neither factor alone, so the model has to
              weigh the trade-off, and at least 4 regions are feasible.
    binding = the best region with NO constraints is ruled out by residency or the latency cap."""
    sub = {k: regions[k] for k in feas}
    _, bc, _, _ = decide(sub, city, "ANY", None, 1.0, 0.0)
    _, bl, _, _ = decide(sub, city, "ANY", None, 0.0, 1.0)
    hard = best not in (bc, bl) and len(feas) >= 4
    _, free_best, _, _ = decide(regions, city, "ANY", None, wc, wl, wp=wp)
    return ("hard" if hard else "easy"), (free_best not in set(feas))


def make_scenarios(regions, n=100, seed=7, hard_share=0.4):
    """Scenarios with a fixed seed (same call, same set). The pool is every combination of user city, residency rule,
    latency cap and priority profile that leaves at least 2 feasible regions. The sample is stratified so that about
    `hard_share` of it is hard (compromise answers), because random sampling gives mostly easy ones."""
    rng = random.Random(seed)
    profiles = dict(WEIGHTS)
    if has_prices(regions):
        profiles.update(PRICE_WEIGHTS)
    combos = [(c, rz, rt, wn) for c in USER_SITES for rz in RESIDENCY for rt in (60, 120, None) for wn in profiles]
    rng.shuffle(combos)
    pool = []
    for c, rz, rt, wn in combos:
        wc, wl, wp = profiles[wn]
        feas, best, acc, gap = decide(regions, c, rz, rt, wc, wl, wp=wp)
        if not feas or len(feas) < 2:
            continue
        diff, binding = classify(regions, c, rz, rt, wc, wl, wp, feas, best)
        pool.append((c, rz, rt, wn, wc, wl, wp, feas, best, acc, gap, diff, binding))
    hard = [p for p in pool if p[11] == "hard"]
    easy = [p for p in pool if p[11] == "easy"]
    n_hard = min(len(hard), round(n * hard_share))
    chosen = hard[:n_hard] + easy[: max(0, n - n_hard)]
    chosen = chosen[:n]
    rng.shuffle(chosen)
    return [Scenario(f"S{i:03d}", *p[:3], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12])
            for i, p in enumerate(chosen)]
