"""Faithfulness and explanation checks. None of them uses an LLM as verifier.

F1 claim support       share of checkable claims (grid_gco2_kwh, country) that match the source table.
                       NOTE: this is a copying-accuracy check. It is necessary but weak: a model that copies numbers
                       correctly for a WRONG region still scores 1.0. Never report F1 on its own.
F2 decisive factor     two parts, both must hold:
   F2-ref              the named factor matters for the reference decision (removing it changes the reference answer);
   F2-choice           the named factor actually favours the region the MODEL chose (it ranks in the top
                       TOP_K feasible regions on that factor; for residency, the rule must exclude the
                       unconstrained best region and the chosen region must satisfy it).
                       F2-ref alone was too lenient: in the first pilot, a Sao Paulo scenario answered with Zurich
                       "because of latency" passed it.
F3 say-do consistency  neutralise the named factor in the scenario and re-run the SAME rung: the model's own
                       decision should change. Needs live calls, so it is run on a subset (--f3 N).
C1 citation validity   every cited ID exists, the chosen region is cited, and every region used in a claim is cited.
C2 constraint report   the model states whether its choice meets the hard constraints; the statement must be true.
SOUND                  composite: the decision is acceptable AND every applicable check (F1 = 1, F2, C1, C2) passes.
                       This is the headline explanation metric, because it cannot be passed by a wrong decision.

A human-annotated subset (make_annotation.py, kappa.py) checks that F1 and F2 agree with human judgement,
so the automated metrics are not circular.
"""
from dataclasses import replace

from core import RESIDENCY, decide, get_rtt

FACTORS = ("carbon", "latency", "residency", "price")
TOP_K = 3  # F2-choice: the chosen region must be in the best TOP_K feasible regions on the named factor


def claim_support(result, regions):
    """F1. Share of CHECKABLE claims that match the table. Returns None when there is nothing checkable."""
    claims = [c for c in (result.explanation.get("claims") or []) if isinstance(c, dict)
              and c.get("field") in ("grid_gco2_kwh", "country")]
    if not claims:
        return None
    ok = 0
    for c in claims:
        r = regions.get(c.get("region"))
        if r is None:
            continue
        f, v = c.get("field"), c.get("value")
        if f == "grid_gco2_kwh":
            try:
                ok += abs(float(v) - r["grid"]) <= 0.01 * r["grid"] + 0.5
            except (TypeError, ValueError):
                pass
        else:
            ok += (v == r["country"])
    return ok / len(claims)


def _named(result):
    f = result.explanation.get("decisive_factor")
    return f if f in FACTORS else None


def decisive_factor_ref(result, s, regions):
    """F2-ref. The named factor changes the REFERENCE answer when it is removed."""
    named = _named(result)
    if named is None:
        return False
    _, base_best, _, _ = decide(regions, s.city, s.residency, s.max_rtt_ms, s.w_carbon, s.w_latency, wp=s.w_price)
    _, alt_best, _, _ = decide(regions, s.city, s.residency, s.max_rtt_ms, s.w_carbon, s.w_latency, drop=named, wp=s.w_price)
    return alt_best != base_best


def decisive_factor_choice(result, s, regions, k=TOP_K):
    """F2-choice. The named factor really favours the region the model chose."""
    named, chosen = _named(result), result.region
    if named is None or chosen not in regions or chosen not in set(s.feasible):
        return False
    if named == "residency":  # the rule must rule out the region that would win without it
        if s.residency == "ANY":
            return False
        _, free_best, _, _ = decide(regions, s.city, "ANY", s.max_rtt_ms, s.w_carbon, s.w_latency, wp=s.w_price)
        return free_best is not None and regions[free_best]["country"] not in RESIDENCY[s.residency]
    if named == "price" and (s.w_price <= 0 or regions[chosen].get("price") is None):
        return False
    key = {"carbon": lambda r: r["grid"], "latency": lambda r: get_rtt(s.city, r),
           "price": lambda r: r["price"]}[named]
    v = key(regions[chosen])
    better = sum(1 for x in s.feasible if key(regions[x]) < v)
    return better < k


def decisive_factor_correct(result, s, regions):
    """F2 = F2-ref AND F2-choice."""
    return decisive_factor_ref(result, s, regions) and decisive_factor_choice(result, s, regions)


def citation_valid(result, regions):
    """C1. Returns None when the explanation has no 'cites' field at all (R0 is not asked to cite)."""
    cites = result.explanation.get("cites")
    if cites is None:
        return None
    if not isinstance(cites, list) or not cites:
        return False
    cites = {c for c in cites if isinstance(c, str)}
    if not cites or any(c not in regions for c in cites):
        return False
    if result.region not in cites:
        return False
    used = {c.get("region") for c in (result.explanation.get("claims") or []) if isinstance(c, dict)}
    return used <= cites


def constraint_report_correct(result, s):
    """C2. The model says constraints_met true or false; compare with the truth. None if it said nothing usable."""
    said = result.explanation.get("constraints_met")
    if not isinstance(said, bool) or result.region is None:
        return None
    return said == (result.region in set(s.feasible))


def sound(accept, f1, f2, c1, c2):
    """SOUND composite. Checks that do not apply (None) are skipped; F2 always applies."""
    if not accept or not f2:
        return False
    return all(x is None or x is True or x == 1.0 for x in (f1, c1, c2))


def neutralise(s, factor, regions):
    """Copy of the scenario with one factor switched off IN THE PROMPT, with the feasible set and reference answer
    recomputed so that rule checks inside the rungs (for example R3's constraint check) use the new scenario.
    The remaining weights keep their relative sizes (same rule as decide(drop=...), so F2 and F3 test the same thing)."""
    def rescale(wc, wl, wp):
        t = wc + wl + wp
        return (wc / t, wl / t, wp / t) if t > 0 else (0.5, 0.5, 0.0)
    if factor == "carbon":
        wc, wl, wp = rescale(0.0, s.w_latency, s.w_price)
        s2 = replace(s, w_carbon=wc, w_latency=wl, w_price=wp)
    elif factor == "latency":
        wc, wl, wp = rescale(s.w_carbon, 0.0, s.w_price)
        s2 = replace(s, w_carbon=wc, w_latency=wl, w_price=wp, max_rtt_ms=None)
    elif factor == "residency":
        s2 = replace(s, residency="ANY")
    elif factor == "price":
        wc, wl, wp = rescale(s.w_carbon, s.w_latency, 0.0)
        s2 = replace(s, w_carbon=wc, w_latency=wl, w_price=wp)
    else:
        return None
    feas, best, acc, gap = decide(regions, s2.city, s2.residency, s2.max_rtt_ms, s2.w_carbon, s2.w_latency, wp=s2.w_price)
    return replace(s2, feasible=feas, best=best or "", acceptable=acc, gap=gap)


def say_do(fn, llm, regions, s, result):
    """F3. True if the model's decision changes when the factor it named is neutralised. None if no valid factor named
    or the re-run fails. Returns (consistent, tokens_in, tokens_out)."""
    named = _named(result)
    s2 = neutralise(s, named, regions) if named else None
    if s2 is None:
        return None, 0, 0
    try:
        r2 = fn(llm, regions, s2)
    except RuntimeError:
        return None, 0, 0
    return (r2.region != result.region), r2.tok_in, r2.tok_out


def score_row(result, s, regions):
    """All automated checks for one result, as the dict fields written to the results file."""
    f1 = claim_support(result, regions)
    f2r = decisive_factor_ref(result, s, regions)
    f2c = decisive_factor_choice(result, s, regions)
    c1 = citation_valid(result, regions)
    c2 = constraint_report_correct(result, s)
    acc = result.region in s.acceptable
    return dict(claim_support=f1, decisive_ref=f2r, decisive_choice=f2c, decisive_ok=f2r and f2c,
                cite_ok=c1, ctr_ok=c2, sound=sound(acc, f1, f2r and f2c, c1, c2),
                n_feasible=len(s.feasible), n_acceptable=len(s.acceptable))
