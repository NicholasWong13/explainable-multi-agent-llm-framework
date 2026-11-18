"""R6: dual-process rung, following the production pattern in the article Dr. Zeyad shared
(Skeptical AI, "LLM vs JEV: The Dual-Process AI Stack", Medium, 21 Sep 2026).

The article's pattern, and how each step is implemented here:
  1. Deterministic rules      Hard constraints (residency, latency cap) are applied by code, not by a model.
                              `prefilter=True` passes only the feasible regions on to System 1.
  2. System 1 (Jev Choice)    A typed choice over the candidate regions that returns ONE region ID and a confidence,
                              and no text. Jev's Choice type accepts up to 255 candidates; we have at most 45.
  3. Confidence gate          p >= tau keeps the System 1 answer. p < tau (the article uses 0.65) or an answer
                              outside the candidates escalates to System 2. The article sends p >= 0.90 straight to a
                              deterministic service; the 0.65-0.90 band is not specified there, and we keep it on
                              System 1 (state this in the paper).
  4. System 2                 R2, the always-on multi-agent system (carbon, latency and compliance agents + synthesiser).
  5. Explanation layer        System 1 cannot explain itself (the article: it is "strictly non-generative"), so a
                              separate explanation agent ALWAYS writes the rationale for the fixed decision.
  6. Egress verification      The article checks LLM text against ground truth before delivery. We run the checks that
                              need no reference answer (F1 claim support, F2-choice, C1 citations, C2 constraint report)
                              and set `flagged=True` for human review when any of them fails.

STAND-IN WARNING. We have no Jev access. `StandInJev` imitates the Choice interface with a short LLM call that may only
return {"region", "confidence"}. That call is still autoregressive and its confidence is the model's own stated
confidence, which is NOT calibrated. So R6 with the stand-in tests the ARCHITECTURE (rules + typed choice + gate +
separate explainer + egress check), not Jev's speed or calibration. Results must say this. To use the real model,
implement `RealJev.choice()` from TypeSafe AI's own documentation; do not guess their API.

Ablation: `prefilter=False` gives System 1 every region, so the effect of the rule engine can be separated from the
effect of the typed choice.

Reference: https://medium.com/@skeptical_ai/llm-vs-jev-the-dual-process-ai-stack-1eb29f95d1dc
           TypeSafe AI Jev: https://aijev.org/
"""
from core import decide

TAU = 0.65  # escalation threshold from the article's diagram (p < 0.65 -> System 2)
CONF = {"tau": TAU, "prefilter": True, "jev": None}  # set by run_real.py (--r6-tau, --r6-no-prefilter)


class StandInJev:
    """Imitates a Jev Choice call with a constrained, text-free LLM call. Not calibrated; see module docstring."""

    def __init__(self, llm):
        self.llm = llm
        self.tok_in = self.tok_out = 0

    def choice(self, state, candidates, scenario=None):
        """Returns (choice or None, p, usage). `choice` is None when the reply is not one of the candidates."""
        from ladder import _parse
        prompt = (f"{state}\nCandidate region IDs: {', '.join(candidates)}\n"
                  'Reply with ONLY this JSON and no other text: {"region": "<one candidate ID>", "confidence": <0..1>}')
        text, u = self.llm.complete(prompt, scenario=scenario, mode="choice", max_tokens=40)
        reg, d, ok = _parse(text)
        try:
            p = min(max(float(d.get("confidence", 0)), 0.0), 1.0)
        except (TypeError, ValueError):
            p = 0.0
        return (reg if ok and reg in candidates else None), p, u


class RealJev:
    """Placeholder for a real TypeSafe AI Jev client. Fill in choice() from their documentation once you have access.
    It must return (choice or None, calibrated p, {"in": tokens_in, "out": tokens_out})."""

    def __init__(self, api_key):
        self.api_key = api_key

    def choice(self, state, candidates, scenario=None):
        raise NotImplementedError("Implement from TypeSafe AI's documentation; no public API was available to us.")


def rule_engine(regions, s):
    """Step 1: the deterministic hard-constraint filter (same rules as the ground truth, no model involved)."""
    feas, _, _, _ = decide(regions, s.city, s.residency, s.max_rtt_ms, s.w_carbon, s.w_latency, wp=s.w_price)
    return feas


def egress_check(result, s, regions):
    """Step 6: checks that need no reference answer. Returns (flagged, details)."""
    from faith import citation_valid, claim_support, constraint_report_correct, decisive_factor_choice
    d = dict(f1=claim_support(result, regions), f2_choice=decisive_factor_choice(result, s, regions),
             c1=citation_valid(result, regions), c2=constraint_report_correct(result, s))
    ok = (d["f1"] in (None, 1.0)) and d["f2_choice"] and d["c1"] is not False and d["c2"] is not False
    return (not ok), d


def r6_dual_process(llm, regions, s, tau=None, prefilter=None, jev=None):
    from ladder import _rows, explain_layer, r2_multi, retrieve
    tau = CONF["tau"] if tau is None else tau
    prefilter = CONF["prefilter"] if prefilter is None else prefilter
    jev = jev or CONF["jev"] or StandInJev(llm)

    cands = rule_engine(regions, s) if prefilter else sorted(regions)
    rows = [r for r in retrieve(regions, s) if r["region"] in set(cands)]
    state = f"{s.text()}\nRegion data:\n{_rows(rows)}"
    pick, p, u = jev.choice(state, cands, scenario=s)
    tin, tout, calls = u["in"], u["out"], 1

    if pick is not None and p >= tau:
        from ladder import Result
        base = Result(pick, {"region": pick, "confidence": p}, tin, tout, calls, escalated=False, parse_ok=True)
        route = f"System 1 kept: p = {p:.2f} >= tau = {tau:.2f}"
    else:
        base = r2_multi(llm, regions, s)
        base.tok_in += tin
        base.tok_out += tout
        base.calls += calls
        base.escalated = True
        route = (f"System 1 answer outside the candidates" if pick is None else
                 f"System 1 escalated: p = {p:.2f} < tau = {tau:.2f}")

    out = explain_layer(llm, regions, s, base)
    out.flagged, details = egress_check(out, s, regions)
    out.explanation["route"] = route
    out.explanation["egress"] = details
    return out
