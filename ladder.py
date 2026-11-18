"""The baseline ladder:
  R0 single LLM | R1 LLM + sources with citations | R2 always-on multi-agent | R3 selective escalation
  (Dr. Zeyad's four baselines), plus R4 learned router | R5 R3 + separate explainer | R6 dual-process (jev_router.py)
Each rung returns a Result with the chosen region, the explanation JSON and token counts.

v3 fixes after the first real-model run: concrete JSON example (the model copied "a|b|c" literally), exactly one
region, a list of valid region IDs, first-object JSON parsing, short agent answers, and the full table in the
sources (the placeholder retriever was returning irrelevant rows)."""
import json
from dataclasses import dataclass

FORMAT = (
    'Reply with ONE JSON object and nothing else, in this shape (the values here are only an example): '
    '{"region": "example-region-1", "confidence": 0.7, "decisive_factor": "carbon", "constraints_met": true, '
    '"explanation": "one short sentence", "cites": ["example-region-1"], '
    '"claims": [{"region": "example-region-1", "field": "grid_gco2_kwh", "value": 123.4}, '
    '{"region": "example-region-1", "field": "country", "value": "XX"}]}. '
    'Rules: pick exactly ONE region ID from the valid list; decisive_factor is exactly one of carbon, latency, residency, price; '
    '"constraints_met" is true only if your region satisfies the residency rule and the latency limit; '
    '"cites" lists the region IDs you rely on, copied from the sources, and must include your chosen region; '
    '"field" is grid_gco2_kwh or country; "value" is a single number or a two-letter country code.'
)


FORMAT_R0 = FORMAT.replace(", copied from the sources,", "")  # R0 has no sources to cite


@dataclass
class Result:
    region: str | None
    explanation: dict
    tok_in: int = 0
    tok_out: int = 0
    calls: int = 0
    escalated: bool = False
    parse_ok: bool = True
    flagged: bool | None = None  # R6 egress check: True = send to human review; None = rung has no egress check


def _parse(text):
    """Take the first JSON object in the text. Returns (region, dict, parse_ok)."""
    try:
        i = text.index("{")
        d, _ = json.JSONDecoder().raw_decode(text[i:])
        return d.get("region"), d, True
    except Exception:
        return None, {"explanation": text, "claims": []}, False


def _ids(regions):
    return "Valid region IDs: " + ", ".join(sorted(regions))


def _rows(rows):
    return "\n".join(f"[{r['region']}] {r['location']} ({r['country']}) lat {r['lat']} lon {r['lon']} "
                     f"grid carbon {r['grid']} gCO2/kWh, CFE {r['cfe']}"
                     + (f", price {r['price']}" if r.get("price") is not None else "") for r in rows)


def retrieve(regions, scenario, k=None):
    """With about 45 rows the whole table fits in the prompt, so the 'RAG' rungs get every row (k=None).
    A token-overlap retriever was tried first and returned irrelevant rows. Revisit retrieval only if the corpus
    grows (for example residency guidance documents)."""
    rows = list(regions.values())
    return rows if k is None else rows[:k]


def _result(text, usage, calls=1):
    reg, d, ok = _parse(text)
    return Result(reg, d, usage["in"], usage["out"], calls, parse_ok=ok)


def r0_single(llm, regions, s):
    t, u = llm.complete(f"{s.text()}\n{_ids(regions)}\n{FORMAT_R0}", scenario=s, mode="single")
    return _result(t, u)


def r1_rag(llm, regions, s):
    t, u = llm.complete(f"{s.text()}\n{_ids(regions)}\nUse ONLY these sources and cite them as [region-id]:\n"
                        f"{_rows(retrieve(regions, s))}\n{FORMAT}", scenario=s, mode="rag")
    return _result(t, u)


AGENTS = {
    "carbon": "You are the sustainability agent. Rank the best 3 regions on grid carbon intensity.",
    "latency": "You are the performance agent. Rank the best 3 regions on distance and latency from the users.",
    "compliance": "You are the compliance agent. List the regions that break the residency or latency rules.",
}


def r2_multi(llm, regions, s):
    rows = _rows(retrieve(regions, s))
    tin = tout = 0
    notes = []
    for name, role in AGENTS.items():
        t, u = llm.complete(f"{role}\n{s.text()}\n{_ids(regions)}\n{rows}\nAnswer in under 80 words.",
                            scenario=s, mode="agent", max_tokens=200)
        tin, tout = tin + u["in"], tout + u["out"]
        notes.append(f"{name} agent: {t}")
    t, u = llm.complete("You are the synthesiser. Combine the agents' opinions, apply the weights, record dissent.\n"
                        f"{s.text()}\n{_ids(regions)}\nSources (cite them as [region-id]):\n{rows}\n" + "\n".join(notes) + f"\n{FORMAT}", scenario=s, mode="synth")
    res = _result(t, u, calls=len(AGENTS) + 1)
    res.tok_in += tin
    res.tok_out += tout
    return res


def r3_selective(llm, regions, s, tau=0.75):
    """Fast path = R1. Escalate to R2 when the answer cannot be parsed, is not a valid region ID, confidence < tau,
    or the answer breaks a hard constraint."""
    fast = r1_rag(llm, regions, s)
    conf = fast.explanation.get("confidence", 0)
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        conf = 0.0
    bad = (not fast.parse_ok) or fast.region not in regions or fast.region not in set(s.feasible)
    if conf >= tau and not bad:
        return fast
    full = r2_multi(llm, regions, s)
    full.tok_in += fast.tok_in
    full.tok_out += fast.tok_out
    full.calls += fast.calls
    full.escalated = True
    return full


ROUTER = {"model": None, "thr": 0.5}  # set by run_real.py when --router is given


def r4_learned(llm, regions, s):
    """Fast path = R1. Escalate to R2 when the learned router says the fast answer is probably wrong."""
    import router
    fast = r1_rag(llm, regions, s)
    p = router.p_wrong(ROUTER["model"], router.features(s, fast, regions))
    if p <= ROUTER["thr"] and fast.parse_ok and fast.region in regions:
        return fast
    full = r2_multi(llm, regions, s)
    full.tok_in += fast.tok_in
    full.tok_out += fast.tok_out
    full.calls += fast.calls
    full.escalated = True
    return full


def explain_layer(llm, regions, s, base):
    """Separate explanation agent. The decision in `base` is FIXED; this agent only writes the explanation, the
    decisive factor, the claims and the citations. It exists because a fast decision path cannot explain itself
    (Dr. Zeyad's dual-process article). If its output cannot be parsed, the base explanation is kept."""
    rows = _rows(retrieve(regions, s))
    t, u = llm.complete("You are the explanation agent. The decision is already made and you must not change it.\n"
                        f"Chosen region: {base.region}\n{s.text()}\n{_ids(regions)}\n"
                        f"Sources (cite them as [region-id]):\n{rows}\n"
                        f"Explain why {base.region} was chosen.\n{FORMAT}",
                        scenario=s, mode="explain", fixed_region=base.region)
    _, d, ok = _parse(t)
    out = Result(base.region, d if ok else base.explanation, base.tok_in + u["in"], base.tok_out + u["out"],
                 base.calls + 1, escalated=base.escalated, parse_ok=base.parse_ok)
    if ok:
        d["region"] = base.region  # the explainer cannot move the decision
    return out


def r5_explained(llm, regions, s):
    """R3 decision (fast path, escalate when unsure) followed by the separate explanation agent."""
    return explain_layer(llm, regions, s, r3_selective(llm, regions, s))


def r6_dual_process(llm, regions, s):
    """Rules + System 1 typed choice + confidence gate + System 2 + separate explainer + egress check.
    See jev_router.py. With the stand-in, this tests the architecture, not Jev itself."""
    from jev_router import r6_dual_process as run
    return run(llm, regions, s)


RUNGS = {"R0 single LLM": r0_single, "R1 LLM+RAG": r1_rag, "R2 always multi-agent": r2_multi,
         "R3 selective escalation": r3_selective, "R4 learned router": r4_learned,
         "R5 selective + explainer": r5_explained, "R6 dual-process (Jev stand-in)": r6_dual_process}
