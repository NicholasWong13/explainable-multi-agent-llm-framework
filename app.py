"""Streamlit demo interface for the cloud-region selection prototype.

Run:
    streamlit run app.py

Requires:
    pip install streamlit
    Ollama must be running with a model pulled (e.g. ollama pull llama3.1:8b)
"""
import sys
import os
import time
import streamlit as st

# ── page config must be first ────────────────────────────────────────────────
st.set_page_config(
    page_title="Cloud Region Selector",
    page_icon="🌍",
    layout="wide",
)

# ── make sure project modules are importable ─────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

from core import (
    USER_SITES, RESIDENCY, WEIGHTS, load_regions, decide, get_rtt, Scenario
)
from ladder import r0_single, r1_rag, r2_multi, r3_selective, r5_explained, r6_dual_process
from faith import claim_support, decisive_factor_correct, citation_valid, constraint_report_correct
from lara_assessment import CLOUD_REGION_LARA, NOTES
from llm_real import OpenAICompatLLM

# ── constants ─────────────────────────────────────────────────────────────────
RUNG_FUNCS = {
    "R0 — Single LLM (no data)": r0_single,
    "R1 — LLM + region data (RAG)": r1_rag,
    "R2 — Full multi-agent system": r2_multi,
    "R3 — Selective escalation": r3_selective,
    "R5 — Selective + explanation agent": r5_explained,
    "R6 — Dual-process (Jev stand-in)": r6_dual_process,
}

LATENCY_LABELS = {
    60:   "Strict  (≤ 60 ms)",
    120:  "Relaxed (≤ 120 ms)",
    None: "No limit",
}

WEIGHT_LABELS = {
    "carbon-first":    "Carbon first  (80% carbon, 20% latency)",
    "carbon-leaning":  "Carbon leaning (65% carbon, 35% latency)",
    "balanced":        "Balanced  (50% carbon, 50% latency)",
    "latency-leaning": "Latency leaning (35% carbon, 65% latency)",
    "latency-first":   "Latency first  (20% carbon, 80% latency)",
}

# ── load regions once ─────────────────────────────────────────────────────────
@st.cache_resource
def get_regions():
    return load_regions("data/regions.csv")

regions = get_regions()

# ── helpers ───────────────────────────────────────────────────────────────────
def badge(ok, true_label="✅", false_label="❌", none_label="—"):
    if ok is None:
        return none_label
    return true_label if ok else false_label


def confidence_bar(conf):
    try:
        v = float(conf)
    except (TypeError, ValueError):
        return "—"
    filled = round(v * 10)
    bar = "█" * filled + "░" * (10 - filled)
    return f"{bar}  {v:.0%}"


def build_scenario(city, residency, max_rtt, weight_name):
    wc, wl, wp = WEIGHTS[weight_name]
    feas, best, acc, gap = decide(regions, city, residency, max_rtt, wc, wl, wp=wp)
    diff = "easy"
    if feas and best:
        from core import classify
        diff, binding = classify(regions, city, residency, max_rtt, wc, wl, wp, feas, best)
    else:
        binding = False
    return Scenario(
        sid="UI-001", city=city, residency=residency, max_rtt_ms=max_rtt,
        weights_name=weight_name, w_carbon=wc, w_latency=wl, w_price=wp,
        feasible=feas, best=best or "", acceptable=acc, gap=gap,
        difficulty=diff, binding=binding,
    )


def run_rung(fn, llm, scenario):
    t0 = time.time()
    result = fn(llm, regions, scenario)
    elapsed = time.time() - t0
    return result, elapsed


def check_faithfulness(result, scenario):
    return {
        "F1 Claim support":   claim_support(result, regions),
        "F2 Decisive factor": decisive_factor_correct(result, scenario, regions),
        "C1 Citation valid":  citation_valid(result, regions),
        "C2 Constraint OK":   constraint_report_correct(result, scenario),
    }


def region_card(region_id):
    """Return a short summary string for a region."""
    if region_id not in regions:
        return region_id or "—"
    r = regions[region_id]
    return f"{region_id}  ·  {r['location']} ({r['country']})  ·  {r['grid']} gCO₂/kWh"


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/51/Google_Cloud_logo.svg/512px-Google_Cloud_logo.svg.png",
             width=140)
    st.title("Cloud Region Selector")
    st.caption("Multi-agent LLM prototype · Dr. Zeyad research project")

    st.divider()
    st.subheader("⚙️ Model settings")
    base_url = st.text_input("Ollama base URL", value="http://localhost:11434/v1")
    model_name = st.text_input("Model name", value="llama3.1:8b")

    st.divider()
    st.subheader("📍 Scenario")

    city = st.selectbox("User location (city)", list(USER_SITES.keys()), index=0)

    residency = st.selectbox(
        "Data residency rule",
        list(RESIDENCY.keys()),
        format_func=lambda k: {
            "ANY": "ANY — no restriction",
            "EU":  "EU — must stay in EU",
            "UK":  "UK — must stay in UK",
            "US":  "US — must stay in US",
            "CA":  "CA — must stay in Canada",
            "APAC":"APAC — Asia-Pacific only",
        }[k],
        index=5,
    )

    latency_choice = st.selectbox(
        "Latency cap",
        [60, 120, None],
        format_func=lambda v: LATENCY_LABELS[v],
        index=2,
    )

    weight_name = st.selectbox(
        "Priority profile",
        list(WEIGHTS.keys()),
        format_func=lambda k: WEIGHT_LABELS[k],
        index=2,
    )

    st.divider()
    st.subheader("🔬 Rungs to compare")
    selected_rungs = st.multiselect(
        "Select rungs",
        list(RUNG_FUNCS.keys()),
        default=["R1 — LLM + region data (RAG)", "R3 — Selective escalation"],
    )

    run_button = st.button("▶  Run", type="primary", use_container_width=True)

    st.divider()
    st.caption("ℹ️ Ollama must be running.  Start it with `ollama serve` in a terminal.")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA — TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_run, tab_regions, tab_lara = st.tabs(["🔍 Run scenario", "🗺️ Region data", "📊 LARA assessment"])

# ── TAB 1: Run scenario ───────────────────────────────────────────────────────
with tab_run:
    scenario = build_scenario(city, residency, latency_choice, weight_name)

    # Scenario summary
    st.subheader("Scenario")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("City", city)
    col_b.metric("Residency rule", residency)
    col_c.metric("Latency cap", LATENCY_LABELS[latency_choice])
    col_d.metric("Priority", weight_name)

    with st.expander("Full scenario prompt sent to the model"):
        st.code(scenario.text(), language=None)

    # Ground truth
    st.subheader("Ground truth (computed, not from the model)")
    if scenario.best:
        gt_col1, gt_col2, gt_col3 = st.columns(3)
        gt_col1.metric("Best region", scenario.best)
        gt_col2.metric("Feasible regions", len(scenario.feasible))
        gt_col3.metric("Difficulty", scenario.difficulty.upper())
        st.caption(f"Acceptable regions (within 5% of best): {', '.join(scenario.acceptable)}")
        if scenario.binding:
            st.warning("⚠️ The unconstrained best region is ruled out by the residency or latency constraint.")
    else:
        st.error("No feasible regions for this combination. Try relaxing the residency rule or latency cap.")

    st.divider()

    # Run
    if run_button:
        if not selected_rungs:
            st.warning("Select at least one rung in the sidebar.")
        elif not scenario.best:
            st.error("No feasible regions — cannot run.")
        else:
            llm = OpenAICompatLLM(
                base_url=base_url,
                model=model_name,
                json_mode=False,
            )
            st.subheader("Results")
            all_results = {}

            progress = st.progress(0, text="Running…")
            for i, rung_label in enumerate(selected_rungs):
                fn = RUNG_FUNCS[rung_label]
                with st.spinner(f"Running {rung_label}…"):
                    try:
                        result, elapsed = run_rung(fn, llm, scenario)
                        faith = check_faithfulness(result, scenario)
                        all_results[rung_label] = (result, elapsed, faith)
                    except Exception as e:
                        st.error(f"{rung_label} failed: {e}")
                        all_results[rung_label] = None
                progress.progress((i + 1) / len(selected_rungs), text=f"Done: {rung_label}")
            progress.empty()

            # Show a card for each rung
            for rung_label, payload in all_results.items():
                st.markdown(f"### {rung_label}")
                if payload is None:
                    st.error("Failed — check that Ollama is running and the model name is correct.")
                    continue

                result, elapsed, faith = payload
                is_correct = result.region in scenario.acceptable if result.region else False

                r_col1, r_col2, r_col3, r_col4, r_col5 = st.columns([2, 2, 1, 1, 1])
                r_col1.metric(
                    "Chosen region",
                    result.region or "—",
                    delta="✅ Acceptable" if is_correct else "❌ Wrong",
                    delta_color="normal" if is_correct else "inverse",
                )
                r_col2.metric("Ground truth", scenario.best)
                r_col3.metric("Tokens", result.tok_in + result.tok_out)
                r_col4.metric("LLM calls", result.calls)
                r_col5.metric("Time", f"{elapsed:.0f}s")

                if result.escalated:
                    st.info("↑ Escalated to full multi-agent system")

                # Confidence
                conf = result.explanation.get("confidence")
                if conf is not None:
                    st.caption(f"Confidence: {confidence_bar(conf)}")

                # Explanation
                explanation_text = result.explanation.get("explanation", "—")
                decisive = result.explanation.get("decisive_factor", "—")
                st.markdown(f"**Explanation:** {explanation_text}")
                st.caption(f"Decisive factor named: **{decisive}**")

                # Faithfulness checks
                f_col1, f_col2, f_col3, f_col4 = st.columns(4)
                f_col1.metric("F1 Claim support",
                              f"{faith['F1 Claim support']:.0%}" if faith['F1 Claim support'] is not None else "—")
                f_col2.metric("F2 Decisive factor", badge(faith["F2 Decisive factor"]))
                f_col3.metric("C1 Citation valid",  badge(faith["C1 Citation valid"]))
                f_col4.metric("C2 Constraint OK",   badge(faith["C2 Constraint OK"]))

                # Claims detail
                claims = result.explanation.get("claims") or []
                if claims:
                    with st.expander("Factual claims made by the model"):
                        for c in claims:
                            reg = regions.get(c.get("region"), {})
                            field = c.get("field", "")
                            stated = c.get("value")
                            if field == "grid_gco2_kwh":
                                actual = reg.get("grid")
                                ok = actual is not None and abs(float(stated) - actual) <= 0.01 * actual + 0.5
                            elif field == "country":
                                actual = reg.get("country")
                                ok = stated == actual
                            else:
                                actual, ok = "—", None
                            icon = badge(ok)
                            st.write(f"{icon}  **{c.get('region')}** · {field}: stated `{stated}`, actual `{actual}`")

                st.divider()

            # Side-by-side comparison table
            if len(all_results) > 1:
                st.subheader("Comparison")
                import pandas as pd
                rows = []
                for rung_label, payload in all_results.items():
                    if payload is None:
                        continue
                    result, elapsed, faith = payload
                    rows.append({
                        "Rung": rung_label.split("—")[0].strip(),
                        "Chosen": result.region or "—",
                        "Correct": badge(result.region in scenario.acceptable if result.region else False),
                        "Tokens": result.tok_in + result.tok_out,
                        "Calls": result.calls,
                        "Escalated": badge(result.escalated, "Yes", "No"),
                        "Claim support": f"{faith['F1 Claim support']:.0%}" if faith['F1 Claim support'] is not None else "—",
                        "Decisive OK": badge(faith["F2 Decisive factor"]),
                        "Cite OK": badge(faith["C1 Citation valid"]),
                        "Time (s)": f"{elapsed:.0f}",
                    })
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    else:
        st.info("👈 Configure the scenario in the sidebar and click **Run**.")


# ── TAB 2: Region data ────────────────────────────────────────────────────────
with tab_regions:
    st.subheader("All 45 regions")
    st.caption("Carbon intensity data: Google region-carbon-info, 2025 (Apache-2.0). "
               "Lat/lon/country entered by hand — check before publishing.")

    import pandas as pd

    rows = []
    for rid, r in regions.items():
        rtt = get_rtt(city, r)
        rows.append({
            "Region": rid,
            "Location": r["location"],
            "Country": r["country"],
            "Carbon (gCO₂/kWh)": r["grid"],
            "CFE": f"{r['cfe']:.0%}" if r["cfe"] is not None else "—",
            f"RTT from {city} (ms)": f"{rtt:.0f}",
            "Feasible": "✅" if rid in scenario.feasible else "❌",
            "Best": "⭐" if rid == scenario.best else "",
        })

    df = pd.DataFrame(rows).sort_values("Carbon (gCO₂/kWh)")
    st.dataframe(df, width="stretch", hide_index=True)

    st.caption("RTT = distance proxy (great-circle at 2/3 speed of light). "
               "Drop data/latency.csv to use measured values.")


# ── TAB 3: LARA assessment ────────────────────────────────────────────────────
with tab_lara:
    st.subheader("LARA — LLM Agent Readiness Assessment")
    st.caption("Source: Li & Ye, arXiv:2605.16297, Apr 2026. Higher scores mean more human involvement; "
               "weights and thresholds are the paper's financial-services calibration, reused unchanged.")

    lara = CLOUD_REGION_LARA
    l_col1, l_col2, l_col3 = st.columns(3)
    l_col1.metric("LARA level", lara.level)
    l_col2.metric("Weighted score", f"{lara.weighted_score:.2f} / 5.00")
    l_col3.metric("Deployment mode", lara.deployment_mode)

    st.divider()

    dim_col1, dim_col2 = st.columns(2)
    with dim_col1:
        st.markdown("**Dimension scores (1 – 5, higher = harder for an agent)**")
        for label, v in zip(["D1 Cognitive complexity", "D2 Data dependency", "D3 Interaction diversity",
                             "D4 Compliance sensitivity (×1.5)", "D5 Innovation requirement"], lara.dims):
            st.progress(v / 5, text=f"{label}  {v:.1f}")

    with dim_col2:
        st.markdown("**Levels (paper, Table 4)**")
        st.markdown("""
| Level | Score | Deployment mode |
|------|-------|---------------|
| L1 | 1.0–2.0 | Full agent + 5% spot check |
| L2 | 2.1–3.0 | Agent drafts + human approval |
| L3 | 3.1–4.0 | Human-led + agent copilot |
| L4 | 4.1–5.0 | Fully human execution |
""")
        st.info("Floor rule: if D4 (compliance sensitivity) ≥ 4, the task is at least L3 "
                f"(weighted mean alone gives {lara.mean_level}).")

    st.divider()
    st.markdown("**What this means for the study**")
    st.markdown(NOTES)
