# Cloud-region selection: practical prototype (v1)

Decision type: which cloud region should host a workload. Sector: business, for enterprise cloud and GreenOps/FinOps teams (see PROPOSAL_DRAFT.md).
Question studied: does selective escalation keep decision quality close to an always-on multi-agent system at lower cost,
and does it keep explanations faithful to the source data?

## Dual-process architecture (Jev + LARA)
- `jev_router.py`: rung R6, following the article's production pattern: rule engine (hard constraints in code) ->
  System 1 typed choice (region ID + confidence, no text) -> gate (p < 0.65 escalates) -> System 2 (R2) ->
  separate explanation agent -> egress check (F1, F2-choice, C1, C2; failure sets `flagged` for human review).
  System 1 is a STAND-IN (a short LLM call, not calibrated): R6 tests the architecture, not Jev. Implement
  `RealJev.choice()` from TypeSafe AI's documentation if access is obtained. `--r6-no-prefilter` is the ablation.
- `lara_assessment.py`: LARA rubric (Li & Ye, arXiv:2605.16297) with the paper's dimensions, weights, thresholds and
  D4 floor rule. Higher scores = more human involvement. Result: 2.73 -> L2 by the mean, L3 after the floor rule
  (human-led, agent copilot). LARA sets the level of human oversight; it does not pick a rung.

## Requirements
Python 3.10 or later (the code uses `str | None` union syntax). No third-party packages are needed to run the pipeline or tests. Optional packages:
- `pip install codecarbon` — for `--codecarbon` energy measurement
- `pip install scipy` — for the Wilcoxon test in `stats.py`

See `requirements.txt` for pinned versions.

## What is real, what is a stand-in, what is missing

| Item | Status |
|---|---|
| Carbon intensity per region | Real. Google `region-carbon-info`, 2025 file (Apache-2.0), 45 regions. |
| Region lat/lon, country | Entered by hand, approximate. Check before publishing. |
| Latency | Distance proxy (a lower bound). Drop a measured `data/latency.csv` (city,region,rtt_ms) next to regions.csv and it is used automatically. NOT YET COLLECTED. |
| Price | Not included. Drop `data/prices.csv` (region,price) next to regions.csv and price joins the score and two extra priority profiles appear. NOT YET COLLECTED. |
| Residency | Rule-based on the region's country. |
| LLM results | One pilot only (`results_real.jsonl`: llama3.2:3b, 20 scenarios, R0–R3, made with an older scenario generator, so it cannot be re-scored). `llm.py` is a test double that peeks at the answer; its output only proves the pipeline runs. |
| Scope | Google Cloud regions only (the carbon source covers GCP). |

## Ground truth
Hard constraints first (residency, optional latency cap), then a min-max normalised weighted score over the feasible regions.
Regions within 0.05 of the best count as acceptable. Nine user cities, six residency rules, three latency caps, five priority profiles.
Difficulty: easy = the answer is also best on carbon alone or latency alone; hard = a compromise that is best on neither (and at least 4 feasible regions).
Samples are stratified (40% hard). `binding` marks scenarios where the unconstrained best region is ruled out by a constraint.

## Ladder (`ladder.py`)
R0 single LLM | R1 LLM + sources with citations | R2 always-on multi-agent | R3 selective escalation (parse failure, invalid ID,
constraint violation or confidence below 0.75) | R4 learned router (`train_router.py`, trained on separate scenarios, tested on unseen ones) | R5 R3 decision plus a separate explanation agent that cannot change the chosen region | R6 dual-process (`jev_router.py`, no router needed).
Every summary also prints the chance level (accept rate of a random feasible region; 0.20 on the 100-scenario set).

## Faithfulness and explanation checks (`faith.py`, no LLM verifier)
F1 claim support (a copying check; never report it alone) | F2 decisive factor = F2-ref (removing the factor changes the
reference answer) AND F2-choice (the factor favours the region the model chose) | F3 say-do consistency (`--f3 N`) |
C1 citation validity | C2 truthful constraint report | **SOUND** = acceptable decision AND all applicable checks pass
(the headline metric; a wrong decision cannot pass it). F2-choice and SOUND were added because, in the pilot, about half
the wrong answers passed F1, the old F2 and C1. `rescore.py` re-applies the current checks to a saved results file.
Human check: `make_annotation.py`, `ANNOTATION_GUIDE.md`, `kappa.py`.

## Statistics and energy
`stats.py`: bootstrap 95% intervals, exact McNemar, Wilcoxon (needs scipy), effect sizes. Planned comparisons: R3 vs R2 and R3 vs R1.
Energy: CodeCarbon >= 3.3.2, `--codecarbon` per rung (machine mode, so the Ollama server is included), `--measure-idle 30`
for idle power before and after, net energy above idle reported. On Windows CodeCarbon reads CPU energy through the
Windows Energy Meter Interface (RAPL counters) when available, GPU through NVML, and RAM is a constant estimate. If no
CPU counter is available it falls back to an estimate from load and TDP; the run records the method per rung
(`energy_cpu_method`) and prints a warning for estimates. Run `python idle_baseline.py` once to see which method your
machine gets. API models are reported by tokens and latency only.

## Run
```
python tests/test_all.py                      # 28 tests: ground truth, checks, router, explainer, LARA, R6, energy
python run_demo.py                            # pipeline check with the test double + LARA report (not a result)
python lara_assessment.py                     # LARA tier and recommended rung for this task
python run_real.py --model llama3.2:3b --n 20 --no-json-mode   # local model via Ollama (add --no-json-mode for small models)
python run_real.py --base-url https://<provider>/v1 --model <name> --key-env MY_KEY --n 100 --f3 20
python stats.py results_real.jsonl
```
Router (R4): `run_real.py --rungs R1 --start 0 --n 60 --out results_train.jsonl`, then `python train_router.py results_train.jsonl`,
then `run_real.py --rungs R1,R2,R3,R4,R5,R6 --start 60 --n 40 --router router.json`. R6 alone needs no router.
