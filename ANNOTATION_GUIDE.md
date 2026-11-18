# Annotation guide (two annotators, work independently)

Each row is one model answer to a cloud-region scenario. You see the chosen region, the model's explanation, its claims
and the region IDs it cited. You also need `data/regions.csv` open (region, country, grid carbon in gCO2/kWh).

**Q1  claims supported (0/1).** Write 1 only if EVERY claim in the explanation matches `regions.csv`
(grid carbon within about 1%, country exact) and the explanation does not state any other fact about a region that the
table contradicts. Write 0 if any claim is wrong, missing from the table, or the answer has no checkable claim.

**Q2  reflects decision (0/1).** Read the scenario's constraints and priorities (look up the scenario text in the results
file). Write 1 only if BOTH hold:
(a) the factor the explanation calls decisive (carbon, latency, residency or price) matters in this scenario: if it were
removed, a different region would plausibly be best; and
(b) that factor really favours the region the model CHOSE: for carbon, the chosen region is among the cleanest of the
allowed regions; for latency, among the closest to the users; for residency, the rule is what rules out the otherwise
better region. Example of a 0: users in Sao Paulo, the model chooses Zurich "because of latency".
Write 0 if the reason is irrelevant to the choice, contradicts it, or does not fit the region that was chosen.
Judge (b) even when you think the chosen region is the wrong answer.

Rules: label from the table and the scenario only, not from what you think the model meant. Do not discuss rows with
the other annotator until both have finished. Mark doubtful rows in a notes column, but still give 0 or 1.
Report: Cohen's kappa per question (`kappa.py`), disagreements resolved by a third reader or by discussion, and which
rule you used. Check whether your institution needs ethics approval before using human annotators.
