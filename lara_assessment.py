"""LARA (LLM Agent Readiness Assessment) applied to the cloud-region selection task.

Source: Li, M., Ye, X. "Task-Level AI Readiness Assessment for Business Process Management: The T-IPO Model and LARA
Matrix in Financial-Services IT Operations." arXiv:2605.16297, 16 Apr 2026. Definitions below follow the paper's
Table 3 (dimensions), Section 5.3 (formula and thresholds), Table 4 (levels) and Definition 2 (D4 floor rule).

Direction: every dimension is scored 1-5 where HIGHER = HARDER for an agent / more human involvement needed.
A LOW weighted score means HIGH readiness (L1). Nothing is inverted.

  D1 Cognitive Complexity   Bloom-anchored: 1 Remember, 2 Understand, 3 Apply, 4 Analyze, 5 Evaluate/Create   x1.0
  D2 Data Dependency        1 single structured source ... 5 real-time streaming + unstructured external data  x1.0
  D3 Interaction Diversity  1 fully independent ... 5 cross-organisational political negotiation              x1.0
  D4 Compliance Sensitivity 1 no compliance ... 5 maximum; errors may cause legal liability                    x1.5
  D5 Innovation Requirement 1 fully templated ... 5 breakthrough original solution required                    x1.0

  LARA = sum(Di * Wi) / sum(Wi), W = (1, 1, 1, 1.5, 1), sum W = 5.5, range 1.0-5.0.
  L1 [1.0, 2.0]  high substitution        full agent + 5% spot check
  L2 (2.0, 3.0]  assistive augmentation   agent drafts + human approval (100% review)
  L3 (3.0, 4.0]  selective assistance     human-led + agent copilot
  L4 (4.0, 5.0]  human-dominant           fully human execution
  Floor rule: D4 >= 4 -> at least L3, whatever the weighted mean.
  Boundary rule: within 0.15 of a threshold, classify into the higher human-involvement level.

LIMITS (say these in the paper):
  - The 1.5x weight and the thresholds were calibrated on Chinese financial-services IT tasks. The paper says other
    sectors need their own Delphi study and k-means re-estimation of the thresholds. We reuse them unchanged.
  - LARA decides how much HUMAN involvement a task needs. It does not choose between our rungs, and it cannot
    "validate" the ladder design. We use it only to position the task: the output is decision support for a person.
  - The scores below are one person's judgement. The paper used three raters (Fleiss' kappa). Have at least two
    people score the task independently and report agreement before using the result.
"""
from dataclasses import dataclass

W = (1.0, 1.0, 1.0, 1.5, 1.0)
LEVELS = [  # (upper bound, level, verdict, deployment mode)
    (2.0, "L1", "High substitution", "Full agent + 5% spot check"),
    (3.0, "L2", "Assistive augmentation", "Agent drafts + human approval (100% review)"),
    (4.0, "L3", "Selective assistance", "Human-led + agent copilot"),
    (5.0, "L4", "Human-dominant", "Fully human execution"),
]
BOUNDARY = 0.15


@dataclass
class LaraScore:
    d1_cognitive_complexity: float
    d2_data_dependency: float
    d3_interaction_diversity: float
    d4_compliance_sensitivity: float
    d5_innovation_requirement: float
    floor_rule: bool = True  # the paper: enable in regulated settings

    @property
    def dims(self):
        return (self.d1_cognitive_complexity, self.d2_data_dependency, self.d3_interaction_diversity,
                self.d4_compliance_sensitivity, self.d5_innovation_requirement)

    @property
    def weighted_score(self):
        return sum(d * w for d, w in zip(self.dims, W)) / sum(W)

    @property
    def mean_level(self):
        """Level from the weighted mean, with the conservative boundary upgrade."""
        s = self.weighted_score
        for i, (hi, lvl, _, _) in enumerate(LEVELS):
            if s <= hi:
                if i < len(LEVELS) - 1 and hi - s < BOUNDARY:
                    return LEVELS[i + 1][1]
                return lvl
        return "L4"

    @property
    def level(self):
        lvl = self.mean_level
        if self.floor_rule and self.d4_compliance_sensitivity >= 4 and lvl in ("L1", "L2"):
            return "L3"
        return lvl

    @property
    def deployment_mode(self):
        return next(m for _, l, _, m in LEVELS if l == self.level)

    def report(self, title="LARA assessment: cloud-region selection"):
        names = ["D1 Cognitive complexity", "D2 Data dependency", "D3 Interaction diversity",
                 "D4 Compliance sensitivity (x1.5)", "D5 Innovation requirement"]
        lines = [title, "=" * len(title)]
        lines += [f"  {n:34} {d:.1f}" for n, d in zip(names, self.dims)]
        lines += [f"  {'Weighted score':34} {self.weighted_score:.2f}",
                  f"  {'Level from weighted mean':34} {self.mean_level}",
                  f"  {'Level after D4 floor rule':34} {self.level}",
                  f"  {'Deployment mode (paper, Table 4)':34} {self.deployment_mode}"]
        return "\n".join(lines)


# Scores for cloud-region selection (judgement of one rater; see LIMITS).
CLOUD_REGION_LARA = LaraScore(
    d1_cognitive_complexity=4.0,
    # Bloom "Analyze": the hard scenarios need a trade-off between carbon, latency (and price) under hard rules.
    # Easy scenarios are closer to "Apply" (3); we score the task at its harder end.
    d2_data_dependency=2.0,
    # Several structured sources (carbon table, region list, price sheet, latency measurements), updated
    # periodically. Live price or latency feeds would raise this to 3.
    d3_interaction_diversity=2.0,
    # Usually needs sign-off from platform, security/compliance and finance teams in the same organisation.
    d4_compliance_sensitivity=4.0,
    # A wrong region can breach data-residency law (GDPR, national sovereignty rules): legal liability.
    # Not 5, because the hard rules can be checked by code before anything is deployed.
    d5_innovation_requirement=1.0,
    # Choosing among listed regions is a templated decision.
)

NOTES = (
    "The weighted mean alone gives L2 (agent drafts, a human approves every decision). D4 = 4 triggers the floor rule, "
    "giving L3 (human-led, agent as copilot). If D4 were scored 3, the result would be L2. In both cases a person "
    "reviews every recommendation, so the system's output is decision support and the explanation is what the "
    "reviewer acts on. This is why explanation faithfulness is a primary outcome in this study. LARA does not say "
    "which rung of the ladder to use."
)


def sensitivity():
    """How the level moves when each dimension is scored one point lower or higher."""
    base = list(CLOUD_REGION_LARA.dims)
    out = []
    for i in range(5):
        for step in (-1, 1):
            d = base.copy()
            d[i] = min(5.0, max(1.0, d[i] + step))
            if d[i] == base[i]:
                continue
            s = LaraScore(*d)
            out.append((f"D{i + 1} {step:+d}", s.weighted_score, s.level))
    return out


if __name__ == "__main__":
    print(CLOUD_REGION_LARA.report())
    print("\nSensitivity (one dimension changed by one point):")
    for name, sc, lvl in sensitivity():
        print(f"  {name:6} score {sc:.2f} -> {lvl}")
    print("\n" + NOTES)
