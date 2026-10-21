# Report Blueprint — Evaluation

**Phase:** Phase 4 — Finalization & Delivery
**Owner:** Rayyan (Track 2 / Track B), per the blueprint's Phase 4 split: "Write the Evaluation section of the report" and "prepare final metric tables and result plots for the Pareto frontier."
**Status:** Structure and verified numbers, sourced entirely from result docs already in the repo. Prose is written by hand from this, not generated. Matches the format of [`report_methodology_results_blueprint.md`](report_methodology_results_blueprint.md) (Track 1's Methodology/Results sections), so the two read as one report.

How to use this: each numbered item is one paragraph. The bullets are what that paragraph covers, in that order. Numbers in **bold** are verified and sourced; the source doc is named so every claim in the report can be traced back.

---

## Section E — Evaluation

**E1. What the evaluation section has to establish.**
- The report's claims rest on one thing: that fidelity, utility and privacy were scored the same way for every generator, so the comparison is fair.
- State the three axes up front and that each has its own paragraph below: statistical fidelity, downstream utility, and resistance to a membership/attribute-inference attack.
- One sentence on scope: this section describes the protocol and reports what it measured; Track 1's Results section (B1–B6) interprets what the numbers mean for the generators.

**E2. Fidelity: formulas and what they're run against.**
- Jensen–Shannon divergence per column (categorical and binary columns directly, continuous columns after 10-quantile binning), correlation preservation (mean absolute difference between real and synthetic Pearson matrices, upper triangle), and a per-column Kolmogorov–Smirnov pass fraction.
- Scored against the real **train** split, the data the generator actually saw — not the holdout, which is reserved for utility and privacy.
- State why binary columns are not binned: 10-quantile binning collapses a two-valued column into one bin and reads JSD 0 regardless of the real positive rate (P-007).
- Cite the floor: real train vs. real holdout alone scores mean JSD **0.099** and correlation diff **0.203** — sampling noise on 94 rows, not a generator failing.

**E3. Why fidelity and utility have no pass/fail threshold.**
- The protocol originally set absolute thresholds for both (correlation ≤ 0.10/0.20, TSTR within 0.10 of TRTR). Both were dropped on 2026-09-19, and the report should say why rather than silently reporting numbers that look like they cleared a bar.
- Correlation: the real-vs-real floor already reads **0.203**, so no generator could pass against the holdout at any threshold.
- Utility: seed-to-seed sd of TSTR AUROC is **0.15–0.22** on a 35-row holdout with 6 positives, larger than the proposed threshold, so pass/fail would be decided by which seed ran. Utility is reported descriptively, over 20 seeds, never as a single number.

**E4. Downstream utility protocol.**
- Train-on-synthetic, test-on-real (TSTR) against train-on-real, test-on-real (TRTR) as the ceiling; target `hospital_expire_flag` (in-hospital mortality), chosen over `readmit_30d` because the latter has only 2 holdout positives (E's footnote: full comparison in [`eval_protocol.md`](../eval_protocol.md) §1).
- Two classifiers (logistic regression, random forest), AUROC as the metric because of the ~31%/69% class split.
- TRTR ceiling, from the same runs: **0.578** (logistic regression), **0.737** (random forest).
- Every number is mean ± sd over **20 seeds** (42–61); state plainly that a single-seed AUROC is not reported anywhere in this project.

**E5. The membership-inference suite, and why one attack is not enough.**
- Shadow-model attack (8 shadow generators, leave-one-shadow-out logistic regression), run three ways: `numeric4` (4 numeric columns), `gower` (all 53 features), `icd9_codes` (Jaccard on code sets). The **worst of the three** is the reported privacy score.
- State the reason plainly: the 4-column attack alone reads **0.503–0.518** for every arm in the project, including the generator known to memorize, so a report that used it alone would call every generator safe.
- Calibration anchors from `evaluation/mia_calibration.py`: an exact-copy ceiling scores **1.000** on every attack; real holdout rows the generator never saw (the floor) score **0.489–0.503**. Every reported attack number sits between these two anchors.
- Ground the design in the literature: [`membership_inference_survey.md`](membership_inference_survey.md) checked 18 sources against their primary record (arXiv/publisher/PoPETs/PMC) and placed Doppel's attacks in the distance-to-closest-record family used by the SynthEHRella and Nature Communications EHR benchmarks. Five gaps against the literature were identified; three are closed (pooled TPR at low FPR, a per-record exposure report, a density check via `synthetic_coverage`), two are open (a density-based attack proper, a release-only attacker without shadow-model access).

**E6. The positive control: does the attack suite actually catch a generator that leaks?**
- A privacy score is meaningless until it is shown to catch a generator that is known, independently, to memorize. TVAE is that control: Track 1's sweep found it reproduces **52–71%** of its rows as near-copies.
- The suite catches it on two of three attacks: worst-case membership AUROC **0.680** against **0.637–0.651** for the other four arms (p < 0.0001 against each, Welch test, 20 seeds), and pooled TPR at 1% FPR **0.058** against a floor of **0.009** — about 6× the attack's own chance rate.
- The suite also shows where a single attack can mislead: the 4-column attack rates TVAE **0.503**, the *lowest* of all five arms, so a report using only that attack would call the memorizing generator the safest one.
- Mechanism, not just the number: TVAE's synthetic rows sit nearest to only **31 of 94** training rows (its output collapses), against **40–47** for the other four arms, and up to **24** synthetic rows crowd onto a single training row against 8–14 elsewhere.

**E7. A leakage channel that belongs to the approach, not to one model.**
- Codes seen only once in train leak through *every* generator, including CTGAN, which learned no other dependence: `codes_once` AUROC **0.821–0.839** against a floor of **0.481**.
- State the mechanism: a generator can only emit a code it saw in training, so a code that appears in exactly one training admission re-identifies that admission whenever the synthetic output contains it — a property of modeling discrete codes at all, not of any one architecture.
- Contrast with TVAE's own leakage, which is *not* through rare codes (per-record exposure Spearman **−0.011** against once-seen-code count, versus **0.20–0.28** for the other arms) — it leaks through ordinary rows instead, consistent with E6.

**E8. Attribute inference.**
- Design: an attacker trained on synthetic data predicts a withheld attribute from the rest of a row; the privacy-relevant number is the **member gap** (accuracy uplift on real train rows minus uplift on real holdout rows), read against `independent_marginals`, which has no column dependence to exploit.
- Targets: `gender`, `first_careunit`, `age_bucket` — the three attributes with class coverage in both splits. The original target, `ethnicity`, is superseded: one patient holds 15 of 35 holdout admissions, all one ethnicity category absent from train entirely, so the attacker predicted the majority class every run and the result was not scored (P-003).
- Headline number: TVAE's gender gap is **0.183 ± 0.126** (p = 0.0004 against the floor), about **4×** the no-dependence control's **0.042**, so the positive control shows up on this attack too.

**E9. The headline benchmark (Track 2's own harness, verified).**
- 5 arms × 20 seeds = 100 runs, every metric above, one environment (Python 3.11.9, numpy 2.4.6, torch 2.14.0+cpu, ctgan 0.12.1). Full table: [`full_evaluation_result.md`](full_evaluation_result.md).
- State the honest headline, verbatim from that doc: *"on 94 training rows the statistical baseline matches or beats both neural generators on fidelity and privacy, TVAE matches the real-data utility ceiling only by copying training rows, CTGAN produces data no more useful or more private than independent marginals, and no arm dominates on fidelity, utility and privacy together."*
- Table: mean JSD **0.0162–0.0165** for the floor and both copulas vs **0.0937–0.1063** for CTGAN/TVAE; TSTR (random forest) **0.480 → 0.738** across arms; worst-case membership **0.637–0.680**.

**E10. The Pareto frontier.**
- `evaluation/pareto.py` reduces each arm to (fidelity, utility, worst-case privacy), marks non-dominated arms, bootstraps frontier membership and pairwise dominance probability over seeds (2,000 resamples). Full evidence: [`full_pareto_result.md`](full_pareto_result.md).
- On the 5-arm benchmark, **all five arms are non-dominated** (bootstrap frequency **0.81–0.97**, no pairwise dominance probability above **0.16**), so the frontier shows trade-off directions, not a ranking.
- Validated under 8 axis choices (adding correlation, one classifier at a time, one attack at a time, dropping utility entirely) and checked against the TVAE positive control: without the utility axis, TVAE's frontier frequency falls from 0.86 to **0.00**, since every other arm beats it on both remaining axes — it holds its place through utility alone, and that utility is copied signal (E9).

**E11. Status: what is verified, and what is not yet folded in.**
- The 5-arm/100-run benchmark above (E9–E10) is Track 2's own, reproduced on this machine, and is the verified baseline for this report.
- Track 1's Phase 3 tuning (ADR-029) added four arms to `evaluation/arms.py` — a tuned copula, a tuned CTGAN, and two diffusion configs — and ran Track 2's unchanged harness to produce a 9-arm/180-run table, written up in [`final_comparison_result.md`](final_comparison_result.md). That doc is explicit that it is evidence for Track 2 to fold in, not a replacement for this section.
- Those 9-arm numbers are **not yet independently reproduced by Track 2** and are not reported as verified in this section. Headline figures from that run (diffusion privacy leak up to **0.787**, both diffusion arms exceeding the train-on-real utility ceiling with a badly wrong class prior — synthetic mortality **0.81** against real **0.36**) are real findings on record, but belong to Track 1's Results section until Track 2 reruns and incorporates them into `full_evaluation_result.md` and this one.
- Plan: rerun `evaluation.eval_runner` over the four new arms in Track 2's own environment, fold the output into `full_evaluation_result.md` and `full_pareto_result.md`, and update this section's E9–E10 numbers before final submission.

**E12. Final metric tables and result plots for the Pareto frontier.**
- Interactive version: the [React dashboard](../web/) (Compare, Trade-offs and Privacy sections) renders every table above as a chart with a table twin, filters, and tooltips, built from this evaluation's own JSON output.
- Static version, for the written report itself: the tables in E9–E10 and the raw `evaluation.summarize_results` / `evaluation.pareto` output in [`full_evaluation_result.md`](full_evaluation_result.md) and [`full_pareto_result.md`](full_pareto_result.md).
- Outstanding for Phase 4: a static exported figure (PNG/SVG) of the fidelity–utility–privacy scatter for the Word/PDF report, since the dashboard itself can't be embedded in a static document. [TODO: export once the arm rerun in E11 lands, so the figure isn't redone twice.]

---

## Verified numbers

Every figure above traces to one of: [`eval_protocol.md`](../eval_protocol.md) (protocol and formulas), [`full_evaluation_result.md`](full_evaluation_result.md) (5-arm benchmark), [`full_pareto_result.md`](full_pareto_result.md) (frontier), [`positive_control_result.md`](positive_control_result.md) (attack calibration and the TVAE control), [`membership_inference_survey.md`](membership_inference_survey.md) (literature grounding), [`problems_and_decisions.md`](problems_and_decisions.md) (P-003, P-007, ADR-029).

The 9-arm numbers quoted in E11 trace to [`final_comparison_result.md`](final_comparison_result.md) and are marked there, and here, as not yet reproduced by Track 2.

## Landmines

- Do not report a single-seed AUROC anywhere — every utility and privacy number is mean ± sd over 20 seeds.
- Do not call the 4-column membership attack "the privacy score" on its own; it misses the known positive control (E6).
- Do not present TVAE's TSTR AUROC (or either diffusion arm's) as evidence the generator "preserves utility" — in all three cases the number comes from copying or a distorted class prior, not learned signal.
- Do not state a pass/fail fidelity or correlation threshold; both were explicitly dropped (E3) because the real-vs-real floor already fails them.
- Do not present the Pareto frontier as ranking the generators — on every arm set evaluated so far, most or all arms are non-dominated.
- Do not cite the 9-arm `final_comparison_result.md` numbers as Track 2-verified until the E11 rerun happens; attribute them to Track 1's run explicitly if used before then.
