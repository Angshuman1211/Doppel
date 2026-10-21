# Report Blueprint — Evaluation

**Phase:** Phase 4 — Finalization & Delivery
**Owner:** Rayyan (Track 2 / Track B), per the blueprint's Phase 4 split: "Write the Evaluation section of the report" and "prepare final metric tables and result plots for the Pareto frontier."
**Status:** Structure and verified numbers for all nine arms (updated 2026-10-21 after Track 2's own run of the four Phase 3 arms), sourced entirely from result docs in the repo. The prose version is [`report_evaluation_section.md`](report_evaluation_section.md). Prose is written by hand from this, not generated. Matches the format of [`report_methodology_results_blueprint.md`](report_methodology_results_blueprint.md) (Track 1's Methodology/Results sections), so the two read as one report.

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
- Train-on-synthetic, test-on-real (TSTR) against train-on-real, test-on-real (TRTR) as the ceiling; target `hospital_expire_flag` (in-hospital mortality), chosen over `readmit_30d` because the latter has only 2 holdout positives (E's footnote: full comparison in [`eval_protocol.md`](eval_protocol.md) §1).
- Two classifiers (logistic regression, random forest), AUROC as the metric because of the ~31%/69% class split.
- TRTR ceiling, from the same runs: **0.578** (logistic regression), **0.737** (random forest).
- Every number is mean ± sd over **20 seeds** (42–61); state plainly that a single-seed AUROC is not reported anywhere in this project.

**E5. The membership-inference suite, and why one attack is not enough.**
- Shadow-model attack (8 shadow generators, leave-one-shadow-out logistic regression), run three ways: `numeric4` (4 numeric columns), `gower` (all 53 features), `icd9_codes` (Jaccard on code sets). The **worst of the three** is the reported privacy score.
- State the reason plainly: the 4-column attack reads **0.503–0.523** for eight of the nine arms, TVAE and the regularized diffusion model included, so a report that used it alone would call almost every generator safe.
- Calibration anchors from `evaluation/mia_calibration.py`: an exact-copy ceiling scores **1.000** on every attack; real holdout rows the generator never saw (the floor) score **0.489–0.503**. Every reported attack number sits between these two anchors.
- Ground the design in the literature: [`membership_inference_survey.md`](membership_inference_survey.md) checked 18 sources against their primary record and placed Doppel's attacks in the distance-to-closest-record family used by the SynthEHRella and Nature Communications EHR benchmarks. Five gaps against the literature were identified; three are closed (pooled TPR at low FPR, a per-record exposure report, a density check via `synthetic_coverage`), two are open (a density-based attack proper, a release-only attacker without shadow-model access).

**E6. Two positive controls: does the suite catch generators that leak?**
- A privacy score is meaningless until it catches generators known, independently, to memorize. There are now two, both identified by Track 1's in-train check before any attack ran: TVAE (52–71% near-copies) and the 3000-step diffusion model (80%).
- The suite ranks both above every non-copying arm: worst case **0.787** (diffusion) and **0.680** (TVAE) against **0.632–0.656** for the copulas, floor and CTGAN arms; p < 0.0001 against every other arm for each, except TVAE against the tuned copula (p = 0.0002).
- Low false-alarm view: at 1% FPR the Gower attack finds **36%** of the diffusion model's training patients, 21% of the regularized one's and 5.8% of TVAE's, against 1.0–4.3% for the rest and 1% chance.
- The guard is not sufficient: the regularized diffusion model passed Track 1's guard (8.4% near-copies) and still has the second-highest worst case, **0.706**, above TVAE (p < 0.0001).
- Mechanism: TVAE's output sits nearest to **31** of 94 training rows and the regularized diffusion model's to **25**, against 39–47 for non-copying arms; the 3000-step diffusion model spreads over **51** while reproducing them closely.

**E7. A leakage channel that belongs to the approach, not to one model.**
- Codes seen only once in train leak through *every* arm, both CTGAN configurations included: `codes_once` AUROC **0.821–0.853** against a floor of **0.481**.
- Mechanism: a generator can only emit a code it saw in training, so a code in exactly one training admission identifies it whenever the synthetic output contains it.
- The copying generators leak through ordinary records instead: per-record exposure against once-seen codes, Spearman **−0.53** (diffusion) and **−0.01** (TVAE), against 0.12–0.37 for the non-copying arms.

**E8. Attribute inference.**
- Design: an attacker trained on synthetic data predicts a withheld attribute from the rest of a row; the privacy-relevant number is the **member gap** (uplift on real train rows minus uplift on real holdout rows), read against `independent_marginals`.
- Targets: `gender`, `first_careunit`, `age_bucket`. The original target, `ethnicity`, is superseded: 15 of 35 holdout admissions belong to one category absent from train, so the attacker predicted the majority class every run (P-003).
- Headline: the 3000-step diffusion model has the largest gap on all three targets (gender **0.286**, first care unit 0.142, age group 0.221, each p < 0.0001); TVAE's gender gap **0.183** (p = 0.0004), about **4×** the floor's 0.042. 24 uncorrected tests.

**E9. The headline benchmark (Track 2's own harness, verified).**
- 9 arms × 20 seeds = 180 runs, three families: floor; copula at Ledoit-Wolf, 0.25 and 0.10 shrinkage; CTGAN original and tuned; TVAE; diffusion at 3000 steps and at the regularized, guard-passing setting. One environment: three of the first five arms were rerun on the second run date and reproduced exactly. Full table: [`final_evaluation_result.md`](final_evaluation_result.md).
- Honest headline, verbatim from that doc: *"across nine configurations of three families on 94 training rows, the statistical baseline is the only family that is faithful without copying; both neural families fail in one of two ways, by learning no dependence (CTGAN) or by reproducing training patients (TVAE and diffusion), and the diffusion model that copies leaks more than any other arm; utility cannot rank the arms, because the highest scores belong to arms that copy or that generate the wrong rate of deaths."*
- Numbers: mean JSD **0.016** for copulas and floor against 0.042–0.378 for neural arms; tuned copula best correlation (**0.102**), utility 0.686 (vs TVAE 0.704, p = 0.68), worst case 0.656.
- Utility trap: diffusion RF TSTR **0.810** and 0.793 above the 0.737 real-data ceiling, while generating **45%** and **79%** deaths against a real 36%; TVAE generates **21%** and still matches the ceiling.

**E10. The Pareto frontier.**
- `evaluation/pareto.py` reduces each arm to (fidelity, utility, worst-case privacy), marks non-dominated arms, bootstraps membership and pairwise dominance over seeds.
- **8 of 9** on the frontier; the original CTGAN is off it, dominated by the tuned CTGAN in 62% of resamples.
- Sensitivity: without the utility axis, TVAE and both diffusion settings fall to **0.00**; with a correlation axis, the tuned copula is on the frontier in every resample (1.00).

**E11. Agreement with Track 1's run, and what is not claimed.**
- Track 1 ran the same nine arms on another machine ([`final_comparison_result.md`](final_comparison_result.md)): every conclusion and the privacy order agree; numbers match closely but not exactly, even for the deterministic copulas; the cause was not identified.
- Two frontier readings differ between runs and are not used: TVAE's frequency (0.37 vs 0.59) and the tuned copula's frequency without utility (a 0.0001 JSD gap to the floor, p = 0.69).
- Limits: one 94-row dataset, a 35-row holdout with six deaths, seeds resample generators not patients, uncorrected p-values, neural configs tuned once on this data; an AUROC near 0.5 means the attacks found nothing, not that the data is safe.

**E12. Final metric tables and result plots for the Pareto frontier.**
- Interactive version: the [React dashboard](../web) renders the tables as charts with table views, filters and tooltips; its Upload section re-ranks any set of result files a reader supplies.
- Static version for the written report: Figure 1, [`figures/pareto_frontier.png`](figures/pareto_frontier.png) (nine arms, drawn by `evaluation/plot_pareto.py` from `evaluation.pareto`'s own loader), and the tables in [`final_evaluation_result.md`](final_evaluation_result.md).

---

## Verified numbers

Every figure above traces to one of: [`eval_protocol.md`](eval_protocol.md) (protocol and formulas), [`final_evaluation_result.md`](final_evaluation_result.md) (nine-arm benchmark, frontier, calibration of the new arms, coverage, mortality rates), [`full_evaluation_result.md`](full_evaluation_result.md) and [`full_pareto_result.md`](full_pareto_result.md) (the first five arms), [`positive_control_result.md`](positive_control_result.md) (attack calibration and the original arms' controls), [`membership_inference_survey.md`](membership_inference_survey.md) (literature grounding), [`problems_and_decisions.md`](problems_and_decisions.md) (P-003, P-007, ADR-029).

## Landmines

- Do not report a single-seed AUROC anywhere — every utility and privacy number is mean ± sd over 20 seeds.
- Do not call the 4-column membership attack "the privacy score" on its own; it misses both positive controls (E5, E6).
- Do not present TVAE's TSTR AUROC, or either diffusion arm's, as evidence the generator "preserves utility" — in all three cases the number comes from copying or a distorted rate of deaths, not learned signal (E9).
- Do not say any arm beats real data on utility; the diffusion arms' scores above the ceiling come from a wrong prior and a noisy 6-positive holdout.
- Do not present Track 1's in-train memorization guard as a privacy guarantee; the regularized diffusion model passed it and leaks second-most (E6).
- Do not state a pass/fail fidelity or correlation threshold; both were explicitly dropped (E3) because the real-vs-real floor already fails them.
- Do not present the Pareto frontier as ranking the generators — eight of nine arms are non-dominated.
