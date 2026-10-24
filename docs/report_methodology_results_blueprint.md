# Report Blueprint — Methodology and Results

**Phase:** Phase 4 — Finalization & Delivery
**Owner:** Angshuman (Track 1 / Track A), leading these two sections per the blueprint's Phase 4 split
**Status:** Structure and verified numbers. Prose is written by hand from this, not generated.

How to use this: each numbered item is one paragraph. The bullets are what that paragraph covers, in that order. Numbers in **bold** are verified and sourced; the source doc is named so every claim in the report can be traced back.

---

## Section A — Methodology

**A1. The problem the methodology has to solve.**
- Published evaluations of synthetic EHR generators are not comparable: different datasets, attack models, downstream tasks.
- So the design fixes one dataset, one protocol, one environment, and varies only the generator.
- State the consequence up front: this is a comparability study, not an attempt to invent an architecture.

**A2. Dataset and split.**
- MIMIC-III Clinical Database Demo v1.4, 100 patients, **129 admissions × 56 columns**, one row per hospital admission.
- Split **94 train / 35 holdout**, at patient level, not admission level, to stop a patient appearing on both sides (ADR-001).
- Generators see train only. The holdout is reserved for evaluation, which is what makes the privacy and utility numbers meaningful.
- Flag the known skew: one patient (`subject_id` 41976) holds **15 of the 35 holdout admissions**, all male, all Puerto Rican, and contributes **0 of the 6 holdout deaths**. Cite Track 3's finding, and whichever convention the team settles on for the clean comparison set.

**A3. Shared encoding, and why it exists.**
- Every generator trains on one shared 110-column frame built by a single codec, and is decoded by the same code.
- The reason is the study's claim: if each generator preprocessed its own way, a fidelity gap could come from the encoding rather than the model.
- Name the frame's composition: 5 categorical, 61 binary (3 labels, 3 missingness flags, 55 ICD-9 multi-hot), 2 integer, 42 continuous.
- State the constraint that drives every result: **110 modeled columns against 94 training rows**.

**A4. How diagnosis codes are represented.**
- The primary code is categorical. Secondary codes appearing in at least 5 training admissions become **55 multi-hot columns, covering 49% of secondary mentions**.
- The remaining **395 codes** form a tail pool: only the per-admission count is modeled, and codes are drawn by training frequency on decode, independently of the row.
- Give the reason in privacy terms: individual rare codes stay realistic, rare *combinations*, which re-identify a patient, are never reproduced.
- `n_diagnoses` is derived from the decoded list, so the count cannot contradict the list.

**A5. The four generator families.**
- One sentence each: independent marginals (floor), Gaussian copula with shrinkage, CTGAN and TVAE from the `ctgan` library, TabDDPM-style diffusion with Gaussian plus multinomial processes sharing one denoiser.
- Say which are the blueprint's three families and which is the reference floor, so the reader doesn't count four.
- Note the copula needs shrinkage because at d > n the correlation matrix is singular.

**A6. The output contract, and why generation is validated before evaluation.**
- Seven invariants, enforced before any file is written, so a failing run leaves nothing for the next stage.
- Highlight the two that carry weight: IDs never collide with real ones, and **every category and code must have been seen in train**, which makes the validator a holdout-leakage tripwire.
- Mention the manifest: seed, hyperparameters, SHA-256 of the training data, git commit, so every result traces to the run that made it.

**A7. Choosing hyperparameters without touching the holdout.**
- The holdout is the test set, so tuning on it would leak the benchmark.
- Selection runs inside train: 5 patient-level folds, fidelity against the rows fitted, memorization against the unseen fifth.
- Define the two memorization measures in one line each (DCR ratio, near-copy rate) and give the calibration: real unseen rows score **1.000 / 0.059**, exact copies score **0.000 / 1.000**.
- State the guard: a config counts only at `dcr_ratio ≥ 0.95` and `near_copy_rate ≤ 0.10`.
- Say explicitly why fidelity alone is not enough: it is maximized by copying.

**A8. Evaluation protocol (Track 2's, summarized, cite their section).**
- Fidelity: JS divergence per column plus correlation preservation.
- Utility: train on synthetic, test on real holdout, AUROC for in-hospital mortality, against a train-on-real ceiling.
- Privacy: membership inference as the strongest of several attacks, read against an exact-copy ceiling and a never-seen floor, plus attribute inference.
- One line on why everything is reported over **20 seeds**: single-seed utility on this holdout varies with sd ≈ 0.2 AUROC, enough for a signal-free generator to beat the real-data ceiling once.

**A9. Reproducibility and environment.**
- Same seed gives byte-identical output; verified across two Python and numpy versions.
- Pinned dependency versions, one image per stage, Kubernetes Jobs sharing one volume.
- Worth one sentence for the report's honesty: a reproducibility bug was found by a second machine disagreeing on a seed, traced to eigendecomposition of a matrix with repeated eigenvalues, and fixed by factorizing differently (P-008, P-009, P-013).

---

## Section B — Results

**B1. Open with the headline, not the setup.**
- On 94 training rows, the classical baseline is the only arm that is both faithful and non-copying; each neural family fails in one of two ways.
- Say the scope limit in the same breath: this is a small-data finding, not a verdict on the architectures.

**B2. Fidelity and memorization inside train (selection results).**
- Table from [`phase3_tuning_result.md`](phase3_tuning_result.md), best configuration per family.
- Copula at shrinkage 0.10: correlation gap **0.111**, near-copy **0.035**, the best correlation of any arm.
- CTGAN tuned to 64×64: mean JSD **0.103 → 0.076**, correlation stuck at **0.182** against the floor's **0.175**, so no dependence learned at any size or length tested.
- TVAE: fails at every setting, **54% to 95%** near-copies, and gets worse with less training.
- Diffusion: best neural fidelity in the project, mean JSD **0.036**, while **80.4%** of rows are near-copies; the only guard-passing setting scores **0.374**, worse than independent sampling.
- One line on effort, so the negative is credible: **390 fits across two grids**, 14 diffusion settings including dropout, width and weight decay.

**B3. The benchmark against the holdout (Track 2's harness, 9 arms × 20 seeds = 180 runs).**
- Full table from [`final_comparison_result.md`](final_comparison_result.md). Read in this order: fidelity, correlation, utility, privacy, then the frontier.
- Fidelity: the copula family is an order of magnitude ahead of every neural arm, **0.0164** against 0.042 to 0.377.
- Correlation: only the copula beats the independent floor (**0.1021** at λ = 0.10 against **0.1629**). Both CTGAN arms score *above* the floor, so no dependence was learned at either network size.
- Utility: on this split it rises with correlation inside the copula family, **0.487 → 0.678**. **Do not claim this as a finding.** Across 10 redrawn patient-level splits the same comparison is exactly zero (−0.000, p = 1.00), so it belongs to this split and not to the method ([`research/split_study_result.md`](../research/split_study_result.md)).
- Privacy: the same ordering on all three attacks, `tabular_diffusion` **0.787** > `tabular_diffusion_reg` **0.706** > `tvae` **0.681** > copula **0.646–0.657** > floor **0.640** > CTGAN **0.634**.
- State the attack's floor explicitly: a generator with no dependence at all scores **0.640**, so the diffusion leak is 0.787 against 0.64, not against 0.5.
- TVAE is the only arm dominated outright (frontier frequency **0.37**), and its utility comes from copying, so it must never be read as "TVAE preserves utility".

**B3a. Why the frontier needs reading carefully.**
- Eight of nine arms sit on the frontier, which on its own says very little.
- Both diffusion arms hold their place **entirely through the utility axis**: remove it and their frontier frequency goes from ~0.98 to **0.00**.
- Add the correlation axis instead and the tuned copula is on the frontier in **every** bootstrap sample (1.00).
- The lesson for the report: a three-axis frontier inherits the weakness of its weakest axis, and here that axis is utility.

**B3b. The utility axis is distorted, and the report should say how.**
- `tabular_diffusion_reg` posts the project's best utility (**0.775**) while generating data in which **81% of patients die**, against **36%** in real training data, and while scoring the worst fidelity of any arm (**0.377**).
- Explain the mechanism in one line: AUROC is rank-based, so it tolerates a badly wrong class prior, which means a generator can win on utility while being clinically unusable for prevalence, calibration or absolute risk.
- Both diffusion arms also exceed the train-on-real ceiling (**0.737** for random forest). On a holdout with 6 positives this says the ceiling is noisy, not that synthetic beats real.

**B3c. The cheap in-train check predicted the expensive holdout one.**
- Ordering by in-train near-copy rate matches the membership-attack ordering on four of five tuned arms.
- The exception is the honest part: `tabular_diffusion_reg` passes the in-train guard at 0.084 and still carries the second-highest membership risk at 0.706.
- Conclusion to state plainly: the guard is necessary but not sufficient, and it is a selection tool, not a privacy guarantee.

**B4. What the failures look like in the data, not just in metrics.**
- CTGAN: the damage is in continuous columns, hospital stay median **11.9 days against 6.7 real**, while categoricals are reasonable (47 of 65 primary codes present).
- TVAE: mode collapse onto a subset of patients, **20 of 65** primary codes, **34%** of rows reproducing a real patient's key fields.
- Diffusion: the same copying signature, **41.5%** of rows matching a real patient's categorical profile against **5.7%** for real unseen patients.
- This paragraph is what turns "the metric says so" into "here is the mechanism".

**B5. The trade-off the project set out to measure.**
- Fidelity and privacy move against each other across the whole sweep: every configuration that stopped copying lost fidelity, in all three neural families.
- The copula is the exception, and the reason is that it never had the capacity to memorize in the first place.
- This is the paper-shaped finding: at this sample size, flexible generators buy fidelity with memorization.

**B5a. The holdout's single-patient skew, and what was done about it.**
- One patient holds **15 of the 35** holdout rows, all male, all Puerto Rican, and none of the 6 deaths (Track 3's finding).
- Every arm was rerun against the 20-row holdout without them. Utility shifts up **+0.08 to +0.10 for every arm**, the top two are unchanged, and the spread narrows, so no conclusion depends on the choice.
- Attribute inference is the one metric that was genuinely corrupted: on the full holdout the attacker scores **below chance on non-members** (−0.063 to −0.073 on gender), which inflates the member gap by roughly 0.05 to 0.08. Those figures come from the clean holdout instead (ADR-032).
- Worth one line on why this is reported rather than quietly fixed: the artifact was found, measured both ways, and the choice stated.

**B5b. How much of this depends on one split.**
- The benchmark fixes one 80/20 patient-level split and varies only seeds. Redrawing the split 10 times shows holdout size ranges **22 to 40 rows** and deaths **5 to 10**, against the reported 35 and 6.
- A large share of the variance in utility comes from the split rather than the generator (split share **0% to 62%** by arm, highest for the neural arms that copy; corrected 2026-10-24), so a ± over seeds understates the real spread.
- Paired across splits, the structural results hold and get stronger: the copula beats the default and the floor on correlation in **10 of 10** splits (p = 0.002), CTGAN is worse than the floor on correlation in **10 of 10**, and the copying arms beat the tuned copula on utility in 9 and 10 splits.
- One reported gain does not survive, and saying so is the honest version of this section (see B3's note).

**B6. Threats to validity, stated by us rather than by the examiner.**
- 94 rows, 110 modeled columns, 35 holdout rows of which 15 are one patient, 6 positives for the utility task.
- Selection numbers are in-sample by construction; the holdout numbers are the benchmark.
- Single-seed numbers are meaningless here, hence 20 seeds everywhere.
- Privacy is attack-relative: a weak attack reads as privacy, which is why a known-memorizing arm is kept as a positive control.
- Rare code combinations are deliberately not modeled, so rare-code fidelity is understated by design.

---

## Verified numbers

Every figure above traces to one of: [`phase3_tuning_result.md`](phase3_tuning_result.md) (selection), [`sweep_result.md`](sweep_result.md) (Phase 2 sweep), [`baseline_generator_result.md`](baseline_generator_result.md) (baseline), [`full_evaluation_result.md`](full_evaluation_result.md) and [`full_pareto_result.md`](full_pareto_result.md) (Track 2 benchmark), [`A4_generator_architecture.md`](A4_generator_architecture.md) (architecture), [`problems_and_decisions.md`](problems_and_decisions.md) (decisions).

The B3 benchmark numbers come from [`final_comparison_result.md`](final_comparison_result.md): 9 arms × 20 seeds, all run in one environment so the table is internally comparable. It supersedes the five-arm numbers in [`full_evaluation_result.md`](full_evaluation_result.md), which predate the diffusion family and the tuned configurations.

## Landmines

- Do not write "CTGAN and diffusion do not work". Write that they do not work **at n = 94**, and say what would change with more data.
- Do not quote a single-seed AUROC anywhere.
- Do not present TVAE's utility as a success.
- Do not call the in-train selection numbers "results of the benchmark". They choose configurations; the holdout evaluation measures them.
- Do not claim the pipeline runs end to end unattended. Jobs are launched individually and orchestration is manual (P-020).
