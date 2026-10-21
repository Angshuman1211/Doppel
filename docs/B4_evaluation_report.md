# Evaluation Report and Final Comparison

**Phase:** Phase 4 — Finalization & Delivery (blueprint Weeks 12–14), on branch `track2-phase4-evaluation-report`
**Owner:** Rayyan (Track 2 / Track B)
**Status:** Done. The blueprint's two Track 2 tasks for Phase 4, the Evaluation section of the report and the final metric tables and Pareto plot, are written for all nine generator settings, after Track 2 reran the four arms Track 1 added in Phase 3.

---

## Objective

Finish Track 2's part of the report. That needed one thing first: the diffusion model and three tuned settings arrived after Track 2's Phase 3 evaluation, so the comparison covered two of the blueprint's three families. Track 1 ran them through Track 2's harness on his machine ([`final_comparison_result.md`](final_comparison_result.md)) and left the evaluation narrative to Track 2. This phase reran them here, checked the two runs against each other, and wrote the section, the tables and the figure from the result.

---

## What was built

### 1. The nine-arm evaluation

The four Phase 3 arms (`gaussian_copula_shrink010`, `ctgan_small`, `tabular_diffusion`, `tabular_diffusion_reg`) were run over seeds 42 to 61, 80 runs with no failures, alongside the 100 Phase 3 runs. Calibration and coverage were run for the new arms too. Results: [`final_evaluation_result.md`](final_evaluation_result.md).

| Check | Result |
|---|---|
| Are the 24 September and 21 October runs comparable? | Seed 42 of `independent_marginals`, `gaussian_copula` and `tvae` was rerun: all 190 numbers in each file identical, TVAE included across torch 2.14.0 and 2.12.0. Generator code for those arms and the cleaned dataset were unchanged |
| Does it agree with Track 1's run? | Same conclusions and the same privacy order; numbers close but not identical, even for the deterministic copulas. Two fragile frontier readings differ and are not used |
| Track 1's figure of 81% deaths for the regularized diffusion arm | Checked over all 20 seeds: 0.787 against a real 0.362 |

Findings that change the headline: the 3000-step diffusion model leaks more than any other arm (worst case 0.787; at 1% false alarms the attack finds 36% of its training patients); the regularized diffusion model passed Track 1's memorization guard and still leaks second most (0.706); TVAE generates 21% deaths against a real 36%, which no earlier doc had noticed; the tuned copula is the best trade-off; eight of nine arms are on the frontier.

### 2. The Evaluation section of the report

[`report_evaluation_section.md`](report_evaluation_section.md) is the prose, and [`report_evaluation_section_blueprint.md`](report_evaluation_section_blueprint.md) the paragraph-by-paragraph plan with every number sourced, in the same format as Track 1's Methodology and Results plan. Both cover the protocol, why the thresholds were dropped, the attack suite and its two positive controls, the leakage through codes seen once, attribute inference, the benchmark, why utility cannot rank the generators, the frontier, and what the section does not claim. Every claim was checked against the result files; three that did not hold were rewritten before commit.

### 3. Final metric tables and the Pareto figure

[`figures/pareto_frontier.png`](figures/pareto_frontier.png), drawn by `evaluation/plot_pareto.py` from `evaluation.pareto`'s own loader, so the figure cannot disagree with the tables. It uses the dashboard's colours and shapes, and labels only the points that stand clear, with the legend naming all nine. The tables are in [`final_evaluation_result.md`](final_evaluation_result.md).

### 4. The dashboard, for nine arms

The web dashboard now shows all nine arms in every section, with its sentences computed from the data rather than written for a fixed set of arms, and a fourth family colour for diffusion. Its Upload section does everything the Streamlit app does, in the browser, checked against the Python on 28,429 values. Details: [`B3b_web_dashboard.md`](B3b_web_dashboard.md).

### 5. Repairs found while checking Track 2's work from start to end

| Problem | Fix |
|---|---|
| A merge on 14 October dropped eight log entries (ADR-025 to 027, P-014 to 018) that seven docs cite | Restored from `02ce916`, in order, adding lines only |
| The same merge removed Track 2's links from the README | Restored, with the status line and roadmap updated |
| The Data section turned `0389` into `389` (the P-005 trap) | Fixed; its fidelity now matches the result files exactly |
| `generators.validate.check_contract` crashes on text in a numeric column | Logged as P-021 for Track 1; the browser port reports the column instead |
| Loose scripts and docs at the repo root | Moved into `preprocessing/` and `docs/`; every link updated (241, none broken), the Dockerfile copies from the new paths, and the container layout and Kubernetes Job are unchanged |

### Not done

- The parity and browser tests for the Upload section were run from a scratch folder, not committed as a test the team can rerun.
- Nothing was run in Docker or on Kubernetes; the Docker daemon was not running on this machine.
- The report itself is assembled by Track 4; this phase delivers Track 2's section, its tables and its figure.

---

## Commands

```bash
pip install -r generators/requirements-neural.txt
for arm in gaussian_copula_shrink010 ctgan_small tabular_diffusion tabular_diffusion_reg; do
  for seed in $(seq 42 61); do python -m evaluation.eval_runner --generator $arm --seed $seed; done
done
python -m evaluation.summarize_results
python -m evaluation.pareto
python -m evaluation.mia_calibration --workers 6 --arms gaussian_copula_shrink010 ctgan_small tabular_diffusion tabular_diffusion_reg \
       --out results/calibration/mia_calibration_phase3_arms.json
python -m evaluation.synthetic_coverage
python -m evaluation.plot_pareto        # docs/figures/pareto_frontier.png

cd web && npm run export-data && npm run build
```

---

## Key Decisions

**Why rerun the four arms instead of using Track 1's numbers?** The evaluation is Track 2's deliverable, and Track 1's doc says as much. Rerunning also tested whether the result travels between machines: it does in every conclusion, though not to the last digit.

**Why calibrate the new arms into a separate file instead of rerunning all eleven?** The shadow-fit cache is keyed by the generator source, which changed when the diffusion model was added, so a full rerun would refit CTGAN and TVAE for hours. The original arms were shown to reproduce exactly, so their calibration stands and the export merges the two files, refusing to if they overlap or used different training codes.

**Why does the report say utility cannot rank the generators?** The three highest scores belong to generators that copy or that generate the wrong share of deaths. Reporting utility as a ranking would call the leakiest generator the best.

**Why move Track 3's scripts?** You asked for the root to be tidied. The scripts read and write paths relative to where they are run, so moving them changes nothing in how they behave. The validator passes from the new folder and the manifest it generates is identical to the committed one. Their owners should know before their next branch touches them.

---

## Outputs

| Path | What it is |
|---|---|
| [`final_evaluation_result.md`](final_evaluation_result.md) | Nine-arm results, frontier, calibration of the new arms, coverage, mortality, and the comparison with Track 1's run |
| [`report_evaluation_section.md`](report_evaluation_section.md), [`report_evaluation_section_blueprint.md`](report_evaluation_section_blueprint.md) | The Evaluation section and its plan |
| [`figures/pareto_frontier.png`](figures/pareto_frontier.png), `evaluation/plot_pareto.py` | The report's Pareto figure and the script that draws it |
| `results/` | 180 result files and the two calibration files (git-ignored, regenerated by the commands above) |
