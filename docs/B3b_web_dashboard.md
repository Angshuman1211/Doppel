# Web Results Dashboard

**Phase:** Phase 3 — Integration & Evaluation (blueprint Weeks 8–11), on branch `track2-phase3-react-dashboard` (merged as PR #14). The Data section's per-column fidelity, the ICD-9 fix and the Upload section were added in Phase 4, on `track2-phase4-evaluation-report`.
**Owner:** Rayyan (Track 2 / Track B)
**Status:** Done for the five evaluated arms. The four arms added in Phase 3 (ADR-029) are not on the page until their evaluation finishes and the data is re-exported. The full evaluation behind the numbers is in [`B3_full_evaluation.md`](B3_full_evaluation.md).

---

## Objective

Show the evaluation's results to someone who has not read the result docs: which generator is realistic, useful and private, how sure we can be, and what the numbers do not say. The blueprint gives the results dashboard to Track 4, whose Streamlit app (`dashboard/`) handles uploaded files (ADR-030). This dashboard sits beside it. Its main job is to present the finished 20-seed evaluation, so every chart is a mean with its spread and every claim on the page traces to a result doc. Its Upload section also does everything the Streamlit app does, in the browser, so one site covers both.

---

## What was built

### 1. Seven sections

Trimmed on 2026-10-21 to what a reader needs: the attack-by-attack calibration charts, the coverage chart, the attribute-inference table, the frontier table, the sensitivity heatmap, the status matrix and the column and manifest tables moved off the page (the first five are in [`final_evaluation_result.md`](final_evaluation_result.md), the last two in the Upload section), taking the page from about 15,000 to 10,400 px.

| Section | What it shows |
|---|---|
| Overview | The headline number (the diffusion model's worst-case membership AUROC), four summary tiles and three findings, each with its evidence |
| Compare | Fidelity, utility and privacy for all nine arms as dot plots with ±1 sd whiskers, filtered by classifier and by attack |
| Trade-offs | A scatter over any two of the three axes, and one card on what the Pareto frontier does and does not say |
| Privacy | How many training patients an attacker identifies at 1% false alarms, and three reasons the attacks can be trusted |
| Data | The synthetic records themselves: a 10-row preview, per-column fidelity against the real training rows, and a real-versus-synthetic view of any one column |
| Upload | The Streamlit app's three tabs, in the browser: explore an uploaded dataset, compare it to the real one (contract check, fidelity, a memorization signal, per-column divergence, a column view), and rank uploaded evaluation reports on a Pareto frontier. See section 6 |
| Project | The four pipeline steps with their status, the dataset, how to read the results with care, the main documents, and six terms explained |

### 2. Data layer

| File | What it does |
|---|---|
| `web/scripts/export_data.py` | Reads `results/` and the calibration output and writes `web/public/data/dashboard.json` (about 36 KB, aggregates only). Reuses `evaluation.pareto` with the same seed and order as the command line, so the frontier matches [`full_pareto_result.md`](full_pareto_result.md) exactly |
| `web/scripts/export_datasets.py` | Writes one JSON per evaluated dataset (5 arms × 20 seeds) plus `real.json` to `web/public/data/datasets/`. Column fidelity is computed with `evaluation.fidelity_metrics` against the real train split, and checked to match each seed's result file exactly |

Every number on the page comes from those files. The only fixed text is `web/src/lib/content.js` (status matrix, glossary, limitations, doc links).

### 3. Charts and interaction

Plain SVG, no chart library: a dot plot, a scatter with whiskers on both axes, and a heatmap. Colour marks the generator family and shape marks the arm, so no chart depends on colour alone. Every chart has a "Show table" view, a tooltip on hover and keyboard focus, and a note saying which direction is better. The page follows the system's light or dark theme and has a toggle; `?theme=light` or `?theme=dark` forces one.

### 4. Verification

| Check | Result |
|---|---|
| Exported numbers against the result docs | Match [`full_evaluation_result.md`](full_evaluation_result.md), [`full_pareto_result.md`](full_pareto_result.md) and [`positive_control_result.md`](positive_control_result.md) |
| Data-section fidelity against `results/<arm>_seed42.json` | Identical mean JS and correlation gap for all five arms after the ICD-9 fix below |
| Column types against pandas | Same dtype, missing and distinct counts on 5 datasets checked |
| Rendering | Production build checked in headless Edge at 1280 px and 390 px, light and dark; label collisions, an overflowing heatmap and crushed mobile layouts were found this way and fixed |

### 5. A bug found while building it

The first Data-section export read the synthetic CSVs without pinning `icd9_primary` as text, so `0389` became `389`, a different code. It showed as two separate categories in the column view and made every arm's page fidelity differ slightly from its result file. This is the P-005 trap again, the same one the evaluation runner hit in Phase 3 (P-014). The export now reads with `dtype={"icd9_primary": str}`, and the page and the result files agree exactly.

### 6. Upload: the Streamlit app's features in the browser

`dashboard/app.py` was rewritten for ADR-030 into three tabs that work on uploaded files. The Upload section reproduces all three, so nothing needs Python or a server:

| Streamlit tab | In the Upload section |
|---|---|
| Dataset Explorer | Rows, columns and missing values, a 20-row preview (all rows on request), a column table with type, missing and distinct counts, and the uploaded manifest whose name matches |
| Compare to Real Data | The Stage 2 contract check with its exact messages; train, holdout or all rows as the reference; mean JS divergence and correlation gap from Track 2's metric; the share of rows matching a real patient against what real unseen rows reach; per-column divergence worst first; and summary statistics or category shares for any one column |
| Evaluation & Pareto Analysis | A table of every uploaded report, fidelity, utility and privacy per generator with spread over seeds, a utility-against-privacy scatter, the Pareto frontier with each generator marked, the single-seed warning, and every report's full metrics |

Files are read with the browser's File API and never sent anywhere. The logic is ported to JavaScript in `web/src/workbench/`: `csv.js` reads a CSV the way `pd.read_csv(..., dtype={"icd9_primary": str})` does, `fidelity.js` ports `evaluation.fidelity_metrics` and the app's own column rules and profile match, `contract.js` ports `generators.validate.check_contract` line for line, and `reports.js` ports the app's report loading and dominance check. Where the app uses Track 2's and Track 1's code when it runs from the repo, the page always uses the ports, so its numbers are the repo-mode ones.

**Verification.** A test runs the ports and the Python on the same inputs: 9 synthetic files (all nine arms at seed 42) against the train, holdout and whole real file, 27 deliberately broken files for the contract check, and 32 report files including broken ones. All 28,429 values agree to within 1e-9. Then the built page was driven in headless Edge: the project's files were uploaded into each drop zone, including all 180 result files, and 74 numbers read off the page matched the Python, at 1280 px and 390 px, light and dark, with no console errors. Three differences are known and harmless: Python prints the real file's path with `\` on Windows where the page prints `/`, as Python does on Linux; equal per-column divergences can come out in a different order, because numpy's sort leaves exact ties in no particular order; and a broken JSON report's error text is the browser's wording, not Python's.

**Two findings from the port.** pandas' default CSV float reader is not correctly rounded: `0.42000000000000004` in a file reads as `0.42`. A value one bit off can land in a different histogram bin, so a straightforward port disagreed with Python on a few lab columns in most of the nine files until pandas' `precise_xstrtod` was ported too. And Track 1's `check_contract` crashes with a `TypeError` when a numeric column holds text, instead of reporting it, which would put a traceback on the Streamlit page (P-021). The port reports the column's wrong type instead.

**A fourth colour.** Uploaded reports can include the diffusion arms, so the diffusion family got a fourth categorical colour (plum, `#8e3185` light and `#a14497` dark), and two marker shapes were added so all nine arms have a distinct colour and shape. The dataviz skill's palette validator was not available in this session, so the colour was chosen by searching OKLCH hues with an approximation of its checks: in both themes the worst colour-blind separation against blue, orange, aqua and the floor gray is 11.3 (target 8) and the normal-vision separation is at least 18.5 (floor 15). The same approximation flags two pairs in the existing palette (aqua against the floor gray, and blue against aqua under tritanopia in dark mode) that the real validator passed earlier, so it is stricter than the validator and its verdict on those is not final.

### Not done

- The four Phase 3 arms are not on the page yet. When their evaluation finishes, `export_datasets.py` needs their labels and both exports need rerunning.
- Hover tooltips, table toggles and keyboard focus were built but not checked in an interactive browser session, only through the rendered page.
- No automated tests in the repo. The parity test and the browser test above were run by hand from a scratch folder; the browser test was not wired into a script the team can rerun.
- The dataset tables are git-ignored, so a fresh clone shows a note in the Data section until `npm run export-data` is run.

---

## Commands

```bash
cd web
npm install
npm run export-data   # dashboard.json from results/, and the dataset tables from output/synthetic/eval/
npm run dev           # http://localhost:5173
npm run build         # static site in web/dist
```

---

## Key Decisions

**Why a second dashboard next to Streamlit?** The Streamlit app is Track 4's and takes uploaded files, which suits a live demo (ADR-030). The question the report needs answered is different: what did the full evaluation find. That needs the 20-seed aggregates, the calibration anchors and the sensitivity analysis together, laid out for a reader rather than an operator. The Streamlit app was left unchanged.

**Why port the Streamlit features instead of linking to the app?** So the whole demo is one static site that opens anywhere, with no Python environment to set up, which matters on this machine where a fresh `pip install streamlit` pulls a pandas build that Application Control blocks (P-017). Porting meant every metric had to be shown equal to the Python, hence the parity test.

**Why export to static JSON instead of reading `results/` at run time?** The page can then be built once and opened anywhere with no Python, and the export is the single place where numbers are computed. Reusing `evaluation.pareto` and `evaluation.fidelity_metrics` there means the page cannot drift from the command-line output.

**Why keep the dataset tables out of git?** They are synthetic rows from generators that are known to copy training rows (TVAE, and the 3000-step diffusion arm), so they are handled like `output/synthetic/`, which is git-ignored for the same reason. The charts use aggregates only.

**Why per-column fidelity on the Data page?** The Compare section shows one mean per arm. The per-column view shows where a generator is wrong (for TVAE, `icd9_primary` first), which is what the Streamlit app's real-data comparison offers, here for every evaluated file and with the evaluation's own metric code.

---

## Outputs

| Path | What it is |
|---|---|
| `web/` | The Vite and React app; run instructions in [`web/README.md`](../web/README.md) |
| `web/public/data/dashboard.json` | Aggregated results, committed |
| `web/public/data/datasets/` | Per-dataset tables, git-ignored, regenerated by `npm run export-data` |
