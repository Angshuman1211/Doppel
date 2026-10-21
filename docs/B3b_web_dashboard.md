# Web Results Dashboard

**Phase:** Phase 3 — Integration & Evaluation (blueprint Weeks 8–11), on branch `track2-phase3-react-dashboard` (merged as PR #14). The Data section's per-column fidelity and the ICD-9 fix were added in Phase 4, on `track2-phase4-evaluation-report`.
**Owner:** Rayyan (Track 2 / Track B)
**Status:** Done for the five evaluated arms. The four arms added in Phase 3 (ADR-029) are not on the page until their evaluation finishes and the data is re-exported. The full evaluation behind the numbers is in [`B3_full_evaluation.md`](B3_full_evaluation.md).

---

## Objective

Show the evaluation's results to someone who has not read the result docs: which generator is realistic, useful and private, how sure we can be, and what the numbers do not say. The blueprint gives the results dashboard to Track 4, whose Streamlit app (`dashboard/`) handles uploaded files (ADR-030). This dashboard sits beside it and does a different job: it presents the finished 20-seed evaluation, so every chart is a mean with its spread and every claim on the page traces to a result doc.

---

## What was built

### 1. Six sections

| Section | What it shows |
|---|---|
| Overview | The headline number (TVAE's worst-case membership AUROC), four summary tiles and three findings, each with its evidence |
| Compare | Fidelity, utility and privacy per arm as dot plots with ±1 sd whiskers, filtered by classifier and by attack |
| Trade-offs | A scatter over any two of the three axes, the frontier table with bootstrap frequencies, and the eight-way sensitivity heatmap |
| Privacy | Each attack against its ceiling and floor, TPR at 1% FPR, how many training rows each generator's output sits nearest to, the once-seen-code channel, and the attribute-inference member gaps with p-values |
| Data | The synthetic tables themselves: a 20-row preview, column types with distinct and missing counts, the generator manifest, per-column fidelity against the real training rows, and a real-versus-synthetic view of any one column |
| Project | Pipeline steps, a status matrix by track and phase, what the page does not show, and a plain-language glossary |

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

### Not done

- The four Phase 3 arms are not on the page yet. When their evaluation finishes, `export_datasets.py` needs their labels and both exports need rerunning.
- Hover tooltips, table toggles and keyboard focus were built but not checked in an interactive browser session, only through the rendered page.
- No automated tests. The checks above were run by hand.
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
