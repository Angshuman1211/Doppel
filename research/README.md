# `research/` — post-course extension studies

Work done after the course project was graded and merged, to turn the benchmark into a paper. Nothing here is part of the graded pipeline: these scripts import `generators/` and `evaluation/` but never modify them, so the 180-run benchmark in [`docs/final_comparison_result.md`](../docs/final_comparison_result.md) stays exactly as submitted. They follow MD Rayyan's seven-item improvement plan.

Outputs land in `research/output/` and third-party data in `research/data/`, both gitignored: every number here regenerates from a seed (ADR-008). Each study has a `*_result.md` beside it holding the numbers, the method and the limits — those are the durable artefacts, the CSVs are not.

| Item | Study | What it answers |
|---|---|---|
| 1 | [`size_sweep.py`](size_sweep.py) · [result](size_sweep_result.md) | Does memorization track cohort size? 50 → 5,000 patients on UCI Diabetes 130-US. |
| 1b | [`size_attack.py`](size_attack.py) · [result](size_attack_result.md) | The same question measured with the shadow membership attack instead of the near-copy rate (added 2026-10-25 after review). |
| 1c | [`uci_coverage.py`](uci_coverage.py) · [result](size_attack_result.md#coverage-the-mode-collapse-explanation-measured-added-2026-10-25) | How synthetic rows sit on the training rows by cohort size; measures the TVAE mode collapse. |
| 2b | [`split_attack.py`](split_attack.py) · [result](split_attack_result.md) | The membership attack across 10 redrawn patient splits, so the privacy ordering is tested over patients rather than seeds. |
| 2 | [`split_study.py`](split_study.py) · [result](split_study_result.md) | How much of the benchmark's spread is the patient split rather than the generator? |
| 3 | [`domias_attack.py`](domias_attack.py) · [result](domias_attack_result.md) | A density-ratio (DOMIAS) and a release-only membership attack, which the paper's limitations name but never ran. |
| 4 | [`dp_study.py`](dp_study.py) · [result](dp_study_result.md) | What does a formal differential-privacy guarantee cost at n = 94? Uses [`generators/dp_copula.py`](../generators/dp_copula.py). |
| 5 | [`stats_report.py`](stats_report.py) · [result](stats_report_result.md) | Confidence intervals over **patients** instead of seeds, and Holm correction over the family of comparisons. |
| 6 | [`task_study.py`](task_study.py) · [result](task_study_result.md) | Does the utility ranking depend on which outcome is predicted? |
| 7 | [`scale_study_result.md`](scale_study_result.md) | Scaling, via `split_study.py --splits 30`. The full grid is not the bottleneck (machines are), so what got scaled is the split count — and it reversed item 5's utility nulls. |

[`tabular_codec.py`](tabular_codec.py) is shared infrastructure, not a study: a schema-agnostic replacement for `generators/codec.py` so the generators can run on a dataset that is not MIMIC. The generators themselves needed no changes to accept it.

## Running them

All of them take `--workers` and write incrementally where a run is long. Times are for 6 workers on one laptop.

```bash
python -m research.size_sweep  --sizes 50 100 250 500 1000 2500 5000 --seeds 2   # 70 fits
python -m research.split_study --splits 30 --seeds 3                             # 810 fits, ~53 min
python -m research.domias_attack --seeds 10                                      # 60 runs, 96 s
python -m research.domias_attack --seeds 10 --n-synth 47                         # matched to the shadow attack, 206 s
python -m research.dp_study                                                      # DP arms at two budgets
python -m research.stats_report                                                   # 180 fits, 882 s
python -m research.stats_report --from-cache                                      # re-aggregate, no refitting
python -m research.task_study --mimic                                             # the MIMIC task census
python -m research.task_study --sizes 250 1000 --seeds 3                          # 30 fits, 400 s
```

`size_sweep` and `task_study` need the UCI data at `research/data/uci_diabetes.csv`, fetched with `ucimlrepo` id 296 — the static `archive.ics.uci.edu` zip URL now 404s. `stats_report` reads `research/output/split_study.csv` for its Wilcoxon tests and skips them if the split study has not been run.

## What the extension changed about the paper's claims

Short version; each result doc has the evidence and the caveats.

- **Killed**: the copula's utility gain from lower shrinkage (item 2 — exactly zero across splits, a single-split artefact).
- **Scoped**: "utility rewards memorization" holds at n = 94 and fails at n = 1,446 (items 1 and 6). Diffusion's memorization is a cohort-size effect on both the near-copy rate and the membership attack. TVAE keeps a high near-copy rate at every size, but its attack AUROC falls to near the independent setting from 362 rows on, because its rows collapse onto 10 to 23% of the training rows (items 1b and 1c).
- **Weakened**: CTGAN's tuning gain is not significant on the report's utility metric, and no utility margin should be quoted as a number — it moves by 23× across prediction tasks, about 6× without the near-lookup emergency task (item 6) and by a third to a half as splits are added (item 7).
- **Strengthened**: the fidelity and correlation orderings win 30 splits of 30 (item 7); the tuned copula turns out to be at the *measurement floor* of this cohort (item 5); the utility ordering does survive correction once there are enough splits to test it (item 7); and a release-only attacker, run at the same release size as the shadow attack, gives about the same leakage figures as the paper (item 3).
- **Corrected after review**: item 3 first compared a 500-row release against the shadow attack's 47-row releases and concluded the paper's leakage figures were conservative. At matched size the difference is small and mixed, so that claim was withdrawn ([item 3 doc](domias_attack_result.md)). Item 4's first DP generator was not differentially private (ranges and levels read from the data, rare codes decoded from the training pool, correlation noise calibrated per entry); the rebuilt one is, and it is also better on fidelity and brings the membership attack to chance at ε = 1 ([item 4 doc](dp_study_result.md)). Item 5's "± is up to 18× too narrow" compared against the seed standard error rather than the printed standard deviation; against what is printed, fidelity is 1.5 to 4× too narrow and utility is already conservative ([item 5 doc](stats_report_result.md)).
- **Added**: three DP baselines with a real guarantee (item 4). At ε = 8 they cost about 12× in marginal fidelity and keep every run usable, and the ICD-9 code-set attack, which carries the benchmark's leak, scores them at chance (0.50 to 0.51 against 0.64 for the non-private independent setting). Their utility stays at or near chance.
- **Corrected mid-flight**: item 5 concluded that no utility comparison was established, from 10 splits. Item 7 showed that was a power limit — four of six survive at 30. The [item 5 doc](stats_report_result.md) keeps the old column deliberately, because the contrast is the lesson.
