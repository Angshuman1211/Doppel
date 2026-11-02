# Patient-Level DP — the guarantee can be upgraded for 4 to 7% of the fidelity it already costs (2026-10-26, corrected 2026-11-02)

Journal-version work (Track 1 task 5). Every DP arm in this project guarantees (ε, δ)-DP at the level of **one admission**. Reviewers asked for patient-level privacy, which is the guarantee that actually matters: a patient with three admissions is protected three times less by a per-row guarantee, and it is the patient, not the row, who is re-identified.

The task offered two routes — "cap admissions per patient, or count the group-privacy cost". Both are done, because counting the group-privacy cost is what shows why capping is the right answer.

```bash
python -m research.dp_study --seeds 20 --workers 3                     # per-admission, 94 rows
python -m research.dp_study --seeds 20 --workers 4 --cap 1 --cap-seed 0 \
    --out patient_dp_study --arms dp_{marginals,copula,bayesnet}_eps{1,8}       # per-patient, 80 rows, 26 s
# and --cap-seed 1 to 4 with --out patient_dp_study_draw1 ... draw4, for the sensitivity to the kept admission
```

---

## 1. Group privacy is not an option at ε = 8

A mechanism that is (ε, δ)-DP for a one-row change is (kε, δ·(e^{kε}−1)/(e^ε−1))-DP for a k-row change. In the **training split** the most admissions any patient has is **3** — not the 15 quoted in `dp_copula.py`'s docstring, which is subject 41976 and sits entirely in the *holdout*, so the generator never sees it. That makes the group-privacy cost smaller than the module claims, and still unusable:

| Stated per-admission | Implied per-patient at k = 3 |
|---|---|
| (1, 1e-5) | (3, 1.1e-4) |
| (8, 1e-5) | (24, **88.9**) |

At ε = 8 the implied δ exceeds 1, so the statement is **vacuous** — it is not a weak guarantee, it is no guarantee. At ε = 1 it is real but three times weaker in ε, and the arm would have to be advertised as ε = 3.

Worth noting the structural point: *k must be public*. Reading the real maximum off the data to claim a tighter group-privacy factor would itself leak, which is the same class of error that made the first DP implementation not private. Capping makes k public by construction.

## 2. Capping costs 15% of the rows

Keeping at most one admission per patient takes the training cohort from **94 rows to 80**, one per patient, and from 34 deaths to between 29 and 33 depending on which admission is kept. Each patient is then exactly one row, so the mechanisms' existing bounded (replace-one-row) guarantee *is* a replace-one-patient guarantee, at the same ε and the same δ, with no extra noise and no change to any mechanism.

Each patient's kept admission is drawn from a generator seeded by that patient alone. The first version drew from one shared stream and only for patients with more than one admission, so changing one patient could change the kept rows of later patients, and replacing a patient was then not replacing one row. That broke the argument above and was corrected (Track 2 review, 2026-11-02). Every number below comes from the corrected code.

Only 12 of the 80 patients have more than one admission, and the maximum is 3. That is why the cap is cheap here, and it is a property of this cohort rather than of the method — see the limits.

## 3. What the upgrade actually costs

Paired on (arm, seed) against the per-admission study, for the three mechanisms other than AIM at both budgets (120 pairs per draw), over five draws of the kept admissions:

| Metric | Patient-level minus admission-level: mean over draws (range) | Wilcoxon p, draw 0 |
|---|---|---|
| Mean JSD | **+0.0165** (+0.0102 to +0.0199) | 2 × 10⁻²¹ |
| Correlation diff | **+0.0076** (+0.0065 to +0.0110) | 4 × 10⁻²⁰ |
| TSTR RF | +0.0101 (+0.0051 to +0.0145) | 0.90 |
| Gower MIA | +0.0129 (+0.0122 to +0.0139) | 0.0004 |
| Code-set MIA | +0.0004 (0.0000 to +0.0007) | 0.73 |
| Worst of three | +0.0088 (+0.0085 to +0.0092) | 0.0008 |

Per arm, mean JSD averaged over the five draws:

| Arm | Per-admission | Per-patient | Cost |
|---|---|---|---|
| `dp_marginals` ε=8 | 0.1860 | 0.1993 | +0.0133 |
| `dp_copula` ε=8 | 0.1920 | 0.2046 | +0.0126 |
| `dp_bayesnet` ε=8 | 0.2058 | 0.2223 | +0.0165 |
| `dp_marginals` ε=1 | 0.3698 | 0.3894 | +0.0196 |
| `dp_copula` ε=1 | 0.3774 | 0.3960 | +0.0186 |
| `dp_bayesnet` ε=1 | 0.3913 | 0.4095 | +0.0182 |

**The fidelity cost is real but small in context.** It sits on top of a number that DP has already pushed from 0.016 to between 0.19 and 0.41: **4 to 7%** of the divergence the guarantee already costs, depending on which admissions are kept. The draw matters: it moves the cost by a factor of two, so one draw should not be quoted alone. The first version's "about 4%" came from one draw of the uncorrected cap, and it also included AIM.

**Utility does not change detectably** (p = 0.90 on draw 0). Runs with a single-class death column at ε=1 are 34 to 49% over the draws, against 30 to 35% per admission, so the smaller cohort does make the label a little more fragile.

**The attack gets very slightly better, not worse.** Gower MIA rises by about +0.013 (p = 0.0004 on draw 0), the expected direction: 80 training rows instead of 94 means each row is a larger share of what the generator saw. The code-set attack does not move and stays at chance (0.499 to 0.511 per arm over the draws). "Patient-level DP" should not be sold as improving measured leakage. It improves the *guarantee*, which is a different thing.

---

## What this means for the paper

- **Report the DP arms at patient level beside the per-admission numbers.** It is the guarantee reviewers asked for, and it requires no change to any mechanism, only a cap applied to the cohort before fitting.
- **State the group-privacy number.** The relevant maximum in train is 3, and at ε = 8 the resulting δ is 88.9, so that statement is empty.
- The DP comparison's conclusions are unchanged: `dp_marginals` is still the best of these arms on fidelity, no arm beats the independent setting on correlation, and the code-set attack stays at chance.

## Limits

- **AIM's corrected patient-level run is outstanding.** It needs private-pgm: `python -m research.dp_study --seeds 20 --workers 3 --cap 1 --out patient_dp_aim --arms dp_aim_eps1 dp_aim_eps8`. The earlier AIM figures (0.2645 at ε=8, 0.4482 at ε=1) came from the uncorrected cap and before the δ fix, so they are not reported.
- **The cap is cheap because this cohort barely repeats.** 68 of 80 patients have a single admission and the maximum is 3. On a cohort with heavy repeat admissions a cap of 1 would discard much more, and the trade would need measuring again.
- Only `cap=1` is available. Caps above 1 need every mechanism's L2 sensitivity multiplied by the cap, and AIM's ε divided by it, which is not implemented, so `dp_study.py` refuses any other value.
- Five draws of the kept admissions, each with 20 generator seeds. The per-admission baseline is one run.
- JSD is measured against each cohort's own training rows, so the capped runs are scored against 80 rows and not 94.
