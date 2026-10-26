# Patient-Level DP — the guarantee can be upgraded for about 4% of the fidelity it already costs (2026-10-26)

Journal-version work (Track 1 task 5). Every DP arm in this project guarantees (ε, δ)-DP at the level of **one admission**. Reviewers asked for patient-level privacy, which is the guarantee that actually matters: a patient with three admissions is protected three times less by a per-row guarantee, and it is the patient, not the row, who is re-identified.

The task offered two routes — "cap admissions per patient, or count the group-privacy cost". Both are done, because counting the group-privacy cost is what shows why capping is the right answer.

```bash
python -m research.dp_study --seeds 20 --workers 3                     # per-admission, 94 rows
python -m research.dp_study --seeds 20 --workers 3 --cap 1 \
    --out patient_dp_study --arms dp_{marginals,copula,bayesnet,aim}_eps{1,8}   # per-patient, 80 rows
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

Keeping at most one admission per patient takes the training cohort from **94 rows to 80**, one per patient, and from 34 deaths to 30. Each patient is then exactly one row, so the mechanisms' existing bounded (replace-one-row) guarantee *is* a replace-one-patient guarantee, at the same ε and the same δ, with no extra noise and no change to any mechanism.

Only 12 of the 80 patients have more than one admission, and the maximum is 3. That is why the cap is cheap here, and it is a property of this cohort rather than of the method — see the limits.

## 3. What the upgrade actually costs

Paired on (arm, seed) across all four mechanisms at both budgets, 160 pairs:

| Metric | Patient-level minus admission-level | Wilcoxon p |
|---|---|---|
| Mean JSD | **+0.0129** | 2.5 × 10⁻²⁴ |
| Correlation diff | **+0.0058** | 5.3 × 10⁻²¹ |
| TSTR RF | +0.0364 | 0.061 |
| Gower MIA | +0.0070 | 0.034 |

Per arm, the fidelity cost:

| Arm | Per-admission | Per-patient | Cost |
|---|---|---|---|
| `dp_marginals` ε=8 | 0.1849 | 0.1919 | +0.0070 |
| `dp_copula` ε=8 | 0.1898 | 0.1958 | +0.0060 |
| `dp_bayesnet` ε=8 | 0.2034 | 0.2166 | +0.0132 |
| `dp_aim` ε=8 | 0.2458 | 0.2645 | +0.0187 |
| `dp_marginals` ε=1 | 0.3676 | 0.3849 | +0.0173 |
| `dp_copula` ε=1 | 0.3747 | 0.3917 | +0.0170 |
| `dp_bayesnet` ε=1 | 0.3947 | 0.4111 | +0.0163 |
| `dp_aim` ε=1 | 0.4406 | 0.4482 | +0.0076 |

**The fidelity cost is real but tiny in context.** It is highly significant — 160 paired runs will detect almost anything — but +0.013 JSD sits on top of a number that DP has already pushed from 0.016 to between 0.19 and 0.44. In relative terms the upgrade costs about **4%** of the divergence the guarantee already costs. Nobody choosing between these arms would change their choice because of it.

**Utility does not change.** +0.036 TSTR at p = 0.061, and in the *favourable* direction. With a per-arm seed spread of 0.13 to 0.21 this is noise, and it should not be reported as capping improving utility.

**One honest wrinkle: the attack gets very slightly better, not worse.** Gower MIA rises +0.0070 (p = 0.034). That is the expected direction — 80 training rows instead of 94 means each row is a larger share of what the generator saw, so membership is marginally easier to detect. The effect is 0.007 against a floor of about 0.50, so it changes nothing practically, but "patient-level DP" should not be sold as improving measured leakage. It improves the *guarantee*, which is a different thing, and this project has already made the mistake of conflating the two.

---

## What this means for the paper

- **Report the DP arms at patient level.** It is the guarantee reviewers asked for, it costs about 4% of the fidelity DP already costs, and it requires no change to any mechanism — only a cap applied to the cohort before fitting.
- **Drop the group-privacy sentence, or state its number.** `dp_copula.py`'s docstring says patient-level privacy "follows by group privacy" and cites up to 15 admissions. Both parts need fixing: the relevant maximum in train is 3, and at ε = 8 the resulting δ is 88.9, i.e. the statement is empty. A reviewer who checks that arithmetic will not be gentle about it.
- The DP comparison's conclusions are unchanged under the upgrade: `dp_marginals` is still the best DP arm on fidelity, AIM still the worst, no DP arm beats the no-dependence floor on correlation, and every one of them puts the attack at chance.

## Limits

- **The cap is cheap because this cohort barely repeats.** 68 of 80 patients have a single admission and the maximum is 3. On a cohort with heavy repeat admissions — any real ICU extract — a cap of 1 would discard much more, and the trade would need measuring again rather than assuming this 4%.
- Only `cap=1` was run. Caps above 1 need the mechanisms' L2 sensitivities multiplied by the cap, which means editing the sensitivity in each measurement; that is not implemented, and `--cap 2` today would silently give a per-admission guarantee on a capped cohort, which is **not** what it looks like. The flag's help text says what it does; the code does not enforce it.
- Which admission is kept is one fixed draw (`CAP_SEED = 0`), shared by every arm and seed so the comparison is paired. Averaging over that choice was not done.
- AIM inherits the upgrade for free only because `cap=1` makes patient-level and row-level the same thing. For caps above 1 AIM would additionally need its own ε divided by the cap, since it does its own internal accounting against a row-level neighbour relation.
