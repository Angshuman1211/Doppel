"""Project Doppel dashboard.

Nothing is read from disk: every dataset and every evaluation report is uploaded through the page, so the
dashboard runs the same way on a laptop, in the container, and in front of an audience with no pipeline
output sitting next to it.

The container image ships this file with pandas only, so the comparison tab computes its own fidelity
numbers. When the dashboard is started from a checkout of the repo, it imports Track 2's metric code and
Track 1's contract validator instead, and says which one produced the numbers on screen.
"""

import json

import numpy as np
import pandas as pd
import streamlit as st

# ICD-9 codes are identifiers, not numbers: read as int, '0389' becomes 389, a different code (P-005).
CODE_COLUMNS = {"icd9_primary": str}

# Columns that identify a row rather than describe a patient, excluded from fidelity comparisons.
ID_COLUMNS = ["subject_id", "hadm_id", "icd9_codes", "split"]

try:  # present when run from the repo, absent in the dashboard image
    from evaluation.fidelity_metrics import run_fidelity_report
    from generators.schema import load_real  # noqa: F401  (import proves the package is importable)
    from generators.validate import check_contract

    REPO_CODE = True
except ImportError:
    REPO_CODE = False


st.set_page_config(page_title="Project Doppel Dashboard", page_icon="📊", layout="wide")

st.title("Project Doppel")
st.subheader("Synthetic EHR Generation Dashboard")
st.write(
    "Upload a synthetic dataset to explore it, add the real cleaned dataset to score fidelity against it, "
    "and upload evaluation reports to compare generators."
)


# ---------- Helpers ----------

def read_csv_upload(upload):
    """Read an uploaded CSV with the ICD-9 dtype pinned."""
    upload.seek(0)
    return pd.read_csv(upload, dtype=CODE_COLUMNS)


def comparable_columns(real, synth):
    shared = [c for c in real.columns if c in synth.columns and c not in ID_COLUMNS]
    categorical = [c for c in shared if real[c].dtype == object or real[c].dropna().nunique() <= 2]
    numeric = [c for c in shared if c not in categorical and pd.api.types.is_numeric_dtype(real[c])]
    return categorical, numeric


def js_divergence(p, q):
    """Jensen-Shannon divergence in bits, 0 when the two distributions match."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = (p + q) / 2

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return (kl(p, m) + kl(q, m)) / 2


def column_divergence(real_col, synth_col, numeric, bins=10):
    """Categorical columns compare value shares; numeric columns are binned on shared quantiles first."""
    real_col, synth_col = real_col.dropna(), synth_col.dropna()
    if real_col.empty or synth_col.empty:
        return np.nan
    if numeric:
        edges = np.unique(np.quantile(pd.concat([real_col, synth_col]), np.linspace(0, 1, bins + 1)))
        if len(edges) < 2:
            return 0.0
        real_counts, _ = np.histogram(real_col, bins=edges)
        synth_counts, _ = np.histogram(synth_col, bins=edges)
    else:
        levels = sorted(set(real_col.astype(str)) | set(synth_col.astype(str)))
        real_counts = real_col.astype(str).value_counts().reindex(levels, fill_value=0).to_numpy()
        synth_counts = synth_col.astype(str).value_counts().reindex(levels, fill_value=0).to_numpy()
    return js_divergence(real_counts / max(real_counts.sum(), 1), synth_counts / max(synth_counts.sum(), 1))


def fidelity_table(real, synth):
    """Per-column divergence plus the correlation gap, computed in the dashboard (pandas only)."""
    categorical, numeric = comparable_columns(real, synth)
    rows = [{"Column": c, "Kind": "categorical", "JS divergence ↓": column_divergence(real[c], synth[c], False)}
            for c in categorical]
    rows += [{"Column": c, "Kind": "numeric", "JS divergence ↓": column_divergence(real[c], synth[c], True)}
             for c in numeric]
    table = pd.DataFrame(rows).sort_values("JS divergence ↓", ascending=False, ignore_index=True)

    real_corr = real[numeric].corr().to_numpy()
    synth_corr = synth[numeric].corr().to_numpy()
    upper = np.triu(np.ones_like(real_corr, dtype=bool), k=1)
    corr_gap = float(np.mean(np.abs(np.nan_to_num(real_corr[upper]) - np.nan_to_num(synth_corr[upper]))))
    return table, corr_gap


def profile_match_rate(real, candidate, columns):
    """Share of candidate rows whose categorical profile exactly matches some real row.

    Matching on every column including lab values is useless here, since two continuous values never
    coincide, so a memorizing generator would score 0%. The categorical profile is where copying shows:
    real unseen patients land near 6% on this dataset, the generators that memorize land above 30%.
    """
    if not columns:
        return np.nan
    real_keys = set(map(tuple, real[columns].astype(str).to_numpy()))
    return float(np.mean([tuple(row) in real_keys for row in candidate[columns].astype(str).to_numpy()]))


def load_report(upload):
    """Pull the three headline metrics out of an uploaded evaluation report."""
    try:
        upload.seek(0)
        result = json.load(upload)
        metrics = result["metrics"]
        tstr = metrics["utility"]["tstr_auroc"]
        if not tstr:
            raise ValueError("no TSTR AUROC values in the report")
        # The evaluation contract has no top-level seed field, so it comes out of run_id (gaussian_copula_seed42).
        label = str(result.get("run_id", upload.name))
        return {
            "generator": result.get("generator_name", upload.name),
            "seed": result.get("seed", label.rsplit("_seed", 1)[-1] if "_seed" in label else "unknown"),
            "fidelity_js": float(metrics["fidelity"]["mean_js_divergence"]),
            "utility_tstr": sum(float(v) for v in tstr.values()) / len(tstr),
            "privacy_worst_mia": float(metrics["privacy"]["membership_worst_case"]["mean_attack_auroc"]),
            "tstr_details": tstr,
            "metrics": metrics,
            "error": None,
        }
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        return {"generator": upload.name, "error": str(error)}


def dominates(a, b):
    """True when a is at least as good as b on all three axes and better on one."""
    a_values = [a["fidelity_js"], -a["utility_tstr"], a["privacy_worst_mia"]]
    b_values = [b["fidelity_js"], -b["utility_tstr"], b["privacy_worst_mia"]]
    return all(x <= y for x, y in zip(a_values, b_values)) and any(x < y for x, y in zip(a_values, b_values))


# ---------- Uploads ----------

with st.sidebar:
    st.header("Data")
    synthetic_uploads = st.file_uploader(
        "Synthetic dataset(s) (CSV)", type="csv", accept_multiple_files=True,
        help="Output of python -m generators.generate, or any CSV with the same columns.",
    )
    real_upload = st.file_uploader(
        "Real cleaned dataset (CSV, optional)", type="csv",
        help="mimic_demo_clean.csv. Needed to score fidelity and to validate against the contract.",
    )
    manifest_uploads = st.file_uploader(
        "Generator manifest(s) (JSON, optional)", type="json", accept_multiple_files=True,
    )
    st.caption(
        "Running from a repo checkout: Track 2's metrics and Track 1's validator are in use."
        if REPO_CODE else
        "Running standalone: fidelity is computed inside the dashboard, and contract validation is unavailable."
    )

synthetic_frames = {}
for upload in synthetic_uploads or []:
    try:
        synthetic_frames[upload.name] = read_csv_upload(upload)
    except (ValueError, pd.errors.ParserError) as error:
        st.sidebar.error(f"{upload.name}: {error}")

real_df = None
if real_upload is not None:
    try:
        real_df = read_csv_upload(real_upload)
    except (ValueError, pd.errors.ParserError) as error:
        st.sidebar.error(f"{real_upload.name}: {error}")

manifests = {}
for upload in manifest_uploads or []:
    try:
        upload.seek(0)
        manifests[upload.name] = json.load(upload)
    except json.JSONDecodeError as error:
        st.sidebar.error(f"{upload.name}: {error}")


# ---------- Tabs ----------

explorer_tab, comparison_tab, evaluation_tab = st.tabs(
    ["Dataset Explorer", "Compare to Real Data", "Evaluation & Pareto Analysis"]
)


with explorer_tab:
    st.header("Synthetic Dataset Explorer")

    if not synthetic_frames:
        st.info("Upload one or more synthetic CSV files in the sidebar to begin.")
    else:
        name = st.selectbox("Dataset", list(synthetic_frames))
        df = synthetic_frames[name]

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing values", int(df.isna().sum().sum()))

        st.subheader("Preview")
        st.dataframe(df.head(20), use_container_width=True)

        st.subheader("Columns")
        st.dataframe(
            pd.DataFrame({
                "Column": df.columns,
                "Data type": df.dtypes.astype(str).values,
                "Missing values": df.isna().sum().values,
                "Distinct values": df.nunique().values,
            }),
            use_container_width=True, hide_index=True,
        )

        matching = [m for filename, m in manifests.items() if name.replace(".csv", "") in filename]
        if matching:
            st.subheader("Generator manifest")
            st.json(matching[0])


with comparison_tab:
    st.header("Synthetic vs Real")

    if not synthetic_frames:
        st.info("Upload a synthetic dataset in the sidebar.")
    elif real_df is None:
        st.info("Upload the real cleaned dataset in the sidebar to compare against it.")
    else:
        name = st.selectbox("Dataset to compare", list(synthetic_frames), key="comparison_dataset")
        synth_df = synthetic_frames[name]

        reference = real_df
        if "split" in real_df.columns:
            which = st.radio("Compare against", ["train split", "holdout split", "all rows"], horizontal=True)
            if which != "all rows":
                reference = real_df[real_df["split"] == which.split()[0]]
            st.caption(f"{len(reference)} real rows in the selected reference.")

        if REPO_CODE:
            errors = check_contract(synth_df, real_df)
            if errors:
                st.error(f"Fails the Stage 2 output contract ({len(errors)} issue(s)):")
                for message in errors:
                    st.write(f"- {message}")
            else:
                st.success("Passes the Stage 2 output contract.")

        table, corr_gap = fidelity_table(reference, synth_df)
        mean_js = float(table["JS divergence ↓"].mean(skipna=True))

        discrete_columns, _ = comparable_columns(reference, synth_df)
        copied = profile_match_rate(reference, synth_df, discrete_columns)
        # Real rows the generator never saw give the honest reference point for that number.
        baseline = None
        if "split" in real_df.columns and {"train", "holdout"} <= set(real_df["split"]):
            baseline = profile_match_rate(
                real_df[real_df["split"] == "train"], real_df[real_df["split"] == "holdout"], discrete_columns
            )

        if REPO_CODE:
            try:
                official = run_fidelity_report(reference, synth_df)
                mean_js, corr_gap = official["mean_js_divergence"], official["correlation_diff"]
                st.caption("Fidelity from Track 2's `evaluation.fidelity_metrics`.")
            except (KeyError, ValueError) as error:
                st.warning(f"Track 2's metric could not run on this file ({error}); showing the dashboard's own.")
        else:
            st.caption("Fidelity computed inside the dashboard, matching the protocol's binned-JS definition.")

        col1, col2, col3 = st.columns(3)
        col1.metric("Mean JS divergence ↓", f"{mean_js:.4f}")
        col2.metric("Correlation gap ↓", f"{corr_gap:.4f}")
        col3.metric(
            "Rows matching a real patient ↓", f"{copied:.1%}",
            delta=None if baseline is None else f"{copied - baseline:+.1%} vs real unseen rows",
            delta_color="inverse",
        )
        st.caption(
            "Lower divergence means the columns look more like the real ones. The correlation gap is the mean "
            "absolute difference between the two correlation matrices. The last number is the share of rows whose "
            + ", ".join(discrete_columns[:4]) + " and the other categorical fields exactly match a real patient"
            + (f"; real rows the generator never saw match at {baseline:.1%}, so that is the level to expect "
               "from a generator that isn't copying." if baseline is not None else
               ". Upload the real dataset with its split column to see the level real unseen rows reach.")
            + " It is a memorization signal, not the full privacy evaluation."
        )

        st.subheader("Per-column divergence")
        st.dataframe(table.style.format({"JS divergence ↓": "{:.4f}"}), use_container_width=True, hide_index=True)
        st.bar_chart(table.set_index("Column")[["JS divergence ↓"]].head(15))

        st.subheader("Column detail")
        column = st.selectbox("Column", table["Column"].tolist())
        kind = table.loc[table["Column"] == column, "Kind"].iloc[0]
        if kind == "numeric":
            summary = pd.DataFrame({
                "Real": reference[column].describe(),
                "Synthetic": synth_df[column].describe(),
            })
            st.dataframe(summary, use_container_width=True)
        else:
            shares = pd.DataFrame({
                "Real": reference[column].astype(str).value_counts(normalize=True),
                "Synthetic": synth_df[column].astype(str).value_counts(normalize=True),
            }).fillna(0)
            st.bar_chart(shares)


with evaluation_tab:
    st.header("Evaluation Results")

    report_uploads = st.file_uploader(
        "Evaluation report(s) (JSON from the evaluation stage)", type="json", accept_multiple_files=True,
    )

    reports, broken = [], []
    for upload in report_uploads or []:
        report = load_report(upload)
        (broken if report.get("error") else reports).append(
            (upload.name, report["error"]) if report.get("error") else report
        )

    if broken:
        with st.expander("Reports that could not be read"):
            for filename, error in broken:
                st.warning(f"{filename}: {error}")

    if not reports:
        st.info("Upload one or more evaluation report JSON files to compare generators.")
    else:
        st.caption(f"{len(reports)} report(s) loaded. Metrics reflect the uploaded reports and seeds.")

        summary = pd.DataFrame([{
            "Generator": r["generator"],
            "Seed": r["seed"],
            "Mean JS divergence ↓": r["fidelity_js"],
            "Mean TSTR AUROC ↑": r["utility_tstr"],
            "Worst-case membership AUROC ↓": r["privacy_worst_mia"],
        } for r in reports])

        st.subheader("Generator comparison")
        number_format = {
            "Mean JS divergence ↓": "{:.4f}",
            "Mean TSTR AUROC ↑": "{:.4f}",
            "Worst-case membership AUROC ↓": "{:.4f}",
        }
        st.dataframe(summary.style.format(number_format), use_container_width=True, hide_index=True)
        st.caption(
            "Lower JS divergence is better fidelity. Higher TSTR AUROC is better downstream utility. Lower "
            "membership AUROC means the attack did worse under this evaluation."
        )

        chart_data = summary.groupby("Generator")[list(number_format)].mean()
        left, right = st.columns(2)
        with left:
            st.subheader("Fidelity")
            st.bar_chart(chart_data[["Mean JS divergence ↓"]])
            st.subheader("Privacy")
            st.bar_chart(chart_data[["Worst-case membership AUROC ↓"]])
        with right:
            st.subheader("Utility")
            st.bar_chart(chart_data[["Mean TSTR AUROC ↑"]])

        st.subheader("Detailed metrics")
        for report in reports:
            with st.expander(f"{report['generator']} — seed {report['seed']}"):
                st.write("TSTR AUROC by classifier")
                st.json(report["tstr_details"])
                st.write("Full evaluation metrics")
                st.json(report["metrics"])

        st.subheader("Pareto frontier")
        st.write(
            "A generator is on the frontier when no other generator is at least as good on fidelity, utility "
            "and privacy together, and strictly better on at least one."
        )

        grouped = {}
        for report in reports:
            grouped.setdefault(report["generator"], []).append(report)

        means = [{
            "Generator": name,
            "Seeds": len(rows),
            "fidelity_js": sum(r["fidelity_js"] for r in rows) / len(rows),
            "utility_tstr": sum(r["utility_tstr"] for r in rows) / len(rows),
            "privacy_worst_mia": sum(r["privacy_worst_mia"] for r in rows) / len(rows),
        } for name, rows in grouped.items()]

        frontier = {
            candidate["Generator"] for candidate in means
            if not any(dominates(other, candidate) for other in means
                       if other["Generator"] != candidate["Generator"])
        }

        st.dataframe(
            pd.DataFrame([{
                "Generator": row["Generator"],
                "Seeds": row["Seeds"],
                "Mean JS divergence": row["fidelity_js"],
                "Mean TSTR AUROC": row["utility_tstr"],
                "Worst-case membership AUROC": row["privacy_worst_mia"],
                "On Pareto frontier": "Yes" if row["Generator"] in frontier else "No",
            } for row in means]).style.format({
                "Mean JS divergence": "{:.4f}",
                "Mean TSTR AUROC": "{:.4f}",
                "Worst-case membership AUROC": "{:.4f}",
            }),
            use_container_width=True, hide_index=True,
        )

        if all(len(rows) == 1 for rows in grouped.values()):
            st.warning(
                "Each generator has one seed here, so frontier membership reflects those single runs. "
                "Upload more seeds to see how much of it is seed noise."
            )
        st.caption("Frontier membership is computed from the uploaded reports; bootstrap frequencies are not.")
