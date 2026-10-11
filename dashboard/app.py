
import json
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Project Doppel Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("Project Doppel")
st.subheader("Synthetic EHR Generation Dashboard")
st.write(
    "Explore synthetic datasets, inspect generator metadata, "
    "and compare fidelity, utility, and privacy evaluation results."
)

SYNTHETIC_DIR = Path("output/synthetic")
RESULTS_DIR = Path("results")


# ---------- Helpers ----------

def load_result(path):
    """Load and validate the minimum fields needed from an evaluation report."""
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        metrics = result["metrics"]

        fidelity = metrics["fidelity"]["mean_js_divergence"]
        utility_values = metrics["utility"]["tstr_auroc"]
        privacy = metrics["privacy"]["membership_worst_case"][
            "mean_attack_auroc"
        ]

        if not utility_values:
            raise ValueError("No TSTR AUROC values found.")

        return {
            "generator": result.get("generator_name", path.stem),
            "seed": result.get(
                "seed",
                path.stem.rsplit("_seed", 1)[-1]
                if "_seed" in path.stem
                else "Unknown",
            ),
            "fidelity_js": float(fidelity),
            "utility_tstr": sum(
                float(value) for value in utility_values.values()
            ) / len(utility_values),
            "privacy_worst_mia": float(privacy),
            "tstr_details": utility_values,
            "metrics": metrics,
            "error": None,
        }
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        return {
            "generator": path.stem,
            "error": str(error),
        }


def dominates(a, b):
    """True when a is at least as good as b on all axes and better on one."""
    a_values = [
        a["fidelity_js"],
        -a["utility_tstr"],
        a["privacy_worst_mia"],
    ]
    b_values = [
        b["fidelity_js"],
        -b["utility_tstr"],
        b["privacy_worst_mia"],
    ]

    return all(x <= y for x, y in zip(a_values, b_values)) and any(
        x < y for x, y in zip(a_values, b_values)
    )


# ---------- Tabs ----------

dataset_tab, evaluation_tab = st.tabs(
    ["Dataset Explorer", "Evaluation & Pareto Analysis"]
)


# ---------- Dataset Explorer ----------

with dataset_tab:
    st.header("Synthetic Dataset Explorer")

    if not SYNTHETIC_DIR.exists():
        st.warning("Synthetic output directory does not exist.")
    else:
        csv_files = sorted(SYNTHETIC_DIR.glob("*.csv"))

        if not csv_files:
            st.info("No synthetic datasets found.")
        else:
            selected_file = st.selectbox(
                "Select a generated dataset",
                csv_files,
                format_func=lambda path: path.name,
            )

            try:
                df = pd.read_csv(selected_file)

                st.success(f"Loaded: {selected_file.name}")

                col1, col2, col3 = st.columns(3)
                col1.metric("Rows", df.shape[0])
                col2.metric("Columns", df.shape[1])
                col3.metric("Missing Values", int(df.isna().sum().sum()))

                st.subheader("Dataset Preview")
                st.dataframe(df.head(20), use_container_width=True)

                st.subheader("Column Information")
                column_info = pd.DataFrame({
                    "Column": df.columns,
                    "Data Type": df.dtypes.astype(str).values,
                    "Missing Values": df.isna().sum().values,
                })
                st.dataframe(column_info, use_container_width=True)

                manifest_file = selected_file.with_suffix(".manifest.json")

                if manifest_file.exists():
                    st.subheader("Generator Manifest")
                    manifest = json.loads(
                        manifest_file.read_text(encoding="utf-8")
                    )
                    st.json(manifest)
                else:
                    st.info("Manifest file not found.")

            except Exception as error:
                st.error(f"Could not load dataset: {error}")


# ---------- Evaluation and Pareto Analysis ----------

with evaluation_tab:
    st.header("Evaluation Results")

    if not RESULTS_DIR.exists():
        st.warning("The results directory does not exist.")
        st.info("Run the evaluation jobs and copy their JSON reports into results/.")
    else:
        report_files = sorted(RESULTS_DIR.glob("*.json"))
        valid_results = []
        invalid_reports = []

        for report_file in report_files:
            report = load_result(report_file)
            if report.get("error"):
                invalid_reports.append((report_file.name, report["error"]))
            else:
                valid_results.append(report)

        if invalid_reports:
            with st.expander("Reports that could not be loaded"):
                for filename, error in invalid_reports:
                    st.warning(f"{filename}: {error}")

        if not valid_results:
            st.info("No valid evaluation reports found in results/.")
        else:
            st.caption(
                f"Loaded {len(valid_results)} evaluation report(s). "
                "Metrics reflect the available reports and seeds."
            )

            summary_rows = []
            for result in valid_results:
                summary_rows.append({
                    "Generator": result["generator"],
                    "Seed": result["seed"],
                    "Mean JS divergence ↓": result["fidelity_js"],
                    "Mean TSTR AUROC ↑": result["utility_tstr"],
                    "Worst-case membership AUROC ↓": result["privacy_worst_mia"],
                })

            summary_df = pd.DataFrame(summary_rows)
            st.subheader("Generator Comparison")
            st.dataframe(
                summary_df.style.format({
                    "Mean JS divergence ↓": "{:.4f}",
                    "Mean TSTR AUROC ↑": "{:.4f}",
                    "Worst-case membership AUROC ↓": "{:.4f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "Lower JS divergence indicates better fidelity. Higher TSTR "
                "AUROC indicates better downstream utility. Lower membership "
                "inference AUROC indicates a weaker attack under this evaluation."
            )

            # Metric charts
            chart_col1, chart_col2 = st.columns(2)

            chart_data = summary_df.set_index("Generator")

            with chart_col1:
                st.subheader("Fidelity")
                st.bar_chart(chart_data[["Mean JS divergence ↓"]])

                st.subheader("Privacy")
                st.bar_chart(
                    chart_data[["Worst-case membership AUROC ↓"]]
                )

            with chart_col2:
                st.subheader("Utility")
                st.bar_chart(chart_data[["Mean TSTR AUROC ↑"]])

            # Per-generator detailed results
            st.subheader("Detailed Evaluation Metrics")
            for result in valid_results:
                with st.expander(
                    f"{result['generator']} — seed {result['seed']}"
                ):
                    st.write("TSTR AUROC by classifier")
                    st.json(result["tstr_details"])
                    st.write("Full evaluation metrics")
                    st.json(result["metrics"])

            # Pareto frontier
            st.subheader("Pareto Frontier")
            st.write(
                "A generator is on the frontier if no other generator "
                "is at least as good in fidelity, utility, and privacy, "
                "and strictly better in at least one."
            )

            # First average the available seeds per generator.
            grouped = {}
            for result in valid_results:
                grouped.setdefault(result["generator"], []).append(result)

            generator_means = []
            for name, rows in grouped.items():
                generator_means.append({
                    "Generator": name,
                    "Seeds": len(rows),
                    "fidelity_js": sum(
                        row["fidelity_js"] for row in rows
                    ) / len(rows),
                    "utility_tstr": sum(
                        row["utility_tstr"] for row in rows
                    ) / len(rows),
                    "privacy_worst_mia": sum(
                        row["privacy_worst_mia"] for row in rows
                    ) / len(rows),
                })

            frontier_names = {
                candidate["Generator"]
                for candidate in generator_means
                if not any(
                    dominates(other, candidate)
                    for other in generator_means
                    if other["Generator"] != candidate["Generator"]
                )
            }

            frontier_df = pd.DataFrame([
                {
                    "Generator": row["Generator"],
                    "Seeds": row["Seeds"],
                    "Mean JS divergence": row["fidelity_js"],
                    "Mean TSTR AUROC": row["utility_tstr"],
                    "Worst-case membership AUROC": row["privacy_worst_mia"],
                    "On Pareto frontier": (
                        "Yes" if row["Generator"] in frontier_names else "No"
                    ),
                }
                for row in generator_means
            ])

            st.dataframe(
                frontier_df.style.format({
                    "Mean JS divergence": "{:.4f}",
                    "Mean TSTR AUROC": "{:.4f}",
                    "Worst-case membership AUROC": "{:.4f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

            if all(len(rows) == 1 for rows in grouped.values()):
                st.warning(
                    "Each generator has only one evaluated seed. "
                    "Frontier membership is based on these observed scores; "
                    "between-seed variability cannot yet be assessed."
                )

            st.caption(
                "This dashboard computes frontier membership from the "
                "loaded reports. It does not calculate bootstrap frequencies."
            )
