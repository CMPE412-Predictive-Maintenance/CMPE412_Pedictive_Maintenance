"""Streamlit demo for the FD001 predictive maintenance project.

Run from the repository root:
    streamlit run 06_deployment/app.py

All data preparation and prediction logic is in pm_deployment.py, shared with
the notebook and the tests.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pm_deployment as pm  # noqa: E402

st.set_page_config(page_title="Engine RUL Demo", layout="wide")


@st.cache_resource(show_spinner="Loading saved models...")
def get_models():
    return pm.load_models()


@st.cache_data(show_spinner="Loading and verifying NASA FD001 test data...")
def get_fleet():
    return pm.load_test_fleet()


@st.cache_data(show_spinner="Scoring all engines...")
def get_fleet_scores(_models, test_df):
    return pm.score_engines(test_df, _models)


def rul_chart(scored, engine_id, cutoff):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(scored["cycle"], scored["predicted_rul"], label="XGBoost (selected)")
    ax.plot(scored["cycle"], scored["baseline_rul"], alpha=0.45, label="Linear Regression (baseline)")
    ax.axhline(pm.CRITICAL_RUL, color="crimson", linestyle="--", linewidth=1,
               label=f"Critical ({pm.CRITICAL_RUL} cycles)")
    ax.axvline(cutoff, color="gray", linewidth=0.8)
    ax.set(title=f"Engine {engine_id}: predicted RUL", xlabel="Cycle", ylabel="Cycles")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def anomaly_chart(scored, engine_id, threshold, cutoff):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(scored["cycle"], scored["anomaly_score"], color="tab:purple", label="Anomaly score")
    ax.axhline(threshold, color="gray", linestyle="--", label="Saved threshold")
    ax.fill_between(scored["cycle"], 0, 1, where=scored["sustained_warning"],
                    transform=ax.get_xaxis_transform(), color="orange", alpha=0.2,
                    label="Sustained warning")
    ax.axvline(cutoff, color="gray", linewidth=0.8)
    ax.set(title=f"Engine {engine_id}: unusual sensor behaviour", xlabel="Cycle",
           ylabel="Score (higher = more unusual)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def show_engine(engine_raw, models, engine_id, true_rul=None):
    """Engine view: RUL and unusual behaviour shown in separate panels."""
    max_cycle = int(engine_raw["cycle"].max())
    cutoff = max_cycle
    if max_cycle > 21:
        cutoff = st.slider("Replay: data available up to cycle", 21, max_cycle, max_cycle,
                           help="Move left to see what the planner would have seen earlier. "
                                "Later cycles are removed before the features are computed.")
    visible = engine_raw[engine_raw["cycle"] <= cutoff]
    scored = pm.score_engines(visible, models)
    now = scored.iloc[-1]
    recent_flags = int(scored.tail(10)["anomaly_flag"].sum())

    st.caption(f"Engine {engine_id} · {len(visible)} observed cycles · latest cycle {cutoff}")
    left, right = st.columns(2)

    with left:
        st.subheader("Predicted remaining useful life")
        status = pm.rul_status(now["predicted_rul"])
        st.metric("XGBoost (selected model)", f"{now['predicted_rul']:.0f} cycles", status,
                  delta_color="off")
        st.write(f"Predicted RUL: {pm.describe_rul(now['predicted_rul'])}.")
        st.caption(f"Baseline for comparison: about {now['baseline_rul']:.0f} cycles.")
        if true_rul is not None and cutoff == max_cycle:
            st.caption(f"Official RUL label for this test engine: {true_rul} cycles "
                       "(shown for evaluation only; a planner would not know it).")
        st.pyplot(rul_chart(scored, engine_id, cutoff))

    with right:
        st.subheader("Unusual sensor behaviour")
        st.metric("Anomaly status", pm.anomaly_status(now["sustained_warning"], recent_flags))
        st.write(f"Latest score {now['anomaly_score']:.3f} (threshold "
                 f"{models.anomaly['threshold']:.3f}) · flagged in last 10 cycles: "
                 f"{recent_flags} · consecutive flagged cycles: {int(now['consecutive_flags'])}")
        st.caption("An unusual pattern is a reason to look closer, not a confirmed fault. "
                   f"A sustained warning needs {models.anomaly['sustained_run']} consecutive "
                   "flagged cycles.")
        st.pyplot(anomaly_chart(scored, engine_id, models.anomaly["threshold"], cutoff))

    action = pm.suggested_action(now["predicted_rul"], now["sustained_warning"])
    box = st.error if action == "Inspect first" else (
        st.warning if action != "Continue monitoring" else st.success)
    box(f"Planner note: **{action}**. Demo rule, to be reviewed by a maintenance planner.")


# ---------------------------------------------------------------------------

st.title("Aircraft engine RUL and unusual-behaviour demo")
st.caption("CMPE 412 · NASA C-MAPSS FD001 (simulated data) · decision support with human review")

problems = pm.check_versions()
if problems:
    st.warning("Library versions differ from the ones used to save the models:\n\n- "
               + "\n- ".join(problems))

try:
    models = get_models()
except ModuleNotFoundError as error:
    st.error(f"Cannot load the saved models: {error}. Install 06_deployment/requirements.txt.")
    st.stop()

tab_engine, tab_fleet, tab_upload, tab_about = st.tabs(
    ["Engine view", "Inspection priority", "New engine record", "About and limitations"])

try:
    test_df, true_rul = get_fleet()
except Exception as error:  # network or verification problem
    test_df, true_rul = None, None
    st.error(f"NASA test data is not available: {error}. The 'New engine record' tab still works.")

with tab_engine:
    if test_df is not None:
        units = sorted(test_df["unit"].unique())
        engine_id = st.selectbox("Select a test engine", units, index=units.index(37),
                                 help="Engine 37 is a late-prediction example from 05 Evaluation.")
        show_engine(test_df[test_df["unit"] == engine_id], models, engine_id,
                    int(true_rul[engine_id]))

with tab_fleet:
    if test_df is not None:
        fleet = pm.engine_summary(get_fleet_scores(models, test_df))
        st.write("All 100 test engines at their latest observed cycle, ranked by predicted RUL.")
        slots = st.number_input("Inspection slots this shift", 1, 100, 10)
        only_warn = st.checkbox("Show only engines with an active sustained warning")
        show_truth = st.checkbox("Show official RUL labels (evaluation view)")
        view = fleet[fleet["sustained_warning"]] if only_warn else fleet
        cols = ["priority", "unit", "last_cycle", "predicted_rul", "rul_status",
                "anomaly_status", "suggested_action"]
        view = view[cols].copy()
        if show_truth:
            view["official_true_rul"] = view["unit"].map(true_rul)
        st.dataframe(view.head(int(slots)).round(1), hide_index=True)
        counts = fleet["suggested_action"].value_counts()
        st.write("Suggested actions across the fleet: "
                 + ", ".join(f"{k}: {v}" for k, v in counts.items()))
        st.caption(f"Demo levels: Critical if predicted RUL ≤ {pm.CRITICAL_RUL}, plan inspection "
                   f"if ≤ {pm.PLAN_RUL}. 'Take a closer look' is shown when a sustained warning "
                   "is active outside the critical range. These are illustrative settings, "
                   "not a validated policy.")

with tab_upload:
    st.write("Upload one engine's record as CSV: one row per cycle with these columns "
             "(extra columns are ignored).")
    st.code(", ".join(pm.required_input_columns(models)))
    sample_path = pm.DEPLOY_DIR / "sample_input" / "new_engine_sample.csv"
    st.download_button("Download a sample file", sample_path.read_bytes(),
                       file_name="new_engine_sample.csv", mime="text/csv")
    uploaded = st.file_uploader("Engine record (CSV)", type="csv")
    use_sample = st.checkbox("Use the sample file", value=uploaded is None)
    source = uploaded if uploaded is not None else (sample_path if use_sample else None)
    if source is not None:
        record = pd.read_csv(source)
        try:
            checked = pm.validate_input(record, models)
        except ValueError as error:
            st.error(f"The record cannot be used: {error}")
        else:
            units = sorted(checked["unit"].unique())
            unit = units[0] if len(units) == 1 else st.selectbox("Engine in file", units)
            show_engine(checked[checked["unit"] == unit], models, unit)

with tab_about:
    card = models.card
    st.markdown(f"""
**How a prediction is made**

1. The engine record is validated and sorted by cycle.
2. For 14 sensors, the 10-cycle mean and the 20-cycle slope are computed from the current
   and earlier cycles only (the same recipe as 04 Modeling).
3. The saved **{card['selected_model']}** model predicts RUL from {len(card['feature_columns'])}
   features. Predictions below 0 are set to 0.
4. Separately, the saved **Isolation Forest** scores the 10-cycle sensor means. A cycle is
   flagged above the saved threshold; 5 consecutive flags make a sustained warning.

**Evaluated performance (05 Evaluation, official test set):** XGBoost MAE 10.97 cycles vs
25.47 for the Linear Regression baseline; 9 of the 10 engines with the lowest RUL were in the
predicted top 10.

**Limitations**

- Simulated FD001 data: one operating condition, one fault mode.
- Late predictions remain: 3 of 25 critical test engines were predicted more than 10 cycles
  late (engine 37: +27 cycles).
- The model was trained with RUL capped at {pm.RUL_CAP} cycles; above that it only indicates
  a healthy engine.
- Unusual sensor behaviour is not a confirmed fault; FD001 has no labelled anomalies.
- Demo thresholds and the planner note are illustrative and not a validated policy.
- Every suggested action is meant to be reviewed by a maintenance planner.
""")
