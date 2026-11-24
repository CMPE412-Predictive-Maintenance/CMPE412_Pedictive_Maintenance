"""Smoke and consistency tests for the deployment demo.

Run from the repository root:
    python -m pytest 06_deployment/tests -v

Tests that need xgboost or the NASA test files are skipped (not failed) when
those are missing, and the reason is printed in the test report.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pm_deployment as pm  # noqa: E402

VALIDATION = pm.REPO_DIR / "03_data_preparation" / "outputs" / "validation.csv"
MODELING_PRED = pm.REPO_DIR / "04_modeling" / "outputs" / "validation_predictions.csv"
MODELING_WARN = pm.REPO_DIR / "04_modeling" / "outputs" / "anomaly_sustained_warnings.csv"
SAMPLE = pm.DEPLOY_DIR / "sample_input" / "new_engine_sample.csv"


class ConstantModel:
    """Stand-in for the XGBoost model when xgboost is not installed.
    It also checks that features arrive in the saved order."""

    def __init__(self, columns):
        self.columns = columns

    def predict(self, X):
        assert list(X.columns) == self.columns
        return np.full(len(X), 50.0)


@pytest.fixture(scope="module")
def card():
    return pm.load_model_card()


@pytest.fixture(scope="module")
def models(card):
    """Real models when xgboost is available, otherwise real baseline and
    anomaly detector with a stand-in for the XGBoost estimator."""
    try:
        return pm.load_models()
    except ModuleNotFoundError:
        anomaly = pm.load_bundle("anomaly_detector")
        selected = {"estimator": ConstantModel(card["feature_columns"]),
                    "feature_columns": card["feature_columns"],
                    "history_features": anomaly["history_features"]}
        return pm.Models(selected, pm.load_bundle("baseline_model"), anomaly, card)


@pytest.fixture(scope="module")
def real_models():
    pytest.importorskip("xgboost", reason="xgboost not installed")
    return pm.load_models()


@pytest.fixture(scope="module")
def validation():
    return pd.read_csv(VALIDATION)


@pytest.fixture(scope="module")
def test_fleet():
    try:
        return pm.load_test_fleet()
    except Exception as exc:  # no network or files missing
        pytest.skip(f"NASA test files not available: {exc}")


# --- Model artifacts ------------------------------------------------------

def test_model_files_exist():
    for name in ["selected_model", "baseline_model", "anomaly_detector"]:
        assert (pm.MODELS_DIR / f"{name}.joblib").is_file()
    assert (pm.MODELS_DIR / "model_card.json").is_file()


def test_feature_recipe_matches_model_card(models, card):
    assert models.selected["feature_columns"] == card["feature_columns"]
    assert len(card["feature_columns"]) == 43
    assert models.recipe == models.anomaly["history_features"]
    assert models.recipe["mean_window"] == 10 and models.recipe["slope_window"] == 20


def test_library_versions_match_saved_models(card):
    problems = [m for m in pm.check_versions(card) if "MUST" in m or "not installed" in m]
    if problems:
        pytest.skip("Version mismatch, results may differ: " + "; ".join(problems))


# --- Feature preparation --------------------------------------------------

def test_slope_matches_polyfit(models, validation):
    feats = pm.prepare_features(validation, models)
    engine = feats[feats["unit"] == feats["unit"].iloc[0]]
    window = engine.iloc[41:61]          # 20 cycles ending at row 60
    expected = np.polyfit(window["cycle"], window["sensor_11"], 1)[0]
    assert np.isclose(engine["slope20_sensor_11"].iloc[60], expected)


def test_features_use_no_future_cycles(models, validation):
    """Features at cycle t must not change when later cycles are added."""
    engine = validation[validation["unit"] == 91]
    full = pm.prepare_features(engine, models)
    cut = pm.prepare_features(engine[engine["cycle"] <= 80], models)
    cols = models.selected["feature_columns"]
    pd.testing.assert_frame_equal(full[full["cycle"] <= 80][cols].reset_index(drop=True),
                                  cut[cols].reset_index(drop=True))


def test_input_order_does_not_matter(models, validation):
    engine = validation[validation["unit"] == 91]
    shuffled = engine.sample(frac=1, random_state=0)
    a = pm.score_engines(engine, models)
    b = pm.score_engines(shuffled, models)
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("change, message", [
    (lambda d: d.drop(columns=["sensor_4"]), "Missing columns"),
    (lambda d: d.assign(sensor_2=np.nan), "Missing or non-numeric"),
    (lambda d: pd.concat([d, d.head(1)]), "only once"),
    (lambda d: d[d["cycle"] != 15], "no gaps"),
    (lambda d: d[d["cycle"] > 5], "start at cycle 1"),
    (lambda d: d.assign(cycle=d["cycle"].where(d["cycle"] != 1, 1.9)), "'cycle' must contain whole numbers"),
    (lambda d: d.assign(unit=d["unit"] + 0.5), "'unit' must contain whole numbers"),
    (lambda d: d.assign(sensor_2=d["sensor_2"].where(d["cycle"] != 10, np.inf)), "Non-finite"),
])
def test_bad_input_is_rejected(models, validation, change, message):
    engine = validation[validation["unit"] == 91].head(30)
    with pytest.raises(ValueError, match=message):
        pm.validate_input(change(engine), models)


def test_sample_input_is_valid(models):
    sample = pd.read_csv(SAMPLE)
    scored = pm.score_engines(sample, models)
    assert len(scored) == len(sample)
    assert (scored["predicted_rul"] >= 0).all()


# --- Same results as 04 Modeling (validation engines) ---------------------

def test_reproduces_modeling_validation_predictions(models, validation):
    scored = pm.score_engines(validation, models)
    ref = pd.read_csv(MODELING_PRED)
    merged = scored.merge(ref, on=["unit", "cycle"], suffixes=("", "_ref"))
    assert len(merged) == len(ref)
    np.testing.assert_allclose(merged["baseline_rul"], merged["baseline_pred"], atol=1e-6)
    np.testing.assert_allclose(merged["anomaly_score"], merged["anomaly_score_ref"], atol=1e-9)
    assert (merged["anomaly_flag"] == merged["anomaly_flag_ref"]).all()


def test_reproduces_modeling_xgboost_predictions(real_models, validation):
    scored = pm.score_engines(validation, real_models)
    ref = pd.read_csv(MODELING_PRED)
    merged = scored.merge(ref, on=["unit", "cycle"])
    np.testing.assert_allclose(merged["predicted_rul"], merged["selected_pred"], atol=1e-3)


def test_reproduces_modeling_first_sustained_warning(models, validation):
    scored = pm.score_engines(validation, models)
    first = scored[scored["sustained_warning"]].groupby("unit")["cycle"].min()
    ref = pd.read_csv(MODELING_WARN).set_index("unit")["warning cycle"]
    pd.testing.assert_series_equal(first.sort_index(), ref.sort_index(),
                                   check_names=False)


# --- Same results as 05 Evaluation (official test set) --------------------

def test_reproduces_evaluation_test_predictions(real_models, test_fleet):
    test, labels = test_fleet
    summary = pm.engine_summary(pm.score_engines(test, real_models)).set_index("unit").sort_index()
    ref = pd.read_csv(pm.EVAL_OUTPUTS_DIR / "test_rul_predictions.csv").set_index("unit")
    np.testing.assert_allclose(summary["predicted_rul"], ref["xgb_predicted_rul"], atol=1e-3)
    np.testing.assert_allclose(summary["baseline_rul"], ref["baseline_predicted_rul"], atol=1e-6)
    assert (labels.values == ref["true_rul"].values).all()


def test_reproduces_evaluation_anomaly_warnings(models, test_fleet):
    test, _ = test_fleet
    scored = pm.score_engines(test, models)
    ref = pd.read_csv(pm.EVAL_OUTPUTS_DIR / "sustained_anomaly_warnings.csv")
    merged = scored.merge(ref, on=["unit", "cycle"], suffixes=("", "_ref"))
    assert len(merged) == len(ref) == 13096
    np.testing.assert_allclose(merged["anomaly_score"], merged["anomaly_score_ref"], atol=1e-9)
    assert (merged["sustained_warning"] == merged["sustained_warning_ref"]).all()


# --- Decision display -----------------------------------------------------

def test_decision_labels():
    assert pm.rul_status(10) == "Critical"
    assert pm.rul_status(45) == "Plan inspection"
    assert pm.rul_status(100) == "Monitor"
    assert pm.suggested_action(20, False) == "Inspect first"
    assert pm.suggested_action(100, True).startswith("Take a closer look")
    assert pm.suggested_action(100, False) == "Continue monitoring"


def test_engine_summary_ranks_by_predicted_rul(models, validation):
    summary = pm.engine_summary(pm.score_engines(validation, models))
    assert summary["priority"].tolist() == list(range(1, 21))
    assert summary["predicted_rul"].is_monotonic_increasing
    assert summary["unit"].is_unique
