"""Shared deployment logic for the FD001 predictive maintenance demo.

The notebook (`notebooks/06_deployment_demo.ipynb`), the Streamlit app (`app.py`)
and the tests (`tests/test_deployment.py`) all import this module, so every
entry point prepares data and produces predictions in exactly the same way.

The feature recipe below reproduces `add_history_features` from
`04_modeling/notebooks/04_modeling.ipynb`. The window sizes and the sensor list
are read from the saved model bundles, not hard-coded.
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEPLOY_DIR = Path(__file__).resolve().parent
REPO_DIR = DEPLOY_DIR.parent
MODELS_DIR = REPO_DIR / "04_modeling" / "models"
EVAL_OUTPUTS_DIR = REPO_DIR / "05_evaluation" / "outputs"
INVENTORY_PATH = (REPO_DIR / "02_data_understanding" / "artifacts"
                  / "data_understanding" / "file_inventory.csv")

# The NASA test files are not stored in Git. The demo downloads them once into
# DATA_DIR. Files already downloaded by 05 Evaluation are reused if present.
DATA_DIR = DEPLOY_DIR / "data"
DATA_DIRS = [DATA_DIR, REPO_DIR / "05_evaluation" / "data"]

RAW_COLUMNS = (["unit", "cycle"]
               + [f"setting_{i}" for i in range(1, 4)]
               + [f"sensor_{i}" for i in range(1, 22)])

# ---------------------------------------------------------------------------
# Demo decision settings
# ---------------------------------------------------------------------------
# These thresholds are illustrative demo settings for the simulated planner
# scenario in 01 Business Understanding. They are NOT a validated maintenance
# policy (see 05 Evaluation, section 7.4).
CRITICAL_RUL = 30        # predicted RUL at or below this -> inspect first
PLAN_RUL = 60            # predicted RUL at or below this -> schedule inspection
RUL_CAP = 125            # training cap used in 04 Modeling


# ---------------------------------------------------------------------------
# Loading models
# ---------------------------------------------------------------------------

def load_model_card(models_dir: Path = MODELS_DIR) -> dict:
    with open(models_dir / "model_card.json", "r", encoding="utf-8") as f:
        return json.load(f)


def check_versions(card: dict | None = None) -> list[str]:
    """Return warnings for library versions that differ from the ones used
    to save the models. scikit-learn and xgboost must match exactly."""
    card = card or load_model_card()
    expected = card["versions"]
    messages = []
    import sklearn
    installed = {"scikit-learn": sklearn.__version__,
                 "numpy": np.__version__, "pandas": pd.__version__}
    try:
        import xgboost
        installed["xgboost"] = xgboost.__version__
    except ImportError:
        installed["xgboost"] = None
    for lib in ["scikit-learn", "xgboost", "numpy", "pandas"]:
        have, want = installed.get(lib), expected.get(lib)
        if have is None:
            messages.append(f"{lib} is not installed (models were saved with {want}).")
        elif want and have != want:
            level = "MUST match" if lib in ("scikit-learn", "xgboost") else "recommended"
            messages.append(f"{lib} {have} installed, models saved with {want} ({level}).")
    return messages


def load_bundle(name: str, models_dir: Path = MODELS_DIR) -> dict:
    """Load one saved bundle: 'selected_model', 'baseline_model' or
    'anomaly_detector'. Loading 'selected_model' requires xgboost."""
    return joblib.load(models_dir / f"{name}.joblib")


@dataclass
class Models:
    selected: dict
    baseline: dict
    anomaly: dict
    card: dict

    @property
    def recipe(self) -> dict:
        return self.selected["history_features"]


def load_models(models_dir: Path = MODELS_DIR) -> Models:
    card = load_model_card(models_dir)
    models = Models(selected=load_bundle("selected_model", models_dir),
                    baseline=load_bundle("baseline_model", models_dir),
                    anomaly=load_bundle("anomaly_detector", models_dir),
                    card=card)
    if models.selected["feature_columns"] != card["feature_columns"]:
        raise ValueError("Selected model feature list does not match model_card.json")
    if models.selected["history_features"] != models.anomaly["history_features"]:
        raise ValueError("RUL model and anomaly detector use different history recipes")
    return models


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def read_cmapss_txt(source) -> pd.DataFrame:
    """Read a raw NASA C-MAPSS text file (path or bytes) with named columns."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    return pd.read_csv(source, sep=r"\s+", header=None, names=RAW_COLUMNS)


def _inventory() -> pd.DataFrame:
    return pd.read_csv(INVENTORY_PATH)


def _verify(filename: str, content: bytes, inventory: pd.DataFrame) -> None:
    record = inventory.loc[inventory["file"] == filename].iloc[0]
    if hashlib.sha256(content).hexdigest() != str(record["sha256"]).strip():
        raise ValueError(f"SHA-256 mismatch for {filename}")
    if len(content) != int(record["bytes"]):
        raise ValueError(f"File size mismatch for {filename}")


def ensure_test_data(data_dir: Path = DATA_DIR, verbose: bool = True) -> Path:
    """Return a directory that contains verified test_FD001.txt and
    RUL_FD001.txt. Downloads the official NASA archive only if needed."""
    inventory = _inventory()
    names = ["test_FD001.txt", "RUL_FD001.txt"]

    for folder in [data_dir] + [d for d in DATA_DIRS if d != data_dir]:
        try:
            for name in names:
                _verify(name, (folder / name).read_bytes(), inventory)
            if verbose:
                print(f"Using verified NASA files in {folder}")
            return folder
        except (FileNotFoundError, ValueError):
            continue

    url = inventory.loc[inventory["file"] == "test_FD001.txt", "source"].iloc[0].split("!")[0]
    if verbose:
        print("Downloading the official NASA C-MAPSS archive (about 12 MB)...")
    try:
        import certifi
        import ssl
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = None
    with urllib.request.urlopen(url, timeout=120, context=context) as response:
        archive = response.read()
    with zipfile.ZipFile(io.BytesIO(archive)) as outer:
        inner_name = next(n for n in outer.namelist() if n.endswith("CMAPSSData.zip"))
        inner_bytes = outer.read(inner_name)
    data_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
        for name in names:
            content = inner.read(name)
            _verify(name, content, inventory)
            (data_dir / name).write_bytes(content)
    if verbose:
        print(f"Downloaded and verified NASA files into {data_dir}")
    return data_dir


def load_test_fleet(data_dir: Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Official FD001 test trajectories and the official RUL labels.
    The labels are only used to show the true value next to a prediction."""
    folder = ensure_test_data(data_dir or DATA_DIR, verbose=False)
    test = read_cmapss_txt(folder / "test_FD001.txt")
    labels = pd.read_csv(folder / "RUL_FD001.txt", sep=r"\s+", header=None,
                         names=["true_rul"])["true_rul"]
    labels.index = pd.RangeIndex(1, len(labels) + 1, name="unit")
    return test, labels


# ---------------------------------------------------------------------------
# Input validation and feature preparation
# ---------------------------------------------------------------------------

def required_input_columns(models: Models) -> list[str]:
    """Raw columns a new engine record must contain."""
    needed = set(models.baseline["feature_columns"]) | set(models.recipe["sensors"])
    needed |= {c for c in models.selected["feature_columns"]
               if not c.startswith(("mean", "slope"))}
    return ["unit", "cycle"] + sorted(needed - {"unit", "cycle"},
                                      key=lambda c: RAW_COLUMNS.index(c))


def validate_input(df: pd.DataFrame, models: Models) -> pd.DataFrame:
    """Check a new engine record and return it sorted by unit and cycle.
    Raises ValueError with a readable message if the record is not usable."""
    missing = [c for c in required_input_columns(models) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    data = df.copy()
    cols = required_input_columns(models)
    for c in cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    if data[cols].isna().any().any():
        bad = data[cols].columns[data[cols].isna().any()].tolist()
        raise ValueError(f"Missing or non-numeric values in: {bad}")
    if data.duplicated(["unit", "cycle"]).any():
        raise ValueError("Each (unit, cycle) pair must appear only once")
    if (data["cycle"] < 1).any():
        raise ValueError("Cycle numbers must start at 1")
    data["unit"] = data["unit"].astype(int)
    data["cycle"] = data["cycle"].astype(int)
    return data.sort_values(["unit", "cycle"]).reset_index(drop=True)


def add_history_features(df: pd.DataFrame, recipe: dict) -> pd.DataFrame:
    """Trailing mean and least-squares slope per sensor, computed within each
    engine from the current and earlier cycles only. Same math as 04 Modeling.
    `df` must be sorted by unit and cycle (validate_input does this)."""
    mean_w, slope_w = recipe["mean_window"], recipe["slope_window"]
    out = df.copy()

    def trailing_mean(series, window):
        return series.groupby(out["unit"]).transform(
            lambda v: v.rolling(window, min_periods=1).mean())

    t = out["cycle"].astype(float)
    m_t = trailing_mean(t, slope_w)
    var_t = trailing_mean(t * t, slope_w) - m_t ** 2

    new_columns = {}
    for s in recipe["sensors"]:
        x = out[s]
        new_columns[f"mean{mean_w}_{s}"] = trailing_mean(x, mean_w)
        cov_tx = trailing_mean(t * x, slope_w) - m_t * trailing_mean(x, slope_w)
        new_columns[f"slope{slope_w}_{s}"] = np.where(var_t > 1e-9, cov_tx / var_t, 0.0)
    return pd.concat([out, pd.DataFrame(new_columns, index=out.index)], axis=1)


def prepare_features(df: pd.DataFrame, models: Models) -> pd.DataFrame:
    """Validate a raw record and add the history features the models need."""
    return add_history_features(validate_input(df, models), models.recipe)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def _sustained(flags: pd.Series, units: pd.Series, run: int) -> tuple[pd.Series, pd.Series]:
    """Consecutive flagged cycles within each engine and the sustained flag."""
    groups = flags.groupby(units).transform(lambda s: s.ne(s.shift()).cumsum())
    consecutive = flags.groupby([units, groups]).cumcount() + 1
    consecutive = consecutive.where(flags, 0)
    return consecutive, consecutive >= run


def score_engines(df: pd.DataFrame, models: Models, include_baseline: bool = True) -> pd.DataFrame:
    """Score every cycle of every engine in a raw record.

    Each row answers: "with the data available up to this cycle, what is the
    predicted RUL and is the sensor behaviour unusual?" Because all features
    use only current and earlier cycles, the last row of an engine equals the
    prediction made when that cycle is the newest observation.
    """
    feats = prepare_features(df, models)
    result = feats[["unit", "cycle"]].copy()

    rul = models.selected["estimator"].predict(feats[models.selected["feature_columns"]])
    result["predicted_rul"] = np.clip(rul, 0, None)
    if include_baseline:
        base = models.baseline["estimator"].predict(feats[models.baseline["feature_columns"]])
        result["baseline_rul"] = np.clip(base, 0, None)

    a = models.anomaly
    result["anomaly_score"] = -a["pipeline"].score_samples(feats[a["feature_columns"]])
    result["anomaly_flag"] = result["anomaly_score"] > a["threshold"]
    result["consecutive_flags"], result["sustained_warning"] = _sustained(
        result["anomaly_flag"], result["unit"], int(a["sustained_run"]))
    return result


def rul_status(predicted_rul: float) -> str:
    if predicted_rul <= CRITICAL_RUL:
        return "Critical"
    if predicted_rul <= PLAN_RUL:
        return "Plan inspection"
    return "Monitor"


def anomaly_status(sustained_warning: bool, recent_flags: int) -> str:
    if sustained_warning:
        return "Sustained unusual pattern"
    if recent_flags > 0:
        return "Isolated unusual readings"
    return "Normal pattern"


def suggested_action(predicted_rul: float, sustained_warning: bool) -> str:
    """Planner note that combines both signals while keeping them visible.
    Demo rule only: inspect when predicted RUL <= CRITICAL_RUL, and take a
    closer look when a sustained anomaly warning is active."""
    if predicted_rul <= CRITICAL_RUL:
        return "Inspect first"
    if sustained_warning:
        return "Take a closer look (unusual sensor pattern)"
    if predicted_rul <= PLAN_RUL:
        return "Schedule inspection"
    return "Continue monitoring"


def engine_summary(scored: pd.DataFrame, recent_window: int = 10) -> pd.DataFrame:
    """One row per engine at its latest observed cycle, ranked by predicted RUL."""
    scored = scored.sort_values(["unit", "cycle"])
    recent = (scored.groupby("unit").tail(recent_window)
              .groupby("unit")["anomaly_flag"].sum().rename("recent_flags"))
    last = scored.groupby("unit").tail(1).set_index("unit").join(recent).reset_index()
    last["rul_status"] = last["predicted_rul"].map(rul_status)
    last["anomaly_status"] = [anomaly_status(w, int(n)) for w, n in
                              zip(last["sustained_warning"], last["recent_flags"])]
    last["suggested_action"] = [suggested_action(r, w) for r, w in
                                zip(last["predicted_rul"], last["sustained_warning"])]
    last = last.sort_values(["predicted_rul", "unit"]).reset_index(drop=True)
    last.insert(0, "priority", np.arange(1, len(last) + 1))
    return last.rename(columns={"cycle": "last_cycle"})


def describe_rul(predicted_rul: float) -> str:
    """Text for the planner. Above the training cap the model only says the
    engine looks healthy; it does not estimate long remaining lives."""
    if predicted_rul >= RUL_CAP - 10:
        return (f"about {predicted_rul:.0f} cycles: beyond the {RUL_CAP}-cycle planning "
                "horizon, the model only indicates a healthy engine")
    return f"about {predicted_rul:.0f} cycles"
