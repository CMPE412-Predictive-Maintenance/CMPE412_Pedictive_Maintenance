# 06 — Deployment

## Objective

Show how the models from [04 — Modeling](../04_modeling/README.md), evaluated in [05 — Evaluation](../05_evaluation/README.md), could be used by a maintenance planner in the simulated scenario from [01 — Business Understanding](../01_business_understanding/README.md).

05 — Evaluation concluded that the selected model supports a **controlled, simulation-based demonstration with human review**, not autonomous maintenance decisions. This stage follows that conclusion: it is a decision-support demo, not an operational system.

**Demo flow:** select an engine → prepare its sensor features → show the predicted remaining useful life (RUL) → show unusual sensor behaviour **separately** → rank the fleet for inspection.

## Deliverables

| Path | Content |
|---|---|
| `pm_deployment.py` | Shared module: model loading, data download and verification, input validation, feature preparation, prediction, demo labels |
| `notebooks/06_deployment_demo.ipynb` | Notebook demo (user scenario, fleet ranking, reproducibility check, limitations) |
| `app.py` | Streamlit app with the same logic |
| `tests/test_deployment.py` | Smoke and consistency tests |
| `sample_input/new_engine_sample.csv` | Example of a new engine record (format reference) |
| `outputs/fleet_inspection_status.csv` | Created by the notebook: all 100 test engines with prediction, labels and suggested action |
| `requirements.txt` | Pinned library versions |

No model is trained, tuned or re-selected in this stage. The demo loads the saved files in `04_modeling/models/` exactly as they are.

## How to Run

Python **3.11–3.13** is required (numpy 2.4 does not support older versions). Run all commands from the **repository root**.

### 1. Create an environment and install the pinned versions

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r 06_deployment/requirements.txt
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r 06_deployment/requirements.txt
```

The saved models must be loaded with **scikit-learn 1.6.1** and **XGBoost 3.4.1** (recorded in `04_modeling/models/model_card.json`). Other versions print an `InconsistentVersionWarning` and may give different results; the notebook, the app and the tests all report a version mismatch.

### 2. Run the tests

```bash
python -m pytest 06_deployment/tests -v
```

The first run downloads the official NASA archive (about 12 MB) into `06_deployment/data/` and checks each file against the SHA-256 hash recorded in 02 — Data Understanding. The folder is excluded from Git. If `05_evaluation/data/` already holds verified files, they are reused.

Expected result: **17 passed**. Tests that need XGBoost or the NASA files are *skipped* (not failed) when those are missing, and the reason is printed with `-rs`.

### 3. Open the notebook demo

```bash
jupyter notebook 06_deployment/notebooks/06_deployment_demo.ipynb
```

Run all cells. Change `ENGINE_ID` in section 4 to look at another engine. Section 6 stops with an error if the predictions differ from 05 — Evaluation.

### 4. Start the Streamlit app

```bash
streamlit run 06_deployment/app.py
```

The browser opens at `http://localhost:8501`. Tabs:

| Tab | What it shows |
|---|---|
| Engine view | Select an engine; predicted RUL and unusual sensor behaviour in **separate panels**; a replay slider shows what the planner would have seen at an earlier cycle |
| Inspection priority | All 100 engines ranked by predicted RUL, with a number of inspection slots and an option to show the official labels |
| New engine record | Upload a CSV (or use the sample) and get the same outputs; invalid records are rejected with a message |
| About and limitations | How a prediction is made, evaluated performance, limitations |

## How a prediction is made

The same steps run in the notebook, the app and the tests, because all three call `pm_deployment.py`.

1. **Validate the record.** Required columns (below), numeric values, no missing values, one row per `(unit, cycle)`; sort by cycle.
2. **History features.** For each of the 14 retained sensors, computed within each engine from the current and earlier cycles only:
   - `mean10_<sensor>`: mean of the last 10 cycles,
   - `slope20_<sensor>`: least-squares slope over the last 20 cycles.

   The window sizes and the sensor list are read from the saved bundles, not hard-coded.
3. **Predict RUL** with the saved XGBoost model on the 43 features in the saved order (14 sensors, 14 means, 14 slopes, `cycle`). Predictions below 0 are set to 0. The Linear Regression baseline is shown only for comparison.
4. **Unusual sensor behaviour** (separate signal, not a model input). The saved Isolation Forest pipeline scores the 14 sensor means. A cycle is flagged when the score is above the saved threshold (0.558). Five consecutive flags make a **sustained warning**.

Scaling: XGBoost needs none. The baseline and the anomaly detector keep their fitted `StandardScaler` inside their saved pipelines, so it is applied automatically and never refitted.

Required input columns for a new engine record:

```
unit, cycle, setting_1, setting_2, sensor_2, sensor_3, sensor_4, sensor_6, sensor_7, sensor_8,
sensor_9, sensor_11, sensor_12, sensor_13, sensor_14, sensor_15, sensor_17, sensor_20, sensor_21
```

`setting_1`, `setting_2` and `sensor_6` are needed only by the baseline.

## User scenario

A maintenance planner starts a shift with a limited number of inspection slots (01 — Business Understanding).

1. The **Inspection priority** tab lists the engines with the lowest predicted RUL first.
2. The planner opens one engine, for example **engine 37**. The model predicts about 48 cycles, which is outside the critical range. The separate anomaly panel shows a **sustained unusual pattern**, so the planner note says *take a closer look*. The official label is 21 cycles: this is the largest late prediction among the critical test engines in 05 — Evaluation, and the anomaly signal is what draws attention to it.
3. The planner decides. The demo does not schedule anything automatically.

## Demo labels

| Label | Rule |
|---|---|
| `rul_status` | Critical if predicted RUL ≤ 30, Plan inspection if ≤ 60, otherwise Monitor |
| `anomaly_status` | Sustained unusual pattern (warning active) / Isolated unusual readings (a flag in the last 10 cycles) / Normal pattern |
| `suggested_action` | Inspect first if RUL is critical; otherwise *Take a closer look* if a sustained warning is active; otherwise Schedule inspection (≤ 60) or Continue monitoring |

The 30-cycle level is the critical range used in 01 — Business Understanding and 05 — Evaluation. The 60-cycle level is an illustrative planning level. The combined rule follows the suggestion in 04 — Modeling ("predicted RUL ≤ X **or** a sustained warning") and the finding in 05 — Evaluation that 2 of the 3 severely late predictions had an active sustained warning.

**What the combined rule would change on the official test set** (computed from the 05 — Evaluation output files, retrospectively, for discussion only):

| Rule | Critical engines found (true RUL ≤ 30, 25 engines) | Engines marked for attention |
|---|---|---|
| Predicted RUL ≤ 30 | 18 | 19 |
| Predicted RUL ≤ 30 **or** sustained warning | 24 | 37 |

Adding the anomaly signal finds 6 more critical engines, including engines 37 and 41, at the cost of 18 more engines to look at. Engine 18 (true RUL 28, predicted 39) is not found by either rule. This is the trade-off between late and early warnings described in 01 — Business Understanding. These levels were not tuned and have not been validated as a maintenance policy.

## Tests

| Test | What it checks |
|---|---|
| Model files and feature recipe | All bundles exist; 43 features in the order of `model_card.json`; RUL model and anomaly detector use the same history recipe |
| Library versions | scikit-learn and XGBoost match the saved models (skipped with a message otherwise) |
| Slope formula | The fast slope equals `numpy.polyfit` on a 20-cycle window |
| No future cycles | Features at cycle *t* are unchanged when later cycles are added |
| Input order | Shuffled rows give identical results |
| Bad input | Missing columns, missing values and duplicated cycles are rejected |
| Same results as 04 — Modeling | Baseline predictions, anomaly scores, flags and first sustained-warning cycles on the 20 validation engines; XGBoost predictions when XGBoost is installed |
| Same results as 05 — Evaluation | XGBoost and baseline predictions for the 100 test engines, official labels, anomaly scores and sustained warnings for all 13,096 test cycles |
| Demo labels and ranking | Label rules; ranking is ordered by predicted RUL |

## Limitations

- **Simulated data.** FD001 has one operating condition and one fault mode; results may not transfer to real engines.
- **Late predictions remain.** On the test set, 3 of 25 critical engines were predicted more than 10 cycles late (engine 37: +27 cycles).
- **Planning horizon.** The model was trained with RUL capped at 125 cycles. Above about 115 cycles the demo says the engine looks healthy instead of giving a precise number.
- **Anomaly ≠ fault.** Isolation Forest shows unusual sensor patterns compared with early, healthy cycles. FD001 has no labelled anomalies, so precision and recall cannot be measured.
- **Demo thresholds.** The 30 / 60 cycle levels and the combined rule are illustrative.
- **Fleet assumptions.** The model uses the current cycle index, so it assumes lifetimes similar to the training fleet (128–362 cycles).
- **Human review.** Every suggested action is meant to be reviewed by a maintenance planner.
- **Local demo.** The app runs on one computer. User accounts, monitoring and retraining are out of scope.

## Stage boundary

This stage reuses the saved models and the evaluated results. It does not change the model, the features or the thresholds from 04 — Modeling, and it does not use the official RUL labels for anything other than display and the reproducibility check.
