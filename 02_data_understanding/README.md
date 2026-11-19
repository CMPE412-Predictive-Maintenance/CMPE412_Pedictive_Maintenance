[README.md](https://github.com/user-attachments/files/35086379/README.md)

# 02 — Data Understanding

## Objective

This CRISP-DM stage examines the structure, quality and sensor behavior of NASA C-MAPSS FD001 before data preparation and modeling. It supports the project's maintenance planning objective by identifying patterns and limitations that may matter for Remaining Useful Life (RUL) prediction.

## Deliverables

- [Data Understanding notebook](notebooks/02_data_understanding.ipynb)
- [Data Understanding report](docs/02_data_understanding.md)

The notebook contains the analysis code, explanations and saved outputs. The report summarizes findings and provides a handoff to Data Preparation. Figures and supporting tables are stored in this stage's `artifacts/` directory.

## Dataset and Scope

FD001 contains simulated engine trajectories under one operating condition and one fault mode: high-pressure compressor (HPC) degradation. Each sensor record has 26 columns: an engine identifier, a cycle index, three operating settings and 21 sensor measurements. Cycles are not hours.

| File | Role in this stage |
| --- | --- |
| `train_FD001.txt` | Training trajectories through failure; used for exploratory analysis. |
| `test_FD001.txt` | Truncated test trajectories; checked structurally and reserved for final evaluation. |
| `RUL_FD001.txt` | Remaining cycles after each test engine's last observation; checked structurally and reserved for final evaluation. |

[Official NASA dataset description](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data) · [NASA PCoE repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

## Analysis Performed

- Verified file structure and expected dataset sizes.
- Checked missing and infinite values, duplicate rows and engine–cycle keys.
- Checked cycle order, gaps and starting cycles within each training engine.
- Examined observed training record lengths, selected sensor trajectories and all sensor distributions.
- Identified constant and exploratory low-variability candidates.
- Examined Pearson correlations among nonconstant training sensors.

## Findings

The training data contain **100 engines, 20,631 cycle records and 21 sensors**. Checks found **no missing or infinite values, duplicate rows, duplicate engine–cycle keys, or cycle sequence issues**.

Observed training record lengths range from **128 to 362 cycles**, with a median of **199** and a mean of approximately **206.3**. These are recorded run-to-failure lengths, not physical ages since manufacture.

Sensors **1, 5, 10, 16, 18 and 19** are exactly constant in the explored training data. Sensors **6, 8 and 13** are low-variability candidates under the notebook's stated exploratory thresholds. No columns were removed. Operating setting `setting_3` was also constant and was removed in Data Preparation after confirming it was constant in the training split.

For the three selected engines (**39, 68 and 69**), **sensor_4 and sensor_11 generally rise toward the end of the recorded trajectories, while sensor_7 generally falls**. This observation is limited to these three examples and does not establish a universal fleet-wide trend.

The strongest absolute Pearson correlation is between **sensor_9 and sensor_14**, approximately **0.963**. Correlation does not establish causality or predictive usefulness on its own.

## Implications for Data Preparation

1. Split the official training data by engine, ensuring no engine appears in both training and validation groups.
2. Recompute constant, low-variability and correlation checks on the designated training engines before deciding which columns to remove. Low variability or high correlation alone is not sufficient justification for removal.
3. Consider recent sensor trends, such as trailing-window slopes or changes in moving averages, as features to evaluate. Compute them separately within each engine using only current and earlier cycles.
4. Fit transformations and feature-selection procedures only on the designated training partition. Use validation data for model and parameter selection.
5. Construct RUL targets in Data Preparation and keep targets, future-derived information and engine identifiers out of model inputs.
6. Reserve the official test and RUL files for final evaluation.

## Running the Notebook

Open the notebook in Google Colab and select **Runtime → Run all**. It first attempts to download the NASA archive automatically and resolves nested ZIP files. If both download sources fail, it offers manual upload of the archive or all three required TXT files.

A successful run generates the English report, figures, supporting tables and `data_understanding_results.zip` in the runtime working directory. Download runtime files before ending the session, and save the executed notebook separately.

## Stage Boundary

This stage does not construct RUL targets, create training/validation splits, fit preprocessing transformations, train models or report prediction performance. Those tasks belong to subsequent CRISP-DM stages.
