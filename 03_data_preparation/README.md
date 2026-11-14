# 03 — Data Preparation

## Objective

Prepare the NASA C-MAPSS FD001 training data for Remaining Useful Life
(RUL) prediction and provide reproducible inputs for the modeling team.

## Preparation Plan

- Load `train_FD001.txt` using the column names from Data Understanding
  and repeat the basic data quality checks.
- Calculate RUL for each row as the engine's final recorded cycle minus
  its current cycle. Verify that RUL is nonnegative and equals zero at
  each engine's final cycle.
- Split the data by engine ID into training and validation groups.
  Verify that the groups have no engines in common.
- Use a fixed random seed for the engine-level split and document
  the training/validation ratio.
- Recheck constant and low-variability columns using only the training
  group. Document which columns are retained or removed and why.
  Low variability alone does not justify removing a column.
- Keep engine IDs and cycle indices for tracking and chronological
  ordering. Exclude engine IDs, the RUL target, and each engine's final
  recorded cycle from model inputs. Document whether the current
  cycle index is used as a feature.
- Fit any preprocessing transformations using only the training group
  and apply the fitted transformations to the validation group.
- If recent sensor trends are included as features, calculate them
  separately within each engine using only current and earlier cycles.
  Document window sizes and the treatment of short histories.
- Reserve the official test data and RUL labels for final evaluation.
  Do not use them for feature selection, preprocessing decisions,
  or model tuning.

## Deliverables

- A reproducible `02_data_preparation.ipynb` notebook.
- Training and validation tables with clearly identified model features,
  RUL targets, and tracking columns.
- The engine IDs assigned to each group.
- The model feature list and reasons for any removed columns.
- Documentation of the split ratio, random seed, target definition,
  preprocessing steps, and any features based on past sensor readings.
- A usage note explaining how the modeling team should load and use
  the prepared inputs.

## Results

This section will be completed after running the preparation notebook.

It will report:

- The number of engines and rows in each group.
- Confirmation that training and validation share no engine IDs.
- Results of the RUL and data quality checks.
- Retained and removed columns, with reasons.
- Any fitted preprocessing transformations or features based on past
  sensor readings.
- The locations and structure of the exported files.
