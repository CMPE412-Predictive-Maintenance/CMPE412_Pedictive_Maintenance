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

The FD001 training file contained 20,631 records from 100 engines.
The notebook split the data by engine ID using an 80/20 split and
a random seed of 42.

| Split | Engines | Records |
| --- | ---: | ---: |
| Training | 80 | 16,527 |
| Validation | 20 | 4,104 |

The splits share no engine IDs. RUL is nonnegative for every record
and equals zero at the final recorded cycle of each engine.

Seven columns were constant in the training split and were removed:
`setting_3`, `sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`,
`sensor_18`, and `sensor_19`. The exported tables retain 17 model
features. `unit` and `cycle` are tracking columns, and `RUL` is the target.

This baseline uses current sensor readings and operating settings.
No scaling or rolling features were applied. The official test files
remain reserved for final evaluation.

Running the notebook generates `data_preparation_results.zip`, which
contains the training and validation tables, engine assignments,
feature specification, quality checks, and modeling instructions.
