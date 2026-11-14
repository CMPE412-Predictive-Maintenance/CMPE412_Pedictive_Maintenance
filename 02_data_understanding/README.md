# 02 - Data Understanding

## Objective

In Business Understanding, we defined the maintenance planner's decision:
which engine should be inspected first? In this stage, we explore whether
sensor measurements show patterns that could help estimate an engine's
remaining useful life (RUL).

## Dataset

We use the FD001 subset of NASA's C-MAPSS Turbofan Engine Degradation
Simulation dataset. The records include engine identifiers, operating
cycles, operating settings, and sensor measurements.

Open the [dataset page on Zenodo](https://zenodo.org/records/15346912)
and download `CMAPSSData.zip` from the Files section.

| File | Role |
| --- | --- |
| `train_FD001.txt` | Engine records through failure. Used for exploratory analysis and later split by engine for model training and validation. |
| `test_FD001.txt` | Engine records that end before failure. Reserved for final evaluation. |
| `RUL_FD001.txt` | Actual RUL at the last observed cycle of each test engine. Used for final evaluation. |

## Exploratory Analysis

- Document the columns and count the engines, rows, and sensor columns in the training file.
- Check for missing values, duplicate engine-cycle records, and gaps in cycle sequences.
- Identify constant sensor columns in the training data. Review the findings before deciding which columns to remove during Data Preparation.
- Plot engine lifetimes and sensor readings over time for selected engines.
- Examine sensor distributions and correlations.

## Findings and Implications for Data Preparation

This section will be completed after the exploratory analysis.
