# 05 — Evaluation

## 1. Evaluation Objective & Business Context

This stage evaluates the predictive maintenance models developed during the Modeling phase of the CRISP-DM process.

Our project focuses on supporting maintenance planning by predicting the Remaining Useful Life (RUL) of simulated aircraft engines using NASA C-MAPSS FD001 sensor data.

The main objective is to determine whether the selected XGBoost model can provide sufficiently accurate RUL predictions on the previously unseen official test dataset.

In addition to prediction accuracy, we examine the practical implications of prediction errors for maintenance decisions. In particular, overestimating the remaining useful life of an engine close to failure may delay inspection and increase operational risk.

We also examine anomaly detection outputs to understand whether unusual sensor behavior can support early maintenance warnings.

The evaluation follows the objectives and success criteria defined in Business Understanding and uses the data preparation procedures and trained models established in the previous CRISP-DM stages.

Final test results will be reported only after the evaluation has been completed.