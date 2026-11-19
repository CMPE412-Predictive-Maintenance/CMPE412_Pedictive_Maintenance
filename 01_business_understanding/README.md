[README.md](https://github.com/user-attachments/files/35085863/README.md)
## 1. Business Understanding

### 1.1 Business Problem and Scenario

Unexpected aircraft engine failures can delay flights, disrupt operations, and increase maintenance costs. When the maintenance team has limited inspection capacity, the planner needs to decide which engines should be inspected first.

In our fictional scenario, a maintenance planner starts a shift with one inspection slot and two engines to review. Engine A has an estimated 15 operating cycles remaining, while Engine B has an estimated 60. Based on these estimates, the planner prioritizes Engine A for inspection. An unusual sensor pattern could also give the planner a reason to take a closer look. These figures are illustrative.

Inspecting an engine too early can use capacity needed elsewhere. Waiting too long can increase the risk of unexpected failure and unplanned downtime. The project explores how sensor data can inform that decision.

### 1.2 Project Objectives

The main objective is to support maintenance planning by estimating each engine's **Remaining Useful Life (RUL)** in operating cycles. The model will use sensor measurements available up to the current cycle. The maintenance planner can use these estimates to prioritize inspections.

We will also use **anomaly detection** to flag unusual sensor behavior for further investigation. These flags indicate unusual patterns that may warrant a closer look, but they do not necessarily mean that an engine is failing.

We will compare several machine learning models to assess how accurately they predict RUL.

### 1.3 Dataset and Project Scope

We will use the **NASA C-MAPSS FD001 dataset**, which contains simulated aircraft engine operating conditions and sensor measurements recorded over successive cycles.

We will use this dataset to explore engine behavior, prepare model inputs, and train and evaluate prediction models. The project is an educational prototype that demonstrates how sensor data could support inspection planning in a simulated setting.

The prototype is not intended to replace professional maintenance decisions or operate as a real aircraft maintenance system.

### 1.4 Success Criteria

We will evaluate RUL predictions using two metrics, both reported in operating cycles:

- **Mean Absolute Error (MAE)** measures how far predictions are from the actual RUL on average.
- **Root Mean Squared Error (RMSE)** gives greater weight to large prediction errors.

Lower MAE and RMSE values indicate better prediction performance.

We will use **Linear Regression** as the baseline model. All models will use the same engine-level training and validation split.

We will select the best candidate model based on validation MAE. The official test set will be used only for final evaluation. Our goal is to achieve a lower test MAE than the Linear Regression baseline.

We will also report RMSE to assess the impact of large prediction errors.

We will examine the direction of errors because it matters for maintenance planning:

- **Overestimating RUL:** Can delay an inspection, especially when an engine has few cycles remaining.
- **Underestimating RUL:** Can move an inspection forward unnecessarily and use capacity needed for other engines.

### 1.5 Expected Business Value

The prototype illustrates how RUL estimates and unusual sensor patterns could help a maintenance planner decide which engines to inspect first.

Better prioritization could help the maintenance team use limited inspection capacity more effectively and potentially reduce unexpected operational disruptions.

However, measuring the actual impact on maintenance costs or unplanned downtime would require real operational data. These benefits are expected outcomes rather than proven results.
