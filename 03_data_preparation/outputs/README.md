
# FD001 Data Preparation Handoff

Extract the ZIP. Run the following from the directory containing the exported files.

```python
import json
from pathlib import Path

import pandas as pd

spec = json.loads(Path("feature_spec.json").read_text(encoding="utf-8"))
features = spec["feature_columns"]
target = spec["target_column"]
train = pd.read_csv("train.csv")
validation = pd.read_csv("validation.csv")

X_train, y_train = train[features], train[target]
X_validation, y_validation = validation[features], validation[target]
```

Use the same engine split for model comparison. Use unit as the group key if
running grouped cross-validation within the training set. Fit any scaler in
the model pipeline on training rows only, or on the training fold in each CV split.
For Isolation Forest, use the listed feature columns without the RUL target.
For final evaluation, use the prepared table with one row per official test
engine at its last observed cycle and its official RUL label.

```python
test = pd.read_csv("test_last_cycle.csv")
X_test, y_test = test[features], test[target]
```

Apply the chosen model's fitted preprocessing to `X_test` and score its
predictions against `y_test`. Reserve this table for final evaluation.

For modeling, compare training with the linear RUL target and with the target
capped at 125 cycles using `y_train.clip(upper=125)`. Use the same engine split
and score both against the original validation RUL with MAE and RMSE.

Use the 17 listed features as the baseline. Compare validation MAE after
dropping `sensor_6` (two distinct values, about 98% at the dominant value) and
`setting_1`/`setting_2` (FD001 has a single operating condition). In a separate
experiment, add the current `cycle` as a feature and compare against the same
baseline.
