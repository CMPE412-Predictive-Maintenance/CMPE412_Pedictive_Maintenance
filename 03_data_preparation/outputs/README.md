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
For final evaluation, select the same feature columns from the official test data
and apply the chosen model's fitted preprocessing. Compare predictions at each
test engine's last observed cycle with its official RUL label.
