# Revised preparation validation

All four code cells were executed in a fresh local Python process with IPython using the cached NASA archive. The saved notebook outputs and preparation files were regenerated together. This was a local execution, not a new Google Colab run or a new network download.

- Training: 80 engines, 16,527 rows.
- Validation: 20 engines, 4,104 rows; no shared engine IDs.
- Features: 17; unit/cycle are tracking columns, RUL is the target.
- Test: 100 last-observed-cycle rows, 20 columns (tracking + features + target).
- Each test feature row and official RUL label independently matched the original NASA files in unit-ID order.
- Baseline train/validation data and supporting CSVs matched the previously verified delivery numerically.
- All three input SHA-256 values matched the original NASA files.
- Notebook syntax and local Markdown/image links passed checks.

RUL remains uncapped in exported tables. Capping at 125 is a documented modeling experiment on training targets, scored using original validation RUL. No predictive model or Isolation Forest result is claimed.

The original revised version already contained the principal review updates. This checked version additionally strengthens test sequence validation, validates training IDs 1-100, records all input hashes and target policy, and clarifies the limitations of the anomaly flag-rate criterion. No repository changes were published or merged.
