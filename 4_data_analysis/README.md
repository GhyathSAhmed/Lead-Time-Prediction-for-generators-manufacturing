# 🧠 Lead-Time Prediction — Data Analysis & Modeling

PyCaret Regression Pipeline for Generator Manufacturing

This folder contains all scripts, configuration, and output artifacts used to
build machine-learning models that predict the lead time to finish a generator
based on pre-manufacturing information.

This stage ensuring full reproducibility and transparency.

## 📌 Overview

We use PyCaret Regression to evaluate multiple ML models, select the top
performers, generate evaluation artifacts, and store everything in a timestamped
"run" folder.

This ensures:

- Full separation of modeling runs

- Perfect reproducibility

- No overwriting previous results

- Easy comparison across versions

The goal: Predict the variable

```nginx
target_lead_time_to_finish_days
```

using only features known before production starts.

## 🔧 Pipeline Flow (Detailed)

```mathematica
                         ┌────────────────────┐
                         │ 1. Load Dataset    │
                         │ from 1_datasets/   │
                         └─────────┬──────────┘
                                   │
                                   ▼
                      ┌────────────────────────────┐
                      │ 2. Preprocess For Modeling │
                      │  - Drop leakage columns    │
                      │  - Convert order_date      │
                      │  - Keep target variable    │
                      └─────────┬──────────────────┘
                                 │
                                 ▼
                ┌────────────────────────────────────────┐
                │ 3. PyCaret Setup                       │
                │  - Auto preprocessing                  │
                │  - Train/test split (80/20)            │
                │  - Date feature extraction             │
                │  - Encoding, imputation                │
                └────────────────┬───────────────────────┘
                                 │
                                 ▼
           ┌──────────────────────────────────────────────┐
           │ 4. compare_models()                          │
           │  - Train many algorithms                     │
           │  - Rank by metrics                           │
           │  - Select Top 5 models                       │
           └───────────────────┬──────────────────────────┘
                               │
                               ▼
                ┌─────────────────────────────────┐
                │ 5. Create Run Folder            │
                │ models/run_YYYYMMDD_HHMMSS/     │
                └────────────────┬────────────────┘
                                 │
                                 ▼
         ┌───────────────────────────────────────────────────┐
         │ 6. For Each Top Model:                           │
         │   - Save raw model (.pkl)                        │
         │   - Save metrics                                 │
         │   - Save full predictions                        │
         │   - Save plots (residuals, error, learning...)   │
         │   - Save configuration JSON                      │
         └─────────────────────────┬─────────────────────────┘
                                   │
                                   ▼
                ┌────────────────────────────────────────┐
                │ 7. Finalize Best Model                 │
                │  - Saved separately inside run folder  │
                │    as best_model_<name>.pkl            │
                └────────────────────────────────────────┘
```

## 🎯 Objectives

This script does:

- Loads prepared dataset from 1_datasets/
- Drops leakage columns
- Converts order_date to datetime
- Trains & evaluates models via PyCaret
- Compares and ranks models
- Saves top 5 model artifacts
- Saves the best model in the root of the run folder
- Saves:

  - Metrics
  - Predictions
  - Plots
  - Model config
  - Model object (.pkl)

## 📂 Output Structure (Per Run)

Every execution creates a completely isolated folder:

```lua
4_data_analysis/
└── models/
    └── run_20251206_204501/
        ├── models_comparison.csv
        ├── best_model_gradient_boosting_regressor.pkl   ← 🏆 Final best model
        ├── model_gradient_boosting_regressor/
        │     ├── model.pkl
        │     ├── config.json
        │     ├── metrics.csv
        │     ├── predictions.csv
        │     ├── residuals.png
        │     ├── error.png
        │     ├── learning.png
        │     └── feature.png
        ├── model_random_forest_regressor/
        ├── model_catboost_regressor/
        ├── model_extra_trees_regressor/
        └── model_lightgbm_regressor/
```

## 🧹 Preprocessing Rules

### Kept Columns

- All usable features not derived from future information

- order_date only

#### Dropped Columns (Leakage)

| Type   | Columns                               |
| --------- | ----------------------------|
| Date columns not known at prediction time | `receiving_ckd_date`, `finishing_date`,`shipping_date`|
| Derived lead times| `lead_time_to_finish`, `lead_time_to_ship`, `lead_time_from_ckd`|

### 🤖 PyCaret Configuration

The script internally sets:

| Setting             | Value                            |
| ------------------- | -------------------------------- |
| Train/test split    | 80 / 20                          |
| Cross-validation    | 5-fold                           |
| Session seed        | 42                               |
| date_features       | `["order_date"]`                 |
| Engineered features | Day, month, year, week, etc.     |
| Encoding            | Automatic                        |
| Imputation          | Automatic                        |
| Feature scaling     | Automatic (if model requires it) |

### 🧪 Model Evaluation

For each model, we save:

| Metric | Description                  |
| ------ | ---------------------------- |
| MAE    | Mean Absolute Error          |
| MSE    | Mean Squared Error           |
| RMSE   | Root Mean Squared Error      |
| R²     | Coefficient of determination |
| MAPE   | Percentage error             |
| RMSLE  | Log loss error               |

Additionally, PyCaret generates:

- Residuals plot

- Error plot

- Feature importance

- Learning curve

### 💾 Saved Artifacts

For EVERY model (top 5):

```lua
model.pkl
config.json
metrics.csv
predictions.csv
residuals.png
error.png
learning.png
feature.png
```

For BEST model ONLY:

```php-template
best_model_<name>.pkl
```

Stored directly in the run directory.

### ▶️ Running the Script

From the project root:

```bash
python 4_data_analysis/model_lead_time_pycaret.py
```

Artifacts will appear inside:

```bash
4_data_analysis/models/run_YYYYMMDD_HHMMSS/
```

### 🔁 Reproducibility

This script guarantees:

- No dataset modification
- Each run stored independently
- Fixed random seed
- Full config saved as JSON
- All metrics saved

Anyone can re-run the script and reproduce the results exactly.
