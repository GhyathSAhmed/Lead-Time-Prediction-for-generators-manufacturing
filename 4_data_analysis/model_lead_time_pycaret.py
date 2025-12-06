"""
model_lead_time_pycaret.py

Data Analysis script: Train a regression model to predict generator lead time
to finish using PyCaret, following the project Data Analysis Guide.

This script:

1. Reads the prepared dataset from `0_datasets/ELO2_lead_time_prepared.csv`.
2. Drops columns that cause data leakage:
   - All date columns except `order_date`
   - All lead-time columns (since they are derived from dates)
3. Converts `order_date` from object to datetime and lets PyCaret handle it
   as a date feature.
4. Uses PyCaret's regression module to:
   - Shuffle and split the data into 80% train / 20% test
   - Encode categorical columns
   - Train and compare multiple regression models
   - Select and finalize the best model
5. Saves the best model to a `models/` folder.

It does NOT modify or overwrite any dataset in `0_datasets/`.

Typical usage (from project root):
    python 3_data_analysis/model_lead_time_pycaret.py
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
from pycaret.regression import (
    compare_models,
    finalize_model,
    plot_model,
    predict_model,
    pull,
    save_model,
    setup,
)

# Suppress pandas / PyCaret FutureWarnings that cause noisy logs and encoding issues
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
)


# On Windows, make stdout/stderr UTF-8 to avoid UnicodeEncodeError in logs

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Project root is assumed to be one level above this script's folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# According to the Data Analysis Guide, we read from 1_datasets/
DATA_DIR = PROJECT_ROOT / "1_datasets"
INPUT_FILENAME = "ELO2_lead_time_prepared.csv"

# Where to store trained models (inside this analysis folder)
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"

TARGET_COLUMN = "target_lead_time_to_finish_days"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def ensure_models_dir() -> None:
    """
    Create the folder used to store trained models if it does not exist.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    """
    Load the prepared ELO2 dataset from `1_datasets`.

    Returns:
        A pandas DataFrame with the prepared dataset.

    Raises:
        FileNotFoundError: If the dataset file is not found.
    """
    path = DATA_DIR / INPUT_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Prepared dataset not found at: {path}")
    df = pd.read_csv(path)
    return df


def preprocess_for_modeling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset for modeling:

    - Drop leakage-prone columns:
        * All date columns except 'order_date'
        * All lead-time columns (derived from dates)
    - Convert 'order_date' from object to datetime.

    The target column is left in the DataFrame because PyCaret expects it.

    Args:
        df: Raw prepared DataFrame as loaded from disk.

    Returns:
        A cleaned DataFrame ready to be passed to PyCaret's `setup`.
    """
    df = df.copy()

    # Convert order_date to datetime (currently typed as object)
    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

    # Columns we explicitly want to drop (if present)
    date_cols_to_drop = [
        "receiving_ckd_date",
        "finishing_date",
        "shipping_date",
    ]

    lead_time_cols_to_drop = [
        "lead_time_to_finish",
        "lead_time_to_ship",
        "lead_time_from_ckd",
    ]

    cols_to_drop = []
    for col in date_cols_to_drop + lead_time_cols_to_drop:
        if col in df.columns:
            cols_to_drop.append(col)

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # Safety check: ensure target column exists
    if TARGET_COLUMN not in df.columns:
        raise KeyError(
            f"Target column '{TARGET_COLUMN}' not found in DataFrame columns: "
            f"{list(df.columns)}"
        )

    print("[PREPROCESS] Dropped columns:", cols_to_drop)
    print("[PREPROCESS] Remaining columns:", list(df.columns))

    return df


def run_pycaret_regression(
    df: pd.DataFrame,
    n_top_models: int = 5,
) -> tuple:
    """
    Initialize PyCaret, compare multiple models, and return:
        - finalized best model
        - list of top models (ordered, best first)
        - comparison metrics DataFrame from PyCaret
        - experiment configuration used for setup()

    Args:
        df: Prepared modeling DataFrame.
        n_top_models: Number of top models to keep and save artifacts for.

    Returns:
        final_model, top_models, comparison_df, experiment_config
    """
    print("[PYCARET] Initializing regression setup...")

    # Detect date column(s) – we expect 'order_date' to be present as datetime
    date_features = [col for col in df.columns if "order_date" in col]

    # Configuration that we also want to persist
    experiment_config = {
        "target": TARGET_COLUMN,
        "train_size": 0.8,
        "session_id": 42,
        "fold": 5,
        "date_features": date_features or [],
        "n_jobs": 1,
    }

    exp = setup(
        data=df,
        target=experiment_config["target"],
        train_size=experiment_config["train_size"],
        session_id=experiment_config["session_id"],
        fold=experiment_config["fold"],
        date_features=experiment_config["date_features"] or None,
        n_jobs=experiment_config["n_jobs"],
        verbose=False,
    )

    print("[PYCARET] Comparing models...")
    top_models = compare_models(n_select=n_top_models)

    # Get comparison metrics table (all models that PyCaret tried)
    comparison_df = pull().copy()

    # Ensure top_models is a list
    if not isinstance(top_models, list):
        top_models = [top_models]

    # Finalize the best model (first in the list)
    best_model = top_models[0]
    final_model = finalize_model(best_model)
    print("[PYCARET] Model training and selection complete.")

    return final_model, top_models, comparison_df, experiment_config


def evaluate_model(model) -> pd.DataFrame:
    """
    Use PyCaret's predict_model to evaluate the finalized model on the holdout
    set created during setup.

    Args:
        model: Finalized PyCaret regression model.

    Returns:
        A DataFrame with predictions on the holdout set.
    """
    print("[PYCARET] Generating predictions and evaluation metrics...")
    results = predict_model(model)
    print("[PYCARET] Predictions and evaluation metrics finished successfully.")
    return results


def save_best_model(model) -> Path:
    """
    Save the finalized best model into the `models` directory (legacy helper).

    Returns:
        Path to the saved model file (without extension as per PyCaret's API).
    """
    ensure_models_dir()
    model_path_prefix = MODELS_DIR / "lead_time_best_model"
    save_model(model, str(model_path_prefix))
    print(f"[MODEL] Best model saved with prefix: {model_path_prefix}")
    return model_path_prefix


def create_run_directory() -> Path:
    """
    Create a unique directory for this analysis run inside `MODELS_DIR`.

    Returns:
        Path to the created run directory, e.g.
        4_data_analysis/models/run_20250318_203015
    """
    ensure_models_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = MODELS_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[RUN] Saving all models, metrics, and plots to: {run_dir}")
    return run_dir


def save_model_with_artifacts(
    model,
    model_name: str,
    run_dir: Path,
    include_plots: bool = True,
) -> None:
    """
    Save a single model, its configuration, holdout metrics, and plots into
    `run_dir/model_<name>/`.

    Artifacts saved:
        - model.pkl                  (full PyCaret pipeline)
        - model_config.json          (estimator class + hyperparameters + repr)
        - predictions.csv            (holdout predictions from predict_model)
        - metrics_summary.csv        (PyCaret metrics from pull())
        - *.png plots from plot_model (residuals, error, feature, learning)
    """
    # Folder name
    safe_name = (
        model_name.lower().replace(" ", "_").replace("/", "_").replace("\\", "_")
    )
    model_dir = run_dir / f"model_{safe_name}"
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"[MODEL] Saving artifacts for: {model_name} -> {model_dir}")

    # ---------------------------
    # 0) Extract base estimator & configuration
    # ---------------------------
    if hasattr(model, "steps") and len(model.steps) > 0:
        base_estimator = model.steps[-1][1]
        base_name = model.steps[-1][0]
    else:
        base_estimator = model
        base_name = model.__class__.__name__

    # Try to get hyperparameters
    params_raw = {}
    try:
        params_raw = base_estimator.get_params(deep=True)
    except (AttributeError, TypeError) as e:
        print(f"[WARN] Could not extract hyperparameters for {model_name}: {e}")
        params_raw = {}

    def make_json_safe(obj):
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        if isinstance(obj, dict):
            return {str(k): make_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [make_json_safe(v) for v in obj]
        return repr(obj)

    params_safe = make_json_safe(params_raw)

    model_config = {
        "model_name_label": model_name,
        "pipeline_step_name": base_name,
        "estimator_class": base_estimator.__class__.__name__,
        "estimator_module": base_estimator.__class__.__module__,
        "estimator_params": params_safe,
        "full_pipeline_repr": repr(model),
    }

    with (model_dir / "model_config.json").open("w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)

    # ---------------------------
    # 1) Save model (pkl)
    # ---------------------------
    model_path_prefix = model_dir / "model"
    save_model(model, str(model_path_prefix))
    print(f"[MODEL] Saved model to: {model_path_prefix}.pkl")

    # ---------------------------
    # 2) Evaluate on holdout & save metrics
    # ---------------------------
    results = predict_model(model)  # holdout predictions
    metrics_df = pull().copy()  # PyCaret metrics summary for this call

    results.to_csv(model_dir / "predictions.csv", index=False)
    metrics_df.to_csv(model_dir / "metrics_summary.csv", index=False)
    print(f"[MODEL] Saved predictions & metrics for: {model_name}")

    # ---------------------------
    # 3) Save plots
    # ---------------------------
    if include_plots:
        cwd = os.getcwd()
        try:
            os.chdir(model_dir)
            for plot_type in ["residuals", "error", "feature", "learning"]:
                try:
                    plot_model(model, plot=plot_type, save=True)
                    print(f"[MODEL] Saved plot '{plot_type}' for: {model_name}")
                except (ValueError, RuntimeError, FileNotFoundError) as e:
                    print(
                        f"[WARN] Could not generate plot '{plot_type}' for {model_name}: {e}"
                    )
        finally:
            os.chdir(cwd)


def save_run_best_model(model, best_model_name: str, run_dir: Path) -> Path:
    """
    Save the best model of this run directly inside `run_dir`, parallel to
    the individual model folders.

    Example output:
        run_20251206_203000/
            best_model_gradient_boosting_regressor.pkl
    """
    # Make the name filesystem-safe
    safe_name = (
        best_model_name.lower().replace(" ", "_").replace("/", "_").replace("\\", "_")
    )

    model_path_prefix = run_dir / f"best_model_{safe_name}"
    save_model(model, str(model_path_prefix))
    print(f"[RUN] Saved best model for this run at: {model_path_prefix}.pkl")
    return model_path_prefix


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------


def run_analysis() -> None:
    """
    Run the full modeling pipeline:

        1. Create a unique run directory.
        2. Load dataset from `1_datasets`.
        3. Drop leakage columns and convert types (especially order_date).
        4. Initialize PyCaret regression and train models.
        5. Save:
            - global comparison table
            - experiment configuration
            - per-model configs, metrics, predictions, and plots.
    """
    # 1. Create run directory
    run_dir = create_run_directory()

    # 2. Load
    df_raw = load_dataset()

    # 3. Preprocess (drop unused dates, lead times, convert order_date)
    df_model = preprocess_for_modeling(df_raw)

    # 4. Train with PyCaret
    final_model, top_models, comparison_df, experiment_config = run_pycaret_regression(
        df_model, n_top_models=5
    )

    # Save global comparison table
    comparison_path = run_dir / "models_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"[RUN] Saved models comparison table to: {comparison_path}")

    # Save experiment configuration
    exp_config_path = run_dir / "experiment_config.json"
    with exp_config_path.open("w", encoding="utf-8") as f:
        json.dump(experiment_config, f, indent=2)
    print(f"[RUN] Saved experiment configuration to: {exp_config_path}")

    # 5. Save artifacts for each top model
    model_names = comparison_df["Model"].tolist()
    for i, model in enumerate(top_models):
        name = model_names[i] if i < len(model_names) else f"model_{i + 1}"
        save_model_with_artifacts(model, name, run_dir)

    # 6. Save the single best model of this run in the run folder itself
    #    (we assume row 0 of comparison_df is the best model used for final_model)
    if len(model_names) > 0:
        best_model_name = model_names[0]
    else:
        best_model_name = "best_model"

    save_run_best_model(final_model, best_model_name, run_dir)

    print("[DONE] Analysis finished successfully. All artifacts saved.")


def main() -> None:
    """
    Entry point for running the analysis script from the command line.
    """
    run_analysis()


if __name__ == "__main__":
    main()
