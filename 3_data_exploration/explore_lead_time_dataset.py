"""
explore_lead_time_dataset.py

Exploratory Data Analysis (EDA) script for the cleaned ELO2 lead-time dataset.

This script follows the Data Exploration Guide:

1. Reads prepared datasets from `0_datasets/`.
2. Explores and understands the dataset:
   - Generates visualizations saved as image files.
   - Computes descriptive statistics and saves them to disk.
3. DOES NOT modify any dataset in `0_datasets/`.

Outputs (all created in this script's folder under `data_exploration_outputs/`):
    - plots/
        - hist_target_lead_time_to_finish_days.png
        - hist_lead_time_to_ship.png
        - hist_lead_time_from_ckd.png
        - bar_engine_type.png
        - bar_engine_model_top20.png
        - bar_alternator_type.png
        - bar_genset_size.png
        - corr_heatmap_numeric.png
    - summaries/
        - dataset_info.txt
        - numeric_describe.csv
        - missing_values.csv
        - categorical_value_counts.txt
        - correlation_matrix.csv

Typical usage:
    python explore_lead_time_dataset.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Project root is assumed to be one level above this script's folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# According to the Data Exploration Guide, we read from 0_datasets/
DATA_DIR = PROJECT_ROOT / "1_datasets"
INPUT_FILENAME = "ELO2_lead_time_prepared.csv"

# Where to store EDA outputs (inside the same folder as this script)
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / "data_exploration_outputs"
PLOTS_DIR = OUTPUT_ROOT / "plots"
SUMMARY_DIR = OUTPUT_ROOT / "summaries"


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def ensure_output_dirs() -> None:
    """
    Create the folders used to store plots and summaries if they do not exist.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    """
    Load the cleaned ELO2 dataset from `0_datasets`.

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


def save_dataset_info(df: pd.DataFrame) -> None:
    """
    Save high-level dataset information (shape, dtypes, info()) to a text file.
    """
    from io import StringIO

    info_path = SUMMARY_DIR / "dataset_info.txt"

    # Capture df.info() output
    buffer = StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()

    with open(info_path, "w", encoding="utf-8") as f:
        f.write("=== DATASET INFO ===\n\n")
        f.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n")

        f.write("dtypes:\n")
        f.write(df.dtypes.to_string())
        f.write("\n\n")

        f.write("DataFrame.info():\n")
        f.write(info_str)

    print(f"[INFO] Dataset info saved to {info_path}")


def save_numeric_describe(df: pd.DataFrame) -> None:
    """
    Save descriptive statistics for numeric columns to a CSV file.
    """
    numeric_df = df.select_dtypes(include=["number"])
    describe = numeric_df.describe().T  # transpose for readability

    out_path = SUMMARY_DIR / "numeric_describe.csv"
    describe.to_csv(out_path)
    print(f"[INFO] Numeric describe saved to {out_path}")


def save_missing_values(df: pd.DataFrame) -> None:
    """
    Save missing value counts per column to a CSV file.
    """
    missing = df.isna().sum().sort_values(ascending=False)
    out_path = SUMMARY_DIR / "missing_values.csv"
    missing.to_csv(out_path, header=["missing_count"])
    print(f"[INFO] Missing values summary saved to {out_path}")


def save_categorical_value_counts(df: pd.DataFrame, max_levels: int = 50) -> None:
    """
    Save value counts for categorical columns to a text file.

    Args:
        df: Input DataFrame.
        max_levels: Maximum number of levels to display per column.
    """
    cat_df = df.select_dtypes(include=["object"])
    out_path = SUMMARY_DIR / "categorical_value_counts.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        for col in cat_df.columns:
            f.write(f"=== {col} ===\n")
            vc = cat_df[col].value_counts(dropna=False).head(max_levels)
            f.write(vc.to_string())
            f.write("\n\n")

    print(f"[INFO] Categorical value counts saved to {out_path}")


def save_correlation_matrix(df: pd.DataFrame) -> None:
    """
    Compute and save the correlation matrix for numeric columns as CSV.
    """
    numeric_df = df.select_dtypes(include=["number"])
    corr = numeric_df.corr()

    out_path = SUMMARY_DIR / "correlation_matrix.csv"
    corr.to_csv(out_path)
    print(f"[INFO] Correlation matrix saved to {out_path}")


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------


def plot_histogram(
    df: pd.DataFrame, column: str, filename: str, bins: int = 20
) -> None:
    """
    Create and save a histogram for a numeric column.

    Args:
        df: Input DataFrame.
        column: Column name to plot.
        filename: Output image filename.
        bins: Number of histogram bins.
    """
    if column not in df.columns:
        print(f"[WARN] Column '{column}' not in DataFrame; skipping histogram.")
        return

    plt.figure(figsize=(8, 5))
    plt.hist(df[column].dropna(), bins=bins)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.tight_layout()

    out_path = PLOTS_DIR / filename
    plt.savefig(out_path)
    plt.close()
    print(f"[PLOT] Histogram saved to {out_path}")


def plot_bar_counts(
    df: pd.DataFrame, column: str, filename: str, top_n: int | None = None
) -> None:
    """
    Create and save a bar plot of value counts for a categorical column.

    Args:
        df: Input DataFrame.
        column: Column name to plot.
        filename: Output image filename.
        top_n: If provided, limit to the top N most frequent categories.
    """
    if column not in df.columns:
        print(f"[WARN] Column '{column}' not in DataFrame; skipping bar plot.")
        return

    counts = df[column].value_counts().dropna()
    if top_n is not None:
        counts = counts.head(top_n)

    plt.figure(figsize=(10, 5))
    counts.plot(kind="bar")
    plt.title(f"Value counts for {column}")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.tight_layout()

    out_path = PLOTS_DIR / filename
    plt.savefig(out_path)
    plt.close()
    print(f"[PLOT] Bar plot saved to {out_path}")


def plot_correlation_heatmap(
    df: pd.DataFrame, filename: str = "corr_heatmap_numeric.png"
) -> None:
    """
    Create and save a correlation heatmap for numeric variables.

    Args:
        df: Input DataFrame.
        filename: Output image filename.
    """
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] == 0:
        print("[WARN] No numeric columns; skipping correlation heatmap.")
        return

    corr = numeric_df.corr()

    plt.figure(figsize=(8, 6))
    plt.imshow(corr, aspect="auto")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.colorbar()
    plt.title("Correlation Heatmap (Numeric Columns)")
    plt.tight_layout()

    out_path = PLOTS_DIR / filename
    plt.savefig(out_path)
    plt.close()
    print(f"[PLOT] Correlation heatmap saved to {out_path}")


def save_full_correlation_matrix(
    df: pd.DataFrame,
    csv_filename: str = "correlation_matrix_full.csv",
    plot_filename: str = "corr_heatmap_full.png",
) -> None:
    """
    Compute and save a correlation matrix for ALL columns, including categorical.

    Categorical columns are encoded as category codes ONLY for visualization.
    Original dataset remains unchanged.

    Heatmap color scheme:
        Red   → Negative / low correlation
        Yellow→ Moderate correlation
        Green → Strong positive correlation

    Args:
        df: Input DataFrame.
        csv_filename: Output CSV file for correlation matrix.
        plot_filename: Output PNG file for heatmap.
    """
    # Copy so we don't modify the original data
    df_encoded = df.copy()

    # Encode categorical columns as numeric codes for correlation
    for col in df_encoded.columns:
        if df_encoded[col].dtype == "object":
            df_encoded[col] = df_encoded[col].astype("category").cat.codes

    # Compute correlation matrix
    corr = df_encoded.corr()

    # Save correlation matrix to CSV
    out_csv = SUMMARY_DIR / csv_filename
    corr.to_csv(out_csv)
    print(f"[INFO] Full correlation matrix saved to {out_csv}")

    # Plot heatmap with Red-Yellow-Green color scale
    plt.figure(figsize=(14, 12))
    plt.imshow(corr, cmap="RdYlGn", aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Full Correlation Heatmap (Red = Low Corr, Green = High Corr)")
    plt.tight_layout()

    out_plot = PLOTS_DIR / plot_filename
    plt.savefig(out_plot)
    plt.close()
    print(f"[PLOT] Full correlation heatmap saved to {out_plot}")


# ---------------------------------------------------------------------------
# Main EDA pipeline
# ---------------------------------------------------------------------------


def run_eda() -> None:
    """
    Run the full EDA pipeline:

        1. Load dataset from `0_datasets`.
        2. Save dataset info, numeric describe, missing-value summary,
           categorical value counts, and correlation matrix.
        3. Generate and save key plots (histograms, bar charts, correlations).
    """
    ensure_output_dirs()

    # 1. Load dataset
    df = load_dataset()

    # 2. Save statistical summaries
    save_dataset_info(df)
    save_numeric_describe(df)
    save_missing_values(df)
    save_categorical_value_counts(df)
    save_correlation_matrix(df)

    # 3. Plots: lead-time histograms
    plot_histogram(
        df,
        column="target_lead_time_to_finish_days",
        filename="hist_target_lead_time_to_finish_days.png",
    )
    plot_histogram(
        df,
        column="lead_time_to_ship",
        filename="hist_lead_time_to_ship.png",
    )
    plot_histogram(
        df,
        column="lead_time_from_ckd",
        filename="hist_lead_time_from_ckd.png",
    )

    # 4. Plots: categorical distributions
    plot_bar_counts(
        df,
        column="engine_type",
        filename="bar_engine_type.png",
    )
    plot_bar_counts(
        df,
        column="engine_model",
        filename="bar_engine_model_top20.png",
        top_n=20,
    )
    plot_bar_counts(
        df,
        column="alternator_type",
        filename="bar_alternator_type.png",
    )
    plot_bar_counts(
        df,
        column="genset_size",
        filename="bar_genset_size.png",
    )
    plot_bar_counts(
        df,
        column="voltage",
        filename="bar_voltage.png",
    )

    # 5. Correlation heatmap
    plot_correlation_heatmap(df)

    # 6. Full correlation matrix & heatmap (all columns encoded)
    save_full_correlation_matrix(df)


def main() -> None:
    """
    Entry point for running the EDA script from the command line.
    """
    run_eda()


if __name__ == "__main__":
    main()
