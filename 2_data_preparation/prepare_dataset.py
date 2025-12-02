"""
prepare_lead_time_dataset.py

Script to prepare the generator production dataset for lead-time prediction.

This script follows the project data-preparation guidelines:

1. Read raw datasets from `1_datasets/`.
2. Clean and reformat the dataset for later analysis/modeling.
3. Write the processed dataset to a **new** file (does NOT modify the original),
  saved in `1_datasets/` with a helpful file name.

Typical usage:
    python prepare_lead_time_dataset.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Go up one level to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = Path("1_datasets")
PROCESSED_DIR = Path("1_datasets")

RAW_FILENAME = "ELO2_Raw_Data.csv"
PROCESSED_FILENAME = "ELO2_lead_time_prepared.csv"

# Date columns in the raw dataset (exact names before cleaning)
RAW_DATE_COLUMNS: List[str] = [
    "Order Date",
    "Receiving CKD Date",
    "Finishing Date",
    "Shipping Date",
]

# Expected lead-time column we will use as the prediction target
RAW_LEAD_TIME_COLUMN = "LEAD TIME TO FINISH"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def load_raw_dataset(
    raw_dir: Path = RAW_DIR, filename: str = RAW_FILENAME
) -> pd.DataFrame:
    """
    Load the raw ELO2 dataset from the `0_datasets` directory.

    This function does **not** modify the data; it only reads the file.

    Args:
        raw_dir: Directory containing raw datasets (default: `0_datasets`).
        filename: Name of the CSV file to read.

    Returns:
        A pandas DataFrame containing the raw data.

    Raises:
        FileNotFoundError: If the expected CSV file does not exist.
    """
    path = raw_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {path}")

    df = pd.read_csv(path)
    return df


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to snake_case for easier use in analysis and code.

    Transformations:
        - Strip leading/trailing whitespace.
        - Replace newlines with spaces.
        - Lowercase all characters.
        - Replace any non-alphanumeric characters with underscores.
        - Collapse multiple underscores into one.
        - Strip leading/trailing underscores.

    Example:
        "LEAD TIME TO SHIP"  -> "lead_time_to_ship"
        "Freq\\n50? 60?"     -> "freq_50_60"
        "Engine\\nType"      -> "engine_type"

    Args:
        df: Input DataFrame with original column names.

    Returns:
        A new DataFrame with standardized column names.
    """

    def clean(name: str) -> str:
        name = str(name).strip().replace("\n", " ")
        name = name.lower()
        # Replace any non-alphanumeric char with underscore
        out = []
        for ch in name:
            if ch.isalnum():
                out.append(ch)
            else:
                out.append("_")
        name = "".join(out)
        # Collapse repeated underscores
        while "__" in name:
            name = name.replace("__", "_")
        # Strip leading/trailing underscores
        name = name.strip("_")
        return name

    df = df.copy()
    df.columns = [clean(c) for c in df.columns]
    return df


def parse_mixed_date_columns(
    df: pd.DataFrame,
    ddmmyyyy_cols: Iterable[str],
    mmddyyyy_cols: Iterable[str],
) -> pd.DataFrame:
    """
    Parse date columns that use mixed formats.

    Args:
        df: DataFrame after column-name standardization.
        ddmmyyyy_cols: List of columns using DD/MM/YYYY format.
        mmddyyyy_cols: List of columns using MM/DD/YYYY format.

    Returns:
        DataFrame with parsed datetime columns.
    """
    df = df.copy()

    # Parse DD/MM/YYYY (day-first)
    for col in ddmmyyyy_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")

    # Parse MM/DD/YYYY (month-first)
    for col in mmddyyyy_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=False, errors="coerce")

    return df


def add_target_columns(
    df: pd.DataFrame, raw_target_col: str = RAW_LEAD_TIME_COLUMN
) -> pd.DataFrame:
    """
    Add clean target columns for lead-time prediction.

    This function assumes that the raw dataset already contains one or more
    lead-time columns in days (e.g., "LEAD TIME TO finish"). It will:

        1. Use the specified raw target column as the main label.
        2. Create a standardized numeric target column:
               `target_lead_time_to_finish_days`
        3. Optionally keep the original raw column (for reference).

    Args:
        df: DataFrame with the original lead-time column present.
        raw_target_col: Name of the lead-time column in the **original**
            (non-standardized) column naming scheme.

    Returns:
        A new DataFrame with:
            - standardized column names
            - an additional `target_lead_time_to_finish_days` column

    Raises:
        KeyError: If the expected lead-time column is not found (before or
            after standardizing column names).
    """
    # First, standardize column names so we know the target’s cleaned name.
    df = standardize_column_names(df)

    # Compute what the cleaned column name for the raw target should be
    temp = pd.DataFrame(columns=[raw_target_col])
    cleaned_target_name = standardize_column_names(temp).columns[0]

    if cleaned_target_name not in df.columns:
        raise KeyError(
            f"Expected lead-time column '{raw_target_col}' (cleaned as "
            f"'{cleaned_target_name}') not found in dataset columns: {list(df.columns)}"
        )

    df["target_lead_time_to_finish_days"] = pd.to_numeric(
        df[cleaned_target_name], errors="coerce"
    )

    return df


def drop_rows_without_target(
    df: pd.DataFrame,
    target_column: str = "target_lead_time_to_finish_days",
) -> pd.DataFrame:
    """
    Drop rows where the target lead-time is missing.

    This is a simple cleaning step so that downstream modeling scripts can
    assume the target column exists and is non-null.

    Args:
        df: DataFrame containing the target column.
        target_column: Name of the target lead-time column.

    Returns:
        A new DataFrame without rows where the target column is null.
    """
    df = df.copy()
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in DataFrame.")

    before = len(df)
    df = df.dropna(subset=[target_column])
    after = len(df)

    print(
        f"Dropped {before - after} rows with missing '{target_column}'. "
        f"Remaining rows: {after}."
    )
    return df


def save_processed_dataset(
    df: pd.DataFrame,
    processed_dir: Path = PROCESSED_DIR,
    filename: str = PROCESSED_FILENAME,
) -> Path:
    """
    Save the processed dataset to the `1_datasets` directory.

    This creates the directory if it does not already exist and writes the
    DataFrame as a CSV file.

    Args:
        df: Processed DataFrame to save.
        processed_dir: Directory where processed datasets are stored.
        filename: Name of the CSV file to write.

    Returns:
        The full path to the saved CSV file.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / filename
    df.to_csv(output_path, index=False)
    print(f"Processed dataset saved to: {output_path}")
    return output_path


def drop_serial_number_columns(df):
    """
    Remove columns that contain serial numbers, unit IDs, generator IDs,
    or any column that may cause data leakage.

    Leakage columns commonly include:
        - serial number columns
        - unit numbers
        - generator IDs
        - any unique ID that describes the exact unit

    Strategy:
        1. Detect columns that contain certain keywords in their names.
        2. Detect columns with >98% unique values (likely identifiers).
        3. Drop these columns from the dataset.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame without leakage columns.
    """

    df = df.copy()

    # Keywords that indicate ID / serial number columns
    leakage_keywords = [
        "serial",
        "s_no",
        "sno",
        "s.no",
        "unit",
        "unit_no",
        "unitnumber",
        "_no",
        "no.",
        "id",
        "S/No.",
    ]

    # 1. Keyword detection
    keyword_cols = [
        col
        for col in df.columns
        if any(kw in col.lower().replace(" ", "_") for kw in leakage_keywords)
    ]

    # 2. High-uniqueness detection (98%+ unique)
    nearly_unique_cols = [
        col for col in df.columns if df[col].nunique() >= 0.98 * len(df)
    ]

    # Combine
    cols_to_drop = set(keyword_cols + nearly_unique_cols)

    # Print what is being removed (important for transparency)
    if cols_to_drop:
        print("Dropping columns due to leakage risk:")
        for c in cols_to_drop:
            print(f"  - {c}")
    else:
        print("No leakage columns detected.")

    # Drop
    df = df.drop(columns=list(cols_to_drop), errors="ignore")

    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def prepare_lead_time_dataset() -> pd.DataFrame:
    """
    Run the full preparation pipeline for the ELO2 lead-time dataset.

    Steps:
        1. Load raw data from `0_datasets/ELO2_Raw_Data.csv`.
        2. Add the target column `target_lead_time_to_finish_days`.
        3. Parse date columns into datetime.
        4. Drop rows with missing target.
        5. Save the processed dataset to `1_datasets/ELO2_lead_time_prepared.csv`.

    Returns:
        The processed DataFrame (also written to disk).
    """
    # 1. Load raw data (columns are still in original form here)
    raw_df = load_raw_dataset()

    # 2. Add standardized target column and standardize all column names
    df = add_target_columns(raw_df)

    # At this point, column names are standardized; update date-column names
    # to their cleaned versions.
    cleaned_date_cols = standardize_column_names(
        pd.DataFrame(columns=RAW_DATE_COLUMNS)
    ).columns.tolist()
    # 3. Also drop serial number / leakage columns
    df = drop_serial_number_columns(df)

    order_date_col = cleaned_date_cols[0]
    other_date_cols = cleaned_date_cols[1:]

    # 4. Parse mixed formats
    df = parse_mixed_date_columns(
        df, ddmmyyyy_cols=[order_date_col], mmddyyyy_cols=other_date_cols
    )

    # 5. Drop rows without target
    df = drop_rows_without_target(df, target_column="target_lead_time_to_finish_days")

    # sort by order date (if available) for convenience
    if "order_date" in df.columns:
        df = df.sort_values("order_date").reset_index(drop=True)

    # 6. Save processed dataset
    save_processed_dataset(df)

    return df


def main() -> None:
    """
    Entry point for running the data-preparation script from the command line.

    When executed directly, this will prepare the lead-time dataset and save
    the processed file into `1_datasets/ELO2_lead_time_prepared.csv`.
    """
    prepare_lead_time_dataset()


if __name__ == "__main__":
    main()
