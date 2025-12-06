"""
Streamlit demo app for Lead-Time Prediction (PyCaret + Odoo-friendly)

This app:
    - Loads the trained PyCaret regression model
    - Lets a user (manager, planner, etc.) input generator order details
    - Sends those inputs to PyCaret's pipeline, which handles:
        * Encoding (one-hot, target encoding, etc.)
        * Imputation
        * Date feature extraction for `order_date`
    - Returns the predicted `target_lead_time_to_finish_days`.

Usage:
    streamlit run demo_lead_time_app.py
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import streamlit as st
from pycaret.regression import load_model, predict_model

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Project root is assumed to be one level above this script (adjust if needed)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Path to the PyCaret model saved in one of your run folders.
# Example:
#   4_data_analysis/models/run_20251206_204501/best_model_gradient_boosting_regressor
#
# IMPORTANT:
#   PyCaret's load_model expects the *prefix* without ".pkl"
#   ("best_model_gradient_boosting_regressor", not "...pkl").
MODEL_PATH_PREFIX = (
    PROJECT_ROOT
    / "4_data_analysis"
    / "models"
    / "run_20251206_145317"
    / "best_model_gradient_boosting_regressor"
)

# Optional: path to prepared dataset, used here ONLY to get unique values
# for dropdowns. If the file is missing, we just fall back to generic inputs.
PREPARED_DATASET_PATH = PROJECT_ROOT / "1_datasets" / "ELO2_lead_time_prepared.csv"

# Target column name used in your modeling script
TARGET_COLUMN = "target_lead_time_to_finish_days"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


@st.cache_resource
def load_trained_model():
    """
    Load the trained PyCaret model from disk.

    We use `load_model` from PyCaret, which restores:
        - Preprocessing pipeline (encoding, imputation, date features, etc.)
        - The trained estimator

    Returns:
        The loaded PyCaret model pipeline.
    """
    model_path_str = str(MODEL_PATH_PREFIX)
    model = load_model(model_path_str)
    return model


@st.cache_data
def load_reference_dataset() -> Optional[pd.DataFrame]:
    """
    Load the prepared dataset (if available) to extract:
        - Unique values for dropdown lists
        - Typical ranges for numeric features

    Returns:
        A pandas DataFrame if the file exists, otherwise None.
    """
    if PREPARED_DATASET_PATH.exists():
        df = pd.read_csv(PREPARED_DATASET_PATH)
        return df
    return None


def get_unique_values(df: Optional[pd.DataFrame], column: str) -> list:
    """
    Helper to get sorted unique values for a column if the reference dataframe
    exists and contains the column. Otherwise, return an empty list.

    Args:
        df: Reference DataFrame or None.
        column: Column name to extract unique values from.

    Returns:
        A sorted list of unique values, or [] if not available.
    """
    if df is None or column not in df.columns:
        return []
    vals = df[column].dropna().unique().tolist()
    # Sort if they are strings or numbers
    try:
        vals = sorted(vals)
    except TypeError:
        pass
    return vals


def build_input_form(df_ref: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Build the Streamlit UI and return a single-row DataFrame with the
    same column names as used when training the model.

    IMPORTANT:
        We do NOT manually encode anything.
        We only create the raw feature columns (e.g., engine_model,
        alternator_model, genset_size, etc.). PyCaret's pipeline will
        handle the proper encoding and transformations.

    Args:
        df_ref: Reference dataset for populating dropdown options.

    Returns:
        A single-row DataFrame ready to be passed into `predict_model`.
    """
    st.subheader("Enter Generator Order Details")

    # Use two/three columns layout for neat UI
    col1, col2 = st.columns(2)

    # --- Order Date ---
    # PyCaret expects this as a date feature (you used date_features=['order_date'])
    with col1:
        order_date_input = st.date_input(
            "Order Date",
            value=date.today(),
            help="Date when the order was created.",
        )

    # Try to load unique categorical values from reference dataset:
    genset_sizes = get_unique_values(df_ref, "genset_size")
    engine_models = get_unique_values(df_ref, "engine_model")
    alternator_models = get_unique_values(df_ref, "alternator_model")
    canopy_sizes = get_unique_values(df_ref, "canopy_size")
    voltages = get_unique_values(df_ref, "voltage")
    engine_types = get_unique_values(df_ref, "engine_type")
    alternator_types = get_unique_values(df_ref, "alternator_type")
    controller_models = get_unique_values(df_ref, "controller_model")
    freqs = get_unique_values(df_ref, "freq_50_60")

    # --- Genset Size ---
    with col1:
        if genset_sizes:
            genset_size = st.selectbox(
                "Genset Size",
                genset_sizes,
                help="Genset size/category as recorded in the dataset.",
            )
        else:
            genset_size = st.text_input(
                "Genset Size",
                help="Enter genset size (e.g., '250 kVA').",
            )

    # --- Engine Model ---
    with col2:
        if engine_models:
            engine_model = st.selectbox(
                "Engine Model",
                engine_models,
                help="Choose engine model (as in the dataset).",
            )
        else:
            engine_model = st.text_input(
                "Engine Model",
                help="Enter engine model name (e.g., 'Perkins 1106A-70TAG1').",
            )

    # --- Alternator Model ---
    with col1:
        if alternator_models:
            alternator_model = st.selectbox(
                "Alternator Model",
                alternator_models,
                help="Choose alternator model (as in the dataset).",
            )
        else:
            alternator_model = st.text_input(
                "Alternator Model",
                help="Enter alternator model (e.g., 'Stamford UCI274C').",
            )

    # --- Canopy Size ---
    with col2:
        if canopy_sizes:
            canopy_size = st.selectbox(
                "Canopy Size",
                canopy_sizes,
                help="Type/size of canopy (if recorded).",
            )
        else:
            canopy_size = st.text_input(
                "Canopy Size",
                help="Enter canopy size/type (or leave blank if none).",
            )

    # --- Voltage ---
    with col1:
        if voltages:
            voltage = st.selectbox(
                "Voltage",
                voltages,
                help="Nominal output voltage configuration.",
            )
        else:
            voltage = st.text_input(
                "Voltage",
                help="Enter voltage (e.g., '400/230V').",
            )

    # --- Engine Type ---
    with col2:
        if engine_types:
            engine_type = st.selectbox(
                "Engine Type",
                engine_types,
                help="Engine configuration/type (e.g., brand-based or series).",
            )
        else:
            engine_type = st.text_input(
                "Engine Type",
                help="Enter engine type/category.",
            )

    # --- Alternator Type ---
    with col1:
        if alternator_types:
            alternator_type = st.selectbox(
                "Alternator Type",
                alternator_types,
                help="Alternator family/type as in dataset.",
            )
        else:
            alternator_type = st.text_input(
                "Alternator Type",
                help="Enter alternator type/family.",
            )

    # --- Controller Model ---
    with col2:
        if controller_models:
            controller_model = st.selectbox(
                "Controller Model",
                controller_models,
                help="Genset controller (e.g., DSE 7320, DSE 8610, ComAp IG-200).",
            )
        else:
            controller_model = st.text_input(
                "Controller Model",
                help="Enter controller model.",
            )

    # --- Frequency (50/60 Hz) ---
    with col1:
        if freqs:
            freq_50_60 = st.selectbox(
                "Frequency (50/60 Hz)",
                freqs,
                help="Operating frequency of the genset.",
            )
        else:
            freq_50_60 = st.selectbox(
                "Frequency (50/60 Hz)",
                options=[50, 60],
                help="Select frequency (50 or 60 Hz).",
            )

    # Build a single-row DataFrame with the SAME column names as training.
    # IMPORTANT: Do not rename these columns; they must match `df_model` used
    # in your PyCaret `setup()` call.
    input_dict = {
        "order_date": pd.to_datetime(order_date_input),
        "genset_size": genset_size,
        "engine_model": engine_model,
        "alternator_model": alternator_model,
        "canopy_size": canopy_size,
        "voltage": voltage,
        "engine_type": engine_type,
        "alternator_type": alternator_type,
        "controller_model": controller_model,
        "freq_50_60": freq_50_60,
    }

    input_df = pd.DataFrame([input_dict])

    return input_df


def make_prediction(model, input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Use PyCaret's `predict_model` to generate a prediction for the given input.

    We pass a single-row DataFrame with raw features. PyCaret will:
        - Apply the same preprocessing pipeline as during training
        - Return a DataFrame that includes:
            - All original columns
            - A 'Label' column with the predicted target

    Args:
        model: Loaded PyCaret model pipeline.
        input_df: Single-row DataFrame with raw features.

    Returns:
        DataFrame with prediction results from `predict_model`.
    """
    preds = predict_model(model, data=input_df)
    return preds


def extract_prediction_column(
    preds: pd.DataFrame, target_column: str
) -> Tuple[str, float]:
    """
    Figure out which column in `preds` contains the prediction.

    PyCaret can use different names for the prediction column:
      - 'prediction_label'  (PyCaret 3.x)
      - 'Label'             (older docs / examples)
      - Sometimes a custom name

    This helper:
      1. Tries common names.
      2. Falls back to the last numeric column if needed.
      3. Raises a clear error if nothing works.

    Args:
        preds: DataFrame returned by `predict_model`.
        target_column: The target column name used in setup (for logging).

    Returns:
        (column_name, first_value_as_float)
    """
    # Most likely names first
    candidate_cols = [
        "prediction_label",  # PyCaret 3 default
        "Label",  # older PyCaret default
        f"{target_column}_prediction",
        f"{target_column}_pred",
    ]

    # 1) Try known names
    for col in candidate_cols:
        if col in preds.columns:
            return col, float(preds[col].iloc[0])

    # 2) Fallback: last numeric column (PyCaret often appends the prediction at the end)
    numeric_cols = preds.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        col = numeric_cols[-1]
        return col, float(preds[col].iloc[0])

    # 3) Nothing found -> error
    raise ValueError(
        "Could not find prediction column in `predict_model` results.\n"
        f"Available columns: {list(preds.columns)}\n"
        "Tried: 'prediction_label', 'Label', and a few target-based guesses."
    )


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------


def main() -> None:
    st.title("Lead-Time Prediction Demo for Generator Manufacturing")
    st.markdown(
        """
This app demonstrates the **lead-time prediction model** trained using PyCaret.

- Input the key configuration of a generator order.
- The model will predict:  
  **`target_lead_time_to_finish_days`** — estimated days from order to finishing.

This demo is intended for:
- Top management (Factory Manager, General Manager)
- System administrator, for planning ERP (Odoo) integration
- Production / planning teams for validation and feedback
"""
    )

    # Load reference dataset (for dropdowns) and model
    with st.spinner("Loading model and reference data..."):
        df_ref = load_reference_dataset()
        model = load_trained_model()

    st.success("Model loaded successfully.")

    # Build input form
    input_df = build_input_form(df_ref)

    st.markdown("### Review Your Input")
    st.dataframe(input_df)

    if st.button("Predict Lead Time"):
        with st.spinner("Generating prediction..."):
            preds = make_prediction(model, input_df)

        # # Debug (optional): see what columns PyCaret returned
        # st.caption("Prediction result columns:")
        # st.code(str(list(preds.columns)))

        # Try to extract prediction column robustly
        try:
            pred_col_name, predicted_days = extract_prediction_column(
                preds, TARGET_COLUMN
            )
        except ValueError as e:
            st.markdown("## 🔮 Prediction Result")
            st.error(str(e))
            with st.expander("Show raw prediction output"):
                st.dataframe(preds)
            return  # stop here

        st.markdown("## 🔮 Prediction Result")

        st.metric(
            label="Estimated Lead Time to Finish (days)",
            value=f"{predicted_days:.1f}",
        )

        with st.expander("Show raw prediction output"):
            st.dataframe(preds)


if __name__ == "__main__":
    main()
