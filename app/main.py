# app/main.py

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "processed" / "air_quality_clean.csv"
MODEL_PATH = BASE_DIR / "models" / "classification_model.joblib"
SCALER_PATH = BASE_DIR / "models" / "classification_scaler.pkl"
FEATURES_PATH = BASE_DIR / "models" / "classification_features.json"

PAGE_TITLE = "Air Quality Category Prediction"


# ---------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    # Make sure DateTime exists if we want to use it for charts
    if "DateTime" in df.columns:
        df["DateTime"] = pd.to_datetime(df["DateTime"])
    return df


@st.cache_resource
def load_model_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Handle both list and dict formats for features json
    with open(FEATURES_PATH, "r") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        feature_names = raw.get("features") or raw.get("feature_names") or list(raw.keys())
    else:
        feature_names = raw

    feature_names = list(feature_names)
    return model, scaler, feature_names


def predict_category(model, scaler, feature_names, values_dict):
    # values_dict: {feature_name: value}
    x = pd.DataFrame([values_dict])[feature_names]
    x_scaled = scaler.transform(x)
    pred = model.predict(x_scaled)[0]
    return pred


# ---------------------------------------------------------
# UI helpers
# ---------------------------------------------------------
def make_professional_header():
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.title(PAGE_TITLE)
    st.caption("© 2025 ClearSkies — Machine Learning Air Quality Analysis Dashboard")


# ---------------------------------------------------------
# Main app
# ---------------------------------------------------------
def main():
    make_professional_header()

    df = load_data()
    model, scaler, feature_names = load_model_artifacts()

    # Columns we want to show in the table
    display_cols = []

    # Show Date/Time first if present
    if "Date" in df.columns:
        display_cols.append("Date")
    if "Time" in df.columns:
        display_cols.append("Time")

    # Then numeric feature columns (only ones that actually exist in df)
    display_cols += [c for c in feature_names if c in df.columns]

    # Add an internal checkbox column for row selection
    view_df = df[display_cols].copy()
    view_df.insert(0, "Use_for_prediction", False)

    st.subheader("Data (click a checkbox in a row to use it for prediction)")
    edited_df = st.data_editor(
        view_df,
        key="data_table",
        height=420,
        width="stretch",
        hide_index=True,
    )

    # Determine which row (if any) is currently selected
    selected_rows = edited_df[edited_df["Use_for_prediction"] == True]

    selected_row = selected_rows.iloc[-1] if not selected_rows.empty else None

    # ------------------------------------------------------------------
    # Prediction section
    # ------------------------------------------------------------------
    st.subheader("Predict Air Quality Category")

    if selected_row is not None:
        st.info(
            f"Using row index **{int(selected_row.name)}** from the data table "
            "to auto-fill prediction inputs."
        )

    # Build default values for the inputs
    defaults = {}
    for col in feature_names:
        if col in df.columns:
            if selected_row is not None:
                defaults[col] = float(selected_row[col])
            else:
                # Fallback: median of column
                defaults[col] = float(df[col].median())
        else:
            # Safety fallback if feature not in df (should not normally happen)
            defaults[col] = 0.0

    with st.form("prediction_form"):
        cols = st.columns(3)
        input_values = {}

        for i, feat in enumerate(feature_names):
            col = cols[i % 3]
            value = float(defaults.get(feat, 0.0))
            input_values[feat] = col.number_input(
                feat,
                value=value,
                format="%.2f",
            )

        submit = st.form_submit_button("Predict")

    if submit:
        category = predict_category(model, scaler, feature_names, input_values)
        st.success(f"Predicted Air Quality Category: **{category}**")

        # Show a quick summary of inputs used
        summary_df = pd.DataFrame([input_values])
        st.markdown("**Input summary used for this prediction**")
        st.dataframe(summary_df, use_container_width=True)

    # ------------------------------------------------------------------
    # Optional visualizations (only when requested)
    # ------------------------------------------------------------------
    st.subheader("Visualize Air Quality (on demand)")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Show pollutant trends over time"):
            if "DateTime" in df.columns:
                time_df = df.set_index("DateTime")
                available = [c for c in ["CO(GT)", "NO2(GT)", "PT08.S5(O3)"] if c in time_df.columns]
                if available:
                    st.line_chart(time_df[available])
                else:
                    st.warning("Pollutant columns for the trend chart are not available in the dataset.")
            else:
                st.warning("DateTime column not found in data; cannot plot time series.")

    with col2:
        if st.button("Show Air Quality category distribution"):
            if "AQ_Category" in df.columns:
                counts = df["AQ_Category"].value_counts().sort_index()
                st.bar_chart(counts)
            else:
                st.warning("Column 'AQ_Category' not found in the data.")


if __name__ == "__main__":
    main()
