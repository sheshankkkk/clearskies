import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import plotly.graph_objects as go
import plotly.express as px


st.set_page_config(page_title="ClearSkies — Air Quality Prediction", layout="wide")

# ------------------------------------------------------
# LOAD MODEL + SCALER + METADATA
# ------------------------------------------------------
MODEL_PATH = "models/classification_model.joblib"
SCALER_PATH = "models/classification_scaler.pkl"
FEATURE_META = "models/classification_features.json"

with open(FEATURE_META, "r") as f:
    meta = json.load(f)

FEATURES = meta["features"]  # 9 features
TARGET = meta["target"]

# Load model dict
MODEL_DICT = joblib.load(MODEL_PATH)
clf = MODEL_DICT["model"]       # <---- FIXED: Extract actual ML model
scaler = joblib.load(SCALER_PATH)

# ------------------------------------------------------
# LOAD DATASET
# ------------------------------------------------------
RAW_DATA_PATH = "data/raw/AirQualityUCI.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(RAW_DATA_PATH, sep=";")
    df = df.replace(",", ".", regex=True)
    df = df.apply(pd.to_numeric, errors="ignore")

    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    df["Time"] = pd.to_datetime(df["Time"], format="%H.%M.%S", errors="coerce").dt.time

    # Keep only Date, Time & model features
    df = df[["Date", "Time"] + FEATURES]

    return df.dropna()

df = load_data()

df_display = df.copy()

# ------------------------------------------------------
# UI HEADER
# ------------------------------------------------------
st.title("ClearSkies — Air Quality Category Prediction")
st.caption("© 2025 ClearSkies — Machine Learning Air Quality Analysis Dashboard")


# ------------------------------------------------------
# ROW CLICK → AUTO-FILL
# ------------------------------------------------------
st.subheader("Data (click a row to auto-fill prediction inputs)")

clicked = st.dataframe(
    df_display,
    on_select="rerun",
    selection_mode="single-row",
    width="stretch"
)

selected_row = None

if clicked and "selection" in clicked and clicked.selection.rows:
    selected_row = clicked.selection.rows[0]


# ------------------------------------------------------
# PREDICTION FORM
# ------------------------------------------------------
st.subheader("Predict Air Quality Category")

cols = st.columns(3)

input_values = {}

# Pre-fill values if row selected
if selected_row is not None:
    auto_vals = df_display.iloc[selected_row][FEATURES].tolist()
else:
    auto_vals = [0.0] * len(FEATURES)

# Build numeric inputs
for i, feature in enumerate(FEATURES):
    input_values[feature] = st.number_input(
        feature,
        value=float(auto_vals[i]),
        step=1.0,
        format="%.2f"
    )

# ------------------------------------------------------
# PREDICTION BUTTON
# ------------------------------------------------------
if st.button("Predict Air Quality Category", type="primary"):

    try:
        X = np.array([input_values[f] for f in FEATURES]).reshape(1, -1)

        # Scale input
        X_scaled = scaler.transform(X)

        # Predict with actual model
        prediction = clf.predict(X_scaled)[0]

        st.success(f"Predicted Air Quality Category: **{prediction}**")

    except Exception as e:
        st.error(f"Error while making prediction: {e}")

# =========================================================
# VISUALIZATION SECTION
# =========================================================

st.markdown("## Visualizations")

# --------------------------
# Helper function: category → color
# --------------------------
def category_color(cat):
    mapping = {
        "Good": "#2ecc71",
        "Moderate": "#f1c40f",
        "Unhealthy": "#e67e22",
        "Very Unhealthy": "#e74c3c",
        "Hazardous": "#8e44ad"
    }
    return mapping.get(cat, "#95a5a6")  # default gray


# --------------------------
# Visualization A — HOURLY TREND (selected pollutant)
# --------------------------

with st.expander("Visualization A — Hourly Trend for Selected Day", expanded=False):

    # Choose a date
    unique_dates = df["Date"].unique()
    selected_date = st.selectbox("Select a date", unique_dates)

    day_df = df[df["Date"] == selected_date].copy()

    if day_df.empty:
        st.warning("No data for this date.")
    else:
        fig_hour = go.Figure()

        fig_hour.add_trace(go.Scatter(
            x=day_df["Time"],
            y=day_df["CO(GT)"],
            mode="lines+markers",
            line=dict(color="#3498db", width=3),
            marker=dict(size=6),
            name="CO(GT)"
        ))

        fig_hour.update_layout(
            height=400,
            template="plotly_dark",
            xaxis_title="Time of Day",
            yaxis_title="CO(GT)",
            title=f"Hourly CO Levels on {selected_date}",
            margin=dict(l=40, r=20, t=60, b=40)
        )

        st.plotly_chart(fig_hour, use_container_width=True)


# ---------------------------------------------------------
# BEST TIME OF THE DAY (Based on Lowest CO(GT))
# ---------------------------------------------------------

st.subheader("Best Time of the Day")

# Ensure time is properly formatted
df["Time"] = df["Time"].astype(str).str[:5]  # keep HH:MM

# Extract hour
df["Hour"] = df["Time"].str.split(":").str[0].astype(int)

# Compute hourly averages
hourly_avg = df.groupby("Hour")["CO(GT)"].mean().reset_index()

# Find best hour
best_hour = hourly_avg.loc[hourly_avg["CO(GT)"].idxmin()]
best_hour_label = f"{int(best_hour['Hour']):02d}:00"
best_value = round(best_hour["CO(GT)"], 2)

# Display result
st.success(
    f"✅ Best Time of the Day: **{best_hour_label}**\n"
    f"🌿 Lowest average CO(GT): **{best_value}**"
)

# Hourly bar chart
fig_hour = px.bar(
    hourly_avg,
    x="Hour",
    y="CO(GT)",
    template="plotly_dark",
    labels={"Hour": "Hour of Day", "CO(GT)": "Avg CO(GT)"},
    title="Hourly Air Quality Profile",
)

fig_hour.add_vline(
    x=int(best_hour["Hour"]),
    line_width=3,
    line_color="lightgreen",
    annotation_text="Best",
    annotation_position="top"
)

st.plotly_chart(fig_hour, use_container_width=True)

