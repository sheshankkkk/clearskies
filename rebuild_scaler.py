import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
import joblib

# Load saved training metadata
with open("models/classification_features.json", "r") as f:
    meta = json.load(f)

FEATURES = meta["features"]

print("\n[INFO] Features used for model:", FEATURES)

# Load processed dataset
df = pd.read_csv("data/processed/air_quality_clean.csv")

print("[INFO] Loaded cleaned dataset:", df.shape)

# Keep only model features
df_model = df[FEATURES].copy()

# Impute missing values (if any)
df_model = df_model.fillna(df_model.mean())

print("[INFO] Final training data shape for scaler:", df_model.shape)

# Rebuild scaler
scaler = StandardScaler()
scaler.fit(df_model)

# Save scaler
joblib.dump(scaler, "models/classification_scaler.pkl")

print("\n[SUCCESS] NEW SCALER BUILT AND SAVED → models/classification_scaler.pkl")
