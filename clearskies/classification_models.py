"""
Classification model training for ClearSkies Air Quality Project
---------------------------------------------------------------
Trains 5 ML models, evaluates them, and saves the best model + scaler + feature names.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# --------------------------------------------------------
# LOAD & CLEAN DATA
# --------------------------------------------------------
def load_clean_data():
    df = pd.read_csv("data/processed/clean_air_quality.csv")

    # Remove target & invalid columns
    required_cols = [
        "CO(GT)", "PT08.S1(CO)", "NMHC(GT)", "PT08.S2(NMHC)",
        "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)",
        "PT08.S5(O3)", "T", "RH", "AH"
    ]

    df = df[required_cols + ["AQ_Category"]]  # keep target at end
    df = df.dropna()

    return df


# --------------------------------------------------------
# TRAIN / TEST SPLIT + SCALING
# --------------------------------------------------------
def prepare_data(df):

    feature_cols = [
        "CO(GT)", "PT08.S1(CO)", "NMHC(GT)", "PT08.S2(NMHC)",
        "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)",
        "PT08.S5(O3)", "T", "RH", "AH"
    ]

    X = df[feature_cols].to_numpy()
    y = df["AQ_Category"]

    # 80% train / 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols


# --------------------------------------------------------
# TRAIN ALL MODELS
# --------------------------------------------------------
def train_all_models(X_train, y_train, X_test, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Decision Tree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC()
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\n----- Training {name} -----")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print(f"Accuracy: {acc}")
        print(report)

        results[name] = acc
        trained_models[name] = model

    return results, trained_models


# --------------------------------------------------------
# SAVE BEST MODEL + SCALER + FEATURES
# --------------------------------------------------------
def save_best_model(results, trained_models, scaler, feature_cols):
    best_model_name = max(results, key=results.get)
    best_model = trained_models[best_model_name]

    os.makedirs("models", exist_ok=True)

    print(f"\n===== BEST MODEL: {best_model_name} ({results[best_model_name]}) =====")

    joblib.dump(best_model, "models/classification_model.joblib")
    joblib.dump(scaler, "models/classification_scaler.pkl")

    with open("models/classification_features.json", "w") as f:
        json.dump(feature_cols, f)

    print("\nSaved:")
    print(" - models/classification_model.joblib")
    print(" - models/classification_scaler.pkl")
    print(" - models/classification_features.json")


# --------------------------------------------------------
# MAIN
# --------------------------------------------------------
def main():
    print("\n=== Loading Clean Data ===")
    df = load_clean_data()
    print(f"Dataset Loaded: {df.shape}")

    print("\n=== Preparing Data (80% Train, 20% Test) ===")
    X_train, X_test, y_train, y_test, scaler, feature_cols = prepare_data(df)

    print("\n=== Training Models ===")
    results, trained_models = train_all_models(X_train, y_train, X_test, y_test)

    print("\n=== Saving Best Model ===")
    save_best_model(results, trained_models, scaler, feature_cols)

    print("\n=== Training Complete ===")


if __name__ == "__main__":
    main()
