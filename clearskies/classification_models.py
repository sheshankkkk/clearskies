from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed" / "air_quality_clean.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


def load_processed():
    df = pd.read_csv(PROCESSED_DATA)

    # Remove non-numeric columns
    df = df.drop(columns=[c for c in ["Date", "Time", "DateTime"] if c in df.columns], errors="ignore")

    return df


def create_aqi_category(df):
    """
    Create a synthetic AQI value using 3 major pollutant indicators.
    """
    df["AQI_raw"] = (
        df["CO(GT)"].fillna(0) +
        df["NO2(GT)"].fillna(0) +
        df["PT08.S5(O3)"].fillna(0)
    )

    df["AQ_Category"] = pd.cut(
        df["AQI_raw"],
        bins=[-1, 100, 300, 600, 1000, 5000],
        labels=["Good", "Moderate", "Unhealthy", "Very_Unhealthy", "Hazardous"]
    )

    df = df.dropna(subset=["AQ_Category"])

    return df


def train_classification(df):
    df = create_aqi_category(df)

    # Identify usable numeric features
    numeric_cols = df.select_dtypes(include=[float, int]).columns.tolist()

    # Remove helper columns
    numeric_cols = [c for c in numeric_cols if c not in ["AQI_raw"]]

    # Keep columns with meaningful data (at least 100 non-NaN values)
    valid_features = [c for c in numeric_cols if df[c].notna().sum() > 100]

    print("\n[INFO] Using features:", valid_features)

    df = df[valid_features + ["AQ_Category"]]
    df = df.dropna()

    print("[INFO] Final shape after filtering:", df.shape)

    if df.shape[0] == 0:
        raise ValueError("Dataset is empty after filtering. Check preprocess step.")

    X = df[valid_features]
    y = df["AQ_Category"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier(),
        "NaiveBayes": GaussianNB(),
        "SVM": SVC()
    }

    results = {}

    for name, model in models.items():
        print(f"\n----- Training {name} -----")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        print("Accuracy:", acc)
        print(classification_report(y_test, preds))

        results[name] = acc
        joblib.dump(model, MODEL_DIR / f"{name}_AQ.pkl")

    joblib.dump(scaler, MODEL_DIR / "classification_scaler.pkl")

    return results


if __name__ == "__main__":
    df = load_processed()
    results = train_classification(df)

    print("\n===== FINAL RESULTS =====")
    for model, acc in results.items():
        print(model, "→", acc)
