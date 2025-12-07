from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed" / "air_quality_clean.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


def load_processed():
    df = pd.read_csv(PROCESSED_DATA)

    # Remove non-numeric columns
    df = df.drop(columns=[c for c in ["Date", "Time", "DateTime"] if c in df.columns], errors="ignore")

    # Keep only numeric columns
    df = df.select_dtypes(include=[float, int])

    # Drop columns with almost no valid data
    df = df.loc[:, df.isna().sum() < (len(df) * 0.8)]

    # Drop remaining NaNs
    df = df.dropna()

    return df


def perform_pca(df, n_components=2):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)

    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(scaled)

    # Save PCA model
    joblib.dump(pca, MODEL_DIR / "pca_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "pca_scaler.pkl")

    return reduced, pca.explained_variance_ratio_


def perform_kmeans(df, k=4):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)

    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(scaled)

    # Save KMeans model
    joblib.dump(kmeans, MODEL_DIR / "kmeans_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "kmeans_scaler.pkl")

    return clusters


if __name__ == "__main__":
    df = load_processed()

    print("Running PCA...")
    reduced, variance = perform_pca(df)
    print("Explained variance:", variance)

    print("\nRunning K-Means clustering...")
    clusters = perform_kmeans(df)

    # Save cluster results
    results = pd.DataFrame({
        "PC1": reduced[:, 0],
        "PC2": reduced[:, 1],
        "Cluster": clusters
    })

    results_path = PROJECT_ROOT / "data" / "processed" / "cluster_results.csv"
    results.to_csv(results_path, index=False)

    print("\nCluster results saved →", results_path)
    print(results.head())
