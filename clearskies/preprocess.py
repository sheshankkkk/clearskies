from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "AirQualityUCI.csv"
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed" / "air_quality_clean.csv"


def load_raw():
    df = pd.read_csv(RAW_DATA, sep=";")

    # Remove last empty column (UCI dataset quirk)
    if df.columns[-1] == "":
        df = df.iloc[:, :-1]

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:

    # Convert -200 values to NaN across all numeric columns
    df.replace(-200, np.nan, inplace=True)

    # Drop columns that are completely empty
    df = df.dropna(axis=1, how="all")

    # Fix Date/Time formatting
    df["Date"] = df["Date"].astype(str)
    df["Time"] = df["Time"].astype(str)

    # Convert "18.00.00" → "18:00:00"
    df["Time"] = df["Time"].str.replace(".", ":", regex=False)

    df["DateTime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        dayfirst=True,
        errors="coerce"
    )

    # Remove invalid timestamps
    df = df[df["DateTime"].notna()]

    # Convert all remaining numeric columns
    for col in df.columns:
        if col not in ["Date", "Time", "DateTime"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill missing sensor data with median (NOT dropping!)
    df = df.fillna(df.median(numeric_only=True))

    df = df.sort_values("DateTime").reset_index(drop=True)

    return df


def save_processed(df: pd.DataFrame):
    PROCESSED_DATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA, index=False)
    print(f"Processed dataset saved → {PROCESSED_DATA}")
    print("Final shape:", df.shape)


if __name__ == "__main__":
    raw = load_raw()
    clean = preprocess(raw)
    save_processed(clean)
    print(clean.head())
