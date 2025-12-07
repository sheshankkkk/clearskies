from pathlib import Path
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "AirQualityUCI.csv"


def load_raw_air_quality() -> pd.DataFrame:
    """
    Load the raw Air Quality UCI dataset from data/raw.
    """
    if not DATA_RAW_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {DATA_RAW_PATH}")

    # UCI air quality CSV is separated by semicolons
    df = pd.read_csv(DATA_RAW_PATH, sep=";")

    print(f"Loaded raw data with shape: {df.shape}")
    return df


if __name__ == "__main__":
    # Quick smoke test when running: python -m clearskies.data_loader
    df = load_raw_air_quality()
    print(df.head())
