from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed" / "air_quality_clean.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


def load_processed():
    return pd.read_csv(PROCESSED_DATA)


def train_regression_models(df: pd.DataFrame, target_column: str):
    """
    Train regression models to predict a single pollutant.
    """
    features = df.drop(columns=[target_column, "DateTime"])
    target = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
        "KNNRegressor": KNeighborsRegressor()
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds, squared=False)
        r2 = r2_score(y_test, preds)

        results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}

        joblib.dump(model, MODEL_DIR / f"{name}_{target_column}.pkl")

    return results


if __name__ == "__main__":
    df = load_processed()

    # Choose pollutant to predict
    pollutant = "CO(GT)"     # change this to NO2(GT), PT08.S1(CO), etc.

    metrics = train_regression_models(df, pollutant)
    print(f"Results for predicting: {pollutant}")
    for model_name, result in metrics.items():
        print(model_name, result)
