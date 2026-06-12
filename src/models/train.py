import os
import joblib
import pandas as pd
import mlflow
import mlflow.xgboost

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


DATA_PATH = "data/processed/afl_features_latest.csv"
MODEL_DIR = "models"
MODEL_PATH = "models/xgb_goal_model.pkl"

TARGET = "Goals"

FEATURES = [
    "Year",
    "Disposals",
    "Marks",
    "Behinds",
    "HitOuts",
    "Tackles",
    "Rebounds",
    "Inside50s",
    "Clearances",
    "Clangers",
    "Frees",
    "FreesAgainst",
    "ContestedMarks",
    "MarksInside50",
    "OnePercenters",
    "GoalAssists",
    "%Played",
    "Height",
    "Weight",
    "BMI",
    "AvgTemp",
    "TempRange",
    "IsRainy",
    "Age"
]


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=[TARGET])

    X = df[FEATURES].fillna(0)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )

    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        random_state=42
    )

    mlflow.set_experiment("AFL_Goal_Prediction")

    with mlflow.start_run():
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        mlflow.log_param("model_type", "XGBRegressor")
        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("max_depth", 5)
        mlflow.log_param("learning_rate", 0.05)
        mlflow.log_param("target", TARGET)

        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

        mlflow.xgboost.log_model(
            model,
            "model",
            registered_model_name="AFL_Goal_Predictor")

        joblib.dump(model, MODEL_PATH)

        print("Model training complete.")
        print(f"MSE: {mse:.4f}")
        print(f"R2: {r2:.4f}")
        print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
