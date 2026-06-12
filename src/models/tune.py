import os
import joblib
import optuna
import pandas as pd
import mlflow
import mlflow.xgboost

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


DATA_PATH = "data/processed/afl_features_latest.csv"
MODEL_DIR = "models"
BEST_MODEL_PATH = "models/xgb_goal_model_optuna.pkl"

TARGET = "Goals"

FEATURES = [
    "Year", "Disposals", "Marks", "Behinds", "HitOuts", "Tackles",
    "Rebounds", "Inside50s", "Clearances", "Clangers", "Frees",
    "FreesAgainst", "ContestedMarks", "MarksInside50",
    "OnePercenters", "GoalAssists", "%Played", "Height", "Weight",
    "BMI", "AvgTemp", "TempRange", "IsRainy", "Age"
]


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": 42,
    }

    model = XGBRegressor(**params)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    return mse


def main():
    global X_train, X_test, y_train, y_test

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

    mlflow.set_experiment("AFL_Goal_Prediction_Optuna")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    best_params = study.best_params
    best_params["random_state"] = 42

    best_model = XGBRegressor(**best_params)
    best_model.fit(X_train, y_train)

    preds = best_model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    with mlflow.start_run(run_name="optuna_best_xgb"):
        mlflow.log_params(best_params)
        mlflow.log_metric("best_mse", mse)
        mlflow.log_metric("best_r2", r2)

        mlflow.xgboost.log_model(best_model, "model")
        joblib.dump(best_model, BEST_MODEL_PATH)

    print("Optuna tuning complete.")
    print("Best parameters:", best_params)
    print(f"Best MSE: {mse:.4f}")
    print(f"Best R2: {r2:.4f}")
    print(f"Best model saved to {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
