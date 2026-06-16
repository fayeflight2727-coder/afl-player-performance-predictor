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


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=[TARGET])

    X = df[FEATURES].fillna(0)
    y = df[TARGET]

    # 60% train, 20% validation, 20% test
    # Test set is only used once at the very end.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, shuffle=False
    )

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
            "objective": "reg:squarederror",
        }

        model = XGBRegressor(**params)
        model.fit(X_train, y_train)

        val_preds = model.predict(X_val)
        val_mse = mean_squared_error(y_val, val_preds)

        return val_mse

    mlflow.set_experiment("AFL_Goal_Prediction_Optuna")

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=20)

    best_params = study.best_params
    best_params["random_state"] = 42
    best_params["objective"] = "reg:squarederror"

    # Refit final model on train + validation data
    best_model = XGBRegressor(**best_params)
    best_model.fit(X_train_val, y_train_val)

    # Final untouched test evaluation
    test_preds = best_model.predict(X_test)
    test_mse = mean_squared_error(y_test, test_preds)
    test_r2 = r2_score(y_test, test_preds)

    with mlflow.start_run(run_name="optuna_best_xgb_validation_tuned"):
        mlflow.log_params(best_params)
        mlflow.log_metric("validation_best_mse", study.best_value)
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("test_r2", test_r2)

        mlflow.xgboost.log_model(
            best_model,
            "model",
            registered_model_name="AFL_Goal_Predictor"
        )

        joblib.dump(best_model, BEST_MODEL_PATH)

    print("Optuna tuning complete.")
    print("Best parameters:", best_params)
    print(f"Best Validation MSE: {study.best_value:.4f}")
    print(f"Final Test MSE: {test_mse:.4f}")
    print(f"Final Test R2: {test_r2:.4f}")
    print(f"Best model saved to {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
