import mlflow.pyfunc
import joblib
from typing import Optional

# Global variable that holds the loaded model in memory.
# Stays None until load_model() is successfully called at startup.
model = None
model_version = "unknown"


def load_model():
    """
    Loads the trained XGBoost model from MLflow registry at API startup.
    Tries the Production stage first, falls back to the local .pkl file
    if MLflow is unavailable.
    """
    global model, model_version

    try:
        # Attempt to load the latest Production model from MLflow registry.
        # Model was registered as "AFL_Goal_Predictor" in train.py.
        model = mlflow.pyfunc.load_model("models:/AFL_Goal_Predictor/Production")
        model_version = "Production"
        print("Model loaded from MLflow registry (Production)")

    except Exception as e:
        print(f"MLflow load failed: {e}")
        print("Falling back to local .pkl file...")

        try:
            # Fall back to the locally saved .pkl file from train.py.
            # Path: models/xgb_goal_model.pkl at the repo root.
            model = joblib.load("models/xgb_goal_model.pkl")
            model_version = "local"
            print("Model loaded from local .pkl file")

        except Exception as e2:
            print(f"Local model load also failed: {e2}")
            model = None
            model_version = "unknown"


def get_model():
    """
    Returns the currently loaded model.
    Called by the predict endpoint in main.py to access the model.
    """
    return model


def get_model_version():
    """
    Returns the version string of the currently loaded model.
    Included in every PredictionOutput response so callers know
    which model version produced the result.
    """
    return model_version


def is_model_loaded() -> bool:
    """
    Returns True if the model is loaded and ready to serve predictions.
    Used by the GET /ready endpoint in main.py to report readiness status.
    """
    return model is not None
