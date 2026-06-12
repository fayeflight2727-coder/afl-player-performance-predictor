import mlflow.pyfunc
import joblib
import signal

model = None
model_version = "unknown"


def _timeout_handler(signum, frame):
    raise TimeoutError("MLflow connection timed out")


def load_model():
    global model, model_version

    # Try MLflow with a 10 second timeout
    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(10)
        model = mlflow.pyfunc.load_model("models:/AFL_Goal_Predictor/Production")
        signal.alarm(0)
        model_version = "Production"
        print("✓ Model loaded from MLflow registry (Production)")

    except Exception as e:
        signal.alarm(0)
        print(f"⚠ MLflow load failed: {e}")
        print("→ Falling back to local .pkl file...")

        try:
            model = joblib.load("models/xgb_goal_model.pkl")
            model_version = "local"
            print("✓ Model loaded from local .pkl file")

        except Exception as e2:
            print(f"✗ Local model load also failed: {e2}")
            model = None
            model_version = "unknown"


def get_model():
    return model


def get_model_version():
    return model_version


def is_model_loaded() -> bool:
    return model is not None
