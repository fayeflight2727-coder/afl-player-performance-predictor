import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from src.api.schemas import (
    PlayerInput,
    PredictionOutput,
    HealthResponse,
    ReadyResponse
)
from src.api.dependencies import (
    load_model,
    get_model,
    get_model_version,
    is_model_loaded
)


class ModelWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        raw = self.model.predict(X)
        if isinstance(raw, str):
            raw = raw.strip("[]").split()
        return np.array(raw, dtype=float)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting AFL Prediction API...")
    load_model()
    yield
    print("Shutting down AFL Prediction API...")


app = FastAPI(
    title="AFL Goal Prediction API",
    description="Predicts AFL player goal-scoring output using a trained XGBoost model.",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.get("/ready", response_model=ReadyResponse)
def ready():
    loaded = is_model_loaded()
    return ReadyResponse(
        status="ready" if loaded else "not ready",
        model_loaded=loaded
    )


@app.post("/predict", response_model=PredictionOutput)
def predict(player: PlayerInput):
    if not is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Try again shortly or check /ready."
        )
    input_dict = player.dict()
    input_df = pd.DataFrame([input_dict]).rename(columns={"Percent_Played": "%Played"})
    try:
        wrapped = ModelWrapper(get_model())
        prediction = wrapped.predict(input_df)
        predicted_goals = float(prediction[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    return PredictionOutput(
        predicted_goals=predicted_goals,
        model_version=get_model_version()
    )


@app.post("/predict/explain")
def predict_explain(player: PlayerInput, position: str = "Forward"):
    if not is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Try again shortly or check /ready."
        )
    input_dict = player.dict()
    input_df = pd.DataFrame([input_dict]).rename(columns={"Percent_Played": "%Played"})
    try:
        wrapped = ModelWrapper(get_model())
        from src.visualization.explainability import generate_explanation_dict
        explanation = generate_explanation_dict(
            model=wrapped,
            X_instance=input_df,
            X_background=input_df,
            position=position,
            player_id="api_request",
            top_n=10
        )
        return explanation
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@app.get("/monitoring/drift")
def monitoring_drift():
    try:
        from src.monitoring.drift import check_data_drift
        reference = pd.read_csv("data/processed/afl_features_latest.csv")
        reference_train = reference[reference["Year"] <= 2022]
        current = reference[reference["Year"] >= 2023]
        drift_features = [
            "Height", "Weight", "BMI", "Age", "Disposals",
            "Clearances", "Marks", "Tackles", "Inside50s"
        ]
        result = check_data_drift(reference_train, current, drift_features)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drift check failed: {str(e)}")
