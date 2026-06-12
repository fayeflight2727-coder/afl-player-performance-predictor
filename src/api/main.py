import pandas as pd
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


# ---------------------------------------------------------------------------
# APP STARTUP
# Loads the XGBoost model from MLflow (or local fallback) when the API starts.
# Runs once before the API begins accepting requests.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# GET /health
# Confirms the API server is running.
# Does not check whether the model is loaded - just confirms the app is alive.
# Used by Docker and CI/CD pipelines to verify the container started correctly.
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


# ---------------------------------------------------------------------------
# GET /ready
# Confirms the API is ready to serve predictions.
# Returns model_loaded=True only if the XGBoost model was successfully loaded.
# Used by CI/CD smoke tests after deployment to verify full readiness.
# ---------------------------------------------------------------------------
@app.get("/ready", response_model=ReadyResponse)
def ready():
    loaded = is_model_loaded()
    return ReadyResponse(
        status="ready" if loaded else "not ready",
        model_loaded=loaded
    )


# ---------------------------------------------------------------------------
# POST /predict
# Accepts 24 player features and returns predicted goal output.
# Input is validated automatically by Pydantic via the PlayerInput schema.
# Percent_Played is remapped to %Played to match the model's training feature names.
# Returns predicted_goals (float) and the model version that produced the result.
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PredictionOutput)
def predict(player: PlayerInput):

    if not is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Try again shortly or check /ready."
        )

    # Convert input to DataFrame
    input_dict = player.dict()

    # Remap Percent_Played back to %Played to match training feature names
    input_dict["%Played"] = input_dict.pop("Percent_Played")

    input_df = pd.DataFrame([input_dict])

    # Run prediction
    try:
        prediction = get_model().predict(input_df)
        predicted_goals = float(prediction[0])
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

    return PredictionOutput(
        predicted_goals=predicted_goals,
        model_version=get_model_version()
    )


# ---------------------------------------------------------------------------
# POST /predict/explain
# Placeholder endpoint for SHAP-based prediction explanations.
# Will be implemented once src/visualization/explainability.py is ready.
# Returns the same prediction as /predict plus a feature importance breakdown.
# ---------------------------------------------------------------------------
@app.post("/predict/explain")
def predict_explain(player: PlayerInput):
    raise HTTPException(
        status_code=501,
        detail="Not implemented yet. Waiting on explainability.py."
    )


# ---------------------------------------------------------------------------
# GET /monitoring/drift
# Placeholder endpoint for feature drift monitoring.
# Will be implemented once drift.py is finalised.
# Returns PSI scores for key features: Height, Weight, BMI, Age, Disposals, Clearances.
# ---------------------------------------------------------------------------
@app.get("/monitoring/drift")
def monitoring_drift():
    raise HTTPException(
        status_code=501,
        detail="Not implemented yet. Waiting on drift.py."
    )
