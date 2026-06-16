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
        from src.visualization.explainability import generate_explanation_dict

        # NOTE: pass the raw model here, not ModelWrapper. get_explainer() in
        # explainability.py picks the SHAP explainer by checking
        # type(model).__name__ against strings like "xgb"/"xgboost" -- a
        # ModelWrapper-wrapped object's type name is "ModelWrapper", which
        # matches none of those checks and silently falls through to the
        # slow, approximate KernelExplainer fallback instead of the intended
        # TreeExplainer. generate_explanation_dict() never calls model.predict()
        # directly (it reconstructs the prediction from baseline + sum(SHAP)),
        # so ModelWrapper's string-output guard is not needed on this path.
        raw_model = get_model()

        # Load background sample from training data so SHAP has a baseline to compare against.
        # Using the input row as its own background causes all SHAP values to be 0.0.
        background_df = pd.read_csv("data/processed/afl_features_latest.csv")
        background_df = background_df[background_df["Year"] <= 2022]

        # Keep only the 24 features the model was trained on
        model_features = [
            "Year", "Disposals", "Marks", "Behinds", "HitOuts", "Tackles",
            "Rebounds", "Inside50s", "Clearances", "Clangers", "Frees",
            "FreesAgainst", "ContestedMarks", "MarksInside50", "OnePercenters",
            "GoalAssists", "%Played", "Height", "Weight", "BMI",
            "AvgTemp", "TempRange", "IsRainy", "Age"
        ]
        background_sample = background_df[model_features].dropna().sample(50, random_state=42)

        explanation = generate_explanation_dict(
            model=raw_model,
            X_instance=input_df,
            X_background=background_sample,
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
