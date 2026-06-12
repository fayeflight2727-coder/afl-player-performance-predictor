from pydantic import BaseModel, Field


# Defines the structure of the incoming prediction request.
# Each field maps to one of the 24 features the XGBoost model was trained on.
class PlayerInput(BaseModel):
    Year: int = Field(..., example=2024)
    Disposals: int = Field(..., example=20)
    Marks: int = Field(..., example=5)
    Behinds: int = Field(..., example=2)
    HitOuts: int = Field(..., example=0)
    Tackles: int = Field(..., example=4)
    Rebounds: int = Field(..., example=3)
    Inside50s: int = Field(..., example=4)
    Clearances: int = Field(..., example=3)
    Clangers: int = Field(..., example=2)
    Frees: int = Field(..., example=1)
    FreesAgainst: int = Field(..., example=1)
    ContestedMarks: int = Field(..., example=2)
    MarksInside50: int = Field(..., example=2)
    OnePercenters: int = Field(..., example=1)
    GoalAssists: int = Field(..., example=1)
    Percent_Played: float = Field(..., example=85.0)
    Height: int = Field(..., example=185)
    Weight: int = Field(..., example=85)
    BMI: float = Field(..., example=24.8)
    AvgTemp: float = Field(..., example=18.5)
    TempRange: float = Field(..., example=8.0)
    IsRainy: int = Field(..., example=0)
    Age: int = Field(..., example=24)


# Defines what the API sends back after a successful prediction.
# predicted_goals is the raw number output from the XGBoost regression model (predicted number of goals).
# model_version tracks which version of AFL_Goal_Predictor from MLflow was used.
class PredictionOutput(BaseModel):
    predicted_goals: float
    model_version: str


# Used by the GET /health endpoint to confirm the server is running.
# Returns a simple status string e.g. "ok".
class HealthResponse(BaseModel):
    status: str


# Used by the GET /ready endpoint to confirm the server is ready to serve predictions.
# status indicates overall readiness e.g. "ready" or "not ready".
# model_loaded confirms whether the XGBoost model has been successfully loaded from MLflow.
class ReadyResponse(BaseModel):
    status: str
    model_loaded: bool
