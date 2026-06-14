# API Documentation

**Project:** AFL Player Performance Predictor   
**Last Updated:** 2026-06-14  
**Base URL (local):** `http://localhost:8000`  
**API Framework:** FastAPI 0.110+  
**Model:** XGBoost regression, `model_version: "local"`

---

## Overview

The AFL Prediction API accepts per-game player statistics and returns a predicted goal output for that player. The model was trained on AFL data from 2012–2022 using chronological splits (validate 2023–2024, test 2025).

All endpoints return JSON. Predictions are regression outputs (continuous float), not classification probabilities.

---

## Endpoints

### GET /health

Confirms the API server is running. Does **not** check whether the model is loaded.

**Request**
```
GET /health
```

**Response** `200 OK`
```json
{"status": "ok"}
```

---

### GET /ready

Confirms the API is ready to serve predictions. Returns `model_loaded: true` only if the XGBoost model was successfully loaded at startup.

**Request**
```
GET /ready
```

**Response — model loaded** `200 OK`
```json
{"status": "ready", "model_loaded": true}
```

**Response — model not yet loaded** `200 OK`
```json
{"status": "not ready", "model_loaded": false}
```

---

### POST /predict

Accepts 24 player features and returns the model's predicted goal output for that game.

**Request**
```
POST /predict
Content-Type: application/json
```

#### Request Body Schema

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `Year` | int | Season year | `2025` |
| `Disposals` | int | Total kicks + handballs | `22` |
| `Marks` | int | Total marks taken | `6` |
| `Behinds` | int | Behinds scored in match | `1` |
| `HitOuts` | int | Hitouts from ruck contests | `0` |
| `Tackles` | int | Defensive tackles | `4` |
| `Rebounds` | int | Defensive rebounds | `1` |
| `Inside50s` | int | Times moved ball inside 50m arc | `5` |
| `Clearances` | int | Clearances from stoppages | `3` |
| `Clangers` | int | Errors / turnovers | `2` |
| `Frees` | int | Free kicks won | `1` |
| `FreesAgainst` | int | Free kicks conceded | `1` |
| `ContestedMarks` | int | Marks taken under pressure | `1` |
| `MarksInside50` | int | Marks inside the 50m arc | `3` |
| `OnePercenters` | int | Spoils, shepherds, smothers | `2` |
| `GoalAssists` | int | Goals assisted | `1` |
| `Percent_Played` | float | Percentage of match played (renamed from `%Played` for API compatibility) | `85.0` |
| `Height` | int | Player height in cm | `185` |
| `Weight` | int | Player weight in kg | `87` |
| `BMI` | float | Body Mass Index (`Weight / (Height/100)²`) | `25.4` |
| `AvgTemp` | float | Average game-day temperature °C | `18.5` |
| `TempRange` | float | Max − Min temperature °C | `8.0` |
| `IsRainy` | int | `1` if Rainfall > 0, else `0` | `0` |
| `Age` | int | Player age at time of game (`Year − BirthYear`) | `26` |

**Note:** `PrimaryPosition` and `Goals` are not part of the request schema. All 24 features above are required — the API returns a validation error if any are missing.

#### Response Schema

| Field | Type | Description |
|-------|------|-------------|
| `predicted_goals` | float | Predicted goal output for this player-game |
| `model_version` | string | Model version identifier from MLflow registry |

---

#### Example Requests and Real Responses

**Forward player** (22 disposals, 6 marks, 3 MarksInside50 — active scorer profile)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Year":2025,"Disposals":22,"Marks":6,"Behinds":1,"HitOuts":0,"Tackles":4,"Rebounds":1,"Inside50s":5,"Clearances":3,"Clangers":2,"Frees":1,"FreesAgainst":1,"ContestedMarks":1,"MarksInside50":3,"OnePercenters":2,"GoalAssists":1,"Percent_Played":85,"Height":185,"Weight":87,"BMI":25.4,"AvgTemp":18.5,"TempRange":8.0,"IsRainy":0,"Age":26}'
```
```json
{"predicted_goals": 2.1389107704162598, "model_version": "local"}
```

**Midfield player** (28 disposals, 8 clearances — contest/clearance profile)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Year":2025,"Disposals":28,"Marks":4,"Behinds":0,"HitOuts":0,"Tackles":6,"Rebounds":2,"Inside50s":3,"Clearances":8,"Clangers":3,"Frees":2,"FreesAgainst":2,"ContestedMarks":1,"MarksInside50":0,"OnePercenters":1,"GoalAssists":2,"Percent_Played":100,"Height":182,"Weight":83,"BMI":25.1,"AvgTemp":18.5,"TempRange":8.0,"IsRainy":0,"Age":25}'
```
```json
{"predicted_goals": 0.5338261127471924, "model_version": "local"}
```

**Ruck player** (35 hitouts, tall and heavy — contested ruck specialist)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Year":2025,"Disposals":18,"Marks":5,"Behinds":1,"HitOuts":35,"Tackles":3,"Rebounds":2,"Inside50s":2,"Clearances":5,"Clangers":2,"Frees":1,"FreesAgainst":1,"ContestedMarks":2,"MarksInside50":1,"OnePercenters":3,"GoalAssists":1,"Percent_Played":100,"Height":201,"Weight":105,"BMI":25.9,"AvgTemp":18.5,"TempRange":8.0,"IsRainy":0,"Age":27}'
```
```json
{"predicted_goals": 0.22037772834300995, "model_version": "local"}
```

**Defender player** (8 rebounds, high OnePercenters — defensive specialist)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Year":2025,"Disposals":20,"Marks":7,"Behinds":0,"HitOuts":0,"Tackles":3,"Rebounds":8,"Inside50s":1,"Clearances":2,"Clangers":1,"Frees":1,"FreesAgainst":2,"ContestedMarks":1,"MarksInside50":0,"OnePercenters":5,"GoalAssists":0,"Percent_Played":100,"Height":186,"Weight":88,"BMI":25.4,"AvgTemp":18.5,"TempRange":8.0,"IsRainy":0,"Age":28}'
```
```json
{"predicted_goals": 0.05483396351337433, "model_version": "local"}
```

**Prediction range interpretation:**

| Predicted Goals | Interpretation |
|-----------------|----------------|
| < 0.2 | Defender / non-scoring specialist |
| 0.2 – 0.6 | Ruck / midfield contributor |
| 0.6 – 1.5 | Midfield with forward role |
| 1.5 – 3.0 | Active forward |
| > 3.0 | Elite forward / high-impact game |

---

### POST /predict/batch

**Status:** Not yet implemented (returns `501 Not Implemented`)

Planned to accept an array of `PlayerInput` objects and return predictions for all in a single request.

---

### POST /predict/explain

Returns a SHAP-based explanation for a single player prediction. Accepts the same 24-feature body as `/predict`, plus an optional `position` query parameter.

**Request**
```
POST /predict/explain?position=Forward
Content-Type: application/json
```

| Query param | Type | Default | Values |
|-------------|------|---------|--------|
| `position` | string | `"Forward"` | `Forward`, `Midfield`, `Ruck`, `Defender` |

Request body is identical to `POST /predict` (see schema above).

**Example response** `200 OK`
```json
{
  "player_id": "api_request",
  "position": "Forward",
  "target": "Total_Score",
  "baseline": 0.51,
  "prediction": 2.14,
  "top_features": [
    {"feature": "MarksInside50", "value": 3.0, "shap_value": 0.82, "direction": "positive"},
    {"feature": "GoalAssists",   "value": 1.0, "shap_value": 0.44, "direction": "positive"},
    {"feature": "Behinds",       "value": 1.0, "shap_value": 0.31, "direction": "positive"},
    {"feature": "Disposals",     "value": 22.0,"shap_value": 0.28, "direction": "positive"},
    {"feature": "HitOuts",       "value": 0.0, "shap_value": -0.12,"direction": "negative"}
  ]
}
```

**Implemented in:** `src/visualization/explainability.py` → `generate_explanation_dict()`

---

### GET /monitoring/drift

Returns PSI and drift detection results for 9 key features, comparing the training period (≤2022) against the current period (2023–2025).

**Request**
```
GET /monitoring/drift
```

**Example response** `200 OK`
```json
{
  "timestamp": "2026-06-14T19:41:55",
  "drift_results": {
    "Weight": {"psi": 0.148, "drift_detected": true,  "severity": "moderate"},
    "BMI":    {"psi": 0.091, "drift_detected": false,  "severity": "low"},
    "Height": {"psi": 0.012, "drift_detected": false,  "severity": "low"},
    "Age":    {"psi": 0.034, "drift_detected": false,  "severity": "low"}
  }
}
```

Features monitored: `Height`, `Weight`, `BMI`, `Age`, `Disposals`, `Clearances`, `Marks`, `Tackles`, `Inside50s`

**Implemented in:** `src/monitoring/drift.py` → `check_data_drift()`

---

## Error Responses

| Status | When | Example |
|--------|------|---------|
| `422 Unprocessable Entity` | Missing or wrong-type field in request body | `{"detail": [{"loc": ["body", "Year"], "msg": "field required"}]}` |
| `503 Service Unavailable` | Model not yet loaded at startup | `{"detail": "Model is not loaded. Try again shortly or check /ready."}` |
| `500 Internal Server Error` | Unexpected error during prediction | `{"detail": "Prediction failed: <error message>"}` |
| `501 Not Implemented` | Endpoint exists but not yet built | `{"detail": "Not implemented yet."}` |

---

## Known Issues and Limitations

- **Single model**: The current API routes all positions through one XGBoost model. Position-specific routing (Forward→TotalScore, Midfield→Clearances, etc.) is planned for a future phase.
- **`%Played` rename**: The feature pipeline uses `%Played` as the column name; the API accepts `Percent_Played` and renames it internally before passing to the model.
- **`Goals` excluded**: `Goals` is not a model input (it is part of the Forward model's target `Total_Score = Goals×6 + Behinds`; including it would be data leakage).
- **`PrimaryPosition` excluded**: Not yet a model input. Position routing is done at the application level, not feature level.

---

## Running the API Locally

```bash
# With Docker (recommended)
docker compose up

# Without Docker
uvicorn src.api.main:app --reload --port 8000
```

Interactive docs available at `http://localhost:8000/docs` (Swagger UI) once running.
