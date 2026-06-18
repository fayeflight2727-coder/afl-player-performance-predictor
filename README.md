# AFL Player Performance Predictor

> **McGill University MMA8 · INSY 684 Enterprise Data Science & ML in Production**

A production-grade machine learning system that predicts goals scored per player per AFL game. Built on causal inference findings from Course 1, the system serves predictions via a FastAPI REST API with SHAP-based explainability, automated drift monitoring, a fairness audit, and a full CI/CD pipeline.

---

## Team

| Name | Role | GitHub |
|------|------|--------|
| Tracy Chan | Backend / DevOps | [@tracychanty](https://github.com/tracychanty) |
| Yanxin Li | ML Engineer | [@yanxinlii](https://github.com/yanxinlii) |
| Faye Wu | Data Engineer | [@fayeflight2727-coder](https://github.com/fayeflight2727-coder) |
| Tia Qiu | ML Analyst / PM | [@TiaQiu1016](https://github.com/TiaQiu1016) |

---

## Business Problem

AFL clubs make multi-million dollar decisions on player selection, in-game substitution, opponent analysis, and recruitment — largely based on intuition and scout reports. This system provides a data-backed decision-support layer: given a player's match statistics and physical attributes, how many goals are they predicted to score, and why?

**Core questions answered:**
1. What is this player's predicted goal output for this game?
2. Which features drove that prediction? (SHAP explainability)
3. Is the model equally accurate across positions, ages, and teams? (Fairness audit)
4. Is the model still reliable as player populations shift over time? (Drift monitoring)

---

## Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | XGBRegressor |
| Target | Goals per player per game |
| R² | **0.4890** |
| MAE | **0.4293 goals** |
| RMSE | 0.6262 goals |
| Training data | 101,692 rows, 2012–2025 (chronological 80/20 split) |
| Test set | 25,424 rows, 2018–2025 |

**Position prediction range:**

| Position | Sample Profile | Predicted Goals |
|----------|---------------|-----------------|
| Forward | 22 disposals, 3 MarksInside50, 85% played | 2.14 |
| Midfield | 28 disposals, 8 clearances, 100% played | 0.53 |
| Ruck | 18 disposals, 35 hitouts, 100% played | 0.22 |
| Defender | 20 disposals, 8 rebounds, 100% played | 0.05 |

---

## Solution Architecture

```
Raw Data (Kaggle AFL Stats)
         │
         ▼
┌─────────────────────────┐
│   Feature Pipeline      │  src/features/build_features.py
│   24 production         │  Physical attributes, game context,
│   features engineered   │  in-game stats, weather
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   Model Training        │  src/models/train.py
│   XGBRegressor          │  MLflow experiment tracking
│   + Hyperparameter      │  Registered as AFL_Goal_Predictor
│   Tuning (Optuna)       │  src/models/tune.py
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   Explainability        │  src/visualization/explainability.py
│   SHAP TreeExplainer    │  Exact SHAP values per prediction
│   Fairness Audit        │  src/visualization/fairness_audit.py
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   API Serving           │  src/api/main.py  (FastAPI)
│   POST /predict         │  Goal prediction
│   POST /predict/explain │  Prediction + SHAP breakdown
│   GET  /monitoring/drift│  PSI drift scores
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   Monitoring            │  src/monitoring/drift.py
│   PSI drift detection   │  Weekly automated checks
│   BMI: 0.327 (HIGH)     │  Retrain triggered at PSI > 0.2
│   Weight: 0.148 (MOD)   │
└─────────────────────────┘
```

---

## Causal Findings (Course 1)

These findings from Course 1 causal analysis inform the feature choices and model interpretation.

### Average Treatment Effects (ATE)

| Treatment | Position | Outcome | ATE | Interpretation |
|-----------|----------|---------|-----|----------------|
| Height | Ruck | HitOuts | **+3.66** | Taller rucks win significantly more hitouts |
| Weight | Ruck | HitOuts | **+4.22** | Heavier rucks dominate contests |
| BMI | Forward | Goals | **+0.64** | Physicality benefits scoring |

### Rule Change Impact

| Rule | Effect |
|------|--------|
| 6-6-6 (2019) | +2,226% increase in height effect for Rucks — structurally changed positional roles |
| Rotation Caps | −97% reduction in weight advantage for Midfielders |

### Heterogeneous Treatment Effects (HTE)

Age moderates physical attribute effects significantly:

| Segment | Height → HitOuts (Ruck) |
|---------|------------------------|
| Young (<23) | **+8.16** |
| Prime (23–28) | +0.95 |
| Veteran (>28) | −2.21 |

---

## Fairness Audit

The model was audited across 27 groups in 4 dimensions on the chronological test set (n = 25,424).

| Group | Result | Key Finding |
|-------|--------|-------------|
| Forward | FLAGGED | MAE ratio 1.40× — harder to predict |
| Midfield | FLAGGED | R² gap 0.23 — low explained variance |
| Ruck | PASS | — |
| Defender | PASS | — |
| Age (all 3 segments) | PASS | Age parity holds |
| Pre-6-6-6 era | FLAGGED | R² gap 0.157; n=568 — monitoring signal |
| Carlton, Richmond | FLAGGED | Under-represented profiles |
| 16 other teams | PASS | No significant disparity |

Full results: `reports/fairness_report.md` · Methodology: `docs/fairness_audit_framework.md`

---

## Repository Structure

```
afl-player-performance-predictor/
│
├── data/
│   ├── raw/                         # Source CSVs from Kaggle (players.csv, stats.csv)
│   └── processed/                   # afl_features_latest.csv (feature pipeline output)
│
├── src/
│   ├── features/
│   │   ├── build_features.py        # Feature pipeline
│   │   └── validate_for_training.py # Pre-training data validation
│   │
│   ├── models/
│   │   ├── train.py                 # Production training script
│   │   └── tune.py                  # Optuna hyperparameter tuning
│   │
│   ├── api/
│   │   ├── main.py                  # FastAPI application
│   │   ├── schemas.py               # Pydantic request/response schemas
│   │   └── dependencies.py          # Model loading (MLflow / local pkl)
│   │
│   ├── monitoring/
│   │   └── drift.py                 # PSI drift detection
│   │
│   └── visualization/
│       ├── explainability.py        # SHAP TreeExplainer integration
│       └── fairness_audit.py        # Fairness audit across groups
│
├── tests/
│   ├── conftest.py
│   ├── test_api_health.py
│   ├── test_predict.py
│   ├── test_explain.py
│   └── test_explainability.py       # Unit tests (no API required)
│
├── reports/
│   ├── fairness_report.md
│   ├── fairness_metrics.csv
│   └── figures/
│       ├── fairness/                # Fairness audit plots
│       └── shap/                    # SHAP waterfall plots by position
│
├── docs/
│   ├── api_documentation.md
│   ├── feature_catalog.md
│   ├── model_card.md
│   ├── fairness_audit_framework.md
│   └── data_pipeline.md
│
├── models/
│   └── xgb_goal_model.pkl           # Trained XGBoost model
│
├── .github/
│   └── workflows/
│       ├── ci.yml                   # Lint + tests on PR
│       └── cd.yml                   # Deploy on merge to main
│
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
└── requirements.txt
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Server health check |
| `GET` | `/ready` | Model load readiness check |
| `POST` | `/predict` | Predict goals for a player profile |
| `POST` | `/predict/explain` | Predict + return top 10 SHAP values |
| `GET` | `/monitoring/drift` | PSI drift scores for 9 key features |

Full request/response schemas and example curl commands: `docs/api_documentation.md`

---

## Running Locally

### With Docker (recommended)

```bash
docker compose up
```

API available at `http://localhost:8000` · Swagger UI at `http://localhost:8000/docs`

### Without Docker

```bash
git clone https://github.com/fayeflight2727-coder/afl-player-performance-predictor.git
cd afl-player-performance-predictor
pip install -r requirements.txt
uvicorn src.api.main:app --reload --port 8000
```

### Run Tests

```bash
pytest tests/ -v
# or
make test
```

---

## CI/CD

- Every pull request triggers lint (flake8) + 41 tests via GitHub Actions
- Smoke tests skip automatically if API is not running (`@pytest.mark.skipif`)
- Merge to main triggers Docker image rebuild and API redeployment

---

## Prior Work

This project builds on our Course 1 causal analysis:
[fayeflight2727-coder/AFL-prediction](https://github.com/fayeflight2727-coder/AFL-prediction)

---

## Dataset

**Source:** [Kaggle AFL Stats Dataset](https://www.kaggle.com/datasets/stoney71/aflstats)
- `players.csv`: Player physical attributes (height, weight, DOB, position)
- `stats.csv`: Per-game match statistics (disposals, marks, goals, etc.)
- Coverage: 2012–2025, men's senior AFL

---

*McGill University MMA8 · INSY 684 · For educational purposes*
