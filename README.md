# AFL Player Performance Predictor — Production ML System

> **McGill University MMA8 · INSY 684 Enterprise Data Science & ML in Production**

A production-grade machine learning system that predicts goal-scoring probability and player performance for AFL (Australian Football League) players. Built on causal inference findings from Course 1, this system brings the model to production with a FastAPI serving layer, automated drift monitoring, SHAP explainability, and a full CI/CD pipeline.

---

## Team

| Name | Role | GitHub |
|------|------|--------|
| Tracy Chan | Backend / DevOps | [@tracychanty](https://github.com/tracychanty) |
| Yanxin Li | ML Engineer | [@yanxinlii](https://github.com/yanxinlii) |
| Faye Wu | Data Engineer | [@fayeflight2727-coder](https://github.com/fayeflight2727-coder) |
| Tia Qiu | ML Analyst / PM | [@TiaQiu1016](https://github.com/TiaQiu1016) |

**GitHub Repository:** [fayeflight2727-coder/afl-player-performance-predictor](https://github.com/fayeflight2727-coder/afl-player-performance-predictor)

---

## Business Problem

AFL teams make multi-million dollar decisions on player selection, recruitment, and in-game strategy — yet most decisions still rely on intuition rather than evidence. Our system gives coaching departments a data-backed decision-support tool: given a player's attributes and game context, how likely are they to score, and *why*?

**Core questions we answer:**
1. What is this player's predicted performance in this game context?
2. Which factors drive that prediction? (SHAP explainability)
3. Is the model fair across different positions and teams?
4. Is the model still reliable as player data drifts over time?

---

## Solution Architecture

```
Raw Data (AFL Stats)
        │
        ▼
┌─────────────────────┐
│  Feature Pipeline   │  Great Expectations validation
│  src/features/      │  Lag features, causal interactions
│  (Data Engineer)    │  Feature store (SQLite)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Model Training     │  Position-specific LassoCV / XGBoost
│  src/models/        │  AutoML (Optuna/Hyperopt)
│  (ML Engineer)      │  MLflow experiment tracking
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Explainability     │  SHAP force plots, waterfall charts
│  src/visualization/ │  Global feature importance dashboard
│  (ML Analyst)       │  Fairness audit across positions/teams
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  API Serving        │  FastAPI  POST /predict
│  src/api/           │  POST /predict/batch
│  (Backend/DevOps)   │  GET /explain/{player_id}
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Monitoring         │  PSI / KS drift detection
│  src/monitoring/    │  Daily automated checks
│  (Data Engineer)    │  GitHub Actions cron
└─────────────────────┘
```

---

## Key Results (from Course 1 Causal Analysis)

These findings from our causal inference work directly inform the feature engineering and model interpretation in this production system.

### Average Treatment Effects (ATE)

| Treatment | Position | Outcome | ATE (XGBoost) | Interpretation |
|-----------|----------|---------|---------------|----------------|
| Height | Ruck | HitOuts | **+3.66** | Taller rucks win significantly more hitouts |
| Weight | Ruck | HitOuts | **+4.22** | Heavier rucks dominate contests |
| BMI | Forward | TotalScore | **+0.64** | Physicality benefits all positions |
| BMI | Midfield | Clearances | **+0.62** | Modern game rewards physicality everywhere |
| Height | Forward | TotalScore | +0.06 | Minimal height benefit for forwards |
| Home | Ruck | HitOuts | +0.20 | Home advantage is ruck-specific |

### Rule Change Impact

| Rule | Position | Change | Effect |
|------|----------|--------|--------|
| 6-6-6 (2019) | Ruck | +2,226% | Made height dramatically more valuable |
| Rotation Caps | Midfield | −97% | Nearly eliminated weight advantage for mids |
| Stand Rule | Midfield | ~0% | Negligible effect on clearances |

### Heterogeneous Treatment Effects (HTE)

Key finding: **age and career stage moderate the physical attribute advantages significantly.**
- Young rucks (<23): Height ATE = +8.16 hitouts (strongest effect)
- Prime forwards (23–28): Height ATE = +0.84 (positive)
- Veteran forwards (>28): Height ATE = −0.51 (diminishing returns)

### Predictive Model Performance

| Position | Target | R² (Test) | MAE (Test) | Model |
|----------|--------|-----------|------------|-------|
| Forward | TotalScore | 0.49 | 4.37 | LassoCV |
| Midfield | Clearances | 0.52 | 1.42 | LassoCV |

**Validation strategy:** Chronological split — train through 2022, validate 2023–2024, test 2025.

**Top predictors:**
- Forward: `MarksInside50` (coeff: +9.75), `Frees`, `Disposals`
- Midfield: `Disposals` (coeff: +2.54), efficiency > volume

---

## Repository Structure

```
afl-player-performance-predictor/
│
├── src/
│   ├── features/
│   │   ├── build_features.py        # Feature pipeline (Data Engineer)
│   │   └── causal_interactions.py   # Height×Ruck, Weight×Midfield features
│   │
│   ├── models/
│   │   ├── train.py                 # Production training script (ML Engineer)
│   │   └── tune.py                  # AutoML hyperparameter tuning
│   │
│   ├── api/
│   │   ├── main.py                  # FastAPI application (Backend/DevOps)
│   │   ├── schemas.py               # Pydantic request/response schemas
│   │   └── dependencies.py          # Model loading from MLflow registry
│   │
│   ├── monitoring/
│   │   └── drift.py                 # PSI/KS drift detection (Data Engineer)
│   │
│   └── visualization/
│       └── explainability.py        # SHAP integration (ML Analyst)
│
├── tests/                           # Unit tests (all members)
│
├── reports/
│   └── data_quality/                # Great Expectations validation results
│
├── docs/
│   ├── kpis_and_success_metrics.md  # Project KPIs
│   ├── feature_catalog.md           # Feature definitions for stakeholders
│   ├── model_card.md                # Model performance and limitations
│   ├── fairness_audit_framework.md  # Bias audit methodology
│   └── stakeholder_template.md      # Communication template
│
├── .github/
│   └── workflows/
│       ├── ci.yml                   # Lint + unit tests on PR
│       └── cd.yml                   # Deploy on merge to main
│
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── requirements.txt
└── .env.example
```

---

## Production Deployment

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Single player prediction |
| `POST` | `/predict/batch` | Multiple players |
| `GET`  | `/explain/{player_id}` | SHAP explanation for a prediction |
| `GET`  | `/health` | Health check |
| `GET`  | `/ready` | Readiness check |

### Monitoring
- Drift detection runs daily at 2 AM UTC
- PSI and KS tests on incoming feature distributions
- Alerts sent to `#ml-monitoring` Slack channel

### CI/CD
- Push to `main` triggers deployment
- All PRs require lint + unit tests to pass
- At least 1 code review approval required before merge

---

## Running Locally

### With Docker (recommended)

```bash
docker build -t afl-predictor .
docker run -p 8000:8000 afl-predictor
```

Or with docker-compose (includes MLflow + monitoring services):

```bash
docker-compose up
```

### Without Docker

```bash
# Clone the repository
git clone https://github.com/fayeflight2727-coder/afl-player-performance-predictor.git
cd afl-player-performance-predictor

# Install dependencies
pip install -r requirements.txt

# Copy environment variables
cp .env.example .env

# Run the API
uvicorn src.api.main:app --reload --port 8000
```

### Run Tests

```bash
# Via Makefile
make test

# Or directly
pytest tests/ -v
```

---

## Branch Naming Convention

```
feature/phase{N}-{short-description}
# e.g.
feature/phase5-shap-explainability
feature/phase2-feature-pipeline
fix/model-drift-threshold
```

---

## Prior Work

This project builds directly on our Course 1 project. Original analysis available at:
[fayeflight2727-coder/AFL-prediction](https://github.com/fayeflight2727-coder/AFL-prediction)
- Completed causal inference (H1–H5) with ATE/HTE analysis
- Predictive model with Streamlit dashboard
- [Live Dashboard](https://team5-afl-performance-analysis.streamlit.app/)

---

## Dataset

**Source:** [Kaggle AFL Stats Dataset](https://www.kaggle.com/datasets/stoney71/aflstats)
- `players.csv`: Individual player statistics per game (kicks, handballs, marks, tackles, etc.)
- `stats.csv`: Match outcomes, scores, venues, attendance, team-level statistics
- Coverage: 100+ years of AFL history; primary analysis window 2012–2025

---

*McGill University MMA8 · INSY 684 · For educational purposes*
