# AFL Player Performance Predictor — Final Presentation Draft
**McGill MMA8 INSY 684 | June 2026**

---

## SLIDE 1 — Title
**AFL Player Performance Predictor**
*Predicting, Explaining, and Auditing Goals in the Australian Football League*

McGill University — MMA8 INSY 684
June 2026

| Member | Role | GitHub ID |
|--------|------|-----------|
| Yanxin Li | ML Engineer | @yanxinlii |
| Faye | Data Engineer | @fayeflight2727-coder |
| Tracy Chan | Backend Engineer | @tracychanty |
| Tia Qiu | ML Analyst / PM | @tiaqiu1016 |

GitHub Repo: github.com/fayeflight2727-coder/afl-player-performance-predictor

---

## SLIDE 2 — Business Problem
**Can data predict who scores?**

AFL clubs spend millions on recruitment and selection decisions — largely based on coach intuition and scout reports.

Our system answers 3 questions from match statistics alone:
1. **What** will this player score in today's game?
2. **Why** did the model predict that? (SHAP explainability)
3. **Is the model fair** across positions, ages, and teams?

Use cases: pre-game lineup selection, in-game substitution, opponent weakness analysis, recruitment targeting.

---

## SLIDE 3 — Data Overview
**Dataset: Kaggle AFL Stats**
- 127,116 player-game rows | 2012–2025 | Men's senior AFL
- Sources: `players.csv` (physical attributes) + `stats.csv` (match stats)

**Target variable:** Goals per player per game
- Mean = 0.51 goals | Most players score 0 or 1 per game — highly skewed

**Train/test split:** Chronological 80/20 (no shuffle) to prevent look-ahead bias
- Training: 2020–2022 | Test: 2023–2025

---

## SLIDE 4 — Course 1 Causal Insights
**Why do physical attributes matter? (ATE findings)**

| Physical Attribute | Position | Causal Effect |
|-------------------|----------|---------------|
| Height | Ruck | +3.66 hitouts per 1 SD increase |
| Weight | Ruck | +4.22 hitouts per 1 SD increase |
| BMI | Forward | +0.64 goals per 1 SD increase |

**Heterogeneous Treatment Effects (HTE):**
- Young rucks (<23): Height ATE = **+8.16** hitouts → 8× stronger than veterans (−2.21)
- 6-6-6 rule change (2019): **+2,226% increase** in height effect for rucks — rule structurally changed the game

These causal findings validated which features to prioritize in the predictive model.

---

## SLIDE 5 — Solution Architecture
**End-to-end system overview**

```
[Raw Data (Kaggle)]
       ↓
[Feature Pipeline]  ←  src/features/build_features.py
       ↓
[XGBoost Model]     ←  models/xgb_goal_model.pkl  (MLflow Registry)
       ↓
[FastAPI Service]   ←  POST /predict | POST /predict/explain
       ↓
[Monitoring]        ←  GET /monitoring/drift  (PSI/KS drift detection)
```

**Tech stack:** Python · FastAPI · Docker · MLflow · XGBoost · SHAP · GitHub Actions · pytest

Each component is independently deployable, version-controlled, and observable.

---

## SLIDE 6 — Feature Engineering
**24 production features across 4 categories**

| Category | Features |
|----------|----------|
| Physical | Height, Weight, BMI, Age |
| Game context | Year, %Played, AvgTemp, TempRange, IsRainy |
| In-game stats | Disposals, Marks, Behinds, HitOuts, Tackles, Rebounds, Inside50s, Clearances, Clangers, Frees, FreesAgainst, ContestedMarks, MarksInside50, OnePercenters, GoalAssists |

**Key engineering decisions:**
- `Disposals = Kicks + Handballs` — using the aggregate avoids multicollinearity
- `%Played` retained — partial-game appearances are a major confounder for goal counts
- `PrimaryPosition` excluded from model features — implicit in the stats (high HitOuts = Ruck, high MarksInside50 = Forward)

---

## SLIDE 7 — Feature Engineering (Continued)
**Causal-informed feature construction**
- Age × position interaction terms from Course 1 HTE analysis inform which feature combinations to monitor
- Era flag logic: model trained on 2020–2025 only — post-6-6-6 era data — to avoid structural shift

**Data leakage controls:**
- Chronological split (`shuffle=False`) — no future stats leak into training
- Goals excluded from features (target variable)
- No rolling averages that look beyond the current game

---

## SLIDE 8 — Model Training
**Algorithm:** XGBRegressor (XGBoost gradient boosting)

| Hyperparameter | Value |
|----------------|-------|
| n_estimators | 300 |
| max_depth | 5 |
| learning_rate | 0.05 |
| random_state | 42 |

**Why XGBoost?**
- Handles mixed numeric features without scaling
- Natively compatible with SHAP TreeExplainer (fast, exact SHAP values)
- Outperforms LassoCV baseline in R² while remaining interpretable

**MLflow tracking:** Every run logged — parameters, metrics, artifacts, model version

---

## SLIDE 9 — AutoML & Hyperparameter Tuning
**Hyperparameter search via Optuna / Hyperopt**

- Search space: n_estimators [100–500], max_depth [3–8], learning_rate [0.01–0.2], subsample [0.6–1.0]
- Objective: minimize MAE on chronological validation fold
- Best trial selected and registered to MLflow Model Registry

**Version history:**

| Version | Training Data | R² | MAE |
|---------|--------------|-----|-----|
| v2.0 | 2012–2025 | 0.37 | — |
| v2.1 (current) | 2020–2025 | **0.489** | **0.4293** |

v2.1 retrained on recent seasons only → **+32% improvement in R²** by removing out-of-distribution pre-2019 data.

---

## SLIDE 10 — MLflow Experiment Tracking
**Reproducibility and model governance**

- Experiment: `AFL_Goal_Prediction`
- Registered model: `AFL_Goal_Predictor` (staging → production)
- Each run logs: hyperparameters, train/val MAE, R², RMSE, feature list, model artifact
- Model loaded at API startup: `mlflow.pyfunc.load_model("models:/AFL_Goal_Predictor/Production")`
- Fallback to `models/xgb_goal_model.pkl` for local development without MLflow server

---

## SLIDE 11 — Explainability: What is SHAP?
**SHAP (SHapley Additive exPlanations)**

For any single prediction:
```
predicted_goals = baseline + SHAP(MarksInside50) + SHAP(Disposals) + ... + SHAP(Age)
```

- **Baseline** = average prediction over background dataset (≈ 0.43 goals)
- Each feature's SHAP value = its contribution pushing prediction up or down
- TreeExplainer: exact SHAP values for XGBoost (no approximation)

**Why it matters for coaches:**
- Not just "2.14 goals predicted" — but *why*: "MarksInside50 added +1.75, HitOuts subtracted −0.12"
- Actionable: identify which player attributes to target in selection and training plans

---

## SLIDE 12 — Explainability: Forward Example
**POST /predict/explain — Forward profile (22 disposals, 3 MarksInside50, 85% played)**

| Feature | SHAP Value | Direction |
|---------|-----------|-----------|
| MarksInside50 | +1.747 | ↑ Goals |
| GoalAssists | +0.312 | ↑ Goals |
| Inside50s | +0.198 | ↑ Goals |
| HitOuts | −0.089 | ↓ Goals |
| Weight | −0.054 | ↓ Goals |

- **Baseline:** 0.4281 goals
- **Predicted:** 2.1389 goals
- **MarksInside50 dominates** — consistent with Course 1 LassoCV coefficient (+9.75)

---

## SLIDE 13 — API & Deployment
**FastAPI — 3 production endpoints**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/predict` | POST | Predict goals for a player profile |
| `/predict/explain` | POST | Predict + return top 10 SHAP values |
| `/monitoring/drift` | GET | PSI/KS drift scores for all features |

**Sample response — POST /predict:**
```json
{"predicted_goals": 2.1389, "model_version": "2.1"}
```

**Deployment:**
- Docker container → `docker build -t afl-predictor . && docker run -p 8000:8000 afl-predictor`
- Model loaded from MLflow registry at startup
- API validated with 41 smoke tests (pytest + requests)

---

## SLIDE 14 — Explainability: Live API
**Endpoint: POST /predict/explain**

```json
{
  "player_id": "api_request",
  "position": "Forward",
  "baseline": 0.4281,
  "prediction": 2.1389,
  "target": "Goals",
  "top_features": [
    {"feature": "MarksInside50", "shap_value": 1.747, "direction": "positive"},
    {"feature": "GoalAssists",   "shap_value": 0.312, "direction": "positive"},
    ...
  ]
}
```

Available for all 4 position profiles.
41 smoke tests verify endpoint correctness and SHAP non-zero values.

---

## SLIDE 15 — API & Deployment (Position Profiles)
**Model handles all 4 positions without explicit routing**

| Position | Predicted Goals | Key Driver |
|----------|-----------------|------------|
| Forward | 2.14 | High MarksInside50 |
| Midfield | 0.53 | High Clearances |
| Ruck | 0.22 | High HitOuts |
| Defender | 0.05 | High Rebounds |

The model implicitly learns positional patterns from the statistics themselves — no position-specific routing needed.

---

## SLIDE 16 — Solution Architecture: Deployment View
**Docker + MLflow + GitHub Actions — production-ready infrastructure**

- Model served inside a **Docker container** (reproducible, portable environment)
- **MLflow** tracks all experiment runs under `AFL_Goal_Prediction`; model registered as `AFL_Goal_Predictor`
- **GitHub Actions** runs lint + 41 smoke tests on every pull request
- Model loads from MLflow registry at startup; falls back to local `.pkl` if registry unavailable
- Drift monitor runs daily — triggers retraining alert when PSI > 0.25

This infrastructure reflects the full ML production lifecycle: build → register → serve → monitor → retrain.

---

## SLIDE 17 — CI/CD Pipeline
**Automated quality gate on every commit**

```
Pull Request → GitHub Actions triggers:
  1. Lint (flake8)
  2. Unit tests — test_explainability.py (11 tests, no API needed)
  3. Smoke tests — test_api_health, test_predict, test_explain (30 tests, requires live API)

Merge to main → Docker image rebuilt → API redeployed
```

**Test coverage:**
- 41 tests total: health checks, schema validation, SHAP correctness, drift endpoint
- `test_explainability.py`: pure unit tests using LinearRegression mock — runs in CI without Docker
- Smoke tests skip automatically if API is not running (`@pytest.mark.skipif`)

---

## SLIDE 18 — Fairness Audit: Methodology
**Ensuring the model treats all player groups equitably**

**Groups audited:**
1. Primary Position (Forward / Midfield / Ruck / Defender)
2. Age Segment (Young <23 / Prime 23–28 / Veteran >28)
3. Rule-change Era (Pre-6-6-6 <2019 / Post-6-6-6 2019+)
4. Team (18 AFL clubs)

**Flagging thresholds (adapted from industry fairness standards):**
- MAE ratio > 1.3× overall → group is harder to predict accurately
- R² gap > 0.10 vs overall → model explains much less variance for this group
- Statistical significance: Mann-Whitney U test, p < 0.05

**Test set:** Chronological 20% holdout, n = 25,424 observations

---

## SLIDE 19 — Fairness Audit: Results
**Overall model:** MAE = 0.4293 | R² = 0.489

| Group | Result | Key Finding |
|-------|--------|-------------|
| Forward | FLAGGED | MAE ratio = 1.40× — forwards harder to predict precisely |
| Midfield | FLAGGED | R² gap = 0.23 — model explains much less midfield variance |
| Ruck | PASS | — |
| Defender | PASS | — |
| Young / Prime / Veteran | PASS (all 3) | Age parity holds despite Course 1 HTE findings |
| Pre-6-6-6 era | FLAGGED | Out-of-distribution: model trained on 2020+ only |
| Carlton, Richmond | FLAGGED | Unusual player profiles under-represented in training |

**Total flagged: 5 groups across 27 tested**

**Recommended mitigations:** Re-weight Forward/Midfield training samples · Add era indicator features (`Post666`) · Re-audit after next retraining

---

## SLIDE 20 — Monitoring & Drift Detection
**GET /monitoring/drift — daily PSI/KS checks**

Comparing training distribution (≤2022) vs current season data (2023–2025):

| Feature | PSI | KS p-value | Status |
|---------|-----|------------|--------|
| Weight | 0.148 | — | ⚠ Moderate drift |
| Height | <0.10 | — | Stable |
| Disposals | <0.10 | — | Stable |
| ... (9 features total) | | | |

**Weight drift interpretation:** Modern AFL players are trending lighter — the 2023–2025 cohort has a shifted Weight distribution vs training data.

**Retraining trigger:** PSI > 0.25 on any key feature OR R² drops below threshold on recent match predictions.

---

## SLIDE 21 — Monitoring: Drift Types
**4 types of drift monitored (ML Engineering framework)**

| Drift Type | What it means | How we detect it |
|------------|--------------|-----------------|
| Feature drift | Input stats distribution changes | PSI / KS test |
| Target drift | Goals distribution shifts | PSI on Goals |
| Concept drift | Relationship between features and goals changes | R² degradation on recent data |
| Prediction drift | Model output distribution shifts | PSI on predicted_goals |

Current status: **Feature drift only (Weight, moderate)**. No concept or target drift detected.

---

## SLIDE 22 — Business Value: KPIs Met
**What we set out to deliver vs what we achieved**

| KPI | Target | Achieved |
|-----|--------|----------|
| Model R² | ≥ 0.40 | **0.489** ✓ |
| API latency | < 200ms | < 200ms ✓ |
| Test suite | ≥ 80% pass rate | 41/41 (100%) ✓ |
| Explainability | Per-prediction SHAP | Live endpoint ✓ |
| Fairness audit | All groups evaluated | 27 groups tested ✓ |
| Drift monitoring | PSI/KS live | Live endpoint ✓ |
| CI/CD | Automated on PR | GitHub Actions ✓ |

---

## SLIDE 23 — Business Value: Coaching Use Cases
**Turning model output into coaching decisions**

**Pre-game selection:**
- Run /predict for shortlisted players → rank by predicted goals → inform Forward line selection

**In-game substitution:**
- If a Forward has high MarksInside50 SHAP → keep on field even if disposal count is low

**Opponent analysis:**
- Profile opposing team's key Forwards → predict their goal output → design defensive assignments

**Recruitment:**
- Score draft prospects on historical stats → identify undervalued young Forwards with high MarksInside50

**Model tells you the what and the why — coaches decide what to do with it.**

---

## SLIDE 24 — Lessons Learned & Next Steps
**What we learned building a production ML system**

| Lesson | Detail |
|--------|--------|
| Column order matters in production | Python dict.pop() + re-add moves keys to end — breaks feature alignment silently |
| SHAP needs a background dataset | Same row as prediction → all SHAP = 0.0; 50 training rows required |
| Chronological splits are non-negotiable | Random splits produce inflated R²; time series data requires time-aware splits |
| Retraining on recent data beats more data | v2.0 (2012–2025): R²=0.37 → v2.1 (2020–2025): R²=0.49 |

**Next steps:**
- Position-specific sub-models (Forward model, Ruck model) to address Midfield R²=0.26
- Add era indicator features (`Post666`, `RotEra`) to fix Pre-6-6-6 fairness flag
- Causal interaction terms (height×ruck, age×position) as production features
- Expand to women's AFL (AFLW) once labelled data is available

---

*Presentation by: Yanxin Li · Faye · Tracy Chan · Tia Qiu*
*McGill MMA8 INSY 684 — June 2026*
