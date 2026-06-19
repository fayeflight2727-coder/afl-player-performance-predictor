# Model Card — AFL Player Performance Predictor

**Version:** 2.0 (Production)
**Status:** Deployed — Phase 5 (API live, SHAP explainability endpoint live)

---

## Model Details

### Overview
A single XGBoost regression model that predicts goals scored per player per game. Trained on physical attributes, game context, and in-game statistics. The model implicitly handles position differences through the statistics themselves (e.g. high HitOuts signals a Ruck; high MarksInside50 signals a Forward).

### Model Architecture

| Property | Detail |
|----------|--------|
| Algorithm | XGBRegressor (XGBoost gradient boosting) |
| Target | `Goals` — goals scored per player per game |
| Input features | 24 features (see Feature Inputs section below) |
| Hyperparameters | `n_estimators=300`, `max_depth=5`, `learning_rate=0.05`, `random_state=42` |
| Model file | `models/xgb_goal_model.pkl` |

**Note on architecture:** The production model is a single regression model predicting `Goals` for all positions. Position-specific routing (Forward→TotalScore, Midfield→Clearances, etc.) from the original Course 1 design was not carried into this production build. The model implicitly learns position differences from the input statistics (e.g. high `HitOuts` indicates a Ruck; high `MarksInside50` indicates a Forward).

**Explainability:** SHAP TreeExplainer applied via `src/visualization/explainability.py`.

**Experiment tracking:** Runs logged to MLflow experiment `AFL_Goal_Prediction`. Model registered as `AFL_Goal_Predictor`.

### Hyperparameter Tuning (Optuna)

Automated hyperparameter search was run using Optuna (20 trials, `src/models/tune.py`).

| Parameter | Baseline | Optuna Best |
|-----------|----------|-------------|
| `n_estimators` | 300 | 196 |
| `max_depth` | 5 | 4 |
| `learning_rate` | 0.05 | 0.0976 |
| `subsample` | 1.0 (default) | 0.7962 |
| `colsample_bytree` | 1.0 (default) | 0.9980 |

**Results:** Optuna best: R²=0.4893, MSE=0.3920 vs baseline R²=0.4890, MSE=0.3922. The improvement in predictive accuracy is marginal (ΔR²=0.0003).

**Decision:** Baseline model retained in production. However, the Optuna-suggested parameters reveal a meaningful insight: a shallower tree structure (`max_depth=4` vs 5), fewer estimators (196 vs 300), and row subsampling (`subsample=0.7962`) produce nearly identical accuracy with a simpler, more regularized model. This reduces overfitting risk and improves generalization to unseen player profiles — a signal that the baseline is not underfit but that further complexity does not help.

### Feature Inputs (24 features)

`Year`, `Disposals`, `Marks`, `Behinds`, `HitOuts`, `Tackles`, `Rebounds`, `Inside50s`, `Clearances`, `Clangers`, `Frees`, `FreesAgainst`, `ContestedMarks`, `MarksInside50`, `OnePercenters`, `GoalAssists`, `%Played`, `Height`, `Weight`, `BMI`, `AvgTemp`, `TempRange`, `IsRainy`, `Age`

See `docs/feature_catalog.md` for definitions. `Goals` itself is excluded (target variable). `PrimaryPosition` is excluded (not yet used as a feature).

---

## Intended Use

### Primary Use Case
Decision support for AFL coaching departments:
- Pre-game player selection and lineup optimization
- In-game substitution decisions
- Opponent weakness analysis
- Recruitment and talent identification

### Out-of-Scope Uses
- This model is not intended for gambling or betting applications
- Not designed for real-time in-play prediction (latency not optimized for sub-second decisions)
- Not validated for junior/women's AFL leagues (training data covers men's senior AFL only)

---

## Training Data

| Property | Detail |
|----------|--------|
| Source | Kaggle AFL Stats Dataset (`players.csv`, `stats.csv`) |
| Coverage | Full processed dataset: 127,116 rows (2012–2025) |
| Train / test split | Chronological 80/20 split (`shuffle=False`) on the full dataset |
| Preprocessing | Feature pipeline via `src/features/build_features.py`; missing values filled with 0 |
| Data leakage controls | No future statistics used; chronological split prevents look-ahead bias |

---

## Evaluation Results

### Test Set Performance

| Metric | Value | Notes |
|--------|-------|-------|
| **R²** | **0.4890** | Model explains ~49% of variance in goals scored |
| **MAE** | **0.4293 goals** | Average prediction error under half a goal |
| **RMSE** | **0.6262 goals** | |
| Mean actual goals | 0.51 | Low baseline — most players score 0 or 1 goals per game |

*Evaluated on chronological 20% holdout of the full 2012-2025 dataset (≈2018–2025, n=25,424).*

### Prediction Range by Position Profile

Based on API testing with representative player profiles:

| Position | Sample Input Profile | Predicted Goals |
|----------|---------------------|-----------------|
| Forward | 22 disposals, 3 MarksInside50, 85% played | 2.14 |
| Midfield | 28 disposals, 8 clearances, 100% played | 0.53 |
| Ruck | 18 disposals, 35 hitouts, 100% played | 0.22 |
| Defender | 20 disposals, 8 rebounds, 100% played | 0.05 |

The model correctly assigns higher goal predictions to Forwards and near-zero predictions to Defenders without explicit position routing.

### Baseline Comparison
XGBRegressor outperforms a naive mean baseline (R²=0) and is retained over LassoCV for SHAP compatibility. Full ablation study not yet completed.

---

## Key Feature Importances

Feature importances below are from the Course 1 causal analysis (LassoCV coefficients). XGBoost SHAP-based importances are available via the `POST /predict/explain` endpoint.

### From Course 1 Analysis (LassoCV coefficients)

| Position | Top Feature | Coefficient | Notes |
|----------|-------------|-------------|-------|
| Forward | `MarksInside50` | +9.75 | Dominant predictor of goal scoring |
| Forward | `Disposals` | positive | Volume contributor |
| Midfield | `Disposals` | +2.54 | Primary driver of clearances (Course 1 Midfield→Clearances model; production model predicts Goals) |
| Midfield | `Marks` | negative | Efficiency over volume matters (same caveat) |

For live XGBoost SHAP values on any prediction, call `POST /predict/explain` — returns top 10 features by |SHAP| impact.

---

## Fairness & Limitations

### Known Disparities
Based on HTE analysis from Course 1:

| Segment | Finding | Risk |
|---------|---------|------|
| Young rucks (<23) | Height effect 8× stronger than veterans | Model may overestimate young ruck potential |
| Veteran forwards (>28) | Height and weight effects become *negative* | Model may underestimate experienced forwards |
| Pre/post-2019 data | 6-6-6 rule structurally changed ruck performance | **Confirmed, not hypothetical** — fairness audit shows the model mixes pre- and post-rule-change data without era features, and the Pre-6-6-6 group is flagged for significantly higher error (see Fairness Audit Status below) |

### Fairness Audit Status
- [x] Position-group performance parity audit — 2 flagged (Forward MAE 1.40×, Midfield R² gap 0.23)
- [x] Team-group performance parity audit — 2 flagged (Carlton, Richmond)
- [x] Age-segment performance parity audit — all PASS (Young, Prime, Veteran)
- [x] Rule-change era audit — Pre-6-6-6 era flagged (R² gap = 0.157, triggered flag; MAE ratio = 1.21× is below the 1.3× threshold; n=568, treat as monitoring signal) — consistent with model mixing pre/post-rule-change data without era indicator features
- [ ] Bias mitigation (position re-weighting, era indicator features) — future work

See `reports/fairness_report.md` for full results and `docs/fairness_audit_framework.md` for methodology.

---

## Ethical Considerations

- **Player privacy:** All data is publicly available match statistics. No private health or biometric data is used beyond what appears in public AFL records.
- **Causal vs. predictive:** The causal analysis (Course 1) established *why* physical attributes matter. The predictive model uses these as features, not to make causal claims in individual predictions.
- **Coaching decisions:** Predictions are decision *support*, not decision *replacement*. Coaches retain full authority over selection.

---

## Deployment

- **Serving:** FastAPI via Docker container
- **Registry:** MLflow Model Registry (staging → production)
- **Endpoint:** `POST /predict` — see `src/api/main.py`
- **Monitoring:** Feature drift checked weekly via `src/monitoring/drift.py` (PSI-based); Slack/email alerts on detection
- **Retraining trigger:** PSI > 0.1 → alert team, schedule review; PSI > 0.2 → auto-trigger retraining pipeline

### Current Drift Status

| Feature | PSI Score | Status | Severity |
|---------|-----------|--------|----------|
| BMI | 0.327 | DRIFT | High |
| Weight | 0.148 | DRIFT | Moderate |
| Height | 0.012 | Stable | Low |
| Age | 0.030 | Stable | Low |
| Disposals | 0.014 | Stable | Low |
| Clearances | 0.003 | Stable | Low |
| Marks | 0.003 | Stable | Low |
| Tackles | 0.012 | Stable | Low |
| Inside50s | 0.001 | Stable | Low |

BMI and Weight drift is attributed to players getting heavier over time, consistent with AFL rotation cap changes reducing weight disadvantage. A retraining validation was run comparing the full dataset model (R²=0.489) against a recent-data-only model (2020–2025, R²=0.481): goal prediction performance did not degrade, so the full-dataset model is retained as the production deployment.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-02 | Course 1 — notebook-based, Streamlit dashboard |
| 2.0 | 2026-06 | Production refactor — FastAPI, MLflow, SHAP, CI/CD |
| 2.1 | 2026-06 | Drift monitoring integrated (weekly PSI); BMI/Weight drift detected but goal prediction stable — full 2012-2025 model retained (R²=0.489 vs 0.481 on recent-only retrain) |
