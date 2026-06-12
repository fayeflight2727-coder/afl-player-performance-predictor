# Model Card — AFL Player Performance Predictor

**Version:** 2.0 (Production)
**Owner:** Tia Qiu (ML Analyst/PM)
**Last Updated:** 2026-06-12
**Status:** In development (Phase 3 — training in progress)

---

## Model Details

### Overview
A set of four position-specific regression models that predict player performance outcomes in AFL matches. Models are trained using a combination of physical attributes, game context, causal interaction features, and rolling performance statistics.

### Model Architecture
| Position | Target | Algorithm | Hyperparameter Tuning |
|----------|--------|-----------|----------------------|
| Forward | TotalScore | LassoCV (primary) / XGBoost | Optuna / Hyperopt |
| Midfield | Clearances | LassoCV (primary) / XGBoost | Optuna / Hyperopt |
| Ruck | HitOuts | LassoCV (primary) / XGBoost | Optuna / Hyperopt |
| Defender | Rebounds | LassoCV (primary) / XGBoost | Optuna / Hyperopt |

**Explainability:** SHAP (SHapley Additive exPlanations) applied to all models. See `src/visualization/explainability.py`.

**Experiment tracking:** All runs logged to MLflow. Model registry manages staging → production promotion.

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
| Coverage | 2012–2022 (training), 2023–2024 (validation) |
| Records | ~50,000+ player-game observations |
| Preprocessing | Outlier removal (3.0× IQR on training data only), time-based split |
| Data leakage controls | No future statistics used; lag features computed from past games only; `TimeSeriesSplit` for CV |

---

## Evaluation Results

### Test Set Performance (2025 season)

| Position | Target | R² | MAE | RMSE |
|----------|--------|-----|-----|------|
| Forward | TotalScore | **0.49** | 4.37 | 5.72 |
| Midfield | Clearances | **0.52** | 1.42 | 1.88 |
| Ruck | HitOuts | TBD | TBD | TBD |
| Defender | Rebounds | TBD | TBD | TBD |

*Ruck and Defender results to be updated after Phase 3 completion.*

### Baseline Comparison
Models compared against OLS/Lasso baseline and Random Forest. LassoCV selected as primary for interpretability. XGBoost retained for SHAP analysis.

---

## Key Feature Importances

### Forward Position
1. `MarksInside50` — coefficient +9.75 (dominant predictor)
2. `Frees` — positive
3. `Disposals` — positive
4. `BMI` / `BMISquared` — nonlinear physical effect

### Midfield Position
1. `Disposals` — coefficient +2.54 (primary driver)
2. `Inside50s` — positive
3. `Marks` — *negative* (efficiency over volume matters)
4. `Post666_x_Height` — rule-change era interaction

---

## Fairness & Limitations

### Known Disparities
Based on HTE analysis from Course 1:

| Segment | Finding | Risk |
|---------|---------|------|
| Young rucks (<23) | Height effect 8× stronger than veterans | Model may overestimate young ruck potential |
| Veteran forwards (>28) | Height and weight effects become *negative* | Model may underestimate experienced forwards |
| Post-2019 data | 6-6-6 rule structurally changed ruck performance | Mixing pre/post data without era features causes bias |

### Fairness Audit Status
- [ ] Position-group performance parity audit (Phase 5)
- [ ] Team-group performance parity audit (Phase 5)
- [ ] Age-segment performance parity audit (Phase 5)
- [ ] Bias mitigation if disparities found (Phase 5)

See `docs/fairness_audit_framework.md` for methodology.

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
- **Monitoring:** Feature drift checked daily via `src/monitoring/drift.py`
- **Retraining trigger:** PSI > 0.25 on key features OR R² drops below threshold on recent matches

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-02 | Course 1 — notebook-based, Streamlit dashboard |
| 2.0 | 2026-06 | Production refactor — FastAPI, MLflow, SHAP, CI/CD |
