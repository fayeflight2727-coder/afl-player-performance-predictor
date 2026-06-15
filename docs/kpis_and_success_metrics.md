# Project KPIs & Success Metrics

**Project:** AFL Player Performance Predictor — Production ML System

---

## 1. Model Performance KPIs

These are the minimum thresholds the model must meet to be considered production-ready.

| Metric | Position | Baseline (Course 1) | Target | Critical Threshold |
|--------|----------|---------------------|--------|--------------------|
| R² | Forward (TotalScore) | 0.49 | ≥ 0.55 | < 0.40 triggers retraining |
| MAE | Forward (TotalScore) | 4.37 | ≤ 4.00 | > 5.50 triggers alert |
| R² | Midfield (Clearances) | 0.52 | ≥ 0.57 | < 0.42 triggers retraining |
| MAE | Midfield (Clearances) | 1.42 | ≤ 1.30 | > 1.80 triggers alert |
| Calibration Error | All positions | — | ≤ 0.05 ECE | > 0.10 triggers review |

**Validation strategy:** Time-based split only (no random shuffle). Train ≤ 2022, validate 2023–2024, test 2025.

---

## 2. System / Engineering KPIs

| KPI | Target | Measurement |
|-----|--------|-------------|
| API latency (p95) | ≤ 200ms | Measured via `/health` endpoint logs |
| API uptime | ≥ 99% | Docker health check + monitoring |
| CI pipeline pass rate | ≥ 95% of PRs | GitHub Actions |
| Test coverage | ≥ 80% | `pytest --cov` |
| Docker image build time | ≤ 5 min | GitHub Actions workflow |

---

## 3. Data Quality KPIs

| KPI | Target | Tool |
|-----|--------|------|
| Feature completeness | ≥ 95% non-null for key features | Great Expectations |
| Schema validation pass rate | 100% on ingestion | Great Expectations suite |
| Feature drift (PSI) | PSI < 0.2 (stable), alert if > 0.25 | `src/monitoring/drift.py` |
| Feature drift (KS test) | p-value > 0.05 (no significant drift) | `src/monitoring/drift.py` |

---

## 4. Explainability & Fairness KPIs

| KPI | Target | Notes |
|-----|--------|-------|
| SHAP coverage | 100% of predictions have explanation | `src/visualization/explainability.py` |
| Position-group performance parity | MAE ratio across positions ≤ 1.3× | Fairness audit |
| Team-group performance parity | MAE ratio across teams ≤ 1.5× | Fairness audit |
| Demographic parity (age groups) | R² gap ≤ 0.10 across age buckets | See HTE results |

**Rationale from HTE findings:** Young rucks (<23) have dramatically different treatment effects (Height ATE = +8.16) vs. veterans. The model must perform reliably across all segments, not just on aggregate.

---

## 5. Project Delivery KPIs

| Milestone | Target Date | Owner | Status |
|-----------|-------------|-------|--------|
| Phase 1: Infrastructure & docs | Week 1 | All | In progress |
| Phase 2: Feature pipeline complete | Week 2 | B (lead) | In progress |
| Phase 3: Model training + MLflow | Week 2–3 | A (lead) | In progress |
| Phase 4: API serving + Docker | Week 3 | C (lead) | — |
| Phase 5: SHAP + fairness audit | Week 3–4 | D (lead) | — |
| Phase 6: Drift monitoring live | Week 4 | B (lead) | — |
| Phase 7: CI/CD pipeline complete | Week 4 | C (lead) | — |
| Phase 8: Final presentation | Week 4 | D (lead) | — |

---

## 6. Business Value KPIs

These connect the technical work to the coaching department's decision-making.

| Business Outcome | Proxy Metric | How We Measure |
|-----------------|--------------|----------------|
| Coaching can identify undervalued players | Model flags players where predicted > recent actual | Player-level prediction output |
| Reduce selection error | Prediction R² on holdout matches 2025 season | Test set evaluation |
| Explain *why* a player is predicted high/low | SHAP top-3 features per prediction | Explainability output |
| Identify position-specific advantages | Causal features (Height×Ruck, BMI interactions) carry positive coefficient | Feature importance |

---

## Definitions

- **PSI (Population Stability Index):** Measures distribution shift between training and live data. PSI < 0.1 = no change, 0.1–0.2 = slight, > 0.2 = significant drift.
- **KS Test:** Kolmogorov-Smirnov test for distributional difference between two samples.
- **ECE (Expected Calibration Error):** How well predicted probabilities match actual frequencies.
- **MAE (Mean Absolute Error):** Average absolute difference between prediction and ground truth.
