# Fairness Audit Framework

**Project:** AFL Player Performance Predictor

---

## Purpose

This document defines the methodology for auditing the AFL Player Performance Predictor for bias and fairness issues. It establishes:
1. Which groups to audit
2. Which fairness metrics to compute
3. Thresholds that trigger mitigation
4. How findings will be reported

---

## Groups Under Audit

### 1. Position Groups
The production model is a **single** XGBRegressor predicting `Goals` for every player, regardless of position — there are no separate per-position sub-models or targets. (The original Course 1 design used position-specific targets; that approach was not carried into production. See `docs/model_card.md`.) We audit whether the *quality* of this single model's predictions is equitable across positions.

| Group | N (full-dataset test set) | Primary metric |
|-------|----------------------------|----------------|
| Forward | ~8,136 | Goals MAE |
| Midfield | ~8,503 | Goals MAE |
| Ruck | ~1,634 | Goals MAE |
| Defender | ~7,151 | Goals MAE |

### 2. Team Groups
Predictions must not systematically favor or penalize players from certain teams. We check whether high-performing clubs' players are better-predicted than players from struggling clubs.

### 3. Age / Career Stage Groups
Based on HTE findings, treatment effects vary strongly with career stage:

| Segment | Age Criterion |
|---------|---------------|
| Young | Age < 23 |
| Prime | Age 23–28 |
| Veteran | Age > 28 |

### 4. Rule-Change Era Groups
Pre- vs. post-rule-change predictions should be validated separately, since the 6-6-6 rule (2019) structurally changed how physical attributes relate to performance.

| Era | Seasons |
|-----|---------|
| Pre-6-6-6 | Before 2019 |
| Post-6-6-6 | 2019 onwards |

**Current status: active and flagged.** The production model is trained on the full 2012-2025 dataset (a reported 2020+-only retrain could not be verified — see `docs/model_card.md`), so this dimension is fully testable. The Pre-6-6-6 group is currently flagged (R² gap = 0.157 > 0.10 threshold; MAE ratio = 1.21× is below the 1.3× threshold and did not trigger the flag). Note: n=568 for Pre-6-6-6 in the test set — all from ~2018 due to chronological split — so this result should be treated as a monitoring signal rather than a confirmed disparity.

---

## Fairness Metrics

### Predictive Parity
The model should predict equally well across all groups.

| Metric | Formula | Threshold |
|--------|---------|-----------|
| MAE Ratio | `MAE_group_i / MAE_overall` | Flag if > 1.3× |
| R² Gap | `R²_best_group − R²_worst_group` | Flag if > 0.10 |
| RMSE Ratio | `RMSE_group_i / RMSE_overall` | Flag if > 1.3× |

### Individual Fairness via SHAP
For any two players with similar observable statistics, their SHAP explanations should be similar. Large SHAP divergence for similar players signals a fairness concern.

---

## Audit Process

### Step 1 — Segment Performance Evaluation
Run model predictions on the chronological 20% test holdout of the full dataset (currently spans 2018–2025, n=25,424). Compute MAE, R², RMSE for each audit group.

```python
# implemented in src/visualization/fairness_audit.py
for group in ['Forward', 'Midfield', 'Ruck', 'Defender']:
    group_df = meta[meta['PrimaryPosition'] == group]
    compute_metrics(group_df['actual'], group_df['predicted'])
```

### Step 2 — SHAP-Based Fairness Check
Compare mean |SHAP| feature importance across groups. Flag features that systematically explain predictions differently for different groups — this catches individual fairness issues that predictive parity alone can miss.

```python
# implemented via plot_fairness_comparison() in src/visualization/explainability.py,
# called from src/visualization/fairness_audit.py for Position and Age Segment groups
plot_fairness_comparison(model, X_with_groups, X_background, group_col="PrimaryPosition", ...)
```

### Step 3 — Statistical Testing
For each flagged group, use a statistical test to confirm the disparity is not random noise.

- **Mann-Whitney U test** for continuous metric differences across groups
- Significance threshold: p < 0.05

### Step 4 — Document Findings
All findings documented in the Fairness Report (output of Phase 5).

---

## Mitigation Strategies

If disparities are found, the following mitigations are evaluated in order:

| Issue | Mitigation |
|-------|-----------|
| High MAE for one position group | Re-weight training samples for that position |
| Systematic over/under-prediction for an age group | Add age×position interaction feature |
| Rule-era disparity | Currently flagged for Pre-6-6-6. Add era indicator features (`Post666`, `RotEra`); train separate era models if gap persists after that |
| SHAP shows irrelevant feature driving group predictions | Remove or regularize that feature; re-evaluate |

---

## Output Deliverables (Phase 5)

| Deliverable | Format | Location |
|-------------|--------|----------|
| Fairness audit report | Markdown | `reports/fairness_report.md` |
| Per-group performance table | CSV | `reports/fairness_metrics.csv` |
| Predictive parity comparison plots (MAE/R² by group) | PNG | `reports/figures/fairness/position_comparison.png`, `age_segment_comparison.png`, `era_comparison.png`, `team_comparison.png` |
| SHAP individual fairness comparison plots (feature reliance by group) | PNG | `reports/figures/fairness/fairness_overall_primaryposition.png`, `fairness_overall_agesegment.png` |
| Corrected model (if needed) | MLflow run | MLflow registry |

---

## Tools

| Tool | Use |
|------|-----|
| SHAP | Feature-level explanation and fairness comparison |
| `src/visualization/explainability.py` | Production SHAP integration |
| pandas / scipy | Statistical testing |
| matplotlib | Visualization |

---

## References

- SHAP: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions" (2017)
- Fairness definitions: Barocas, Hardt & Narayanan, *Fairness and Machine Learning* (2019)
- HTE/ATE results: Course 1 causal analysis findings (Height/Weight/BMI effects, 6-6-6 rule change), which were produced in the separate Course 1 project, not included in this repository. Summarized in `docs/model_card.md`.
