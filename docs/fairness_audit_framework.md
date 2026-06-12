# Fairness Audit Framework

**Project:** AFL Player Performance Predictor
**Owner:** Tia Qiu (ML Analyst/PM)
**Phase:** 5 (Week 3–4) — Lead deliverable
**Last Updated:** 2026-06-12

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
The model trains four separate sub-models (Forward, Midfield, Ruck, Defender). We audit whether the *quality* of predictions is equitable across positions.

| Group | N (approx.) | Primary metric |
|-------|-------------|----------------|
| Forward | ~6,375 | TotalScore MAE |
| Midfield | ~8,056 | Clearances MAE |
| Ruck | ~1,664 | HitOuts MAE |
| Defender | ~9,331 | Rebounds MAE |

### 2. Team Groups
Predictions must not systematically favor or penalize players from certain teams. We check whether high-performing clubs' players are better-predicted than players from struggling clubs.

### 3. Age / Career Stage Groups
Based on HTE findings, treatment effects vary strongly with career stage:

| Segment | Age / Games Criterion |
|---------|-----------------------|
| Young | Age < 23 |
| Prime | Age 23–28 |
| Veteran | Age > 28 |
| Rookie | < 50 career games |
| Established | 50–150 career games |
| Late-career | > 150 career games |

### 4. Rule-Change Era Groups
Pre- vs. post-rule-change predictions should be validated separately, since the 6-6-6 rule (2019) structurally changed how physical attributes relate to performance.

| Era | Seasons |
|-----|---------|
| Pre-6-6-6 | Before 2019 |
| Post-6-6-6 | 2019 onwards |

---

## Fairness Metrics

### Predictive Parity
The model should predict equally well across all groups.

| Metric | Formula | Threshold |
|--------|---------|-----------|
| MAE Ratio | `MAE_group_i / MAE_overall` | Flag if > 1.3× |
| R² Gap | `R²_best_group − R²_worst_group` | Flag if > 0.10 |
| RMSE Ratio | `RMSE_group_i / RMSE_overall` | Flag if > 1.3× |

### Demographic Parity (for ranking use cases)
If the model is used to rank players, high-ranked predictions should not be disproportionately from one group.

| Metric | Definition | Threshold |
|--------|-----------|-----------|
| Top-quintile representation | % of group in top 20% predictions vs. group's population % | Flag if disparity > 15% |

### Individual Fairness via SHAP
For any two players with similar observable statistics, their SHAP explanations should be similar. Large SHAP divergence for similar players signals a fairness concern.

---

## Audit Process

### Step 1 — Segment Performance Evaluation
Run model predictions on test set (2025 season). Compute MAE, R², RMSE for each audit group.

```python
# pseudocode
for group in ['Forward', 'Midfield', 'Ruck', 'Defender']:
    group_df = test_df[test_df['Position'] == group]
    compute_metrics(group_df['actual'], group_df['predicted'])
```

### Step 2 — SHAP-Based Fairness Check
Compare SHAP feature importance distributions across groups. Flag features that systematically explain predictions differently for different groups.

```python
# pseudocode — in src/visualization/explainability.py
shap_values_by_group = {}
for group in audit_groups:
    shap_values_by_group[group] = compute_shap(model, group_df)
plot_shap_comparison(shap_values_by_group)
```

### Step 3 — Statistical Testing
For each flagged group, use a statistical test to confirm the disparity is not random noise.

- **Mann-Whitney U test** for continuous metric differences across groups
- **Chi-square test** for categorical representation
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
| Rule-era disparity | Ensure era indicator features are included; train separate era models if gap persists |
| SHAP shows irrelevant feature driving group predictions | Remove or regularize that feature; re-evaluate |

---

## Output Deliverables (Phase 5)

| Deliverable | Format | Location |
|-------------|--------|----------|
| Fairness audit report | Markdown | `reports/fairness_report.md` |
| Per-group performance table | CSV | `reports/fairness_metrics.csv` |
| SHAP comparison plots | PNG | `reports/figures/fairness/` |
| Corrected model (if needed) | MLflow run | MLflow registry |

---

## Tools

| Tool | Use |
|------|-----|
| SHAP | Feature-level explanation and fairness comparison |
| `src/visualization/explainability.py` | Production SHAP integration |
| pandas / scipy | Statistical testing |
| matplotlib / plotly | Visualization |

---

## References

- SHAP: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions" (2017)
- Fairness definitions: Barocas, Hardt & Narayanan, *Fairness and Machine Learning* (2019)
- HTE results: `reports/tables/hte_results.csv` (Course 1)
- ATE results: `reports/tables/ate_results.csv` (Course 1)
