# Stakeholder Communication Template

**Project:** AFL Player Performance Predictor

---

## Weekly Progress Update Template

Use this template for weekly progress presentations to the professor and class.

```
## AFL Predictor — Week [N] Progress Update
Date: [DATE]
Team: Tracy Chan (Backend), Yanxin Li (ML Engineer), [Member B] (Data Engineer), Tia Qiu (Analyst/PM)

### Completed This Week
- [Member A]: [what was done]
- [Member B]: [what was done]
- [Member C]: [what was done]
- [Member D]: [what was done]

### In Progress
- [task] — [owner] — [ETA]

### Blockers
- [blocker] — [owner] — [how to resolve]

### Next Week Plan
- [Member A]: [planned work]
- [Member B]: [planned work]
- [Member C]: [planned work]
- [Member D]: [planned work]

### Key Metrics This Week
| Metric | Last Week | This Week | Target |
|--------|-----------|-----------|--------|
| Model R² (Forward) | — | — | ≥ 0.55 |
| Model R² (Midfield) | — | — | ≥ 0.57 |
| Test coverage | — | — | ≥ 80% |
| Open GitHub issues | — | — | 0 critical |

### GitHub Activity
- Commits this week: [N]
- PRs merged: [N]
- Issues closed: [N]
- Issues opened: [N]
```

---

## Phase Completion Report Template

Use this when a phase is complete.

```
## Phase [N] Complete — [Phase Name]

### What Was Built
[2–3 sentences describing the deliverable]

### Key Technical Decisions
- [decision and rationale]
- [decision and rationale]

### Results / Metrics
[Table or list of key numbers]

### How This Connects to Business Value
[1–2 sentences connecting technical output to coaching use case]

### What's Next (Phase [N+1])
[1 sentence on the next phase owner and goal]
```

---

## GitHub Issue Template

Use this structure when creating issues for tasks.

```
**Title:** [phase][role] Short description
# e.g.: [phase5][analyst] Implement SHAP force plots for Forward model

**Description:**
What needs to be done and why.

**Acceptance Criteria:**
- [ ] Criterion 1
- [ ] Criterion 2

**Owner:** @[github_handle]
**Phase:** [N]
**Labels:** phase-N, role-[analyst|ml-engineer|data-engineer|backend]
**Milestone:** Phase [N] — [Week]
```

---

## Final Presentation Outline (Section 5.9)

**Total:** ~25 slides

### Per-Slide Content Guide

**Slide 1 — Team & Project Title**
- Team name, course (McGill MMA8 INSY 684)
- Table: Member name | Role | GitHub ID
- GitHub repo: github.com/fayeflight2727-coder/afl-player-performance-predictor

**Slide 2 — Business Problem**
- AFL teams spend millions on player recruitment based on intuition
- Our system: given a player's stats + game context → predict goals + explain why
- 3 core questions: What will this player score? Why? Is the model fair?

**Slides 3–4 — Data & Causal Insights**
- Dataset: Kaggle AFL Stats, 127K player-game rows, 2012–2025
- Course 1 ATE findings: Height→Ruck (+3.66 hitouts), Weight→Ruck (+4.22), BMI→Forward (+0.64)
- HTE: young rucks (<23) show 8× stronger height effect than veterans
- 6-6-6 rule change (2019): +2,226% effect on ruck height value

**Slides 5–6 — Solution Architecture**
- System diagram from README (Feature Pipeline → Model → API → Monitoring)
- Tech stack: FastAPI, Docker, MLflow, GitHub Actions, XGBoost, SHAP

**Slides 7–8 — Feature Engineering**
- 24 production features: physical, positional, weather, in-game stats
- Key decisions: Disposals = Kicks + Handballs (avoid multicollinearity), %Played retained
- Lag features, causal interaction terms (height×ruck, weight×midfield)

**Slides 9–11 — Model Training & AutoML**
- XGBRegressor, n_estimators=300, max_depth=5, lr=0.05
- Hyperparameter tuning via Optuna/Hyperopt
- MLflow experiment tracking: AFL_Goal_Prediction, model registry
- v2.1 retraining: 2020–2025 data → R² improved 0.37 → 0.49 (+32%)

**Slides 11–12 — Explainability (SHAP)**
- Slide 11: What is SHAP — baseline vs. prediction, push/pull of features
- Slide 12: Forward example — MarksInside50 adds +1.75 goals (dominant feature, confirms Course 1 coeff +9.75)

**Slides 13–15 — API & Deployment**
- Slide 13: FastAPI endpoints — POST /predict, POST /predict/explain, GET /monitoring/drift
- Slide 14: Live API — POST /predict/explain JSON response with top 10 SHAP features per prediction
- Slide 15: Position profiles — Forward 2.14, Midfield 0.53, Ruck 0.22, Defender 0.05 (no explicit routing)

**Slide 16 — Solution Architecture: Deployment View**
- Docker container, MLflow model registry (local fallback to .pkl)
- GitHub Actions runs lint + 41 smoke tests on every PR

**Slide 17 — CI/CD Pipeline**
- GitHub Actions: lint + unit tests on every PR, deploy on merge to main
- 41 smoke tests covering all endpoints and explainability module

**Slides 18–19 — Fairness & Ethical AI**
- Slide 18: Audit methodology — 4 groups (position, age, era, team), thresholds MAE ratio >1.3×, R² gap >0.10
- Slide 19: Results — 5 flagged groups: Forward (MAE 1.40×), Midfield (R² gap 0.23), Pre-2019 era, Carlton, Richmond. Age segments all PASS. Recommended mitigations.

**Slides 20–21 — Monitoring & Drift**
- PSI/KS tests on 9 features comparing train (≤2022) vs current (2023–2025)
- Weight drift detected (PSI=0.148, moderate) — players getting lighter in modern game
- Retraining trigger: PSI > 0.25 OR R² drops below threshold

**Slides 22–23 — Business Value & Impact**
- Slide 22: KPIs met — API latency <200ms, 41 tests passing, drift monitoring live
- Slide 23: Coaching use cases — pre-game selection, opponent analysis, recruitment. Model predicts 2.14 goals for active Forward vs 0.05 for Defender (position awareness without explicit routing)

**Slide 24 — Lessons Learned & Next Steps**
- Lessons: column order bugs matter in production, background dataset required for SHAP, chronological splits critical for sports data
- Next steps: position-specific models, era indicator features, causal interaction terms in production pipeline
