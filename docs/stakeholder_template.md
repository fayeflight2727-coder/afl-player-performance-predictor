# Stakeholder Communication Template

**Project:** AFL Player Performance Predictor
**Owner:** Tia Qiu (ML Analyst/PM)
**Last Updated:** 2026-06-12

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

Structure for the final presentation, following professor's EnterpriseDataScience Section 5.9 guidelines.

### Slide Structure

| Section | Slides | Owner |
|---------|--------|-------|
| 1. Team & Project Title | 1 | D (Tia) |
| 2. Business Problem | 2 | D (Tia) |
| 3. Data & Causal Insights (Course 1) | 3 | D (Tia) |
| 4. Solution Architecture | 2 | C (Tracy) |
| 5. Feature Engineering | 2 | B |
| 6. Model Training & AutoML | 3 | A (Yanxin) |
| 7. Explainability (SHAP) | 3 | D (Tia) — LEAD |
| 8. Fairness Audit | 2 | D (Tia) — LEAD |
| 9. API & Deployment | 2 | C (Tracy) |
| 10. Monitoring & Drift | 2 | B |
| 11. CI/CD Pipeline | 1 | C (Tracy) |
| 12. Business Value & Impact | 2 | D (Tia) |
| 13. Live Demo | — | D (coordinates) |
| 14. Lessons Learned & Next Steps | 1 | All |

**Total:** ~26 slides

### Demo Script
1. Show live API: `POST /predict` with a sample player payload
2. Show SHAP explanation: `GET /explain/{player_id}` — "why did the model predict this?"
3. Show fairness dashboard: performance parity across positions
4. Show drift monitoring: PSI scores for current season data
