"""
Fairness audit for AFL Player Performance Predictor.
Evaluates predictive parity across position, age segment, era, and team groups.

Outputs:
    reports/fairness_report.md
    reports/fairness_metrics.csv
    reports/figures/fairness/*.png

Usage:
    python src/visualization/fairness_audit.py
"""

import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server use
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH = "models/xgb_goal_model.pkl"
DATA_PATH  = "data/processed/afl_features_latest.csv"
TARGET     = "Goals"

FEATURES = [
    "Year", "Disposals", "Marks", "Behinds", "HitOuts", "Tackles", "Rebounds",
    "Inside50s", "Clearances", "Clangers", "Frees", "FreesAgainst",
    "ContestedMarks", "MarksInside50", "OnePercenters", "GoalAssists",
    "%Played", "Height", "Weight", "BMI", "AvgTemp", "TempRange", "IsRainy", "Age"
]

MAE_RATIO_THRESHOLD  = 1.3
R2_GAP_THRESHOLD     = 0.10
PVALUE_THRESHOLD     = 0.05

Path("reports").mkdir(exist_ok=True)
FIGURES_DIR = Path("reports/figures/fairness")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Load model and data ───────────────────────────────────────────────────────

print("Loading model and data...")
model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH).dropna(subset=[TARGET])

# Model v2.1 was trained only on 2020-2025 data (see docs/model_card.md).
# Match that same filter here so the audit's test set reflects what the
# model actually saw, instead of re-splitting the full 2012-2025 dataset
# and accidentally testing on pre-2020 rows the model was never trained on.
df = df[df["Year"] >= 2020].reset_index(drop=True)

X = df[FEATURES].fillna(0)
y = df[TARGET]

split     = int(len(X) * 0.8)
X_test    = X.iloc[split:].copy()
y_test    = y.iloc[split:].copy()
meta      = df.iloc[split:].copy()
meta["predicted"] = model.predict(X_test)
meta["actual"]    = y_test.values
meta["error"]     = np.abs(meta["predicted"] - meta["actual"])

overall_mae  = mean_absolute_error(meta["actual"], meta["predicted"])
overall_rmse = np.sqrt(mean_squared_error(meta["actual"], meta["predicted"]))
overall_r2   = r2_score(meta["actual"], meta["predicted"])

print(f"Overall  MAE={overall_mae:.4f}  RMSE={overall_rmse:.4f}  R²={overall_r2:.4f}  n={len(meta):,}\n")

# ── Metric helper ─────────────────────────────────────────────────────────────

def group_metrics(df_group, group_col, group_val):
    if len(df_group) < 20:
        return None
    mae  = mean_absolute_error(df_group["actual"], df_group["predicted"])
    rmse = np.sqrt(mean_squared_error(df_group["actual"], df_group["predicted"]))
    r2   = r2_score(df_group["actual"], df_group["predicted"])
    _, pvalue = stats.mannwhitneyu(
        df_group["error"],
        meta[meta[group_col] != group_val]["error"],
        alternative="two-sided"
    )
    flagged = (mae / overall_mae > MAE_RATIO_THRESHOLD or
               rmse / overall_rmse > MAE_RATIO_THRESHOLD or
               abs(overall_r2 - r2) > R2_GAP_THRESHOLD)
    return {
        "group_col":  group_col,
        "group":      str(group_val),
        "n":          len(df_group),
        "mae":        round(mae, 4),
        "rmse":       round(rmse, 4),
        "r2":         round(r2, 4),
        "mae_ratio":  round(mae / overall_mae, 3),
        "r2_gap":     round(overall_r2 - r2, 3),
        "p_value":    round(pvalue, 4),
        "significant": pvalue < PVALUE_THRESHOLD,
        "flagged":    flagged,
    }

results = []

# ── 1. Position ───────────────────────────────────────────────────────────────

print("Position groups:")
for pos in ["Forward", "Midfield", "Ruck", "Defender"]:
    g = meta[meta["PrimaryPosition"] == pos]
    r = group_metrics(g, "PrimaryPosition", pos)
    if r:
        results.append(r)
        flag = " *** FLAGGED" if r["flagged"] else ""
        print(f"  {pos:10s}  n={r['n']:5,}  MAE={r['mae']:.4f}  R²={r['r2']:.4f}  ratio={r['mae_ratio']:.2f}{flag}")

# ── 2. Age segment ────────────────────────────────────────────────────────────

print("\nAge segments:")
meta["AgeSegment"] = pd.cut(meta["Age"], bins=[0,22,28,99],
                             labels=["Young (<23)", "Prime (23-28)", "Veteran (>28)"])
for seg in ["Young (<23)", "Prime (23-28)", "Veteran (>28)"]:
    g = meta[meta["AgeSegment"] == seg]
    r = group_metrics(g, "AgeSegment", seg)
    if r:
        results.append(r)
        flag = " *** FLAGGED" if r["flagged"] else ""
        print(f"  {seg:18s}  n={r['n']:5,}  MAE={r['mae']:.4f}  R²={r['r2']:.4f}  ratio={r['mae_ratio']:.2f}{flag}")

# ── 3. Rule-change era ────────────────────────────────────────────────────────

print("\nRule-change era:")
meta["Era"] = meta["Year"].apply(lambda y: "Post-6-6-6 (2019+)" if y >= 2019 else "Pre-6-6-6 (<2019)")
era_counts = meta["Era"].value_counts()
era_applicable = era_counts.get("Pre-6-6-6 (<2019)", 0) >= 20 and era_counts.get("Post-6-6-6 (2019+)", 0) >= 20

if era_applicable:
    for era in ["Pre-6-6-6 (<2019)", "Post-6-6-6 (2019+)"]:
        g = meta[meta["Era"] == era]
        r = group_metrics(g, "Era", era)
        if r:
            results.append(r)
            flag = " *** FLAGGED" if r["flagged"] else ""
            print(f"  {era:22s}  n={r['n']:5,}  MAE={r['mae']:.4f}  R²={r['r2']:.4f}  ratio={r['mae_ratio']:.2f}{flag}")
else:
    print("  Not applicable — model trained/tested only on 2020+ data, so every")
    print("  row is already Post-6-6-6. No pre-2019 rows remain to compare against.")

# ── 4. Team ───────────────────────────────────────────────────────────────────

print("\nTeam groups:")
team_results = []
for team in sorted(meta["Team"].dropna().unique()):
    g = meta[meta["Team"] == team]
    r = group_metrics(g, "Team", team)
    if r:
        team_results.append(r)
results.extend(team_results)

if team_results:
    tdf = pd.DataFrame(team_results)
    worst = tdf.loc[tdf["mae"].idxmax()]
    best  = tdf.loc[tdf["mae"].idxmin()]
    n_flag = len(tdf[tdf["flagged"]])
    flagged_names = ", ".join(tdf[tdf["flagged"]]["group"].tolist()) or "None"
    print(f"  Best:    {best['group']}  MAE={best['mae']:.4f}")
    print(f"  Worst:   {worst['group']}  MAE={worst['mae']:.4f}  ratio={worst['mae_ratio']:.2f}")
    print(f"  Flagged: {n_flag} teams — {flagged_names}")

# ── Save CSV ──────────────────────────────────────────────────────────────────

metrics_df = pd.DataFrame(results)
metrics_df.to_csv("reports/fairness_metrics.csv", index=False)
print(f"\nSaved reports/fairness_metrics.csv ({len(metrics_df)} rows)")

# ── Comparison plots ────────────────────────────────────────────────────────


def plot_group_comparison(df, title, filename, horizontal=False):
    """Bar chart comparing MAE and R² across groups, with overall reference lines.
    Flagged groups are highlighted in red, passing groups in green.
    """
    if df.empty:
        return
    df = df.sort_values("mae", ascending=horizontal)
    colors = ["#e74c3c" if f else "#2ecc71" for f in df["flagged"]]

    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, 0.35 * len(df)) if horizontal else 5))

    if horizontal:
        axes[0].barh(df["group"], df["mae"], color=colors)
        axes[0].axvline(overall_mae, color="black", linestyle="--", linewidth=1, label=f"Overall MAE={overall_mae:.3f}")
        axes[1].barh(df["group"], df["r2"], color=colors)
        axes[1].axvline(overall_r2, color="black", linestyle="--", linewidth=1, label=f"Overall R²={overall_r2:.3f}")
    else:
        axes[0].bar(df["group"], df["mae"], color=colors)
        axes[0].axhline(overall_mae, color="black", linestyle="--", linewidth=1, label=f"Overall MAE={overall_mae:.3f}")
        axes[1].bar(df["group"], df["r2"], color=colors)
        axes[1].axhline(overall_r2, color="black", linestyle="--", linewidth=1, label=f"Overall R²={overall_r2:.3f}")
        for ax in axes:
            ax.tick_params(axis="x", rotation=30)

    axes[0].set_title("MAE by group (lower is better)")
    axes[0].legend(fontsize=8)
    axes[1].set_title("R² by group (higher is better)")
    axes[1].legend(fontsize=8)
    fig.suptitle(title)
    plt.tight_layout()
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── Write report ──────────────────────────────────────────────────────────────

def md_table(df, cols):
    rows = ["| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if c == "flagged":
                cells.append("**YES**" if v else "no")
            elif isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)

pos_df  = metrics_df[metrics_df["group_col"] == "PrimaryPosition"]
age_df  = metrics_df[metrics_df["group_col"] == "AgeSegment"]
era_df  = metrics_df[metrics_df["group_col"] == "Era"]
team_df = metrics_df[metrics_df["group_col"] == "Team"]

total_flagged    = len(metrics_df[metrics_df["flagged"]])
pos_flagged      = pos_df[pos_df["flagged"]]
age_flagged      = age_df[age_df["flagged"]]
era_flagged      = era_df[era_df["flagged"]]
n_teams_flagged  = len(team_df[team_df["flagged"]]) if len(team_df) > 0 else 0
flagged_teams    = ", ".join(team_df[team_df["flagged"]]["group"].tolist()) if len(team_df) > 0 else "None"

print("\nGenerating comparison plots...")
plot_group_comparison(pos_df,  "Fairness Audit — Position Group Comparison",      "position_comparison.png")
plot_group_comparison(age_df,  "Fairness Audit — Age Segment Comparison",         "age_segment_comparison.png")
plot_group_comparison(era_df,  "Fairness Audit — Rule-Change Era Comparison",     "era_comparison.png")
plot_group_comparison(team_df, "Fairness Audit — Team Comparison (18 AFL Clubs)", "team_comparison.png", horizontal=True)

cols = ["group", "n", "mae", "r2", "mae_ratio", "r2_gap", "p_value", "flagged"]

report = f"""# Fairness Audit Report

**Project:** AFL Player Performance Predictor
**Author:** Tia Qiu (ML Analyst/PM)
**Date:** 2026-06-15
**Model:** XGBRegressor — `models/xgb_goal_model.pkl` (v2, trained 2020–2025)
**Framework:** `docs/fairness_audit_framework.md`

---

## Overall Model Performance (Baseline)

| Metric | Value |
|--------|-------|
| MAE | {overall_mae:.4f} goals |
| RMSE | {overall_rmse:.4f} goals |
| R² | {overall_r2:.4f} |
| Test set | {len(meta):,} player-game observations ({int(meta['Year'].min())}–{int(meta['Year'].max())}) |

**Flagging thresholds:** MAE ratio > 1.3×  |  R² gap > 0.10  |  p < 0.05 (Mann-Whitney U)

---

## 1. Position Group Audit

{md_table(pos_df, cols)}

![Position group comparison](figures/fairness/position_comparison.png)

**Findings:**
"""
if len(pos_flagged) == 0:
    report += "- No position groups exceed thresholds. Predictive parity holds across Forward, Midfield, Ruck, and Defender.\n"
else:
    for _, row in pos_flagged.iterrows():
        sig = f"statistically significant (p={row['p_value']:.4f})" if row["significant"] else "not statistically significant"
        report += f"- **{row['group']}** flagged: MAE ratio={row['mae_ratio']:.2f}×, R² gap={row['r2_gap']:.3f} — {sig}.\n"

report += f"""
---

## 2. Age Segment Audit

{md_table(age_df, cols)}

![Age segment comparison](figures/fairness/age_segment_comparison.png)

**Findings:**
"""
if len(age_flagged) == 0:
    report += "- No age segments exceed thresholds. Age-based predictive parity holds.\n"
else:
    for _, row in age_flagged.iterrows():
        sig = f"statistically significant (p={row['p_value']:.4f})" if row["significant"] else "not statistically significant"
        report += f"- **{row['group']}** flagged: MAE ratio={row['mae_ratio']:.2f}×, R² gap={row['r2_gap']:.3f} — {sig}.\n"

report += """
**Context from Course 1 HTE:** Age strongly moderates physical attribute effects (young rucks show Height ATE=+8.16 vs. −2.21 for veterans). Some age-related prediction variance is expected.

---

## 3. Rule-Change Era Audit

"""
if era_df.empty:
    report += (
        "**Not applicable.** The model is trained and tested only on 2020+ data "
        "(see `docs/model_card.md`), so every row in the test set is already "
        "Post-6-6-6 — there are no pre-2019 rows left to compare against. "
        "This audit dimension cannot be evaluated for this model version.\n"
    )
else:
    report += md_table(era_df, cols)
    report += "\n\n![Rule-change era comparison](figures/fairness/era_comparison.png)\n"
    report += "\n**Findings:**\n"
    if len(era_flagged) == 0:
        report += "- No era groups exceed thresholds. The model generalises across pre- and post-6-6-6 rule eras.\n"
    else:
        for _, row in era_flagged.iterrows():
            sig = f"statistically significant (p={row['p_value']:.4f})" if row["significant"] else "not statistically significant"
            report += f"- **{row['group']}** flagged: MAE ratio={row['mae_ratio']:.2f}× — {sig}.\n"

if len(team_df) > 0:
    worst_team = team_df.loc[team_df["mae"].idxmax()]
    best_team  = team_df.loc[team_df["mae"].idxmin()]
    report += f"""
---

## 4. Team Group Audit

| | |
|--|--|
| Teams audited | {len(team_df)} |
| Best-predicted | {best_team['group']} (MAE={best_team['mae']:.4f}) |
| Worst-predicted | {worst_team['group']} (MAE={worst_team['mae']:.4f}, ratio={worst_team['mae_ratio']:.2f}×) |
| Teams flagged | {n_teams_flagged} |
| Flagged teams | {flagged_teams} |

Full team results in `reports/fairness_metrics.csv`.

![Team comparison](figures/fairness/team_comparison.png)

**Findings:**
"""
    if n_teams_flagged == 0:
        report += "- No teams exceed the 1.3× MAE threshold. Team-level predictive parity holds.\n"
    else:
        report += f"- {n_teams_flagged} team(s) flagged. Recommend checking whether flagged teams have unusual player profiles under-represented in training data.\n"

report += f"""
---

## Summary

| Audit Group | Groups Tested | Flagged | Result |
|-------------|--------------|---------|--------|
| Position | {len(pos_df)} | {len(pos_flagged)} | {"NEEDS REVIEW" if len(pos_flagged) > 0 else "PASS"} |
| Age Segment | {len(age_df)} | {len(age_flagged)} | {"NEEDS REVIEW" if len(age_flagged) > 0 else "PASS"} |
| Rule-Change Era | {len(era_df)} | {len(era_flagged)} | {"N/A — no pre-2019 data in test set" if era_df.empty else ("NEEDS REVIEW" if len(era_flagged) > 0 else "PASS")} |
| Team | {len(team_df)} | {n_teams_flagged} | {"NEEDS REVIEW" if n_teams_flagged > 0 else "PASS"} |

**Total flagged groups: {total_flagged}**

---

## Recommended Actions

"""
if total_flagged == 0:
    report += """All groups pass fairness thresholds. No mitigations required at this time.

Recommended next steps:
- Re-run this audit after any model retraining
- Complete SHAP-based individual fairness check once `/predict/explain` background is stable
- Add era indicator features (`Post666`, `RotEra`) in next model iteration to explicitly account for rule-change effects
"""
else:
    report += """Based on flagged groups:

1. **Re-weight training samples** for flagged position/age groups
2. **Add age×position interaction features** if young-player error persists
3. **Add era indicator features** (`Post666`, `RotEra`) to explicitly model rule-change effects
4. **Re-run audit** after any mitigation to verify improvement
"""

report += """
---

## Methodology

- **Test set:** Chronological 20% holdout (last 20% of rows by time)
- **Statistical test:** Mann-Whitney U (two-sided), each group vs. rest
- **Significance threshold:** p < 0.05
- **Minimum group size:** 20 observations
- **SHAP individual fairness:** Pending — use `POST /predict/explain` for per-player checks

*Generated by `src/visualization/fairness_audit.py`*
"""

with open("reports/fairness_report.md", "w") as f:
    f.write(report)

print("Saved reports/fairness_report.md")
print(f"\nAudit complete. Total flagged groups: {total_flagged}")
