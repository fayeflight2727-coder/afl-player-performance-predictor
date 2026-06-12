# Feature Catalog

**Project:** AFL Player Performance Predictor
**Owner:** Tia Qiu (ML Analyst/PM)
**Last Updated:** 2026-06-12
**Source:** `src/features/build_features.py` (Faye Wu) + causal analysis results from Course 1

---

## Overview

This catalog documents every feature in the production pipeline. It is intended for:
- Stakeholders who want to understand what drives predictions
- Engineers extending the feature pipeline
- Auditors reviewing the fairness of model inputs

**Pipeline file:** `src/features/build_features.py`
**Output dataset:** `data/processed/afl_features_latest.csv`
**Scaler:** `RobustScaler` (handles outliers better than StandardScaler; applied per position)

---

## 1. Physical Attribute Features

Derived during player data processing in `AFLDataPreprocessor.add_players_derived_features()`.

| Feature | Type | Formula / Source | Causal Finding |
|---------|------|-----------------|----------------|
| `Height` | Continuous | Raw from `players.csv` | Strong positive for Rucks (ATE=+3.66 hitouts); minimal for Forwards |
| `Weight` | Continuous | Raw from `players.csv`; rows with `Weight=0` removed | Strong positive for Rucks (ATE=+4.22); negative for veteran Forwards |
| `BMI` | Derived | `Weight / (Height/100)²` | Positive for ALL positions — modern game rewards physicality |
| `Age` | Derived | `Year − BirthYear` (from `Dob`) | Moderates physical attribute effects (see HTE section) |

**Note:** `BMISquared` and `AgeSquared` appeared in the Course 1 notebook but are not yet in the production pipeline. To be added in a future feature engineering iteration.

**Key HTE finding** (from `reports/tables/hte_results.csv`):

| Segment | Height → HitOuts (Ruck) |
|---------|------------------------|
| Young (<23) | **+8.16** |
| Prime (23–28) | +0.95 |
| Veteran (>28) | −2.21 |

---

## 2. Position Feature

Standardized in `AFLDataPreprocessor.finalize_player_positions()`.

| Feature | Type | Logic |
|---------|------|-------|
| `PrimaryPosition` | Categorical | **Ruck-Priority rule:** any multi-position listing containing "Ruck" → `Ruck`; otherwise take first listed position. Players with no position are excluded. |

Values: `Forward`, `Midfield`, `Ruck`, `Defender`

---

## 3. Weather Features

Derived in `AFLDataPreprocessor.refine_weather_features()` from `games.csv`. Rows missing weather data are dropped before merging.

| Feature | Type | Formula | Notes |
|---------|------|---------|-------|
| `AvgTemp` | Continuous | `(MaxTemp + MinTemp) / 2` | °C |
| `TempRange` | Continuous | `MaxTemp − MinTemp` | Day temperature variation |
| `IsRainy` | Binary | `1 if Rainfall > 0 else 0` | Passed through unscaled |

---

## 4. In-Game Performance Features

These are per-game statistics from `stats.csv`. The following columns were **explicitly removed** from the pipeline as redundant, leaky, or uninformative:

| Removed Column | Reason |
|----------------|--------|
| `Kicks`, `Handballs` | Rolled into `Disposals` to avoid multicollinearity |
| `ContestedPossessions`, `UncontestedPossessions` | Redundant |
| `BrownlowVotes` | Not available before the game |
| `Bounces` | Sparse / low signal |
| `Subs` | Mostly missing |
| `Round`, `GameNumber` | Not informative for the model |
| `PlayerName` | Non-numeric identifier |

**Features retained:**

| Feature | Type | Description |
|---------|------|-------------|
| `Disposals` | Continuous | Total kicks + handballs (Kicks and Handballs dropped; this is their sum) |
| `Marks` | Continuous | Contested + uncontested marks |
| `Goals` | Continuous | Goals scored in the match |
| `Behinds` | Continuous | Behinds scored in the match |
| `HitOuts` | Continuous | Hitouts from ruck contests (target for Ruck model; feature for others) |
| `Tackles` | Continuous | Defensive tackles |
| `Rebounds` | Continuous | Defensive rebounds (target for Defender model; feature for others) |
| `Inside50s` | Continuous | Times player moved ball inside 50m arc |
| `Clearances` | Continuous | Clearances from stoppages (target for Midfield model; feature for others) |
| `Clangers` | Continuous | Errors / turnovers |
| `Frees` | Continuous | Free kicks won |
| `FreesAgainst` | Continuous | Free kicks conceded |
| `ContestedMarks` | Continuous | Marks taken under physical pressure |
| `MarksInside50` | Continuous | Marks inside the 50m arc — **top Forward predictor** (Course 1 coeff: +9.75) |
| `OnePercenters` | Continuous | Spoils, shepherds, smothers |
| `GoalAssists` | Continuous | Goals assisted |
| `%Played` | Continuous | Percentage of match played |

---

## 5. Target Variables (by Position Model)

| Position Model | Target Column | Description |
|---------------|---------------|-------------|
| Forward | `Total_Score` | `Goals × 6 + Behinds × 1` — computed in `merge_datasets()` |
| Midfield | `Clearances` | Clearances from stoppages |
| Ruck | `HitOuts` | Hitouts from ruck contests |
| Defender | `Rebounds` | Defensive rebounds |

---

## 6. Planned Features (Not Yet Implemented)

These features are referenced in the role assignment and Course 1 findings but are not yet in `build_features.py`. They are planned for a future commit.

| Feature | File | Status |
|---------|------|--------|
| `IsHome` | `build_features.py` | Not yet derived — game merge brings `AwayTeam` but `IsHome` is not computed from it |
| `TeamQuality` | `build_features.py` | Not yet implemented |
| `Post666`, `PostStand`, `RotEra_medium`, `RotEra_low` | `build_features.py` | Era indicators from Course 1 not yet added |
| `BMISquared`, `AgeSquared` | `build_features.py` | Nonlinear terms from Course 1 not yet added |
| `Height_x_Ruck`, `Weight_x_Midfield`, etc. | `src/features/causal_interactions.py` | File does not exist yet |
| `Disposals_lag3/5`, `Total_Score_lag3/5`, `Form_trend` | `build_features.py` | Lag/rolling features not yet implemented |

---

## 7. Data Quality Outputs

`build_features.py` automatically saves three quality reports to `reports/data_quality/`:

| File | Contents |
|------|----------|
| `data_quality_report.json` | Row count, column count, year range, unique players/games, missing values, position distribution |
| `column_summary.csv` | Per-column dtype, null count, null %, unique value count |
| `dataset_summary.txt` | Human-readable summary with position distribution |
