# Feature Catalog

**Project:** AFL Player Performance Predictor
**Owner:** Tia Qiu (ML Analyst/PM)
**Last Updated:** 2026-06-12
**Source:** Derived from `Cleaned_Data.ipynb` / `Data_Preprocessing.ipynb` and causal analysis results

---

## Overview

This catalog documents every feature used in the production model. It is intended for:
- Stakeholders who want to understand what drives predictions
- Engineers implementing the feature pipeline (`src/features/build_features.py`)
- Auditors reviewing the fairness of model inputs

Features are organized by category. Position-specific features are clearly marked.

---

## 1. Physical Attribute Features

These are the primary causal drivers identified in Course 1 analysis.

| Feature | Type | Description | Source Column | Causal Finding |
|---------|------|-------------|---------------|----------------|
| `Height` | Continuous | Player height in cm | `players.csv: height` | Strong positive for Rucks (ATE=+3.66 hitouts), minimal for Forwards |
| `Weight` | Continuous | Player weight in kg | `players.csv: weight` | Strong positive for Rucks (ATE=+4.22 hitouts); negative for veteran Forwards |
| `BMI` | Derived | Weight / (Height/100)² | Computed | Positive for ALL positions — modern game rewards physicality |
| `BMISquared` | Derived | BMI² | Computed | Captures nonlinear BMI effects on scoring |
| `Age` | Continuous | Player age at time of match | Computed from DOB + match date | Moderates physical attribute effects (see HTE) |
| `AgeSquared` | Derived | Age² | Computed | Captures career-arc nonlinearity (peak ~23–28) |

**Key HTE finding:** The benefit of Height and Weight changes with career stage:
- Young rucks (<23): Height ATE = **+8.16** hitouts
- Prime rucks (23–28): Height ATE = +0.95
- Veteran rucks (>28): Height ATE = **−2.21** (declining)

---

## 2. Causal Interaction Features

These interaction terms encode the causal relationships identified in Course 1 and are located in `src/features/causal_interactions.py`.

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `Height_x_Ruck` | `Height × IsRuck` | 6-6-6 rule increased ruck height advantage by 2,226% |
| `Weight_x_Midfield` | `Weight × IsMidfield` | Rotation caps reduced midfield weight advantage by 97% — must model pre/post era |
| `BMI_x_Position` | `BMI × PositionCode` | BMI effect varies by position despite being universally positive |
| `Post666_x_Height` | `Post666 × Height` | Captures the structural break in height value after 2019 6-6-6 rule |
| `RotEra_x_Weight` | `RotEra × Weight` | Captures how rotation caps changed weight's importance for mids |

---

## 3. Rule Change / Era Indicators

| Feature | Type | Description | Value |
|---------|------|-------------|-------|
| `Post666` | Binary | After 2019 6-6-6 rule introduction | 0/1 |
| `PostStand` | Binary | After stand rule change | 0/1 |
| `RotEra_medium` | Binary | Medium rotation cap era | 0/1 |
| `RotEra_low` | Binary | Low rotation cap era (most restrictive) | 0/1 |

---

## 4. Contextual / Match Features

| Feature | Type | Description | Source |
|---------|------|-------------|--------|
| `IsHome` | Binary | Whether the player's team is home | Match data |
| `TeamQuality` | Continuous | Opponent-adjusted team strength score | Derived from win rates |
| `AvgTemp` | Continuous | Average temperature at venue (°C) | Weather data |
| `TempRange` | Continuous | Temperature range during match | Weather data |
| `IsRainy` | Binary | Rain indicator for match day | Weather data |

**Note:** `IsHome` effect is ruck-specific (ATE = +0.20 hitouts). Effect near zero for other positions.

---

## 5. Universal In-Game Performance Features

These are per-game statistics from the current or recent games.

| Feature | Type | Description | Notes |
|---------|------|-------------|-------|
| `Disposals` | Continuous | Total kicks + handballs | Primary midfield driver (coeff +2.54) |
| `Marks` | Continuous | Contested + uncontested marks | Negative association for mids (efficiency matters) |
| `Clangers` | Continuous | Errors / turnovers | Negative predictor |
| `Frees` | Continuous | Free kicks won | Positive for forwards |
| `FreesAgainst` | Continuous | Free kicks conceded | Negative predictor |
| `PctPlayed` | Continuous | % of game time played | Scales all other stats |
| `GamesPlayed` | Continuous | Career games played at time of match | Proxy for experience |

---

## 6. Position-Specific Features

These are only used for predictions within the relevant position model.

### Forward
| Feature | Description | Importance |
|---------|-------------|------------|
| `MarksInside50` | Marks taken inside the 50m arc | **Top predictor** (coeff +9.75) |
| `Tackles` | Defensive tackles | Secondary predictor |

### Midfield
| Feature | Description | Importance |
|---------|-------------|------------|
| `Inside50s` | Times player moved ball inside 50 | Key predictor |
| `GoalAssists` | Goals assisted | High importance |

### Ruck
| Feature | Description | Importance |
|---------|-------------|------------|
| `Clearances` | Clearances from congested stoppages | Key predictor |
| `ContestedMarks` | Marks under physical pressure | Secondary predictor |

### Defender
| Feature | Description | Importance |
|---------|-------------|------------|
| `OnePercenters` | Spoils, shepherds, smothers | Key predictor |
| `Tackles` | Defensive tackles | Key predictor |

---

## 7. Lag / Rolling Features (Production Additions)

These are new features added for the production system that were not in the Course 1 model.

| Feature | Description | Window |
|---------|-------------|--------|
| `Disposals_lag3` | Rolling average disposals | Last 3 games |
| `Disposals_lag5` | Rolling average disposals | Last 5 games |
| `TotalScore_lag3` | Rolling average score contribution | Last 3 games |
| `TotalScore_lag5` | Rolling average score contribution | Last 5 games |
| `Form_trend` | Slope of score over last 5 games | Derived |

**Note:** Lag features must be computed using only past data to prevent leakage. `TimeSeriesSplit` is used in cross-validation.

---

## 8. Target Variables (by Position Model)

| Position Model | Target Variable | Description |
|---------------|-----------------|-------------|
| Forward | `TotalScore` | Goals × 6 + Behinds × 1 |
| Midfield | `Clearances` | Clearances from stoppages |
| Ruck | `HitOuts` | Hitouts from ruck contests |
| Defender | `Rebounds` | Defensive rebounds / rebound 50s |

---

## Feature Store Location

Features are cached in the feature store at:
- Implementation: `src/features/build_features.py`
- Storage: Feast / SQLite (see `docker-compose.yml` for service config)
- Validation: Great Expectations suite in `reports/data_quality/`
