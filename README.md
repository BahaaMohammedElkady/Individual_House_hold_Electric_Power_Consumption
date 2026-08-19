# Household Power Consumption Forecasting

A time-series regression case study on the UCI *Individual Household Electric Power Consumption* dataset — from raw sensor data to a leakage-free, deployment-realistic forecasting model.

---

## Overview

This project builds a regression pipeline to forecast a single household's active power consumption using ~4 years of minute-level smart meter data. Beyond producing a working model, the project is structured as a rigorous case study in **preventing and catching data leakage** — a theme that shaped several key decisions throughout.

**Final result:** XGBoost forecasting power consumption **60 minutes ahead**, achieving **R² = 0.357**, **MAE = 0.486 kW**, using only features genuinely available before the prediction timestamp.

---

## Dataset

| | |
|---|---|
| Source | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) (id=235) |
| Location | Sceaux, France (7km from Paris) |
| Period | December 2006 – November 2010 (~4 years) |
| Sampling rate | 1-minute intervals |
| Instances | 2,075,259 |
| Raw features | 9 (Date, Time, Global Active/Reactive Power, Voltage, Global Intensity, 3 Sub-metering channels) |
| Missing data | ~1.25% (25,979 rows) |
| License | CC BY 4.0 |

---

## Project Structure

```
├── Raw_Datasets/           # Untouched original data
├── Cleaned_Datasets/       # Processed data, trained models, scalers
├── Charts/                 # All visualizations (15 exported charts)
├── Reports/                # Results tables, feature lists, this README
└── notebook.ipynb          # Full step-by-step analysis
```

---

## Pipeline Summary

### 1. Data Preparation
- Fixed a mixed-dtype issue: 6 numeric columns were loaded as strings due to empty-string missing-value encoding; corrected with `pd.to_numeric(errors='coerce')`.
- Combined `Date` + `Time` into a proper `datetime` index. Verified **zero gaps and zero duplicates** across all 2,075,259 expected one-minute timestamps.

### 2. Missing Value Analysis & Imputation
Gap-length analysis revealed a bimodal structure:

| Gap type | Count | Total minutes | % of missing data |
|---|---|---|---|
| Short/medium (≤6h) | 64 | 441 | ~2% |
| Long (>6h) | 7 | 25,538 | ~98% |

A **hybrid imputation strategy** was applied:
- **Short gaps** → time-based linear interpolation
- **Long gaps** → weekly seasonal fill (value from the same weekday/time, 7 days prior), preserving realistic daily consumption cycles instead of flattening multi-day outages

### 3. Feature Engineering
- Cyclical calendar encodings (`hour_sin/cos`, `month_sin/cos`), `is_weekend`, `day_of_week`
- Lag features at multiple resolutions (1, 5, 15, 30, 60, 1440 minutes)
- Rolling statistics (60-min and 180-min mean, 60-min std)
- `RobustScaler` applied to all continuous predictors (chosen over `StandardScaler` due to heavy right-skew in appliance-driven sub-metering data)

### 4. Data Leakage — Two Issues Caught and Fixed
This project deliberately documents two leakage incidents as a demonstration of proper validation discipline:

1. **`unmetered_power`**, a feature derived directly from the dataset's own documentation formula, was algebraically reconstructable back into the target (`Global_active_power`). Initial R² of **1.000** flagged the problem immediately. **Fixed by removal.**
2. **`Global_intensity`** and **`Global_reactive_power`** are near-deterministic physical functions of the target (P ≈ V × I), measured *simultaneously* rather than knowable in advance. Removing them was necessary once the project's forecasting use case was confirmed. **Fixed by removal**, along with `Voltage` and `Sub_metering_1/2/3` — all simultaneous smart-meter readings unavailable at true forecast time.

### 5. Modeling & Horizon Reframing
An initial 1-minute-ahead forecast was found to be trivially dominated by persistence (`gap_lag_1min` = 98.7% of Random Forest's feature importance), making Random Forest and Linear Regression nearly indistinguishable (R² 0.941 vs 0.940). The problem was reframed to a **60-minute-ahead forecast** — a more realistic and actionable horizon for demand-response or load-shifting use cases.

---

## Results

| Model | Horizon | MAE (kW) | RMSE (kW) | R² |
|---|---|---|---|---|
| Linear Regression | 1 min | 0.083 | 0.218 | 0.940 |
| Random Forest | 1 min | 0.080 | 0.215 | 0.941 |
| Linear Regression | 60 min | 0.544 | 0.775 | 0.237 |
| Random Forest | 60 min | 0.499 | 0.742 | 0.300 |
| XGBoost (12 features) | 60 min | 0.489 | 0.716 | 0.348 |
| **XGBoost (17 features) — Final** | **60 min** | **0.486** | **0.711** | **0.357** |

At the 60-minute horizon, model sophistication meaningfully mattered: XGBoost outperformed Linear Regression by **+12 points of R²** and Random Forest by **+6 points**, driven by its ability to capture nonlinear interactions between recency and time-of-day signals.

---

## Key Insight: The Forecast Ceiling Is Real, Not a Modeling Failure

The final model explains ~36% of variance an hour ahead — a modest number by design, not by shortfall. Visual inspection of predicted-vs-actual plots shows the model reliably tracks the *general consumption regime* (rising into an active period, settling into a low-usage overnight lull) but cannot anticipate the exact minute an individual appliance switches on. That event is inherently unpredictable from historical and calendar data alone for a single household — genuine improvement beyond this point would likely require external signals (weather, occupancy/calendar data) not available in this dataset.

---

## Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · matplotlib · seaborn · joblib

---

## Reproducing This Project

```bash
pip install ucimlrepo xgboost pandas numpy scikit-learn matplotlib seaborn
```

Run `notebook.ipynb` sequentially — each step is self-contained with markdown documentation explaining the rationale before every code cell. The final trained model is saved at `Cleaned_Datasets/final_xgboost_60min_model.pkl`.

---

## Author

Dr. Bahaa — Pharmacist & Data Science Practitioner
