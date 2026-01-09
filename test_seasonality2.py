"""
Robust seasonality analysis for Stage-2 (EMA + wearable) data using relative time.

Purpose
-------
Estimate the strength of approximately annual (~365-day) periodic structure
across Stage-2 variables, despite the absence of absolute calendar dates.

Key outputs
-----------
1. Per-variable seasonality strength (R^2 of sinusoidal model)
2. Domain-level aggregate seasonality
3. Global aggregate seasonality score
4. Clear textual interpretation

Method
------
For each numeric Stage-2 variable x(t), fit:
    x(t) ~ sin(2πt / 365) + cos(2πt / 365)
and compute R^2 as a measure of seasonality strength.

Seasonality is treated as a background modulator, not a primary driver.
"""

import pandas as pd
import numpy as np

# --------------------
# Configuration
# --------------------

DATA_DIR = "../WARND_data/data"
STAGE2_X = f"{DATA_DIR}/stage2_train_x.csv"

ID_COL = "external_id"
TIME_COL = "day_num"
CYCLE_LENGTH = 365

# Domain mapping (editable and transparent)
DOMAIN_MAP = {
    "affect": [
        "sad_d", "cheerful_d", "motivated_d", "depressed_e",
        "anhedonia_e", "content_e"
    ],
    "energy_fatigue": [
        "tired_d", "sleep_dur_tot_s", "sleep_qual_m"
    ],
    "stress": [
        "stress_w", "stressed_d", "overwhelm_d"
    ],
    "activity": [
        "day_total_steps", "day_active_seconds", "intraday_steps"
    ],
}

MIN_OBS = 5000  # minimum observations required to fit seasonality

# --------------------
# Load data
# --------------------

print("Loading Stage-2 data...")
x2 = pd.read_csv(STAGE2_X, low_memory=False)

if TIME_COL not in x2.columns:
    raise ValueError(f"Required time column '{TIME_COL}' not found.")

x2 = x2.dropna(subset=[TIME_COL])
x2["annual_phase"] = x2[TIME_COL] % CYCLE_LENGTH

# --------------------
# Prepare sinusoidal regressors
# --------------------

sin_term = np.sin(2 * np.pi * x2["annual_phase"] / CYCLE_LENGTH)
cos_term = np.cos(2 * np.pi * x2["annual_phase"] / CYCLE_LENGTH)
X_season = np.column_stack([sin_term, cos_term])

# --------------------
# Identify numeric variables
# --------------------

numeric_cols = x2.select_dtypes(include="number").columns.tolist()
numeric_cols = [
    c for c in numeric_cols
    if c not in [TIME_COL, "annual_phase"]
]

print(f"Total numeric Stage-2 variables: {len(numeric_cols)}")

# --------------------
# Compute seasonality per variable
# --------------------

seasonality_rows = []

for var in numeric_cols:
    y = x2[var].values
    valid = ~np.isnan(y)

    if valid.sum() < MIN_OBS:
        continue

    yv = y[valid]
    Xv = X_season[valid]

    # Center variable
    yv = yv - np.mean(yv)

    # Fit sinusoidal regression
    beta, _, _, _ = np.linalg.lstsq(Xv, yv, rcond=None)
    y_hat = Xv @ beta

    ss_total = np.sum(yv ** 2)
    ss_resid = np.sum((yv - y_hat) ** 2)

    r2 = 1 - ss_resid / ss_total if ss_total > 0 else 0.0

    seasonality_rows.append({
        "variable": var,
        "seasonality_r2": r2,
    })

seasonality_df = pd.DataFrame(seasonality_rows)
seasonality_df = seasonality_df.sort_values("seasonality_r2", ascending=False)

print(f"\nVariables with valid seasonality estimates: {len(seasonality_df)}")

# --------------------
# Domain-level aggregation
# --------------------

domain_rows = []

for domain, vars_in_domain in DOMAIN_MAP.items():
    subset = seasonality_df[
        seasonality_df["variable"].isin(vars_in_domain)
    ]

    if subset.empty:
        continue

    domain_rows.append({
        "domain": domain,
        "mean_seasonality_r2": subset["seasonality_r2"].mean(),
        "max_seasonality_r2": subset["seasonality_r2"].max(),
        "n_variables": len(subset),
    })

domain_df = pd.DataFrame(domain_rows)

# --------------------
# Global aggregate seasonality
# --------------------

global_seasonality = seasonality_df["seasonality_r2"].mean()

# --------------------
# Reporting
# --------------------

print("\n================ Seasonality Summary (Stage-2) ================\n")

print("Top 10 most seasonal variables:")
print(seasonality_df.head(10))

print("\nDomain-level seasonality:")
print(domain_df)

print("\nGlobal aggregate seasonality (mean R^2 across variables):")
print(f"{global_seasonality:.4f}")



print("\nEnd of robust seasonality analysis.")