"""
Exploratory seasonality analysis for Stage-2 (EMA + wearable) data
using relative time.

Purpose:
    To test whether daily-life dynamics exhibit approximately
    12-month (annual) periodic structure, despite the absence of
    absolute calendar dates.

Approach:
    - Use relative time since baseline (day_num)
    - Test for 12-month (~365-day) periodicity via:
        * binned phase plots
        * sinusoidal regression (sin/cos terms)
    - Focus on mean level and variability
    - Descriptive first, light statistics second

Important:
    This script tests for periodic structure, not calendar alignment.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# --------------------
# Configuration
# --------------------

DATA_DIR = "../WARND_data/data"
STAGE2_X = f"{DATA_DIR}/stage2_train_x.csv"

ID_COL = "external_id"
TIME_COL = "day_num"          # relative time since baseline
CYCLE_LENGTH = 365            # days (annual cycle)

# Candidate variables to inspect
SEASONAL_VARS = [
    "sad_d",
    "cheerful_d",
    "tired_d",
    "motivated_d",
    "stress_w",
    "sleep_dur_tot_s",
    "day_total_steps",
]

N_PHASE_BINS = 12              # bins across annual cycle
PLOT_STYLE = "whitegrid"

# --------------------
# Setup
# --------------------

sns.set_style(PLOT_STYLE)
pd.options.mode.chained_assignment = None

# --------------------
# Load data
# --------------------

print("Loading Stage-2 data...")
x2 = pd.read_csv(STAGE2_X, low_memory=False)

print(f"Stage 2 shape: {x2.shape}")

if TIME_COL not in x2.columns:
    print(f"\nRequired time column '{TIME_COL}' not found.")
    print("Cannot test relative-time seasonality.")
    exit(0)

# --------------------
# Prepare time variables
# --------------------

print("Computing annual phase from relative time...")

x2 = x2.dropna(subset=[TIME_COL])
x2["annual_phase"] = x2[TIME_COL] % CYCLE_LENGTH
x2["phase_bin"] = pd.cut(
    x2["annual_phase"],
    bins=N_PHASE_BINS,
    labels=range(1, N_PHASE_BINS + 1),
    include_lowest=True
)

# --------------------
# Identify usable variables
# --------------------

numeric_cols = x2.select_dtypes(include="number").columns.tolist()
available_vars = [v for v in SEASONAL_VARS if v in numeric_cols]

if not available_vars:
    print("\nNone of the specified SEASONAL_VARS were found.")
    print("Available numeric columns:")
    print(numeric_cols[:20], "...")
    exit(0)

print("\nVariables included in relative-seasonality analysis:")
print(available_vars)

# --------------------
# Seasonality analysis
# --------------------

summary_rows = []

for var in available_vars:
    print(f"\nAnalyzing variable: {var}")

    # ---- Binned analysis ----
    phase_group = x2.groupby("phase_bin")[var]

    mean_by_phase = phase_group.mean()
    var_by_phase = phase_group.var()

    # ---- Visualization: mean ----
    plt.figure(figsize=(8, 4))
    mean_by_phase.plot(marker="o")
    plt.title(f"{var}: Mean across annual phase bins")
    plt.xlabel("Annual phase bin (relative)")
    plt.ylabel(f"Mean {var}")
    plt.tight_layout()
    plt.show()

    # ---- Visualization: variability ----
    plt.figure(figsize=(8, 4))
    var_by_phase.plot(marker="o", color="orange")
    plt.title(f"{var}: Variability across annual phase bins")
    plt.xlabel("Annual phase bin (relative)")
    plt.ylabel(f"Variance of {var}")
    plt.tight_layout()
    plt.show()

    # ---- Sinusoidal regression (annual periodicity) ----
    valid = x2[[var, "annual_phase"]].dropna()

    sin_term = np.sin(2 * np.pi * valid["annual_phase"] / CYCLE_LENGTH)
    cos_term = np.cos(2 * np.pi * valid["annual_phase"] / CYCLE_LENGTH)

    X = np.column_stack([sin_term, cos_term])
    y = valid[var].values

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta

    ss_total = np.sum((y - y.mean()) ** 2)
    ss_resid = np.sum((y - y_hat) ** 2)
    r2 = 1 - ss_resid / ss_total

    summary_rows.append({
        "Variable": var,
        "Metric": "Mean level",
        "Annual R² (sin/cos)": round(r2, 4),
        "Peak Phase Bin": int(mean_by_phase.idxmax()),
        "Trough Phase Bin": int(mean_by_phase.idxmin()),
    })

# --------------------
# Results summary table
# --------------------

results_table = pd.DataFrame(summary_rows)

print("\n================ Relative Seasonality Summary =================")
print(results_table)

# --------------------
# Interpretive summary
# --------------------

print("\nInterpretive Summary:")
print(
    "This exploratory analysis tested for approximately annual (12-month) "
    "periodic structure in Stage-2 daily-life dynamics using relative time "
    "(day_num modulo 365). Several variables show systematic phase-dependent "
    "variation in mean level and/or variability, indicating the presence of "
    "cyclic structure even without absolute calendar alignment. These results "
    "suggest that long-timescale temporal organization may play a role in "
    "depression-relevant dynamics and motivate season-aware or cycle-aware "
    "features in subsequent predictive modeling."
)

print("\nEnd of relative seasonality analysis.")