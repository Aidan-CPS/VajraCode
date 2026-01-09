"""
Test script: Nonlinear MML baseline using Stage 1 data only.

This script answers the following question:

    Can static baseline questionnaire data predict the earliest observed
    depression onset, when modeled using a nonlinear but theory-agnostic
    machine-learning model?

Design choices:
- One row per participant
- Outcome = earliest available Stage-3 onset
- Predictors = numeric Stage-1 baseline variables only
- Model = HistGradientBoostingClassifier
- Evaluation = AUC-ROC on a held-out validation set

This script serves as a clean reference baseline against which all
subsequent models (dynamic, theory-guided, yogic, etc.) should be compared.
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATA_DIR = "../WARND_data/data"

STAGE1_X = f"{DATA_DIR}/stage1_train_x.csv"
STAGE3_Y = f"{DATA_DIR}/stage3_train_y.csv"

ID_COL = "external_id"
TIME_COL = "month_number"
TARGET_COL = "onset"

RANDOM_STATE = 42
TEST_SIZE = 0.2


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

print("Loading data...")

x1 = pd.read_csv(STAGE1_X)
y3 = pd.read_csv(STAGE3_Y)

print(f"Stage 1 shape (baseline predictors): {x1.shape}")
print(f"Stage 3 shape (outcomes): {y3.shape}")


# ---------------------------------------------------------------------
# Define prediction target: earliest observed onset per participant
# ---------------------------------------------------------------------

print("Selecting earliest outcome per participant...")

first_outcome = (
    y3.sort_values(TIME_COL)
      .groupby(ID_COL, as_index=False)
      .first()
)

print(f"First outcome table shape: {first_outcome.shape}")


# ---------------------------------------------------------------------
# Merge predictors and outcome
# ---------------------------------------------------------------------

df = x1.merge(
    first_outcome[[ID_COL, TARGET_COL]],
    on=ID_COL,
    how="inner"
)

print(f"Merged dataset shape (one row per participant): {df.shape}")


# ---------------------------------------------------------------------
# Prepare features and target
# ---------------------------------------------------------------------

y = df[TARGET_COL]

# Use numeric baseline features only (theory-agnostic)
X = df.select_dtypes(include="number")
X = X.drop(columns=[TARGET_COL], errors="ignore")

# Drop degenerate features (entirely missing)
all_nan_cols = X.columns[X.isna().all()].tolist()
if all_nan_cols:
    print(f"Dropping all-NaN columns: {all_nan_cols}")
    X = X.drop(columns=all_nan_cols)

print(f"Number of numeric baseline features used: {X.shape[1]}")


# ---------------------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")


# ---------------------------------------------------------------------
# Train nonlinear MML model
# ---------------------------------------------------------------------

print("Training HistGradientBoostingClassifier...")

model = HistGradientBoostingClassifier(
    max_depth=6,
    max_iter=300,
    learning_rate=0.05,
    random_state=RANDOM_STATE
)

model.fit(X_train, y_train)


# ---------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------

y_val_pred = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_val_pred)

print(f"Validation AUC (HGB, Stage 1 only): {auc:.3f}")