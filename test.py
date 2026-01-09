# test.py

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# --------------------
# Paths
# --------------------
DATA_DIR = "../WARND_data/data"

STAGE1_X = f"{DATA_DIR}/stage1_train_x.csv"
STAGE3_Y = f"{DATA_DIR}/stage3_train_y.csv"

ID_COL = "external_id"
TIME_COL = "month_number"
TARGET_COL = "onset"

# --------------------
# Load data
# --------------------
print("Loading data...")
x1 = pd.read_csv(STAGE1_X)
y3 = pd.read_csv(STAGE3_Y)

print(f"Stage 1 shape: {x1.shape}")
print(f"Stage 3 Y shape: {y3.shape}")

# --------------------
# Select earliest outcome per participant
# --------------------
print("Selecting earliest outcome per participant...")
first_outcome = (
    y3.sort_values(TIME_COL)
      .groupby(ID_COL, as_index=False)
      .first()
)

print(f"First outcome shape: {first_outcome.shape}")

# --------------------
# Merge baseline with outcome
# --------------------
df = x1.merge(first_outcome[[ID_COL, TARGET_COL]], on=ID_COL)

print(f"Merged dataset shape: {df.shape}")

# --------------------
# Prepare features and target
# --------------------
y = df[TARGET_COL]

X = df.select_dtypes(include="number")
X = X.drop(columns=[TARGET_COL], errors="ignore")

# Drop columns that are entirely missing
all_nan_cols = X.columns[X.isna().all()].tolist()
if all_nan_cols:
    print(f"Dropping all-NaN columns: {all_nan_cols}")
    X = X.drop(columns=all_nan_cols)


print(f"Using {X.shape[1]} numeric baseline features")

# --------------------
# Train / validation split
# --------------------
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------
# Build preprocessing + model pipeline
# --------------------
pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

# --------------------
# Train model
# --------------------
pipeline.fit(X_train, y_train)

# --------------------
# Evaluate
# --------------------
y_val_pred = pipeline.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_val_pred)

print(f"Validation AUC (baseline, Stage 1 only): {auc:.3f}")