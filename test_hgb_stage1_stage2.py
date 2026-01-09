import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

# --------------------
# Paths
# --------------------
DATA_DIR = "../WARND_data/data"

STAGE1_X = f"{DATA_DIR}/stage1_train_x.csv"
STAGE2_X = f"{DATA_DIR}/stage2_train_x.csv"
STAGE3_Y = f"{DATA_DIR}/stage3_train_y.csv"

ID_COL = "external_id"
TIME_COL = "month_number"
TARGET_COL = "onset"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# --------------------
# Load data
# --------------------
print("Loading data...")
x1 = pd.read_csv(STAGE1_X)
x2 = pd.read_csv(STAGE2_X, low_memory=False)
y3 = pd.read_csv(STAGE3_Y)

print(f"Stage 1 shape: {x1.shape}")
print(f"Stage 2 shape: {x2.shape}")
print(f"Stage 3 Y shape: {y3.shape}")

# --------------------
# Define target: earliest outcome per participant
# --------------------
print("Selecting earliest outcome per participant...")
first_outcome = (
    y3.sort_values(TIME_COL)
      .groupby(ID_COL, as_index=False)
      .first()
)

print(f"First outcome shape: {first_outcome.shape}")

# --------------------
# Aggregate Stage 2 per participant (NO time filtering)
# --------------------
print("Aggregating Stage 2 features (mean + variance, per participant)...")

numeric_cols = x2.select_dtypes(include="number").columns.tolist()

# Remove identifier columns if present
if ID_COL in numeric_cols:
    numeric_cols.remove(ID_COL)

agg_mean = (
    x2.groupby(ID_COL)[numeric_cols]
      .mean()
      .add_suffix("_mean")
)

agg_var = (
    x2.groupby(ID_COL)[numeric_cols]
      .var()
      .add_suffix("_var")
)

x2_agg = pd.concat([agg_mean, agg_var], axis=1).reset_index()

print(f"Aggregated Stage 2 shape: {x2_agg.shape}")

# --------------------
# Merge all predictors
# --------------------
df = (
    x1.merge(x2_agg, on=ID_COL, how="inner")
      .merge(first_outcome[[ID_COL, TARGET_COL]], on=ID_COL)
)

print(f"Final dataset shape: {df.shape}")

# --------------------
# Prepare features and target
# --------------------
y = df[TARGET_COL]

X = df.select_dtypes(include="number")
X = X.drop(columns=[TARGET_COL], errors="ignore")

# Drop all-NaN columns
all_nan_cols = X.columns[X.isna().all()].tolist()
if all_nan_cols:
    print(f"Dropping all-NaN columns: {all_nan_cols}")
    X = X.drop(columns=all_nan_cols)

print(f"Using {X.shape[1]} numeric features (Stage 1 + Stage 2)")

# --------------------
# Train / validation split
# --------------------
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# --------------------
# Train HGB model
# --------------------
print("Training HistGradientBoostingClassifier...")

model = HistGradientBoostingClassifier(
    max_depth=6,
    max_iter=300,
    learning_rate=0.05,
    random_state=RANDOM_STATE
)

model.fit(X_train, y_train)

# --------------------
# Evaluate
# --------------------
y_val_pred = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_val_pred)

print(f"Validation AUC (HGB, Stage 1 + Stage 2, naive): {auc:.3f}")