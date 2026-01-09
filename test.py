import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# ---- Paths (adjust if needed) ----
DATA_DIR = "../WARND_data/data"

STAGE1_X = f"{DATA_DIR}/stage1_train_x.csv"
STAGE3_Y = f"{DATA_DIR}/stage3_train_y.csv"

# ---- Load data ----
print("Loading data...")
x1 = pd.read_csv(STAGE1_X)
y3 = pd.read_csv(STAGE3_Y)

print(f"Stage1 shape: {x1.shape}")
print(f"Stage3 Y shape: {y3.shape}")

# ---- Merge (you may need to adjust ID column name) ----
ID_COL = "id"   # check actual column name

df = x1.merge(y3, on=ID_COL)

# ---- Select a tiny set of features (numeric only) ----
numeric_cols = df.select_dtypes(include="number").columns.tolist()
numeric_cols.remove("y")  # remove target column if named differently

X = df[numeric_cols]
y = df["y"]

print(f"Using {len(numeric_cols)} numeric features")

# ---- Train / validation split ----
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- Scale features ----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# ---- Fit logistic regression ----
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# ---- Evaluate ----
y_val_pred = model.predict_proba(X_val_scaled)[:, 1]
auc = roc_auc_score(y_val, y_val_pred)

print(f"Validation AUC: {auc:.3f}")