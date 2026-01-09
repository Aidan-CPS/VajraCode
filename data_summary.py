# data_summary.py

import pandas as pd

# ---- Paths ----
DATA_DIR = "../WARND_data/data"

STAGE1_X = f"{DATA_DIR}/stage1_train_x.csv"
STAGE3_Y = f"{DATA_DIR}/stage3_train_y.csv"

# ---- Load data ----
print("Loading data...\n")

x1 = pd.read_csv(STAGE1_X)
y3 = pd.read_csv(STAGE3_Y)

# ---- Basic shapes ----
print("Stage 1 (baseline predictors)")
print("Shape:", x1.shape)
print()

print("Stage 3 (outcomes)")
print("Shape:", y3.shape)
print()

# ---- Column inspection ----
print("Stage 1 columns:")
print(x1.columns.tolist())
print()

print("Stage 3 Y columns:")
print(y3.columns.tolist())
print()

# ---- Peek at data ----
print("First 5 rows of Stage 1:")
print(x1.head())
print()

print("First 5 rows of Stage 3 Y:")
print(y3.head())
print()

# ---- Unique counts (helps identify ID columns) ----
print("Unique values per column in Stage 3 Y:")
for col in y3.columns:
    print(f"{col}: {y3[col].nunique()}")