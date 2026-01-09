# Data Directory

This directory contains the data used for longitudinal mental health modeling.

## Structure

- `raw/`: Raw, unprocessed data files (not tracked in version control)
- `processed/`: Processed and cleaned data ready for analysis (not tracked in version control)
- `external/`: External reference data or auxiliary datasets (not tracked in version control)

## Loading Data

Use the `DataLoader` class from `src.data` to load data:

```python
from src.data import DataLoader

loader = DataLoader(data_dir="data/raw")

# Load CSV
dataset = loader.load_csv("your_data.csv", subject_col="subject_id", time_col="timepoint")

# Load Excel
dataset = loader.load_excel("your_data.xlsx", subject_col="subject_id", time_col="timepoint")
```

## Data Format

Longitudinal data should be in long format with the following structure:

| subject_id | timepoint | feature_1 | feature_2 | ... | outcome |
|------------|-----------|-----------|-----------|-----|---------|
| 1          | 0         | 0.5       | 1.2       | ... | 0       |
| 1          | 1         | 0.6       | 1.1       | ... | 0       |
| 2          | 0         | 0.3       | 1.5       | ... | 1       |
| 2          | 1         | 0.4       | 1.4       | ... | 1       |

## Privacy

**Important**: Data files are excluded from version control via `.gitignore`. Never commit sensitive or identifiable data to the repository.
