# VajraCode Overview

## Project Summary

VajraCode is a clean, modular Python research codebase for longitudinal mental-health modeling. It provides a complete workflow from data loading to model evaluation with emphasis on clarity and reproducibility.

## Key Components

### 1. Data Module (`src/data/`)
- **LongitudinalDataset**: Container class for longitudinal data
- **DataLoader**: Load CSV/Excel files or create synthetic data
- **split_temporal()**: Split data by timepoints for temporal validation

### 2. Models Module (`src/models/`)
- **BaseModel**: Abstract base class for all models
- **LinearModel**: Linear regression for continuous outcomes
- **LogisticModel**: Logistic regression for binary classification
- Features: Automatic standardization, coefficient extraction, interpretability

### 3. Evaluation Module (`src/evaluation/`)
- **ModelEvaluator**: Compute metrics for regression and classification
- **cross_validate_temporal()**: Temporal cross-validation
- Metrics: MSE, RMSE, MAE, R², accuracy, precision, recall, F1, AUC

### 4. Visualization Module (`src/visualization/`)
- **plot_longitudinal()**: Visualize temporal trajectories
- **plot_feature_distributions()**: Feature histograms
- **plot_correlation_matrix()**: Correlation heatmap
- **plot_model_performance()**: Model comparison
- **plot_predictions_vs_actual()**: Prediction quality assessment

### 5. Utilities Module (`src/utils/`)
- Configuration loading/saving (YAML/JSON)
- Directory management helpers

## Scripts

### Exploratory Analysis
```bash
python scripts/exploratory/analyze_data.py [options]
```
- Generates dataset summary statistics
- Creates visualization plots
- Saves outputs to specified directory

### Model Training
```bash
python scripts/training/train_baseline.py [options]
```
- Trains baseline models (linear/logistic)
- Performs temporal cross-validation
- Saves results and visualizations

## Testing

Comprehensive test suite with 22 tests covering:
- Data loading and processing
- Model training and prediction
- Evaluation metrics
- Edge cases and error handling

Run tests:
```bash
pytest tests/ -v
```

## Data Format

Expected format: Long-form longitudinal data

| subject_id | timepoint | feature_1 | feature_2 | outcome |
|------------|-----------|-----------|-----------|---------|
| 1          | 0         | 0.5       | 1.2       | 2.3     |
| 1          | 1         | 0.6       | 1.1       | 2.5     |
| 2          | 0         | 0.3       | 1.5       | 1.8     |

## Configuration

Configuration files in `config/` directory allow customization of:
- Data paths and column names
- Model hyperparameters
- Training/evaluation settings
- Output preferences

## Extensibility

The modular design makes it easy to:
1. Add new model types by inheriting from `BaseModel`
2. Add new evaluation metrics to `ModelEvaluator`
3. Add new visualization functions
4. Create custom preprocessing pipelines

## Best Practices

1. **Reproducibility**: Always set random seeds
2. **Documentation**: All functions have comprehensive docstrings
3. **Testing**: Write tests for new functionality
4. **Version Control**: Data files are excluded from git
5. **Modularity**: Keep functions focused and reusable

## Example Workflow

1. Load data:
```python
from src.data import DataLoader
loader = DataLoader()
dataset = loader.load_csv("data.csv")
```

2. Train model:
```python
from src.models import LinearModel
model = LinearModel()
model.fit(X_train, y_train)
```

3. Evaluate:
```python
from src.evaluation import ModelEvaluator
evaluator = ModelEvaluator()
metrics = evaluator.evaluate_regression(y_true, y_pred)
```

4. Visualize:
```python
from src.visualization import plot_predictions_vs_actual
plot_predictions_vs_actual(y_true, y_pred)
```

## Dependencies

Core dependencies:
- numpy, pandas, scipy (data handling)
- scikit-learn (models and metrics)
- matplotlib, seaborn (visualization)
- pytest (testing)
- pyyaml (configuration)

See `requirements.txt` for complete list.

## Future Extensions

Potential areas for expansion:
- Advanced models (random forests, neural networks)
- Time series specific models (ARIMA, state-space)
- Missing data imputation
- Feature engineering pipelines
- Cross-validation strategies
- Model selection and hyperparameter tuning
- Interactive visualizations
