# VajraCode

VajraCode is a clean, modular Python research codebase for longitudinal mental-health modeling. It provides reproducible tools for loading data, exploratory analysis, baseline modeling (linear/logistic regression), evaluation, and visualization.

The codebase emphasizes **clarity**, **modularity**, and **extensibility** over performance, making it ideal for research and educational purposes.

## Features

- 📊 **Data Loading Utilities**: Easy-to-use tools for loading CSV/Excel data and creating synthetic datasets
- 🔍 **Exploratory Analysis**: Scripts for analyzing longitudinal patterns and feature distributions
- 🤖 **Baseline Models**: Simple, interpretable linear and logistic regression models
- 📈 **Evaluation Helpers**: Comprehensive metrics for regression and classification tasks
- 📉 **Visualization Tools**: Beautiful plots for data exploration and model assessment
- 🧪 **Testing Suite**: Unit tests ensuring code reliability
- 📓 **Example Notebooks**: Jupyter notebooks demonstrating the complete workflow

## Project Structure

```
VajraCode/
├── data/                      # Data directory (files not tracked)
│   ├── raw/                   # Raw data files
│   ├── processed/             # Processed data
│   └── external/              # External reference data
├── src/                       # Source code
│   ├── data/                  # Data loading and processing
│   ├── models/                # Baseline models
│   ├── evaluation/            # Evaluation metrics
│   ├── visualization/         # Plotting utilities
│   └── utils/                 # Helper functions
├── scripts/                   # Executable scripts
│   ├── exploratory/           # Exploratory analysis scripts
│   ├── preprocessing/         # Data preprocessing
│   └── training/              # Model training scripts
├── tests/                     # Unit tests
├── notebooks/                 # Jupyter notebooks
├── config/                    # Configuration files
├── docs/                      # Documentation
└── outputs/                   # Output directory for results

```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Aidan-CPS/VajraCode.git
cd VajraCode
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

## Quick Start

### 1. Using Python Scripts

#### Exploratory Data Analysis

```bash
# Analyze synthetic data
python scripts/exploratory/analyze_data.py --output-dir outputs/exploratory

# Analyze your own data
python scripts/exploratory/analyze_data.py --data-path data/raw/mydata.csv --output-dir outputs/exploratory
```

#### Train Baseline Models

```bash
# Train regression model
python scripts/training/train_baseline.py --task regression --output-dir outputs/models

# Train classification model
python scripts/training/train_baseline.py --task classification --output-dir outputs/models

# Train on custom data
python scripts/training/train_baseline.py --data-path data/raw/mydata.csv --task regression --target my_outcome
```

### 2. Using the Python API

```python
from src.data import DataLoader
from src.models import LinearModel, LogisticModel
from src.evaluation import cross_validate_temporal
from src.visualization import plot_longitudinal, plot_predictions_vs_actual

# Load data
loader = DataLoader()
dataset = loader.create_synthetic_data(n_subjects=100, n_timepoints=5)

# Or load from CSV
# dataset = loader.load_csv("data/raw/mydata.csv")

# Explore data
print(dataset.summary())
plot_longitudinal(dataset, n_subjects=10)

# Train model
model = LinearModel()
feature_cols = dataset.get_feature_columns(exclude=['outcome_continuous'])
results = cross_validate_temporal(
    model=model,
    dataset=dataset,
    feature_cols=feature_cols,
    target_col='outcome_continuous',
    train_timepoints=[0, 1, 2],
    test_timepoints=[3, 4],
    task='regression'
)

print(results['metrics'])
```

### 3. Using Jupyter Notebooks

Launch Jupyter and open the example notebook:

```bash
jupyter notebook notebooks/getting_started.ipynb
```

## Data Format

VajraCode expects longitudinal data in **long format** with these key columns:

- **Subject identifier** (e.g., `subject_id`): Unique ID for each subject
- **Time identifier** (e.g., `timepoint`): Time index for each observation
- **Features**: Predictor variables
- **Outcomes**: Target variables (continuous or binary)

Example:

| subject_id | timepoint | feature_1 | feature_2 | outcome |
|------------|-----------|-----------|-----------|---------|
| 1          | 0         | 0.5       | 1.2       | 2.3     |
| 1          | 1         | 0.6       | 1.1       | 2.5     |
| 2          | 0         | 0.3       | 1.5       | 1.8     |

## Running Tests

Run the test suite to verify installation:

```bash
pytest tests/ -v
```

Run tests with coverage:

```bash
pytest tests/ --cov=src --cov-report=html
```

## Documentation

### Core Modules

- **`src.data`**: Data loading utilities and dataset classes
  - `DataLoader`: Load data from CSV/Excel or create synthetic data
  - `LongitudinalDataset`: Container for longitudinal data
  - `split_temporal()`: Split data by timepoints

- **`src.models`**: Baseline prediction models
  - `LinearModel`: Linear regression for continuous outcomes
  - `LogisticModel`: Logistic regression for binary outcomes

- **`src.evaluation`**: Model evaluation tools
  - `ModelEvaluator`: Compute evaluation metrics
  - `cross_validate_temporal()`: Temporal cross-validation

- **`src.visualization`**: Plotting functions
  - `plot_longitudinal()`: Visualize temporal trajectories
  - `plot_feature_distributions()`: Feature distributions
  - `plot_correlation_matrix()`: Feature correlations
  - `plot_model_performance()`: Compare model performance
  - `plot_predictions_vs_actual()`: Prediction quality

### Configuration

Example configuration files are provided in `config/`. You can customize:
- Data paths and column names
- Model hyperparameters
- Training settings
- Evaluation metrics
- Output preferences

## Extending VajraCode

The codebase is designed for easy extension:

1. **Add new models**: Inherit from `BaseModel` in `src/models/`
2. **Add new metrics**: Extend `ModelEvaluator` in `src/evaluation/`
3. **Add new plots**: Add functions to `src/visualization/`
4. **Add preprocessing**: Create modules in `scripts/preprocessing/`

Example of adding a custom model:

```python
from src.models import BaseModel

class MyCustomModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(name="MyCustomModel")
        # Initialize your model
        
    def fit(self, X, y):
        # Training logic
        self.is_fitted = True
        
    def predict(self, X):
        # Prediction logic
        return predictions
```

## Best Practices

1. **Version Control**: Never commit data files; they're excluded in `.gitignore`
2. **Reproducibility**: Always set random seeds for reproducible results
3. **Documentation**: Add docstrings to new functions and classes
4. **Testing**: Write tests for new functionality
5. **Configuration**: Use config files for experiment settings

## Research Focus

VajraCode supports research on:
- Longitudinal patterns in mental health data
- Early detection of mental health conditions
- Theory-agnostic vs. theory-guided modeling approaches
- Model interpretability and feature importance
- Temporal validation strategies

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - See LICENSE file for details

## Citation

If you use VajraCode in your research, please cite:

```bibtex
@software{vajracode2024,
  title={VajraCode: A Research Codebase for Longitudinal Mental Health Modeling},
  author={VajraCode Team},
  year={2024},
  url={https://github.com/Aidan-CPS/VajraCode}
}
```

## Contact

For questions or feedback, please open an issue on GitHub.

## Acknowledgments

This codebase was developed for longitudinal mental health research, emphasizing clarity and reproducibility in modeling depression as a dynamic process.
