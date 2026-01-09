"""VajraCode: A research codebase for longitudinal mental-health modeling.

This package provides utilities for loading, analyzing, and modeling
longitudinal mental health data with a focus on clarity, modularity,
and reproducibility.
"""

__version__ = "0.1.0"
__author__ = "VajraCode Team"

from src.data import DataLoader, LongitudinalDataset
from src.models import LinearModel, LogisticModel
from src.evaluation import ModelEvaluator
from src.visualization import plot_longitudinal, plot_model_performance

__all__ = [
    "DataLoader",
    "LongitudinalDataset",
    "LinearModel",
    "LogisticModel",
    "ModelEvaluator",
    "plot_longitudinal",
    "plot_model_performance",
]
