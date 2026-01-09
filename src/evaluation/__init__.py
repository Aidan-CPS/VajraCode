"""Evaluation utilities for model assessment.

This module provides tools for evaluating model performance on
longitudinal mental health data, including metrics for both
continuous and binary outcomes.
"""

from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


class ModelEvaluator:
    """Utility class for evaluating model performance.
    
    This class provides methods to compute various evaluation metrics
    for both regression and classification tasks.
    """
    
    def __init__(self):
        """Initialize the ModelEvaluator."""
        self.results = {}
        
    def evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "model"
    ) -> Dict[str, float]:
        """Evaluate regression model performance.
        
        Args:
            y_true: True continuous values
            y_pred: Predicted continuous values
            model_name: Name of the model being evaluated
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        metrics = {
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
        }
        
        # Store results
        self.results[model_name] = {
            "task": "regression",
            "metrics": metrics,
            "n_samples": len(y_true)
        }
        
        return metrics
    
    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        model_name: str = "model"
    ) -> Dict[str, Any]:
        """Evaluate classification model performance.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_pred_proba: Predicted probabilities (optional, for AUC)
            model_name: Name of the model being evaluated
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }
        
        # Add AUC if probabilities are provided
        if y_pred_proba is not None:
            # If binary probs, take positive class
            if y_pred_proba.ndim == 2:
                y_pred_proba = y_pred_proba[:, 1]
            metrics["auc"] = roc_auc_score(y_true, y_pred_proba)
        
        # Add confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["true_negatives"] = int(tn)
            metrics["false_positives"] = int(fp)
            metrics["false_negatives"] = int(fn)
            metrics["true_positives"] = int(tp)
        
        # Store results
        self.results[model_name] = {
            "task": "classification",
            "metrics": metrics,
            "n_samples": len(y_true)
        }
        
        return metrics
    
    def compare_models(self) -> pd.DataFrame:
        """Compare all evaluated models.
        
        Returns:
            DataFrame with comparison of all models
        """
        if not self.results:
            return pd.DataFrame()
        
        rows = []
        for model_name, result in self.results.items():
            row = {"model": model_name, "task": result["task"]}
            # Flatten metrics
            for metric_name, metric_value in result["metrics"].items():
                if metric_name != "confusion_matrix":
                    row[metric_name] = metric_value
            rows.append(row)
            
        return pd.DataFrame(rows)
    
    def get_summary(self, model_name: str) -> str:
        """Get a text summary of model performance.
        
        Args:
            model_name: Name of the model to summarize
            
        Returns:
            String summary of model performance
        """
        if model_name not in self.results:
            return f"No results found for model '{model_name}'"
        
        result = self.results[model_name]
        task = result["task"]
        metrics = result["metrics"]
        
        lines = [
            f"Model: {model_name}",
            f"Task: {task}",
            f"Samples: {result['n_samples']}",
            "",
            "Metrics:"
        ]
        
        for metric_name, metric_value in metrics.items():
            if metric_name == "confusion_matrix":
                lines.append(f"  Confusion Matrix:")
                for row in metric_value:
                    lines.append(f"    {row}")
            elif isinstance(metric_value, (int, float)):
                lines.append(f"  {metric_name}: {metric_value:.4f}")
            else:
                lines.append(f"  {metric_name}: {metric_value}")
        
        return "\n".join(lines)


def cross_validate_temporal(
    model: Any,
    dataset: Any,
    feature_cols: List[str],
    target_col: str,
    train_timepoints: List[Any],
    test_timepoints: List[Any],
    task: str = "regression"
) -> Dict[str, Any]:
    """Perform temporal cross-validation.
    
    Train on earlier timepoints and evaluate on later ones to assess
    model generalization to future time periods.
    
    Args:
        model: Model instance with fit() and predict() methods
        dataset: LongitudinalDataset instance
        feature_cols: List of feature column names
        target_col: Target column name
        train_timepoints: List of timepoints for training
        test_timepoints: List of timepoints for testing
        task: Either "regression" or "classification"
        
    Returns:
        Dictionary with evaluation results
    """
    # Split data
    train_data = dataset.data[
        dataset.data[dataset.time_col].isin(train_timepoints)
    ]
    test_data = dataset.data[
        dataset.data[dataset.time_col].isin(test_timepoints)
    ]
    
    # Prepare features and targets
    X_train = train_data[feature_cols].values
    y_train = train_data[target_col].values
    X_test = test_data[feature_cols].values
    y_test = test_data[target_col].values
    
    # Fit model
    model.fit(X_train, y_train, feature_names=feature_cols)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate
    evaluator = ModelEvaluator()
    if task == "regression":
        metrics = evaluator.evaluate_regression(y_test, y_pred, model.name)
    elif task == "classification":
        # Get probabilities if available
        y_pred_proba = None
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
        metrics = evaluator.evaluate_classification(
            y_test, y_pred, y_pred_proba, model.name
        )
    else:
        raise ValueError(f"Unknown task: {task}")
    
    return {
        "metrics": metrics,
        "train_timepoints": train_timepoints,
        "test_timepoints": test_timepoints,
        "n_train": len(train_data),
        "n_test": len(test_data),
    }
