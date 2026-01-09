"""Tests for evaluation utilities."""

import pytest
import numpy as np
from src.evaluation import ModelEvaluator


class TestModelEvaluator:
    """Tests for ModelEvaluator class."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        # Regression data
        self.y_true_reg = np.random.randn(100)
        self.y_pred_reg = self.y_true_reg + np.random.randn(100) * 0.1
        
        # Classification data
        self.y_true_cls = np.random.randint(0, 2, 100)
        self.y_pred_cls = self.y_true_cls.copy()
        # Add some errors
        error_idx = np.random.choice(100, 10, replace=False)
        self.y_pred_cls[error_idx] = 1 - self.y_pred_cls[error_idx]
        self.y_pred_proba = np.random.rand(100)
    
    def test_evaluate_regression(self):
        """Test regression evaluation."""
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_regression(
            self.y_true_reg,
            self.y_pred_reg,
            "test_model"
        )
        
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        
        # Check that metrics are reasonable
        assert metrics['mse'] >= 0
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0
        assert -1 <= metrics['r2'] <= 1
    
    def test_evaluate_classification(self):
        """Test classification evaluation."""
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_classification(
            self.y_true_cls,
            self.y_pred_cls,
            self.y_pred_proba,
            "test_model"
        )
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'auc' in metrics
        assert 'confusion_matrix' in metrics
        
        # Check that metrics are in valid ranges
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1'] <= 1
        assert 0 <= metrics['auc'] <= 1
    
    def test_compare_models(self):
        """Test model comparison."""
        evaluator = ModelEvaluator()
        
        evaluator.evaluate_regression(self.y_true_reg, self.y_pred_reg, "model1")
        evaluator.evaluate_regression(self.y_true_reg, self.y_pred_reg * 1.1, "model2")
        
        comparison = evaluator.compare_models()
        
        assert len(comparison) == 2
        assert 'model' in comparison.columns
        assert 'mse' in comparison.columns
    
    def test_get_summary(self):
        """Test getting summary."""
        evaluator = ModelEvaluator()
        evaluator.evaluate_regression(self.y_true_reg, self.y_pred_reg, "test_model")
        
        summary = evaluator.get_summary("test_model")
        
        assert isinstance(summary, str)
        assert "test_model" in summary
        assert "regression" in summary
