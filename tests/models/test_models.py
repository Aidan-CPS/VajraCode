"""Tests for baseline models."""

import pytest
import numpy as np
from src.models import LinearModel, LogisticModel


class TestLinearModel:
    """Tests for LinearModel class."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.X_train = np.random.randn(100, 5)
        self.y_train = 2.0 + 0.5 * self.X_train[:, 0] + np.random.randn(100) * 0.1
        self.X_test = np.random.randn(20, 5)
    
    def test_init(self):
        """Test initialization."""
        model = LinearModel()
        assert model.name == "LinearModel"
        assert not model.is_fitted
    
    def test_fit_predict(self):
        """Test fitting and prediction."""
        model = LinearModel()
        model.fit(self.X_train, self.y_train)
        
        assert model.is_fitted
        
        predictions = model.predict(self.X_test)
        assert len(predictions) == len(self.X_test)
        assert isinstance(predictions, np.ndarray)
    
    def test_predict_before_fit(self):
        """Test that prediction fails before fitting."""
        model = LinearModel()
        with pytest.raises(RuntimeError):
            model.predict(self.X_test)
    
    def test_get_coefficients(self):
        """Test getting model coefficients."""
        model = LinearModel()
        model.fit(self.X_train, self.y_train)
        
        coefs = model.get_coefficients()
        assert len(coefs) == 5
        assert all(isinstance(v, float) for v in coefs.values())
    
    def test_standardization(self):
        """Test that standardization works."""
        model_std = LinearModel(standardize=True)
        model_no_std = LinearModel(standardize=False)
        
        model_std.fit(self.X_train, self.y_train)
        model_no_std.fit(self.X_train, self.y_train)
        
        # Both should work
        pred_std = model_std.predict(self.X_test)
        pred_no_std = model_no_std.predict(self.X_test)
        
        assert len(pred_std) == len(self.X_test)
        assert len(pred_no_std) == len(self.X_test)


class TestLogisticModel:
    """Tests for LogisticModel class."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.X_train = np.random.randn(100, 5)
        # Create binary labels
        prob = 1 / (1 + np.exp(-(self.X_train[:, 0] + self.X_train[:, 1])))
        self.y_train = (np.random.rand(100) < prob).astype(int)
        self.X_test = np.random.randn(20, 5)
    
    def test_init(self):
        """Test initialization."""
        model = LogisticModel()
        assert model.name == "LogisticModel"
        assert not model.is_fitted
    
    def test_fit_predict(self):
        """Test fitting and prediction."""
        model = LogisticModel()
        model.fit(self.X_train, self.y_train)
        
        assert model.is_fitted
        
        predictions = model.predict(self.X_test)
        assert len(predictions) == len(self.X_test)
        assert all(p in [0, 1] for p in predictions)
    
    def test_predict_proba(self):
        """Test probability prediction."""
        model = LogisticModel()
        model.fit(self.X_train, self.y_train)
        
        proba = model.predict_proba(self.X_test)
        assert proba.shape == (len(self.X_test), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
    
    def test_get_coefficients(self):
        """Test getting model coefficients."""
        model = LogisticModel()
        model.fit(self.X_train, self.y_train)
        
        coefs = model.get_coefficients()
        assert len(coefs) == 5
        assert all(isinstance(v, float) for v in coefs.values())
