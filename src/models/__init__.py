"""Baseline models for longitudinal mental health prediction.

This module provides simple, interpretable baseline models including
linear regression and logistic regression with support for longitudinal data.
"""

from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler


class BaseModel:
    """Base class for all models in VajraCode.
    
    This provides a common interface for all prediction models.
    """
    
    def __init__(self, name: str = "BaseModel"):
        """Initialize the base model.
        
        Args:
            name: Name identifier for the model
        """
        self.name = name
        self.is_fitted = False
        self.feature_names = None
        self.scaler = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model to training data.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target vector of shape (n_samples,)
        """
        raise NotImplementedError("Subclasses must implement fit()")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Predictions of shape (n_samples,)
        """
        raise NotImplementedError("Subclasses must implement predict()")
        
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        return {"name": self.name, "is_fitted": self.is_fitted}


class LinearModel(BaseModel):
    """Linear regression model for continuous outcomes.
    
    This model predicts continuous mental health outcomes (e.g., depression scores)
    using linear regression. Features are standardized by default.
    
    Attributes:
        model: Underlying scikit-learn LinearRegression model
        standardize: Whether to standardize features
        scaler: StandardScaler instance if standardization is used
    """
    
    def __init__(
        self,
        standardize: bool = True,
        fit_intercept: bool = True,
        name: str = "LinearModel"
    ):
        """Initialize the linear model.
        
        Args:
            standardize: Whether to standardize features before fitting
            fit_intercept: Whether to fit an intercept term
            name: Name identifier for the model
        """
        super().__init__(name)
        self.standardize = standardize
        self.model = LinearRegression(fit_intercept=fit_intercept)
        if standardize:
            self.scaler = StandardScaler()
            
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None):
        """Fit the linear model to training data.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target vector of shape (n_samples,)
            feature_names: Optional list of feature names
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        if self.standardize:
            X = self.scaler.fit_transform(X)
            
        self.model.fit(X, y)
        self.is_fitted = True
        self.feature_names = feature_names
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Predicted continuous values of shape (n_samples,)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
            
        X = np.asarray(X)
        if self.standardize:
            X = self.scaler.transform(X)
            
        return self.model.predict(X)
    
    def get_coefficients(self) -> Dict[str, float]:
        """Get model coefficients.
        
        Returns:
            Dictionary mapping feature names (or indices) to coefficients
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting coefficients")
            
        coef_dict = {}
        for i, coef in enumerate(self.model.coef_):
            if self.feature_names:
                key = self.feature_names[i]
            else:
                key = f"feature_{i}"
            coef_dict[key] = coef
            
        return coef_dict
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        params = super().get_params()
        params.update({
            "standardize": self.standardize,
            "intercept": self.model.intercept_ if self.is_fitted else None,
            "coefficients": self.get_coefficients() if self.is_fitted else None,
        })
        return params


class LogisticModel(BaseModel):
    """Logistic regression model for binary outcomes.
    
    This model predicts binary mental health outcomes (e.g., depression diagnosis)
    using logistic regression. Features are standardized by default.
    
    Attributes:
        model: Underlying scikit-learn LogisticRegression model
        standardize: Whether to standardize features
        scaler: StandardScaler instance if standardization is used
    """
    
    def __init__(
        self,
        standardize: bool = True,
        penalty: str = 'l2',
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: Optional[int] = 42,
        name: str = "LogisticModel"
    ):
        """Initialize the logistic model.
        
        Args:
            standardize: Whether to standardize features before fitting
            penalty: Regularization penalty ('l1', 'l2', 'elasticnet', 'none')
            C: Inverse of regularization strength
            max_iter: Maximum number of iterations
            random_state: Random seed for reproducibility
            name: Name identifier for the model
        """
        super().__init__(name)
        self.standardize = standardize
        self.model = LogisticRegression(
            penalty=penalty,
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            solver='lbfgs' if penalty == 'l2' else 'saga'
        )
        if standardize:
            self.scaler = StandardScaler()
            
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None):
        """Fit the logistic model to training data.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Binary target vector of shape (n_samples,)
            feature_names: Optional list of feature names
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        if self.standardize:
            X = self.scaler.fit_transform(X)
            
        self.model.fit(X, y)
        self.is_fitted = True
        self.feature_names = feature_names
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make binary predictions on new data.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Predicted binary values of shape (n_samples,)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
            
        X = np.asarray(X)
        if self.standardize:
            X = self.scaler.transform(X)
            
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Predicted probabilities of shape (n_samples, 2)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
            
        X = np.asarray(X)
        if self.standardize:
            X = self.scaler.transform(X)
            
        return self.model.predict_proba(X)
    
    def get_coefficients(self) -> Dict[str, float]:
        """Get model coefficients.
        
        Returns:
            Dictionary mapping feature names (or indices) to coefficients
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting coefficients")
            
        coef_dict = {}
        for i, coef in enumerate(self.model.coef_[0]):
            if self.feature_names:
                key = self.feature_names[i]
            else:
                key = f"feature_{i}"
            coef_dict[key] = coef
            
        return coef_dict
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        params = super().get_params()
        params.update({
            "standardize": self.standardize,
            "intercept": self.model.intercept_[0] if self.is_fitted else None,
            "coefficients": self.get_coefficients() if self.is_fitted else None,
        })
        return params
