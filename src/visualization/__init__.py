"""Visualization utilities for longitudinal mental health data.

This module provides plotting functions for exploratory data analysis
and model evaluation.
"""

from typing import Optional, List, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Set default plotting style
sns.set_style("whitegrid")
sns.set_palette("husl")


def plot_longitudinal(
    dataset: Any,
    subject_ids: Optional[List[Any]] = None,
    feature_col: str = None,
    n_subjects: int = 10,
    figsize: Tuple[int, int] = (12, 6),
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot longitudinal trajectories for multiple subjects.
    
    Args:
        dataset: LongitudinalDataset instance
        subject_ids: Specific subject IDs to plot (if None, random sample)
        feature_col: Feature column to plot
        n_subjects: Number of subjects to plot if subject_ids not specified
        figsize: Figure size
        title: Plot title
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    if feature_col is None:
        feature_cols = dataset.get_feature_columns()
        if not feature_cols:
            raise ValueError("No feature columns found in dataset")
        feature_col = feature_cols[0]
    
    if subject_ids is None:
        all_subjects = dataset.data[dataset.subject_col].unique()
        subject_ids = np.random.choice(
            all_subjects,
            size=min(n_subjects, len(all_subjects)),
            replace=False
        )
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for subject_id in subject_ids:
        subject_data = dataset.get_subject_data(subject_id)
        subject_data = subject_data.sort_values(dataset.time_col)
        
        ax.plot(
            subject_data[dataset.time_col],
            subject_data[feature_col],
            marker='o',
            alpha=0.6,
            label=f"Subject {subject_id}"
        )
    
    ax.set_xlabel("Timepoint", fontsize=12)
    ax.set_ylabel(feature_col, fontsize=12)
    
    if title is None:
        title = f"Longitudinal Trajectories: {feature_col}"
    ax.set_title(title, fontsize=14)
    
    # Only show legend if not too many subjects
    if len(subject_ids) <= 15:
        ax.legend(loc='best', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_feature_distributions(
    dataset: Any,
    feature_cols: Optional[List[str]] = None,
    n_cols: int = 3,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot distributions of multiple features.
    
    Args:
        dataset: LongitudinalDataset instance
        feature_cols: List of features to plot (if None, plot all)
        n_cols: Number of columns in subplot grid
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    if feature_cols is None:
        feature_cols = dataset.get_feature_columns()[:12]  # Limit to 12
    
    n_features = len(feature_cols)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    if figsize is None:
        figsize = (n_cols * 4, n_rows * 3)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, feature_col in enumerate(feature_cols):
        ax = axes[idx]
        data = dataset.data[feature_col].dropna()
        
        ax.hist(data, bins=30, alpha=0.7, edgecolor='black')
        ax.set_xlabel(feature_col, fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_title(f"Distribution: {feature_col}", fontsize=11)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_correlation_matrix(
    dataset: Any,
    feature_cols: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot correlation matrix of features.
    
    Args:
        dataset: LongitudinalDataset instance
        feature_cols: List of features to include (if None, use all)
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    if feature_cols is None:
        feature_cols = dataset.get_feature_columns()
    
    corr_matrix = dataset.data[feature_cols].corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    
    ax.set_title("Feature Correlation Matrix", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_model_performance(
    results: dict,
    metric: str = "rmse",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot model performance comparison.
    
    Args:
        results: Dictionary mapping model names to evaluation results
        metric: Metric to plot
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    model_names = list(results.keys())
    metric_values = []
    
    for model_name in model_names:
        if metric in results[model_name]:
            metric_values.append(results[model_name][metric])
        else:
            metric_values.append(None)
    
    # Filter out None values
    valid_pairs = [(name, val) for name, val in zip(model_names, metric_values) if val is not None]
    if not valid_pairs:
        raise ValueError(f"Metric '{metric}' not found in any results")
    
    model_names, metric_values = zip(*valid_pairs)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = sns.color_palette("husl", len(model_names))
    bars = ax.bar(range(len(model_names)), metric_values, color=colors, alpha=0.7)
    
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(f"Model Comparison: {metric.upper()}", fontsize=14)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{value:.3f}',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str = "regression",
    figsize: Tuple[int, int] = (8, 8),
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot predicted vs actual values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        task: Either "regression" or "classification"
        figsize: Figure size
        title: Plot title
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if task == "regression":
        ax.scatter(y_true, y_pred, alpha=0.5)
        
        # Add diagonal line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
        
        ax.set_xlabel("Actual Values", fontsize=12)
        ax.set_ylabel("Predicted Values", fontsize=12)
        
        if title is None:
            title = "Predicted vs Actual Values"
        
        # Calculate and display R²
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
    elif task == "classification":
        # For classification, show confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        
        if title is None:
            title = "Confusion Matrix"
    
    ax.set_title(title, fontsize=14)
    ax.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
