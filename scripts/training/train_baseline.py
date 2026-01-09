#!/usr/bin/env python
"""Training script for baseline models on longitudinal data.

This script demonstrates how to:
1. Load and prepare data
2. Train baseline models (linear/logistic regression)
3. Evaluate model performance
4. Save results
"""

import sys
import os
import argparse
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data import DataLoader
from src.models import LinearModel, LogisticModel
from src.evaluation import ModelEvaluator, cross_validate_temporal
from src.visualization import plot_model_performance, plot_predictions_vs_actual
import matplotlib.pyplot as plt


def main():
    """Train and evaluate baseline models."""
    parser = argparse.ArgumentParser(
        description="Train baseline models on longitudinal data"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to data file. If not provided, uses synthetic data."
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["regression", "classification"],
        default="regression",
        help="Prediction task"
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target column name"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/models",
        help="Directory to save outputs"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load or create data
    loader = DataLoader()
    
    if args.data_path:
        print(f"Loading data from {args.data_path}...")
        if args.data_path.endswith('.csv'):
            dataset = loader.load_csv(args.data_path)
        else:
            dataset = loader.load_excel(args.data_path)
    else:
        print("Creating synthetic data...")
        dataset = loader.create_synthetic_data(
            n_subjects=100,
            n_timepoints=5
        )
    
    # Set target column
    if args.target is None:
        if args.task == "regression":
            target_col = "outcome_continuous"
        else:
            target_col = "outcome_binary"
    else:
        target_col = args.target
    
    # Get feature columns
    feature_cols = dataset.get_feature_columns(
        exclude=[target_col, 'outcome_binary', 'outcome_continuous']
    )
    
    print(f"\nTask: {args.task}")
    print(f"Target: {target_col}")
    print(f"Features: {len(feature_cols)} features")
    print(f"Feature names: {feature_cols[:5]}...")
    
    # Prepare temporal split
    all_timepoints = sorted(dataset.data[dataset.time_col].unique())
    n_train = int(len(all_timepoints) * 0.7)
    train_timepoints = all_timepoints[:n_train]
    test_timepoints = all_timepoints[n_train:]
    
    print(f"\nTemporal split:")
    print(f"  Train timepoints: {train_timepoints}")
    print(f"  Test timepoints: {test_timepoints}")
    
    # Create and train model
    if args.task == "regression":
        model = LinearModel(name="LinearRegression")
    else:
        model = LogisticModel(name="LogisticRegression")
    
    print(f"\nTraining {model.name}...")
    results = cross_validate_temporal(
        model=model,
        dataset=dataset,
        feature_cols=feature_cols,
        target_col=target_col,
        train_timepoints=train_timepoints,
        test_timepoints=test_timepoints,
        task=args.task
    )
    
    # Print results
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"Training samples: {results['n_train']}")
    print(f"Test samples: {results['n_test']}")
    print("\nMetrics:")
    for metric_name, metric_value in results['metrics'].items():
        if metric_name != "confusion_matrix":
            print(f"  {metric_name}: {metric_value:.4f}")
    
    # Save results
    results_path = output_dir / f"{args.task}_results.json"
    with open(results_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        serializable_results = {
            'n_train': int(results['n_train']),
            'n_test': int(results['n_test']),
            'train_timepoints': [int(t) for t in results['train_timepoints']],
            'test_timepoints': [int(t) for t in results['test_timepoints']],
            'metrics': {}
        }
        for k, v in results['metrics'].items():
            if k == 'confusion_matrix':
                serializable_results['metrics'][k] = v
            else:
                serializable_results['metrics'][k] = float(v)
        json.dump(serializable_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Plot predictions vs actual
    test_data = dataset.data[dataset.data[dataset.time_col].isin(test_timepoints)]
    X_test = test_data[feature_cols].values
    y_test = test_data[target_col].values
    y_pred = model.predict(X_test)
    
    fig = plot_predictions_vs_actual(
        y_test,
        y_pred,
        task=args.task,
        title=f"{model.name}: Predictions vs Actual"
    )
    plot_path = output_dir / f"{args.task}_predictions.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {plot_path}")
    
    # Print model coefficients
    if hasattr(model, 'get_coefficients'):
        print("\n" + "="*60)
        print("MODEL COEFFICIENTS (Top 10)")
        print("="*60)
        coefs = model.get_coefficients()
        sorted_coefs = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)
        for feature, coef in sorted_coefs[:10]:
            print(f"  {feature}: {coef:.4f}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
