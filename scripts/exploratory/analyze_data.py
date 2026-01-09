#!/usr/bin/env python
"""Exploratory data analysis script for longitudinal mental health data.

This script demonstrates how to:
1. Load longitudinal data
2. Compute basic statistics
3. Visualize temporal patterns
4. Explore feature relationships
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data import DataLoader
from src.visualization import (
    plot_longitudinal,
    plot_feature_distributions,
    plot_correlation_matrix
)
import matplotlib.pyplot as plt


def main():
    """Run exploratory data analysis."""
    parser = argparse.ArgumentParser(
        description="Exploratory data analysis for longitudinal data"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to data file (CSV or Excel). If not provided, uses synthetic data."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/exploratory",
        help="Directory to save output figures"
    )
    parser.add_argument(
        "--n-subjects",
        type=int,
        default=100,
        help="Number of subjects for synthetic data"
    )
    parser.add_argument(
        "--n-timepoints",
        type=int,
        default=5,
        help="Number of timepoints for synthetic data"
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
            n_subjects=args.n_subjects,
            n_timepoints=args.n_timepoints
        )
    
    # Print dataset summary
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    summary = dataset.summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Print basic statistics
    print("\n" + "="*60)
    print("FEATURE STATISTICS")
    print("="*60)
    feature_cols = dataset.get_feature_columns(exclude=['outcome_binary', 'outcome_continuous'])
    print(dataset.data[feature_cols].describe())
    
    # Plot longitudinal trajectories
    print("\nGenerating longitudinal trajectory plots...")
    fig = plot_longitudinal(
        dataset,
        n_subjects=10,
        feature_col='outcome_continuous',
        title="Example Longitudinal Trajectories"
    )
    plt.savefig(output_dir / "longitudinal_trajectories.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'longitudinal_trajectories.png'}")
    
    # Plot feature distributions
    print("\nGenerating feature distribution plots...")
    fig = plot_feature_distributions(
        dataset,
        feature_cols=feature_cols[:9]  # Limit to 9 features
    )
    plt.savefig(output_dir / "feature_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'feature_distributions.png'}")
    
    # Plot correlation matrix
    print("\nGenerating correlation matrix...")
    fig = plot_correlation_matrix(
        dataset,
        feature_cols=feature_cols
    )
    plt.savefig(output_dir / "correlation_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'correlation_matrix.png'}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
