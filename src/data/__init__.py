"""Data loading and processing utilities for longitudinal mental health data."""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import pandas as pd


class LongitudinalDataset:
    """Container for longitudinal mental health data.
    
    This class provides a structured way to store and access longitudinal
    data with multiple timepoints per subject.
    
    Attributes:
        data (pd.DataFrame): The main data containing features and targets
        subject_col (str): Column name for subject identifiers
        time_col (str): Column name for time points
        metadata (dict): Additional metadata about the dataset
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        subject_col: str = "subject_id",
        time_col: str = "timepoint",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize the LongitudinalDataset.
        
        Args:
            data: DataFrame containing the longitudinal data
            subject_col: Name of the column containing subject IDs
            time_col: Name of the column containing timepoints
            metadata: Optional dictionary with dataset metadata
        """
        self.data = data
        self.subject_col = subject_col
        self.time_col = time_col
        self.metadata = metadata or {}
        
    def get_subject_data(self, subject_id: Any) -> pd.DataFrame:
        """Get all timepoints for a specific subject.
        
        Args:
            subject_id: The subject identifier
            
        Returns:
            DataFrame containing all timepoints for the subject
        """
        return self.data[self.data[self.subject_col] == subject_id]
    
    def get_timepoint_data(self, timepoint: Any) -> pd.DataFrame:
        """Get data for all subjects at a specific timepoint.
        
        Args:
            timepoint: The timepoint identifier
            
        Returns:
            DataFrame containing data for all subjects at the timepoint
        """
        return self.data[self.data[self.time_col] == timepoint]
    
    def get_feature_columns(self, exclude: Optional[List[str]] = None) -> List[str]:
        """Get list of feature column names.
        
        Args:
            exclude: List of columns to exclude (in addition to subject and time cols)
            
        Returns:
            List of feature column names
        """
        exclude_cols = {self.subject_col, self.time_col}
        if exclude:
            exclude_cols.update(exclude)
        return [col for col in self.data.columns if col not in exclude_cols]
    
    def summary(self) -> Dict[str, Any]:
        """Generate a summary of the dataset.
        
        Returns:
            Dictionary containing dataset statistics
        """
        return {
            "n_subjects": self.data[self.subject_col].nunique(),
            "n_timepoints": self.data[self.time_col].nunique(),
            "n_observations": len(self.data),
            "n_features": len(self.get_feature_columns()),
            "timepoint_range": (
                self.data[self.time_col].min(),
                self.data[self.time_col].max()
            ),
            "missing_values": self.data.isnull().sum().sum(),
        }


class DataLoader:
    """Utility class for loading longitudinal mental health data.
    
    This class provides methods to load data from various file formats
    and convert them into LongitudinalDataset objects.
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """Initialize the DataLoader.
        
        Args:
            data_dir: Base directory for data files (defaults to ./data)
        """
        if data_dir is None:
            data_dir = os.path.join(os.getcwd(), "data")
        self.data_dir = Path(data_dir)
        
    def load_csv(
        self,
        filepath: str,
        subject_col: str = "subject_id",
        time_col: str = "timepoint",
        **kwargs
    ) -> LongitudinalDataset:
        """Load longitudinal data from a CSV file.
        
        Args:
            filepath: Path to the CSV file (relative to data_dir or absolute)
            subject_col: Name of the subject ID column
            time_col: Name of the timepoint column
            **kwargs: Additional arguments passed to pd.read_csv
            
        Returns:
            LongitudinalDataset object containing the loaded data
        """
        path = Path(filepath)
        if not path.is_absolute():
            path = self.data_dir / path
            
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
            
        data = pd.read_csv(path, **kwargs)
        
        # Validate required columns
        if subject_col not in data.columns:
            raise ValueError(f"Subject column '{subject_col}' not found in data")
        if time_col not in data.columns:
            raise ValueError(f"Time column '{time_col}' not found in data")
            
        metadata = {
            "source_file": str(path),
            "loaded_at": pd.Timestamp.now(),
        }
        
        return LongitudinalDataset(data, subject_col, time_col, metadata)
    
    def load_excel(
        self,
        filepath: str,
        subject_col: str = "subject_id",
        time_col: str = "timepoint",
        sheet_name: int = 0,
        **kwargs
    ) -> LongitudinalDataset:
        """Load longitudinal data from an Excel file.
        
        Args:
            filepath: Path to the Excel file (relative to data_dir or absolute)
            subject_col: Name of the subject ID column
            time_col: Name of the timepoint column
            sheet_name: Sheet name or index to load
            **kwargs: Additional arguments passed to pd.read_excel
            
        Returns:
            LongitudinalDataset object containing the loaded data
        """
        path = Path(filepath)
        if not path.is_absolute():
            path = self.data_dir / path
            
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
            
        data = pd.read_excel(path, sheet_name=sheet_name, **kwargs)
        
        # Validate required columns
        if subject_col not in data.columns:
            raise ValueError(f"Subject column '{subject_col}' not found in data")
        if time_col not in data.columns:
            raise ValueError(f"Time column '{time_col}' not found in data")
            
        metadata = {
            "source_file": str(path),
            "sheet_name": sheet_name,
            "loaded_at": pd.Timestamp.now(),
        }
        
        return LongitudinalDataset(data, subject_col, time_col, metadata)
    
    def create_synthetic_data(
        self,
        n_subjects: int = 100,
        n_timepoints: int = 5,
        n_features: int = 10,
        random_seed: Optional[int] = 42
    ) -> LongitudinalDataset:
        """Create synthetic longitudinal data for testing and demonstration.
        
        Args:
            n_subjects: Number of subjects
            n_timepoints: Number of timepoints per subject
            n_features: Number of feature variables
            random_seed: Random seed for reproducibility
            
        Returns:
            LongitudinalDataset object with synthetic data
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            
        # Create subject and timepoint arrays
        subjects = np.repeat(np.arange(n_subjects), n_timepoints)
        timepoints = np.tile(np.arange(n_timepoints), n_subjects)
        
        # Create feature data with temporal trends
        data_dict = {
            "subject_id": subjects,
            "timepoint": timepoints,
        }
        
        for i in range(n_features):
            # Add some temporal trend and individual variation
            feature_values = (
                0.5 * timepoints +  # temporal trend
                np.random.randn(len(subjects)) +  # noise
                np.repeat(np.random.randn(n_subjects), n_timepoints)  # individual effects
            )
            data_dict[f"feature_{i}"] = feature_values
            
        # Add a binary outcome (e.g., depression indicator)
        # Probability increases with time and some features
        prob = 1 / (1 + np.exp(-(
            -1.0 +
            0.2 * timepoints +
            0.3 * data_dict["feature_0"]
        )))
        data_dict["outcome_binary"] = (np.random.rand(len(subjects)) < prob).astype(int)
        
        # Add a continuous outcome
        data_dict["outcome_continuous"] = (
            2.0 +
            0.5 * timepoints +
            0.3 * data_dict["feature_0"] +
            0.2 * data_dict["feature_1"] +
            np.random.randn(len(subjects))
        )
        
        data = pd.DataFrame(data_dict)
        
        metadata = {
            "type": "synthetic",
            "n_subjects": n_subjects,
            "n_timepoints": n_timepoints,
            "n_features": n_features,
            "random_seed": random_seed,
            "created_at": pd.Timestamp.now(),
        }
        
        return LongitudinalDataset(data, "subject_id", "timepoint", metadata)


def split_temporal(
    dataset: LongitudinalDataset,
    train_timepoints: List[Any],
    test_timepoints: List[Any]
) -> Tuple[LongitudinalDataset, LongitudinalDataset]:
    """Split dataset into train and test sets based on timepoints.
    
    This is useful for temporal validation where we train on earlier
    timepoints and test on later ones.
    
    Args:
        dataset: The dataset to split
        train_timepoints: List of timepoints for training
        test_timepoints: List of timepoints for testing
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    train_data = dataset.data[
        dataset.data[dataset.time_col].isin(train_timepoints)
    ].copy()
    test_data = dataset.data[
        dataset.data[dataset.time_col].isin(test_timepoints)
    ].copy()
    
    train_dataset = LongitudinalDataset(
        train_data,
        dataset.subject_col,
        dataset.time_col,
        {**dataset.metadata, "split": "train"}
    )
    test_dataset = LongitudinalDataset(
        test_data,
        dataset.subject_col,
        dataset.time_col,
        {**dataset.metadata, "split": "test"}
    )
    
    return train_dataset, test_dataset
