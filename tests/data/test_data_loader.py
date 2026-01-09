"""Tests for data loading utilities."""

import pytest
import numpy as np
import pandas as pd
from src.data import DataLoader, LongitudinalDataset, split_temporal


class TestLongitudinalDataset:
    """Tests for LongitudinalDataset class."""
    
    def setup_method(self):
        """Set up test data."""
        self.data = pd.DataFrame({
            'subject_id': [1, 1, 2, 2, 3, 3],
            'timepoint': [0, 1, 0, 1, 0, 1],
            'feature_1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'feature_2': [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
            'outcome': [0, 1, 1, 1, 0, 0]
        })
        self.dataset = LongitudinalDataset(self.data)
    
    def test_init(self):
        """Test initialization."""
        assert self.dataset.subject_col == 'subject_id'
        assert self.dataset.time_col == 'timepoint'
        assert len(self.dataset.data) == 6
    
    def test_get_subject_data(self):
        """Test getting data for a specific subject."""
        subject_data = self.dataset.get_subject_data(1)
        assert len(subject_data) == 2
        assert all(subject_data['subject_id'] == 1)
    
    def test_get_timepoint_data(self):
        """Test getting data for a specific timepoint."""
        timepoint_data = self.dataset.get_timepoint_data(0)
        assert len(timepoint_data) == 3
        assert all(timepoint_data['timepoint'] == 0)
    
    def test_get_feature_columns(self):
        """Test getting feature column names."""
        feature_cols = self.dataset.get_feature_columns(exclude=['outcome'])
        assert 'feature_1' in feature_cols
        assert 'feature_2' in feature_cols
        assert 'subject_id' not in feature_cols
        assert 'timepoint' not in feature_cols
        assert 'outcome' not in feature_cols
    
    def test_summary(self):
        """Test dataset summary."""
        summary = self.dataset.summary()
        assert summary['n_subjects'] == 3
        assert summary['n_timepoints'] == 2
        assert summary['n_observations'] == 6
        assert summary['timepoint_range'] == (0, 1)


class TestDataLoader:
    """Tests for DataLoader class."""
    
    def test_init(self):
        """Test initialization."""
        loader = DataLoader()
        assert loader.data_dir is not None
    
    def test_create_synthetic_data(self):
        """Test synthetic data creation."""
        loader = DataLoader()
        dataset = loader.create_synthetic_data(
            n_subjects=10,
            n_timepoints=3,
            n_features=5
        )
        
        assert len(dataset.data) == 30  # 10 subjects * 3 timepoints
        assert dataset.data['subject_id'].nunique() == 10
        assert dataset.data['timepoint'].nunique() == 3
        
        # Check that features were created
        feature_cols = [col for col in dataset.data.columns if col.startswith('feature_')]
        assert len(feature_cols) == 5
        
        # Check that outcomes exist
        assert 'outcome_binary' in dataset.data.columns
        assert 'outcome_continuous' in dataset.data.columns
    
    def test_synthetic_data_reproducibility(self):
        """Test that synthetic data is reproducible with same seed."""
        loader = DataLoader()
        dataset1 = loader.create_synthetic_data(random_seed=42)
        dataset2 = loader.create_synthetic_data(random_seed=42)
        
        pd.testing.assert_frame_equal(dataset1.data, dataset2.data)


class TestSplitTemporal:
    """Tests for temporal splitting."""
    
    def test_split_temporal(self):
        """Test temporal data splitting."""
        data = pd.DataFrame({
            'subject_id': [1, 1, 1, 2, 2, 2],
            'timepoint': [0, 1, 2, 0, 1, 2],
            'feature': [1, 2, 3, 4, 5, 6],
        })
        dataset = LongitudinalDataset(data)
        
        train_dataset, test_dataset = split_temporal(
            dataset,
            train_timepoints=[0, 1],
            test_timepoints=[2]
        )
        
        assert len(train_dataset.data) == 4
        assert len(test_dataset.data) == 2
        assert all(train_dataset.data['timepoint'].isin([0, 1]))
        assert all(test_dataset.data['timepoint'] == 2)
