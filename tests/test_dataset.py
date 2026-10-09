"""Tests for the Breast Cancer dataset pipeline, preprocessing leakage prevention, and reproducibility."""

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

from src.data.dataset import (
    TARGET_LABEL_MAPPING,
    TARGET_NAMES,
    BreastCancerDataset,
    create_dataloaders,
    prepare_breast_cancer_data,
)


def test_dataset_total_samples_and_shapes():
    """Verify that dataset splits cover all 569 samples and have exactly 30 features."""
    raw_data = load_breast_cancer()
    total_samples, total_features = raw_data.data.shape
    assert total_samples == 569
    assert total_features == 30

    splits = prepare_breast_cancer_data(
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
    )

    n_train = len(splits.y_train)
    n_val = len(splits.y_val)
    n_test = len(splits.y_test)

    # Check sample conservation
    assert n_train + n_val + n_test == total_samples
    assert splits.sample_counts["total"] == total_samples
    assert splits.sample_counts["train"] == n_train
    assert splits.sample_counts["val"] == n_val
    assert splits.sample_counts["test"] == n_test

    # Check array shapes
    assert splits.X_train.shape == (n_train, 30)
    assert splits.X_val.shape == (n_val, 30)
    assert splits.X_test.shape == (n_test, 30)
    assert splits.y_train.shape == (n_train,)
    assert splits.y_val.shape == (n_val,)
    assert splits.y_test.shape == (n_test,)


def test_target_label_mapping_and_semantics():
    """Verify that target label mapping explicitly identifies 0 as malignant and 1 as benign."""
    assert TARGET_LABEL_MAPPING[0] == "malignant"
    assert TARGET_LABEL_MAPPING[1] == "benign"
    assert len(TARGET_LABEL_MAPPING) == 2

    splits = prepare_breast_cancer_data(random_seed=42)
    for partition_name, y_arr in [("train", splits.y_train), ("val", splits.y_val), ("test", splits.y_test)]:
        unique_labels = set(np.unique(y_arr))
        assert unique_labels.issubset({0, 1}), f"Unexpected labels in {partition_name}: {unique_labels}"


def test_split_disjointness():
    """Verify that train, validation, and test splits have completely disjoint index sets."""
    splits = prepare_breast_cancer_data(random_seed=42)

    train_set = set(splits.train_indices.tolist())
    val_set = set(splits.val_indices.tolist())
    test_set = set(splits.test_indices.tolist())

    assert len(train_set.intersection(val_set)) == 0, "Train and validation sets overlap!"
    assert len(train_set.intersection(test_set)) == 0, "Train and test sets overlap!"
    assert len(val_set.intersection(test_set)) == 0, "Validation and test sets overlap!"

    all_indices = train_set.union(val_set).union(test_set)
    assert all_indices == set(range(569)), "Indices do not cover the full dataset!"


def test_split_stratification():
    """Verify that class ratios are preserved across train, validation, and test partitions."""
    raw_data = load_breast_cancer()
    global_benign_ratio = np.mean(raw_data.target == 1)

    splits = prepare_breast_cancer_data(random_seed=42)
    train_benign_ratio = np.mean(splits.y_train == 1)
    val_benign_ratio = np.mean(splits.y_val == 1)
    test_benign_ratio = np.mean(splits.y_test == 1)

    # Allow tiny rounding deviation due to integer sample counts
    assert abs(train_benign_ratio - global_benign_ratio) < 0.02
    assert abs(val_benign_ratio - global_benign_ratio) < 0.03
    assert abs(test_benign_ratio - global_benign_ratio) < 0.03


def test_split_reproducibility():
    """Verify that identical random seeds produce identical splits and data arrays."""
    splits_run1 = prepare_breast_cancer_data(random_seed=123)
    splits_run2 = prepare_breast_cancer_data(random_seed=123)
    splits_run3 = prepare_breast_cancer_data(random_seed=999)

    np.testing.assert_array_equal(splits_run1.train_indices, splits_run2.train_indices)
    np.testing.assert_array_equal(splits_run1.val_indices, splits_run2.val_indices)
    np.testing.assert_array_equal(splits_run1.test_indices, splits_run2.test_indices)
    np.testing.assert_allclose(splits_run1.X_train, splits_run2.X_train)

    # Different seed must yield different split
    assert not np.array_equal(splits_run1.train_indices, splits_run3.train_indices)


def test_prevention_of_preprocessing_leakage():
    """
    Verify that the scaler is fit STRICTLY on training data and not on validation/test data.
    
    Checks:
    1. Scaler mean and scale match training partition statistics exactly.
    2. Scaler mean and scale differ from full dataset statistics (since val/test samples are excluded).
    3. Scaler transformation on X_train results in mean ~0 and std ~1 for X_train.
    """
    splits = prepare_breast_cancer_data(random_seed=42)

    train_mean = splits.X_train_raw.mean(axis=0)
    train_var = splits.X_train_raw.var(axis=0)

    # Scaler must match training set statistics
    np.testing.assert_allclose(splits.scaler.mean_, train_mean, rtol=1e-5)
    np.testing.assert_allclose(splits.scaler.var_, train_var, rtol=1e-5)

    # Scaler mean must not match whole dataset mean (leakage check)
    full_data = load_breast_cancer().data
    full_mean = full_data.mean(axis=0)
    assert not np.allclose(splits.scaler.mean_, full_mean, atol=1e-3), (
        "Scaler mean matches full dataset mean, indicating potential data leakage!"
    )

    # Scaled training data must have mean ~ 0 and std ~ 1
    np.testing.assert_allclose(splits.X_train.mean(axis=0), np.zeros(30), atol=1e-6)
    np.testing.assert_allclose(splits.X_train.std(axis=0), np.ones(30), atol=1e-6)


def test_pytorch_dataset_and_dataloaders():
    """Verify PyTorch Dataset and DataLoader batching and tensor dimensions."""
    splits = prepare_breast_cancer_data(random_seed=42)
    train_loader, val_loader, test_loader = create_dataloaders(splits, batch_size=16)

    # Test train loader
    x_batch, y_batch = next(iter(train_loader))
    assert x_batch.shape == (16, 30)
    assert y_batch.shape == (16, 1)
    assert x_batch.dtype.is_floating_point
    assert y_batch.dtype.is_floating_point

    # Ensure total samples loaded across all batches equals split sizes
    total_train_loaded = sum(len(y) for _, y in train_loader)
    total_val_loaded = sum(len(y) for _, y in val_loader)
    total_test_loaded = sum(len(y) for _, y in test_loader)

    assert total_train_loaded == len(splits.y_train)
    assert total_val_loaded == len(splits.y_val)
    assert total_test_loaded == len(splits.y_test)
