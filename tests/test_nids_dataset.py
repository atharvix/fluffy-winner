"""Unit and integration tests for UNSW-NB15 NIDS dataset pipeline."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.nids_dataset import (
    RAW_CATEGORICAL_FEATURES,
    RAW_INPUT_FEATURES,
    RAW_NUMERICAL_FEATURES,
    TARGET_LABEL_MAPPING,
    TARGET_NAMES,
    create_nids_dataloaders,
    prepare_nids_data,
)


def test_nids_dataset_schema_and_target_mapping():
    """Verify target mapping semantics, class labels, and complete absence of target leakage."""
    assert TARGET_LABEL_MAPPING[0] == "BENIGN"
    assert TARGET_LABEL_MAPPING[1] == "ATTACK"
    assert TARGET_NAMES == ["BENIGN", "ATTACK"]

    # Verify input features do NOT contain identifiers or targets
    assert "id" not in RAW_INPUT_FEATURES
    assert "label" not in RAW_INPUT_FEATURES
    assert "attack_cat" not in RAW_INPUT_FEATURES

    # Verify total feature count
    assert len(RAW_INPUT_FEATURES) == 42
    assert len(RAW_NUMERICAL_FEATURES) == 39
    assert len(RAW_CATEGORICAL_FEATURES) == 3


def test_nids_split_reproducibility():
    """Verify identical random seeds yield bitwise identical data splits and indices."""
    splits_a = prepare_nids_data(sample_size=300, random_seed=42)
    splits_b = prepare_nids_data(sample_size=300, random_seed=42)

    np.testing.assert_array_equal(splits_a.train_indices, splits_b.train_indices)
    np.testing.assert_array_equal(splits_a.val_indices, splits_b.val_indices)
    np.testing.assert_array_equal(splits_a.test_indices, splits_b.test_indices)
    np.testing.assert_array_almost_equal(splits_a.X_train, splits_b.X_train)
    np.testing.assert_array_equal(splits_a.y_train, splits_b.y_train)


def test_nids_split_disjointness():
    """Verify train, validation, and test indices are strictly mutually disjoint."""
    splits = prepare_nids_data(sample_size=500, random_seed=123)

    train_set = set(splits.train_indices)
    val_set = set(splits.val_indices)
    test_set = set(splits.test_indices)

    assert len(train_set.intersection(val_set)) == 0, "Train and Val indices overlap!"
    assert len(train_set.intersection(test_set)) == 0, "Train and Test indices overlap!"
    assert len(val_set.intersection(test_set)) == 0, "Val and Test indices overlap!"


def test_nids_leakage_prevention():
    """Verify ColumnTransformer is fitted strictly on training data partition."""
    splits = prepare_nids_data(sample_size=600, random_seed=42)

    # Preprocessor scaler means should match training raw features, NOT total or test
    num_transformer = splits.preprocessor.named_transformers_["num"]
    fitted_means = num_transformer.mean_

    train_raw_means = splits.X_train_raw[RAW_NUMERICAL_FEATURES].to_numpy().mean(axis=0)
    test_raw_means = splits.X_test_raw[RAW_NUMERICAL_FEATURES].to_numpy().mean(axis=0)

    np.testing.assert_allclose(fitted_means, train_raw_means, rtol=1e-5, atol=1e-5)
    assert not np.allclose(fitted_means, test_raw_means, rtol=1e-4, atol=1e-4)


def test_nids_handling_unseen_categories():
    """Verify that unseen categorical values in inference/test data are handled gracefully without error."""
    splits = prepare_nids_data(sample_size=400, random_seed=42)

    # Create dummy DataFrame with completely novel unseen categories
    novel_df = splits.X_test_raw.iloc[:5].copy()
    novel_df["proto"] = "quantum_proto_unknown"
    novel_df["service"] = "unseen_service_999"
    novel_df["state"] = "UNSEEN_STATE"

    # Should transform without raising errors or creating NaNs
    transformed = splits.preprocessor.transform(novel_df)
    assert transformed.shape == (5, splits.X_train.shape[1])
    assert np.isfinite(transformed).all()


def test_nids_dataloaders_and_batch_shapes():
    """Verify DataLoader yields float32 tensors with expected batch dimension."""
    splits = prepare_nids_data(sample_size=300, random_seed=42)
    train_loader, val_loader, test_loader = create_nids_dataloaders(splits, batch_size=32)

    x_b, y_b = next(iter(train_loader))
    assert isinstance(x_b, torch.Tensor)
    assert isinstance(y_b, torch.Tensor)
    assert x_b.dtype == torch.float32
    assert y_b.dtype == torch.float32
    assert x_b.ndim == 2
    assert y_b.ndim == 2
    assert y_b.shape[1] == 1
    assert x_b.shape[1] == splits.X_train.shape[1]
