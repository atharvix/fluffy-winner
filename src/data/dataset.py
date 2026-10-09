"""Dataset pipeline for Breast Cancer Wisconsin Diagnostic dataset.

Explicit target label mapping:
    0 -> malignant (cancerous / positive for malignancy in clinical context)
    1 -> benign (non-cancerous)

This module handles:
1. Loading the dataset from sklearn.datasets.load_breast_cancer.
2. Stratified and reproducible train/val/test splitting.
3. Fitting preprocessing scaler strictly on training data and transforming splits.
4. Wrapping data into PyTorch Datasets and DataLoaders.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Explicit label mapping - never infer semantics solely from numeric indices
TARGET_LABEL_MAPPING: Dict[int, str] = {
    0: "malignant",
    1: "benign",
}

TARGET_NAMES: List[str] = ["malignant", "benign"]


@dataclass
class DatasetSplits:
    """Container holding raw and scaled dataset splits, scalers, and metadata."""
    X_train_raw: np.ndarray
    X_val_raw: np.ndarray
    X_test_raw: np.ndarray
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    scaler: StandardScaler
    feature_names: List[str]
    target_names: List[str]
    sample_counts: Dict[str, int]
    class_distribution: Dict[str, Dict[str, int]]


class BreastCancerDataset(Dataset):
    """PyTorch Dataset wrapper for breast cancer features and binary targets."""

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        """
        Args:
            features: 2D array of shape (n_samples, n_features).
            labels: 1D array of binary class labels of shape (n_samples,).
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def prepare_breast_cancer_data(
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> DatasetSplits:
    """
    Load Breast Cancer Wisconsin Diagnostic dataset, perform stratified train/val/test split,
    and fit preprocessing (StandardScaler) strictly on training partition to avoid data leakage.

    Args:
        train_ratio: Proportion of dataset for training (e.g. 0.70).
        val_ratio: Proportion of dataset for validation (e.g. 0.15).
        test_ratio: Proportion of dataset for testing (e.g. 0.15).
        random_seed: Fixed random seed for reproducible splitting.

    Returns:
        DatasetSplits dataclass instance containing all split arrays, scaler, and metadata.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(
            f"Split ratios must sum to 1.0, got {train_ratio} + {val_ratio} + {test_ratio} = {train_ratio + val_ratio + test_ratio}"
        )

    bunch = load_breast_cancer()
    X: np.ndarray = bunch.data
    y: np.ndarray = bunch.target
    feature_names: List[str] = list(bunch.feature_names)
    target_names: List[str] = [TARGET_LABEL_MAPPING[i] for i in range(len(TARGET_LABEL_MAPPING))]

    n_samples = len(X)
    indices = np.arange(n_samples)

    # First split: train vs temp (val + test)
    temp_ratio = val_ratio + test_ratio
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        indices,
        y,
        test_size=temp_ratio,
        stratify=y,
        random_state=random_seed,
    )

    # Second split: val vs test from temp
    val_share_of_temp = val_ratio / temp_ratio
    val_idx_rel, test_idx_rel, y_val, y_test = train_test_split(
        np.arange(len(temp_idx)),
        y_temp,
        test_size=(1.0 - val_share_of_temp),
        stratify=y_temp,
        random_state=random_seed,
    )

    val_idx = temp_idx[val_idx_rel]
    test_idx = temp_idx[test_idx_rel]

    X_train_raw = X[train_idx]
    X_val_raw = X[val_idx]
    X_test_raw = X[test_idx]

    # Fit preprocessing scaler ONLY on training data to prevent leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    def get_class_counts(labels: np.ndarray) -> Dict[str, int]:
        counts = {}
        for cls_id, cls_name in TARGET_LABEL_MAPPING.items():
            counts[cls_name] = int(np.sum(labels == cls_id))
        return counts

    sample_counts = {
        "total": int(n_samples),
        "train": int(len(train_idx)),
        "val": int(len(val_idx)),
        "test": int(len(test_idx)),
    }

    class_distribution = {
        "train": get_class_counts(y_train),
        "val": get_class_counts(y_val),
        "test": get_class_counts(y_test),
        "total": get_class_counts(y),
    }

    return DatasetSplits(
        X_train_raw=X_train_raw,
        X_val_raw=X_val_raw,
        X_test_raw=X_test_raw,
        X_train=X_train_scaled,
        X_val=X_val_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        scaler=scaler,
        feature_names=feature_names,
        target_names=target_names,
        sample_counts=sample_counts,
        class_distribution=class_distribution,
    )


def create_dataloaders(
    splits: DatasetSplits,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train, validation, and test splits.

    Args:
        splits: DatasetSplits object.
        batch_size: Batch size for loaders.
        num_workers: Number of workers for data loading.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_dataset = BreastCancerDataset(splits.X_train, splits.y_train)
    val_dataset = BreastCancerDataset(splits.X_val, splits.y_val)
    test_dataset = BreastCancerDataset(splits.X_test, splits.y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
