"""Dataset pipeline for UNSW-NB15 Network Intrusion Detection dataset.

Explicit target label mapping:
    0 -> BENIGN (legitimate network traffic)
    1 -> ATTACK (malicious network traffic / intrusion attempts)

Positive class convention:
    Class 1 (ATTACK) is the target condition of interest in network security.
    High attack recall (detection rate) ensures malicious activities are caught.
    Low false-positive rate (FPR) prevents security operations alert fatigue.

This module provides:
1. Loading and validation of the UNSW-NB15 dataset.
2. Separation of input features from target labels and identifiers (zero target leakage).
3. Stratified, reproducible train / validation / test partitioning before any preprocessing.
4. Train-fitted ColumnTransformer (StandardScaler on numerical features, OneHotEncoder on categoricals).
5. PyTorch Dataset and DataLoader wrappers.
6. Persistence and restoration of fitted preprocessing artifacts for production inference.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset


# Explicit target mapping for Network Intrusion Detection
TARGET_LABEL_MAPPING: Dict[int, str] = {
    0: "BENIGN",
    1: "ATTACK",
}

TARGET_NAMES: List[str] = ["BENIGN", "ATTACK"]

# Features explicitly separated from identifiers and targets
IDENTIFIER_COLUMNS: List[str] = ["id"]
TARGET_COLUMNS: List[str] = ["label", "attack_cat"]

RAW_CATEGORICAL_FEATURES: List[str] = [
    "proto",
    "service",
    "state",
]

RAW_NUMERICAL_FEATURES: List[str] = [
    "dur",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sttl",
    "dttl",
    "sload",
    "dload",
    "sloss",
    "dloss",
    "sinpkt",
    "dinpkt",
    "sjit",
    "djit",
    "swin",
    "stcpb",
    "dtcpb",
    "dwin",
    "tcprtt",
    "synack",
    "ackdat",
    "smean",
    "dmean",
    "trans_depth",
    "response_body_len",
    "ct_srv_src",
    "ct_state_ttl",
    "ct_dst_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_flw_http_mthd",
    "ct_src_ltm",
    "ct_srv_dst",
    "is_sm_ips_ports",
]

RAW_INPUT_FEATURES: List[str] = RAW_NUMERICAL_FEATURES + RAW_CATEGORICAL_FEATURES


@dataclass
class NIDSSplits:
    """Container holding raw and scaled dataset splits, preprocessor, and metadata."""
    X_train_raw: pd.DataFrame
    X_val_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    preprocessor: ColumnTransformer
    raw_feature_names: List[str]
    encoded_feature_names: List[str]
    target_names: List[str]
    sample_counts: Dict[str, int]
    class_distribution: Dict[str, Dict[str, int]]
    dataset_metadata: Dict[str, Any]


class NIDSDataset(Dataset):
    """PyTorch Dataset wrapper for preprocessed network flow features and binary labels."""

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        """
        Args:
            features: 2D array of shape (n_samples, n_encoded_features).
            labels: 1D array of binary class labels (n_samples,).
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def load_raw_unsw_nb15(
    data_path: Union[str, Path] = "data/raw/UNSW_NB15_training-set.csv",
) -> pd.DataFrame:
    """
    Load raw UNSW-NB15 CSV and validate mandatory columns.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw UNSW-NB15 file not found at: {path}. "
            f"Ensure data/raw/UNSW_NB15_training-set.csv has been downloaded."
        )

    df = pd.read_csv(path)

    # Validate essential targets and input columns
    if "label" not in df.columns:
        raise ValueError("Missing mandatory 'label' target column in dataset.")

    missing_inputs = [col for col in RAW_INPUT_FEATURES if col not in df.columns]
    if missing_inputs:
        raise ValueError(f"Dataset missing required input features: {missing_inputs}")

    return df


def prepare_nids_data(
    data_path: Union[str, Path] = "data/raw/UNSW_NB15_training-set.csv",
    sample_size: Optional[int] = 6000,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    deduplicate: bool = False,
    max_categories_per_cat_feature: int = 10,
) -> NIDSSplits:
    """
    Load UNSW-NB15 dataset, perform stratified train/val/test splitting,
    and fit preprocessing (StandardScaler + OneHotEncoder) STRICTLY on training split
    to prevent data leakage.

    Args:
        data_path: Path to raw CSV.
        sample_size: If specified, extract a representative stratified subset of this size.
                     None uses full dataset.
        train_ratio: Fraction for training split (default 0.70).
        val_ratio: Fraction for validation split (default 0.15).
        test_ratio: Fraction for held-out test split (default 0.15).
        random_seed: Random seed for reproducible splitting.
        deduplicate: Whether to remove duplicate flow feature rows prior to splitting.
        max_categories_per_cat_feature: Max top categories for OneHotEncoder (others infrequent).

    Returns:
        NIDSSplits containing all split arrays, preprocessor, and schema metadata.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(
            f"Split ratios must sum to 1.0, got {train_ratio} + {val_ratio} + {test_ratio}"
        )

    df = load_raw_unsw_nb15(data_path)

    # Optional deduplication check
    raw_total_rows = len(df)
    if deduplicate:
        df = df.drop_duplicates(subset=RAW_INPUT_FEATURES, keep="first").reset_index(drop=True)

    # Optional stratified subsetting for bounded compute environments
    if sample_size is not None and sample_size < len(df):
        df, _ = train_test_split(
            df,
            train_size=sample_size,
            stratify=df["label"],
            random_state=random_seed,
        )
        df = df.reset_index(drop=True)

    n_samples = len(df)
    indices = np.arange(n_samples)
    y = df["label"].to_numpy(dtype=int)

    # Split 1: Train vs (Val + Test)
    temp_ratio = val_ratio + test_ratio
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        indices,
        y,
        test_size=temp_ratio,
        stratify=y,
        random_state=random_seed,
    )

    # Split 2: Val vs Test
    val_share = val_ratio / temp_ratio
    val_rel, test_rel, y_val, y_test = train_test_split(
        np.arange(len(temp_idx)),
        y_temp,
        test_size=(1.0 - val_share),
        stratify=y_temp,
        random_state=random_seed,
    )
    val_idx = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]

    # Extract raw feature DataFrames
    X_train_raw = df.iloc[train_idx][RAW_INPUT_FEATURES].copy().reset_index(drop=True)
    X_val_raw = df.iloc[val_idx][RAW_INPUT_FEATURES].copy().reset_index(drop=True)
    X_test_raw = df.iloc[test_idx][RAW_INPUT_FEATURES].copy().reset_index(drop=True)

    # Clean non-finite / missing values if present (UNSW-NB15 clean by default)
    X_train_raw = X_train_raw.fillna(0.0)
    X_val_raw = X_val_raw.fillna(0.0)
    X_test_raw = X_test_raw.fillna(0.0)

    # Build ColumnTransformer:
    # 1. StandardScaler for numerical features
    # 2. OneHotEncoder with handle_unknown='ignore' for categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), RAW_NUMERICAL_FEATURES),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    max_categories=max_categories_per_cat_feature,
                    sparse_output=False,
                ),
                RAW_CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    # Fit STRICTLY on training split to prevent leakage
    X_train_scaled = preprocessor.fit_transform(X_train_raw)
    X_val_scaled = preprocessor.transform(X_val_raw)
    X_test_scaled = preprocessor.transform(X_test_raw)

    encoded_feature_names = list(preprocessor.get_feature_names_out())

    def get_class_counts(labels: np.ndarray) -> Dict[str, int]:
        return {
            TARGET_LABEL_MAPPING[cls_id]: int(np.sum(labels == cls_id))
            for cls_id in TARGET_LABEL_MAPPING
        }

    sample_counts = {
        "raw_total_available": int(raw_total_rows),
        "total_selected": int(n_samples),
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

    dataset_metadata = {
        "dataset_name": "UNSW-NB15",
        "provenance": (
            "Australian Centre for Cyber Security (ACCS), UNSW Canberra. "
            "Authors: Nour Moustafa and Jill Slay (2015)."
        ),
        "source_file": str(data_path),
        "sample_size": sample_size,
        "deduplicated": deduplicate,
        "random_seed": random_seed,
        "input_feature_count_raw": len(RAW_INPUT_FEATURES),
        "encoded_feature_dim": int(X_train_scaled.shape[1]),
        "categorical_columns": RAW_CATEGORICAL_FEATURES,
        "numerical_columns": RAW_NUMERICAL_FEATURES,
        "target_mapping": TARGET_LABEL_MAPPING,
        "positive_class": "ATTACK (1)",
        "negative_class": "BENIGN (0)",
    }

    return NIDSSplits(
        X_train_raw=X_train_raw,
        X_val_raw=X_val_raw,
        X_test_raw=X_test_raw,
        X_train=X_train_scaled.astype(np.float32),
        X_val=X_val_scaled.astype(np.float32),
        X_test=X_test_scaled.astype(np.float32),
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        preprocessor=preprocessor,
        raw_feature_names=RAW_INPUT_FEATURES,
        encoded_feature_names=encoded_feature_names,
        target_names=TARGET_NAMES,
        sample_counts=sample_counts,
        class_distribution=class_distribution,
        dataset_metadata=dataset_metadata,
    )


def create_nids_dataloaders(
    splits: NIDSSplits,
    batch_size: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train, validation, and test splits.
    """
    train_dataset = NIDSDataset(splits.X_train, splits.y_train)
    val_dataset = NIDSDataset(splits.X_val, splits.y_val)
    test_dataset = NIDSDataset(splits.X_test, splits.y_test)

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


def save_preprocessor_artifact(
    preprocessor: ColumnTransformer,
    output_path: Union[str, Path] = "results/network_intrusion/preprocessor.joblib",
) -> Path:
    """Save fitted ColumnTransformer to joblib artifact."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, p)
    return p


def load_preprocessor_artifact(
    artifact_path: Union[str, Path] = "results/network_intrusion/preprocessor.joblib",
) -> ColumnTransformer:
    """Load fitted ColumnTransformer from joblib artifact."""
    p = Path(artifact_path)
    if not p.exists():
        raise FileNotFoundError(f"Preprocessor artifact not found at {p}")
    return joblib.load(p)
