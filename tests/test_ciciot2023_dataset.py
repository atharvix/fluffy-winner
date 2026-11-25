"""Unit tests for CIC-IoT-2023 data loading, preprocessing, and zero-leakage pipeline."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
import torch

from src.data.ciciot2023_dataset import (
    CICIOT2023Dataset,
    CICIOT2023Splits,
    clean_numerical_features,
    create_ciciot2023_dataloaders,
    map_ciciot2023_labels,
    scan_ciciot2023_csvs,
)


def test_scan_ciciot2023_csvs():
    """Verify that scanning data/ciciot2023 returns the expected unique CSV files."""
    csvs = scan_ciciot2023_csvs("data/ciciot2023")
    assert len(csvs) == 63, f"Expected 63 unique CSV files, but found {len(csvs)}"
    # Verify no duplicate filenames
    filenames = [p.name for p in csvs]
    assert len(filenames) == len(set(filenames)), "Duplicate filenames found in scanned CSV list"


def test_map_ciciot2023_labels():
    """Verify label mapping: BENIGN / BenignTraffic -> 0, all attack classes -> 1."""
    raw_labels = pd.Series([
        "BenignTraffic",
        "BENIGN",
        "benign",
        "DDOS-ICMP_FLOOD",
        "MIRAI-GREETH_FLOOD",
        "SQLINJECTION",
        "UPLOADING_ATTACK",
        "Backdoor_Malware",
    ])
    mapped = map_ciciot2023_labels(raw_labels)

    expected = pd.Series([0, 0, 0, 1, 1, 1, 1, 1])
    assert (mapped.values == expected.values).all(), f"Label mapping mismatch: {mapped.values} vs {expected.values}"


def test_clean_numerical_features():
    """Verify inf, -inf, NaN sanitation and float32 precision."""
    df = pd.DataFrame({
        "feat1": [1.0, np.inf, -np.inf, np.nan, 5.0],
        "feat2": [0.0, 10.0, np.nan, 20.0, np.inf],
    })
    feature_cols = ["feat1", "feat2"]
    cleaned = clean_numerical_features(df, feature_cols)

    assert not np.isinf(cleaned.values).any(), "Found remaining infinities after cleaning"
    assert not np.isnan(cleaned.values).any(), "Found remaining NaNs after cleaning"
    assert cleaned.values.dtype == np.float32, f"Expected float32 dtype, got {cleaned.values.dtype}"

    # Verify inf replaced with 0.0 and NaN with 0.0
    assert cleaned.loc[1, "feat1"] == 0.0
    assert cleaned.loc[2, "feat1"] == 0.0
    assert cleaned.loc[3, "feat1"] == 0.0


def test_zero_leakage_scaler():
    """Verify that StandardScaler fits strictly on train, leaving test mean/std unobserved."""
    np.random.seed(42)
    # Train drawn from N(10, 2), Test drawn from N(50, 5)
    train_data = np.random.normal(loc=10.0, scale=2.0, size=(1000, 5)).astype(np.float32)
    test_data = np.random.normal(loc=50.0, scale=5.0, size=(500, 5)).astype(np.float32)

    scaler = StandardScaler()
    scaler.fit(train_data)

    # Scaler mean must match train distribution (~10.0), NOT test (~50.0)
    assert np.allclose(scaler.mean_, 10.0, atol=0.5), f"Train mean expected ~10.0, got {scaler.mean_}"

    transformed_train = scaler.transform(train_data)
    transformed_test = scaler.transform(test_data)

    assert np.allclose(transformed_train.mean(axis=0), 0.0, atol=0.1)
    # Transformed test should have mean around (50 - 10) / 2 = 20
    assert np.allclose(transformed_test.mean(axis=0), 20.0, atol=2.0)


def test_ciciot2023_dataloaders():
    """Verify batching, dimensions, and tensor types in PyTorch DataLoader."""
    n_features = 39
    X_train = np.random.randn(2048, n_features).astype(np.float32)
    y_train = np.random.randint(0, 2, size=(2048,)).astype(np.int64)
    X_val = np.random.randn(512, n_features).astype(np.float32)
    y_val = np.random.randint(0, 2, size=(512,)).astype(np.int64)
    X_test = np.random.randn(1024, n_features).astype(np.float32)
    y_test = np.random.randint(0, 2, size=(1024,)).astype(np.int64)

    scaler = StandardScaler().fit(X_train)
    splits = CICIOT2023Splits(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=[f"feat_{i}" for i in range(n_features)],
        scaler=scaler,
        sample_counts={"train": 2048, "val": 512, "test": 1024, "total": 3584},
        class_distribution={},
        metadata={},
    )

    train_loader, val_loader, test_loader = create_ciciot2023_dataloaders(splits, batch_size=512)

    x_b, y_b = next(iter(train_loader))
    assert x_b.shape == (512, n_features)
    assert y_b.shape == (512, 1)
    assert x_b.dtype == torch.float32
    assert y_b.dtype == torch.float32


def test_load_ciciot2023_benchmark_loader():
    """Verify that loading the cached benchmark produces expected shapes and metadata."""
    from src.data.ciciot2023_dataset import load_ciciot2023_benchmark
    splits = load_ciciot2023_benchmark(cache_dir="data/ciciot2023/processed_parquet")
    assert splits.X_train.shape == (500000, 39)
    assert splits.X_val.shape == (100000, 39)
    assert splits.X_test.shape == (500000, 39)
    assert len(splits.feature_names) == 39
    assert splits.sample_counts["total"] == 1100000
    assert splits.X_train.dtype == np.float32
    assert splits.y_train.dtype == np.int64


def test_stream_dataset_batching(tmp_path):
    """Verify PyTorch IterableDataset streaming from chunked parquet shards."""
    from src.data.ciciot2023_dataset import CICIOT2023StreamDataset

    # Create 2 small test parquet shards
    feature_cols = [f"f_{i}" for i in range(10)]
    for shard_idx in range(2):
        df = pd.DataFrame(np.random.randn(500, 10), columns=feature_cols).astype(np.float32)
        df["binary_label"] = np.random.randint(0, 2, size=500).astype(np.int64)
        df["Label"] = ["BENIGN" if b == 0 else "DDOS-ICMP_FLOOD" for b in df["binary_label"]]
        df.to_parquet(tmp_path / f"test_shard_{shard_idx}.parquet")

    scaler = StandardScaler()
    scaler.fit(np.random.randn(100, 10).astype(np.float32))

    stream_ds = CICIOT2023StreamDataset(
        shard_paths=[tmp_path / "test_shard_0.parquet", tmp_path / "test_shard_1.parquet"],
        feature_names=feature_cols,
        scaler=scaler,
        batch_size=128,
        shuffle_shards=False,
    )

    batches = list(stream_ds)
    assert len(batches) > 0
    x_b, y_b = batches[0]
    assert x_b.shape == (128, 10)
    assert y_b.shape == (128, 1)
    assert x_b.dtype == torch.float32
    assert y_b.dtype == torch.float32


def test_pos_weight_skew_calculation():
    """Verify that pos_weight correctly computes num_benign / num_attack for extreme skew."""
    num_benign = 1000
    num_attack = 40000
    pos_weight = num_benign / num_attack
    assert np.isclose(pos_weight, 0.025)
    # Ensure BCEWithLogitsLoss initializes with pos_weight without error
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    dummy_logits = torch.randn(16, 1)
    dummy_targets = torch.randint(0, 2, (16, 1)).float()
    loss = criterion(dummy_logits, dummy_targets)
    assert not torch.isnan(loss)
    assert loss.item() > 0


