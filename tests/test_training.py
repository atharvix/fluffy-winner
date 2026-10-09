"""Tests for training pipeline, checkpointing, and reproducibility."""

import json
from pathlib import Path
import tempfile
import torch

from src.training.train import set_seed, train_baseline


def test_train_baseline_reproducibility():
    """Verify that training with same seed produces identical training metrics and results."""
    with tempfile.TemporaryDirectory() as tmp_dir1, tempfile.TemporaryDirectory() as tmp_dir2:
        res1 = train_baseline(
            random_seed=42,
            epochs=5,
            patience=5,
            hidden_dims=[32, 16],
            output_dir=tmp_dir1,
        )
        res2 = train_baseline(
            random_seed=42,
            epochs=5,
            patience=5,
            hidden_dims=[32, 16],
            output_dir=tmp_dir2,
        )

        assert res1["metrics"]["best_epoch"] == res2["metrics"]["best_epoch"]
        assert round(res1["metrics"]["test_partition"]["metrics"]["accuracy"], 6) == round(
            res2["metrics"]["test_partition"]["metrics"]["accuracy"], 6
        )
        assert round(res1["metrics"]["test_partition"]["loss"], 6) == round(
            res2["metrics"]["test_partition"]["loss"], 6
        )


def test_saved_artifacts_structure():
    """Verify that all expected artifact files are created and have valid schemas."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        res = train_baseline(
            random_seed=42,
            epochs=3,
            patience=3,
            hidden_dims=[32, 16],
            output_dir=tmp_dir,
        )

        out_path = Path(tmp_dir)
        config_file = out_path / "config.json"
        history_file = out_path / "training_history.json"
        metrics_file = out_path / "metrics.json"
        checkpoint_file = out_path / "best_model.pt"

        assert config_file.exists(), "config.json missing"
        assert history_file.exists(), "training_history.json missing"
        assert metrics_file.exists(), "metrics.json missing"
        assert checkpoint_file.exists(), "best_model.pt missing"

        # Validate config.json
        with open(config_file, "r") as f:
            config = json.load(f)
        assert config["random_seed"] == 42
        assert config["dataset"]["target_label_mapping"]["0"] == "malignant"
        assert config["dataset"]["target_label_mapping"]["1"] == "benign"
        assert config["model"]["num_hidden_layers"] == 2

        # Validate metrics.json segregation
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
        assert "training_partition" in metrics
        assert "validation_partition" in metrics
        assert "test_partition" in metrics
        assert "accuracy" in metrics["test_partition"]["metrics"]
        assert "precision" in metrics["test_partition"]["metrics"]
        assert "recall" in metrics["test_partition"]["metrics"]
        assert "f1" in metrics["test_partition"]["metrics"]
        assert "roc_auc" in metrics["test_partition"]["metrics"]
        assert "confusion_matrix" in metrics["test_partition"]["metrics"]

        # Validate checkpoint loadable
        checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        assert "model_state_dict" in checkpoint
        assert "best_epoch" in checkpoint
        assert "best_val_loss" in checkpoint
