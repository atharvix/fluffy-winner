"""Training script for classical baseline and hybrid quantum models on Wisconsin Breast Cancer dataset."""

import argparse
import copy
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import (
    DatasetSplits,
    create_dataloaders,
    prepare_breast_cancer_data,
)
from src.evaluation.metrics import compute_classification_metrics
from src.models.classical import ClassicalMLP
from src.models.hybrid import HybridQuantumClassifier


def set_seed(seed: int = 42) -> None:
    """Set random seeds across random, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_dataset(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluate model on an entire feature/label array and return loss and metrics.

    Args:
        model: PyTorch model.
        features: 2D numpy array of scaled features.
        labels: 1D numpy array of binary targets.
        criterion: Loss criterion (e.g. BCEWithLogitsLoss).
        device: Device to run evaluation on.

    Returns:
        Tuple of (mean_loss, metrics_dict).
    """
    model.eval()
    x_tensor = torch.tensor(features, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        logits = model(x_tensor)
        loss = criterion(logits, y_tensor).item()
        probs = torch.sigmoid(logits).cpu().numpy().ravel()

    metrics = compute_classification_metrics(y_true=labels, y_pred_probs=probs)
    return float(loss), metrics


def train_baseline(
    random_seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    hidden_dims: Optional[List[int]] = None,
    dropout_rate: float = 0.2,
    use_batch_norm: bool = True,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    epochs: int = 100,
    patience: int = 15,
    output_dir: str = "results",
    device_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train classical baseline MLP with validation-loss-based model checkpointing and early stopping.
    """
    if hidden_dims is None:
        hidden_dims = [64, 32]

    set_seed(random_seed)

    if device_name is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    splits: DatasetSplits = prepare_breast_cancer_data(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
    )

    train_loader, _, _ = create_dataloaders(splits, batch_size=batch_size)

    input_dim = splits.X_train.shape[1]
    model = ClassicalMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    history: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch = -1
    best_model_state: Optional[Dict[str, Any]] = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        train_batches = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            train_batches += 1

        train_loss, train_metrics = evaluate_dataset(
            model=model,
            features=splits.X_train,
            labels=splits.y_train,
            criterion=criterion,
            device=device,
        )
        val_loss, val_metrics = evaluate_dataset(
            model=model,
            features=splits.X_val,
            labels=splits.y_val,
            criterion=criterion,
            device=device,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_roc_auc": val_metrics["roc_auc"],
        }
        history.append(epoch_record)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    final_train_loss, final_train_metrics = evaluate_dataset(
        model=model,
        features=splits.X_train,
        labels=splits.y_train,
        criterion=criterion,
        device=device,
    )
    final_val_loss, final_val_metrics = evaluate_dataset(
        model=model,
        features=splits.X_val,
        labels=splits.y_val,
        criterion=criterion,
        device=device,
    )
    final_test_loss, final_test_metrics = evaluate_dataset(
        model=model,
        features=splits.X_test,
        labels=splits.y_test,
        criterion=criterion,
        device=device,
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    config_data = {
        "random_seed": int(random_seed),
        "dataset": {
            "name": "Breast Cancer Wisconsin Diagnostic",
            "source": "sklearn.datasets.load_breast_cancer",
            "target_label_mapping": {0: "malignant", 1: "benign"},
            "features_count": int(input_dim),
            "feature_names": [str(name) for name in splits.feature_names],
            "sample_counts": {k: int(v) for k, v in splits.sample_counts.items()},
            "class_distribution": {
                split_k: {cls_k: int(cnt) for cls_k, cnt in cls_dict.items()}
                for split_k, cls_dict in splits.class_distribution.items()
            },
            "split_ratios": {
                "train": float(train_ratio),
                "val": float(val_ratio),
                "test": float(test_ratio),
            },
        },
        "model": {
            "type": "ClassicalMLP",
            "input_dim": int(input_dim),
            "hidden_dims": [int(h) for h in hidden_dims],
            "dropout_rate": float(dropout_rate),
            "use_batch_norm": bool(use_batch_norm),
            "num_hidden_layers": int(len(hidden_dims)),
        },
        "training": {
            "epochs_max": int(epochs),
            "epochs_trained": int(len(history)),
            "best_epoch": int(best_epoch),
            "patience": int(patience),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "batch_size": int(batch_size),
            "loss_function": "BCEWithLogitsLoss",
            "optimizer": "AdamW",
            "device": str(device),
        },
    }

    metrics_data = {
        "random_seed": random_seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "training_partition": {
            "loss": final_train_loss,
            "metrics": final_train_metrics,
        },
        "validation_partition": {
            "loss": final_val_loss,
            "metrics": final_val_metrics,
        },
        "test_partition": {
            "loss": final_test_loss,
            "metrics": final_test_metrics,
        },
    }

    config_file = out_path / "config.json"
    history_file = out_path / "training_history.json"
    metrics_file = out_path / "metrics.json"
    checkpoint_file = out_path / "best_model.pt"

    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "config": config_data,
        },
        checkpoint_file,
    )

    return {
        "config": config_data,
        "history": history,
        "metrics": metrics_data,
        "checkpoint_path": str(checkpoint_file),
    }


def train_hybrid(
    random_seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    n_qubits: int = 4,
    n_layers: int = 2,
    encoder_hidden_dim: int = 16,
    learning_rate: float = 0.01,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    epochs: int = 30,
    patience: int = 10,
    output_dir: str = "results/hybrid_quantum",
    device_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train differentiable hybrid quantum-classical classifier with data re-uploading.

    Args:
        random_seed: Fixed seed for reproducibility.
        train_ratio: Training split ratio.
        val_ratio: Validation split ratio.
        test_ratio: Test split ratio.
        n_qubits: Number of qubits (4).
        n_layers: Number of quantum data re-uploading layers.
        encoder_hidden_dim: Dimension of intermediate classical encoder layer.
        learning_rate: Optimizer learning rate.
        weight_decay: L2 penalty.
        batch_size: Mini-batch size.
        epochs: Maximum number of epochs.
        patience: Early stopping patience.
        output_dir: Directory to save artifacts.
        device_name: Device specification (e.g. 'cpu').

    Returns:
        Dictionary with config, history, metrics, and checkpoint path.
    """
    set_seed(random_seed)

    if device_name is None:
        device = torch.device("cpu")  # Quantum simulation default
    else:
        device = torch.device(device_name)

    splits: DatasetSplits = prepare_breast_cancer_data(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
    )

    train_loader, _, _ = create_dataloaders(splits, batch_size=batch_size)

    input_dim = splits.X_train.shape[1]
    model = HybridQuantumClassifier(
        input_dim=input_dim,
        n_qubits=n_qubits,
        n_layers=n_layers,
        encoder_hidden_dim=encoder_hidden_dim,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    history: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch = -1
    best_model_state: Optional[Dict[str, Any]] = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        train_batches = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            train_batches += 1

        train_loss, train_metrics = evaluate_dataset(
            model=model,
            features=splits.X_train,
            labels=splits.y_train,
            criterion=criterion,
            device=device,
        )
        val_loss, val_metrics = evaluate_dataset(
            model=model,
            features=splits.X_val,
            labels=splits.y_val,
            criterion=criterion,
            device=device,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_malignant_sensitivity": val_metrics["malignant_sensitivity"],
            "val_malignant_specificity": val_metrics["malignant_specificity"],
        }
        history.append(epoch_record)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    final_train_loss, final_train_metrics = evaluate_dataset(
        model=model,
        features=splits.X_train,
        labels=splits.y_train,
        criterion=criterion,
        device=device,
    )
    final_val_loss, final_val_metrics = evaluate_dataset(
        model=model,
        features=splits.X_val,
        labels=splits.y_val,
        criterion=criterion,
        device=device,
    )
    final_test_loss, final_test_metrics = evaluate_dataset(
        model=model,
        features=splits.X_test,
        labels=splits.y_test,
        criterion=criterion,
        device=device,
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    config_data = {
        "random_seed": int(random_seed),
        "dataset": {
            "name": "Breast Cancer Wisconsin Diagnostic",
            "source": "sklearn.datasets.load_breast_cancer",
            "target_label_mapping": {0: "malignant", 1: "benign"},
            "features_count": int(input_dim),
            "feature_names": [str(name) for name in splits.feature_names],
            "sample_counts": {k: int(v) for k, v in splits.sample_counts.items()},
            "class_distribution": {
                split_k: {cls_k: int(cnt) for cls_k, cnt in cls_dict.items()}
                for split_k, cls_dict in splits.class_distribution.items()
            },
            "split_ratios": {
                "train": float(train_ratio),
                "val": float(val_ratio),
                "test": float(test_ratio),
            },
        },
        "model": {
            "type": "HybridQuantumClassifier",
            "input_dim": int(input_dim),
            "n_qubits": int(n_qubits),
            "n_layers": int(n_layers),
            "encoder_hidden_dim": int(encoder_hidden_dim),
            "quantum_parameters_count": int(model.quantum_layer.num_parameters),
            "diff_method": "backprop",
            "device_backend": "default.qubit",
        },
        "training": {
            "epochs_max": int(epochs),
            "epochs_trained": int(len(history)),
            "best_epoch": int(best_epoch),
            "patience": int(patience),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "batch_size": int(batch_size),
            "loss_function": "BCEWithLogitsLoss",
            "optimizer": "AdamW",
            "device": str(device),
        },
    }

    metrics_data = {
        "random_seed": random_seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "training_partition": {
            "loss": final_train_loss,
            "metrics": final_train_metrics,
        },
        "validation_partition": {
            "loss": final_val_loss,
            "metrics": final_val_metrics,
        },
        "test_partition": {
            "loss": final_test_loss,
            "metrics": final_test_metrics,
        },
    }

    config_file = out_path / "config.json"
    history_file = out_path / "training_history.json"
    metrics_file = out_path / "metrics.json"
    checkpoint_file = out_path / "best_model.pt"

    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "config": config_data,
        },
        checkpoint_file,
    )

    return {
        "config": config_data,
        "history": history,
        "metrics": metrics_data,
        "checkpoint_path": str(checkpoint_file),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train classical baseline or hybrid quantum model on Breast Cancer dataset."
    )
    parser.add_argument("--model-type", choices=["classical", "hybrid"], default="classical", help="Model type to train.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train set ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation set ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test set ratio.")
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[64, 32], help="Classical hidden layer units.")
    parser.add_argument("--q-layers", type=int, default=2, help="Number of quantum data re-uploading layers.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate (classical).")
    parser.add_argument("--no-batchnorm", dest="batch_norm", action="store_false", help="Disable batch norm.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs.")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save artifacts.")

    args = parser.parse_args()

    if args.model_type == "classical":
        out_dir = args.output_dir if args.output_dir is not None else "results"
        print("=" * 60)
        print("Training Classical Deep Learning Baseline")
        print(f"Random Seed: {args.seed}")
        print(f"Hidden Dimensions: {args.hidden_dims}")
        print(f"Output Directory: {out_dir}")
        print("=" * 60)

        result = train_baseline(
            random_seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            hidden_dims=args.hidden_dims,
            dropout_rate=args.dropout,
            use_batch_norm=args.batch_norm,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            output_dir=out_dir,
        )
    else:
        out_dir = args.output_dir if args.output_dir is not None else "results/hybrid_quantum"
        print("=" * 60)
        print("Training Hybrid Quantum-Classical Classifier")
        print(f"Random Seed: {args.seed}")
        print(f"Quantum Qubits: 4 | Re-uploading Layers: {args.q_layers}")
        print(f"Output Directory: {out_dir}")
        print("=" * 60)

        result = train_hybrid(
            random_seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            n_qubits=4,
            n_layers=args.q_layers,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            output_dir=out_dir,
        )

    test_m = result["metrics"]["test_partition"]["metrics"]
    val_m = result["metrics"]["validation_partition"]["metrics"]
    train_m = result["metrics"]["training_partition"]["metrics"]

    print("\nTraining completed successfully!")
    print(f"Best Epoch: {result['metrics']['best_epoch']} (Val Loss: {result['metrics']['best_val_loss']:.4f})")
    print(f"Train Accuracy: {train_m['accuracy']:.4f} | F1: {train_m['f1']:.4f} | Malignant Sens: {train_m['malignant_sensitivity']:.4f}")
    print(f"Val   Accuracy: {val_m['accuracy']:.4f} | F1: {val_m['f1']:.4f} | Malignant Sens: {val_m['malignant_sensitivity']:.4f}")
    print(f"Test  Accuracy: {test_m['accuracy']:.4f} | F1: {test_m['f1']:.4f} | ROC-AUC: {test_m['roc_auc']:.4f} | Malignant Sens: {test_m['malignant_sensitivity']:.4f} | Malignant Spec: {test_m['malignant_specificity']:.4f}")
    print(f"\nArtifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
