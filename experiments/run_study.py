"""Controlled Experimental Comparison Runner.

Evaluates:
1. Classical MLP Baseline
2. Standard VQC (Single Data Encoding at t=0, L=2)
3. Hybrid Quantum Classifier with Data Re-uploading (L=2)
4. Re-uploading Layer Ablation (L=1, L=2, L=4)

Runs 3 independent random seeds (42, 123, 999) under strict validation-only selection.
Records runtime duration, circuit metrics, parameter breakdowns, and environment metadata.
"""

import copy
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pennylane as qml
import sklearn
import torch
import torch.nn as nn

from src.data.dataset import (
    DatasetSplits,
    create_dataloaders,
    prepare_breast_cancer_data,
)
from src.evaluation.metrics import compute_classification_metrics
from src.models.classical import ClassicalMLP
from src.models.hybrid import HybridQuantumClassifier, StandardVQCClassifier
from src.training.train import evaluate_dataset, set_seed


def get_environment_metadata() -> Dict[str, str]:
    """Capture runtime and dependency version metadata."""
    return {
        "os_platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "pennylane_version": qml.__version__,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
    }


def train_and_evaluate_run(
    model_family: str,
    model_name: str,
    seed: int,
    model_kwargs: Dict[str, Any],
    train_kwargs: Dict[str, Any],
    base_output_dir: str = "results/controlled_study",
) -> Dict[str, Any]:
    """
    Execute a single controlled experiment run.

    Args:
        model_family: 'classical', 'standard_vqc', or 'hybrid_reuploading'
        model_name: Unique descriptive name (e.g. 'Standard_VQC_L2')
        seed: Random seed for splitting and initialization
        model_kwargs: Constructor keyword arguments for the model
        train_kwargs: Training hyperparameters (lr, weight_decay, epochs, patience, batch_size)
        base_output_dir: Parent directory for study results

    Returns:
        Dictionary of run results and metrics.
    """
    set_seed(seed)
    device = torch.device("cpu")

    # Data split (same deterministic split function for every model)
    train_ratio = train_kwargs.get("train_ratio", 0.70)
    val_ratio = train_kwargs.get("val_ratio", 0.15)
    test_ratio = train_kwargs.get("test_ratio", 0.15)
    batch_size = train_kwargs.get("batch_size", 32)
    lr = train_kwargs.get("lr", 0.001)
    weight_decay = train_kwargs.get("weight_decay", 1e-4)
    epochs = train_kwargs.get("epochs", 50)
    patience = train_kwargs.get("patience", 12)

    splits: DatasetSplits = prepare_breast_cancer_data(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=seed,
    )

    train_loader, _, _ = create_dataloaders(splits, batch_size=batch_size)
    input_dim = splits.X_train.shape[1]

    # Instantiate model
    circuit_metadata = {}
    if model_family == "classical":
        hidden_dims = model_kwargs.get("hidden_dims", [64, 32])
        dropout_rate = model_kwargs.get("dropout_rate", 0.2)
        use_batch_norm = model_kwargs.get("use_batch_norm", True)
        model = ClassicalMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
        ).to(device)
        param_breakdown = {
            "total_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "classical_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "quantum_parameters": 0,
        }
        circuit_metadata = {"type": "None (Classical)"}

    elif model_family == "standard_vqc":
        n_layers = model_kwargs.get("n_layers", 2)
        encoder_hidden_dim = model_kwargs.get("encoder_hidden_dim", 16)
        model = StandardVQCClassifier(
            input_dim=input_dim,
            n_qubits=4,
            n_layers=n_layers,
            encoder_hidden_dim=encoder_hidden_dim,
        ).to(device)
        param_breakdown = model.parameter_breakdown
        circuit_metadata = model.quantum_layer.circuit_info

    elif model_family == "hybrid_reuploading":
        n_layers = model_kwargs.get("n_layers", 2)
        encoder_hidden_dim = model_kwargs.get("encoder_hidden_dim", 16)
        model = HybridQuantumClassifier(
            input_dim=input_dim,
            n_qubits=4,
            n_layers=n_layers,
            encoder_hidden_dim=encoder_hidden_dim,
        ).to(device)
        param_breakdown = model.parameter_breakdown
        circuit_metadata = model.quantum_layer.circuit_info
    else:
        raise ValueError(f"Unknown model_family: {model_family}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch = -1
    best_model_state: Optional[Dict[str, Any]] = None
    epochs_no_improve = 0

    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

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

        record = {
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
        history.append(record)

        # Model selection: ONLY based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    duration_seconds = time.perf_counter() - start_time

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Evaluate all partitions with best model
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

    run_dir = Path(base_output_dir) / f"{model_name}_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config_data = {
        "model_name": model_name,
        "model_family": model_family,
        "random_seed": int(seed),
        "environment": get_environment_metadata(),
        "dataset": {
            "name": "Breast Cancer Wisconsin Diagnostic",
            "target_label_mapping": {0: "malignant", 1: "benign"},
            "features_count": int(input_dim),
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
        "model_architecture": {
            "family": model_family,
            "parameters": param_breakdown,
            "circuit_info": circuit_metadata,
            "kwargs": model_kwargs,
        },
        "training": {
            "epochs_max": int(epochs),
            "epochs_trained": int(len(history)),
            "best_epoch": int(best_epoch),
            "patience": int(patience),
            "learning_rate": float(lr),
            "weight_decay": float(weight_decay),
            "batch_size": int(batch_size),
            "duration_seconds": float(duration_seconds),
            "loss_function": "BCEWithLogitsLoss",
            "optimizer": "AdamW",
            "device": str(device),
        },
    }

    metrics_data = {
        "model_name": model_name,
        "random_seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "duration_seconds": float(duration_seconds),
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

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    with open(run_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "config": config_data,
        },
        run_dir / "best_model.pt",
    )

    print(
        f"[{model_name} | Seed {seed}] Best Epoch {best_epoch:2d}/{len(history):2d} "
        f"| Val Loss: {best_val_loss:.4f} | Test Acc: {final_test_metrics['accuracy']:.4f} "
        f"| Malignant Sens: {final_test_metrics['malignant_sensitivity']:.4f} "
        f"| Test AUC: {final_test_metrics['roc_auc']:.4f} ({duration_seconds:.2f}s)"
    )

    return {
        "model_name": model_name,
        "seed": seed,
        "config": config_data,
        "history": history,
        "metrics": metrics_data,
        "run_dir": str(run_dir),
    }


def run_all_experiments(
    seeds: Optional[List[int]] = None,
    output_dir: str = "results/controlled_study",
) -> List[Dict[str, Any]]:
    """
    Run complete controlled comparison matrix across all models and seeds.
    """
    if seeds is None:
        seeds = [42, 123, 999]

    # Models definition with validation-tuned learning rates
    experiment_specs = [
        {
            "model_family": "classical",
            "model_name": "Classical_MLP",
            "model_kwargs": {"hidden_dims": [64, 32], "dropout_rate": 0.2, "use_batch_norm": True},
            "train_kwargs": {"lr": 0.001, "weight_decay": 1e-4, "epochs": 60, "patience": 15, "batch_size": 32},
        },
        {
            "model_family": "standard_vqc",
            "model_name": "Standard_VQC_L2",
            "model_kwargs": {"n_layers": 2, "encoder_hidden_dim": 16},
            "train_kwargs": {"lr": 0.01, "weight_decay": 1e-4, "epochs": 35, "patience": 10, "batch_size": 32},
        },
        {
            "model_family": "hybrid_reuploading",
            "model_name": "Hybrid_Reuploading_L2",
            "model_kwargs": {"n_layers": 2, "encoder_hidden_dim": 16},
            "train_kwargs": {"lr": 0.01, "weight_decay": 1e-4, "epochs": 35, "patience": 10, "batch_size": 32},
        },
        {
            "model_family": "hybrid_reuploading",
            "model_name": "Hybrid_Reuploading_L1",
            "model_kwargs": {"n_layers": 1, "encoder_hidden_dim": 16},
            "train_kwargs": {"lr": 0.01, "weight_decay": 1e-4, "epochs": 35, "patience": 10, "batch_size": 32},
        },
        {
            "model_family": "hybrid_reuploading",
            "model_name": "Hybrid_Reuploading_L4",
            "model_kwargs": {"n_layers": 4, "encoder_hidden_dim": 16},
            "train_kwargs": {"lr": 0.01, "weight_decay": 1e-4, "epochs": 35, "patience": 10, "batch_size": 32},
        },
    ]

    print("=" * 80)
    print("STARTING CONTROLLED EXPERIMENTAL COMPARISON STUDY")
    print(f"Seeds: {seeds}")
    print(f"Total Runs: {len(experiment_specs) * len(seeds)}")
    print(f"Output Directory: {output_dir}")
    print("=" * 80)

    results: List[Dict[str, Any]] = []
    total_start = time.perf_counter()

    for spec in experiment_specs:
        for seed in seeds:
            res = train_and_evaluate_run(
                model_family=spec["model_family"],
                model_name=spec["model_name"],
                seed=seed,
                model_kwargs=spec["model_kwargs"],
                train_kwargs=spec["train_kwargs"],
                base_output_dir=output_dir,
            )
            results.append(res)

    total_duration = time.perf_counter() - total_start
    print("=" * 80)
    print(f"ALL EXPERIMENTAL RUNS COMPLETED in {total_duration:.2f} seconds.")
    print("=" * 80)

    return results


if __name__ == "__main__":
    run_all_experiments()
