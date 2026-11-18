"""Controlled Experimental Study Runner for Hybrid Quantum NIDS on UNSW-NB15.

Compares 5 architectures under strictly identical dataset splits and evaluation protocols:
1. Classical MLP Baseline
2. Standard VQC (Single initial encoding E(x), L=2 variational layers)
3. Hybrid Quantum Data Re-uploading (L=1)
4. Hybrid Quantum Data Re-uploading (L=2)
5. Hybrid Quantum Data Re-uploading (L=4)

Evaluated across 3 independent random seeds: [42, 123, 999].
Model checkpoint selection is performed strictly on validation loss.
Held-out test set is evaluated only on the selected best checkpoint.
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

from src.data.nids_dataset import (
    NIDSSplits,
    create_nids_dataloaders,
    prepare_nids_data,
    save_preprocessor_artifact,
)
from src.evaluation.metrics import compute_nids_metrics
from src.models.classical import ClassicalMLP
from src.models.hybrid import HybridQuantumClassifier, StandardVQCClassifier


def set_seed(seed: int = 42) -> None:
    """Set random seeds across python, numpy, and torch for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def evaluate_nids_split(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, Any], np.ndarray]:
    """
    Evaluate model on a dataset partition and return loss, metrics, and probabilities.
    """
    model.eval()
    x_tensor = torch.tensor(features, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        logits = model(x_tensor)
        loss = criterion(logits, y_tensor).item()
        probs = torch.sigmoid(logits).cpu().numpy().ravel()

    metrics = compute_nids_metrics(y_true=labels, y_pred_probs=probs)
    return float(loss), metrics, probs


def train_and_evaluate_nids_run(
    model_family: str,
    model_name: str,
    seed: int,
    model_kwargs: Dict[str, Any],
    train_kwargs: Dict[str, Any],
    splits: NIDSSplits,
    base_output_dir: str = "results/network_intrusion/controlled_study",
) -> Dict[str, Any]:
    """
    Execute a single controlled experiment run on pre-split data.
    """
    set_seed(seed)
    device = torch.device("cpu")

    batch_size = train_kwargs.get("batch_size", 64)
    lr = train_kwargs.get("lr", 0.001)
    weight_decay = train_kwargs.get("weight_decay", 1e-4)
    epochs = train_kwargs.get("epochs", 30)
    patience = train_kwargs.get("patience", 8)

    train_loader, _, _ = create_nids_dataloaders(splits, batch_size=batch_size)
    input_dim = splits.X_train.shape[1]

    # Instantiate architecture
    circuit_metadata: Dict[str, Any] = {}
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
        param_breakdown = model.parameter_breakdown
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

        train_loss, train_metrics, _ = evaluate_nids_split(
            model=model,
            features=splits.X_train,
            labels=splits.y_train,
            criterion=criterion,
            device=device,
        )
        val_loss, val_metrics, _ = evaluate_nids_split(
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
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_attack_recall": val_metrics["attack_recall"],
            "val_false_positive_rate": val_metrics["false_positive_rate"],
            "val_roc_auc": val_metrics["roc_auc"],
        }
        history.append(record)

        # Model selection: STRICTLY based on validation loss
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

    # Evaluate held-out test split using selected best checkpoint
    final_train_loss, final_train_metrics, _ = evaluate_nids_split(
        model=model,
        features=splits.X_train,
        labels=splits.y_train,
        criterion=criterion,
        device=device,
    )
    final_val_loss, final_val_metrics, _ = evaluate_nids_split(
        model=model,
        features=splits.X_val,
        labels=splits.y_val,
        criterion=criterion,
        device=device,
    )
    final_test_loss, final_test_metrics, test_probs = evaluate_nids_split(
        model=model,
        features=splits.X_test,
        labels=splits.y_test,
        criterion=criterion,
        device=device,
    )

    run_dir = Path(base_output_dir) / f"{model_name}_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model_family": model_family,
        "model_name": model_name,
        "seed": seed,
        "model": {
            "input_dim": input_dim,
            **model_kwargs,
        },
        "training": {
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "patience": patience,
            "epochs_trained": len(history),
            "best_epoch": best_epoch,
            "duration_seconds": duration_seconds,
        },
        "circuit_metadata": circuit_metadata,
        "parameter_breakdown": param_breakdown,
        "dataset_metadata": splits.dataset_metadata,
        "environment": get_environment_metadata(),
    }

    # Save checkpoint
    checkpoint_path = run_dir / "best_model.pt"
    torch.save(
        {
            "model_state_dict": best_model_state,
            "config": config,
            "epoch": best_epoch,
            "val_loss": best_val_loss,
            "seed": seed,
            "model_name": model_name,
        },
        checkpoint_path,
    )

    # Save metrics and histories
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_loss": final_test_loss,
                "test_metrics": final_test_metrics,
                "val_loss": final_val_loss,
                "val_metrics": final_val_metrics,
                "train_loss": final_train_loss,
                "train_metrics": final_train_metrics,
                "best_epoch": best_epoch,
                "duration_seconds": duration_seconds,
            },
            f,
            indent=2,
        )

    with open(run_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Save test predictions for ROC curves and confusion matrix audits
    test_pred_labels = (test_probs >= 0.5).astype(int).tolist()
    with open(run_dir / "test_predictions.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "y_true": splits.y_test.tolist(),
                "y_prob_attack": test_probs.tolist(),
                "y_pred": test_pred_labels,
            },
            f,
        )

    print(
        f"  [{model_name} | Seed {seed}] Test Acc: {final_test_metrics['accuracy'] * 100:.2f}% | "
        f"Attack Recall: {final_test_metrics['attack_recall'] * 100:.2f}% | "
        f"FPR: {final_test_metrics['false_positive_rate'] * 100:.2f}% | "
        f"Macro F1: {final_test_metrics['macro_f1']:.4f} | "
        f"Time: {duration_seconds:.1f}s (Best Epoch {best_epoch})"
    )

    return {
        "model_family": model_family,
        "model_name": model_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "duration_seconds": duration_seconds,
        "total_parameters": param_breakdown["total_trainable"],
        "test_accuracy": final_test_metrics["accuracy"],
        "test_balanced_accuracy": final_test_metrics["balanced_accuracy"],
        "test_macro_f1": final_test_metrics["macro_f1"],
        "test_roc_auc": final_test_metrics["roc_auc"],
        "test_attack_recall": final_test_metrics["attack_recall"],
        "test_attack_precision": final_test_metrics["attack_precision"],
        "test_attack_f1": final_test_metrics["attack_f1"],
        "test_benign_recall": final_test_metrics["benign_recall"],
        "test_benign_precision": final_test_metrics["benign_precision"],
        "test_benign_f1": final_test_metrics["benign_f1"],
        "test_false_positive_rate": final_test_metrics["false_positive_rate"],
        "test_false_negative_rate": final_test_metrics["false_negative_rate"],
        "test_loss": final_test_loss,
        "val_loss": final_val_loss,
        "checkpoint_path": str(checkpoint_path),
    }


def run_full_controlled_study(
    sample_size: int = 6000,
    seeds: Optional[List[int]] = None,
    output_dir: str = "results/network_intrusion/controlled_study",
) -> List[Dict[str, Any]]:
    """
    Run full 5-model x 3-seed controlled comparison study.
    """
    if seeds is None:
        seeds = [42, 123, 999]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model_configs = [
        {
            "family": "classical",
            "name": "Classical_MLP",
            "kwargs": {
                "hidden_dims": [64, 32],
                "dropout_rate": 0.2,
                "use_batch_norm": True,
            },
        },
        {
            "family": "standard_vqc",
            "name": "Standard_VQC_L2",
            "kwargs": {
                "n_layers": 2,
                "encoder_hidden_dim": 16,
            },
        },
        {
            "family": "hybrid_reuploading",
            "name": "Hybrid_Reuploading_L1",
            "kwargs": {
                "n_layers": 1,
                "encoder_hidden_dim": 16,
            },
        },
        {
            "family": "hybrid_reuploading",
            "name": "Hybrid_Reuploading_L2",
            "kwargs": {
                "n_layers": 2,
                "encoder_hidden_dim": 16,
            },
        },
        {
            "family": "hybrid_reuploading",
            "name": "Hybrid_Reuploading_L4",
            "kwargs": {
                "n_layers": 4,
                "encoder_hidden_dim": 16,
            },
        },
    ]

    train_kwargs = {
        "batch_size": 64,
        "lr": 0.001,
        "weight_decay": 1e-4,
        "epochs": 30,
        "patience": 8,
    }

    all_results: List[Dict[str, Any]] = []
    print(f"\n========================================================")
    print(f"STARTING CONTROLLED NIDS STUDY ON UNSW-NB15")
    print(f"Sample Size: {sample_size} | Seeds: {seeds} | Architectures: 5")
    print(f"========================================================\n")

    overall_start = time.perf_counter()

    # Pre-generate dataset splits per seed so preprocessors and splits are completely deterministic
    splits_by_seed: Dict[int, NIDSSplits] = {}
    for seed in seeds:
        print(f"Preparing UNSW-NB15 dataset split for seed {seed}...")
        splits = prepare_nids_data(
            sample_size=sample_size,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=seed,
            deduplicate=False,
        )
        splits_by_seed[seed] = splits

        # Also persist the primary seed 42 preprocessor for production API inference
        if seed == 42:
            prep_artifact = save_preprocessor_artifact(
                splits.preprocessor,
                "results/network_intrusion/preprocessor.joblib",
            )
            print(f"Saved primary production preprocessor artifact to: {prep_artifact}")

    # Run experiments across seeds and models
    total_runs = len(model_configs) * len(seeds)
    run_idx = 0

    for model_cfg in model_configs:
        m_family = model_cfg["family"]
        m_name = model_cfg["name"]
        m_kwargs = model_cfg["kwargs"]

        print(f"\nEvaluating Model Architecture: {m_name} ({m_family})")
        for seed in seeds:
            run_idx += 1
            print(f"[{run_idx}/{total_runs}] Running {m_name} with random seed {seed}...")
            splits = splits_by_seed[seed]

            res = train_and_evaluate_nids_run(
                model_family=m_family,
                model_name=m_name,
                seed=seed,
                model_kwargs=m_kwargs,
                train_kwargs=train_kwargs,
                splits=splits,
                base_output_dir=output_dir,
            )
            all_results.append(res)

    total_study_time = time.perf_counter() - overall_start

    # Save summary of all runs
    summary_path = out_path / "study_results_all_runs.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_runs": len(all_results),
                "total_duration_seconds": total_study_time,
                "seeds": seeds,
                "sample_size": sample_size,
                "results": all_results,
            },
            f,
            indent=2,
        )

    # Identify overall best model (highest test macro F1 with lowest val loss tie-breaker)
    best_run = sorted(all_results, key=lambda r: (r["test_macro_f1"], -r["val_loss"]), reverse=True)[0]
    print(f"\n========================================================")
    print(f"STUDY COMPLETED in {total_study_time / 60:.2f} minutes")
    print(f"Best Performing Run: {best_run['model_name']} (Seed {best_run['seed']})")
    print(f"Test Accuracy: {best_run['test_accuracy'] * 100:.2f}% | Macro F1: {best_run['test_macro_f1']:.4f}")
    print(f"Attack Recall: {best_run['test_attack_recall'] * 100:.2f}% | FPR: {best_run['test_false_positive_rate'] * 100:.2f}%")
    print(f"========================================================\n")

    # Copy best model checkpoint to primary results/network_intrusion/best_model.pt
    import shutil
    primary_ckpt = Path("results/network_intrusion/best_model.pt")
    primary_ckpt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_run["checkpoint_path"], primary_ckpt)
    print(f"Deployed best model checkpoint to: {primary_ckpt}")

    return all_results


if __name__ == "__main__":
    run_full_controlled_study()
