"""Controlled Experimental Study Runner for Hybrid Quantum NIDS on CIC-IoT-2023.

Evaluates 6 architectures across 3 random seeds [42, 123, 999] on a 1.1M-sample partitioned benchmark:
1. Classical MLP Baseline (64 -> 32)
2. Classical Bottleneck MLP Control (16 -> 4 -> 1)
3. Standard VQC (Single initial encoding E(x), L=2 variational layers)
4. Hybrid Quantum Data Re-uploading (L=1)
5. Hybrid Quantum Data Re-uploading (L=2)
6. Hybrid Quantum Data Re-uploading (L=4)

Hardware Acceleration:
- Classical models utilize CUDA (NVIDIA RTX 5060 8GB).
- Quantum circuits utilize PennyLane with automatic lightning.qubit -> default.qubit fallback.
- Batch size: 1024 for high-throughput vectorized backpropagation.

Zero-Leakage & Portability:
- StandardScaler fit strictly on train split (saved to results/ciciot2023/preprocessor.joblib).
- Validation threshold tau* tuned via Youden's J statistic.
- Test metrics evaluated on 500,000 held-out flows using tau*.
- Checkpoints saved with CPU-mapped state_dicts for portable execution on lab PC and laptops.
"""

import argparse
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

from src.data.ciciot2023_dataset import (
    CICIOT2023Splits,
    create_ciciot2023_dataloaders,
    load_ciciot2023_benchmark,
)
from src.evaluation.metrics import compute_nids_metrics, optimize_decision_threshold
from src.models.classical import ClassicalBottleneckMLP, ClassicalMLP
from src.models.hybrid import HybridQuantumClassifier, StandardVQCClassifier


def set_seed(seed: int = 42) -> None:
    """Set random seeds across python, numpy, and torch for strict reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_environment_metadata() -> Dict[str, str]:
    """Capture environment and dependency versions."""
    meta = {
        "os_platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "pennylane_version": qml.__version__,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "cuda_available": str(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        meta["cuda_device"] = torch.cuda.get_device_name(0)
    return meta


def evaluate_split(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
    eval_batch_size: int = 8192,
) -> Tuple[float, Dict[str, Any], np.ndarray]:
    """
    Evaluate model on a dataset partition in vectorized batches of eval_batch_size.
    Returns average loss, metrics dictionary, and predicted probabilities array.
    """
    model.eval()
    n_samples = len(features)
    logits_list: List[torch.Tensor] = []
    total_loss = 0.0

    with torch.no_grad():
        for start_idx in range(0, n_samples, eval_batch_size):
            end_idx = min(start_idx + eval_batch_size, n_samples)
            x_b = torch.tensor(features[start_idx:end_idx], dtype=torch.float32).to(device)
            y_b = torch.tensor(labels[start_idx:end_idx], dtype=torch.float32).unsqueeze(1).to(device)
            b_logits = model(x_b)
            b_loss = criterion(b_logits, y_b).item()
            total_loss += b_loss * (end_idx - start_idx)
            logits_list.append(b_logits.cpu())

    all_logits = torch.cat(logits_list, dim=0)
    loss = total_loss / max(n_samples, 1)
    probs = torch.sigmoid(all_logits).numpy().ravel()

    metrics = compute_nids_metrics(y_true=labels, y_pred_probs=probs, threshold=threshold)
    return float(loss), metrics, probs


def train_and_evaluate_ciciot_run(
    model_family: str,
    model_name: str,
    seed: int,
    model_kwargs: Dict[str, Any],
    train_kwargs: Dict[str, Any],
    splits: CICIOT2023Splits,
    base_output_dir: str = "results/ciciot2023/controlled_study",
) -> Dict[str, Any]:
    """
    Execute a single controlled experiment run on pre-split CIC-IoT-2023 data.
    """
    set_seed(seed)

    # Device assignment: 100% on GPU for both Classical and Quantum models
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_default_device("cuda")
        print(f"[{model_name} | Seed {seed}] Training on CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        torch.set_default_device("cpu")
        print(f"[{model_name} | Seed {seed}] Training on CPU (vectorized simulator)")

    batch_size = train_kwargs.get("batch_size", 1024)
    lr = train_kwargs.get("lr", 0.001)
    weight_decay = train_kwargs.get("weight_decay", 1e-4)
    epochs = train_kwargs.get("epochs", 25)
    patience = train_kwargs.get("patience", 5)

    train_loader, _, _ = create_ciciot2023_dataloaders(splits, batch_size=batch_size)
    input_dim = splits.X_train.shape[1]

    # Model instantiation
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

    elif model_family == "classical_bottleneck":
        hidden_dim = model_kwargs.get("hidden_dim", 16)
        latent_dim = model_kwargs.get("latent_dim", 4)
        model = ClassicalBottleneckMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
        ).to(device)
        param_breakdown = model.parameter_breakdown

    elif model_family == "standard_vqc":
        n_qubits = model_kwargs.get("n_qubits", 4)
        n_layers = model_kwargs.get("n_layers", 2)
        encoder_hidden_dim = model_kwargs.get("encoder_hidden_dim", 16)
        model = StandardVQCClassifier(
            input_dim=input_dim,
            n_qubits=n_qubits,
            n_layers=n_layers,
            encoder_hidden_dim=encoder_hidden_dim,
        ).to(device)
        param_breakdown = model.parameter_breakdown
        circuit_metadata = model.quantum_layer.circuit_info

    elif model_family == "hybrid_reuploading":
        n_qubits = model_kwargs.get("n_qubits", 4)
        n_layers = model_kwargs.get("n_layers", 2)
        encoder_hidden_dim = model_kwargs.get("encoder_hidden_dim", 16)
        model = HybridQuantumClassifier(
            input_dim=input_dim,
            n_qubits=n_qubits,
            n_layers=n_layers,
            encoder_hidden_dim=encoder_hidden_dim,
        ).to(device)
        param_breakdown = model.parameter_breakdown
        circuit_metadata = model.quantum_layer.circuit_info

    else:
        raise ValueError(f"Unknown model_family: {model_family}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop with early stopping on validation loss
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    best_model_state: Optional[Dict[str, torch.Tensor]] = None
    epoch_histories: List[Dict[str, Any]] = []

    t_train_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        n_train_samples = 0
        t_epoch_start = time.time()

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item() * len(x_batch)
            n_train_samples += len(x_batch)

        epoch_duration = time.time() - t_epoch_start
        avg_train_loss = train_loss_total / max(n_train_samples, 1)

        # Validation loss evaluation
        val_loss, val_metrics_interim, _ = evaluate_split(
            model=model,
            features=splits.X_val,
            labels=splits.y_val,
            criterion=criterion,
            device=device,
            threshold=0.5,
        )

        epoch_histories.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 5),
            "val_loss": round(val_loss, 5),
            "val_accuracy": val_metrics_interim["accuracy"],
            "val_macro_f1": val_metrics_interim["macro_f1"],
            "duration_seconds": round(epoch_duration, 2),
        })

        print(
            f"  Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val F1: {val_metrics_interim['macro_f1']:.4f} | "
            f"({epoch_duration:.1f}s)"
        )

        # Check for improvement
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping triggered at epoch {epoch} (best epoch: {best_epoch}).")
                break

    total_train_time = time.time() - t_train_start

    # Restore best checkpoint
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # 1. Validation evaluation & decision threshold optimization (Youden's J)
    val_loss, _, val_probs = evaluate_split(
        model=model,
        features=splits.X_val,
        labels=splits.y_val,
        criterion=criterion,
        device=device,
        threshold=0.5,
    )

    best_tau = optimize_decision_threshold(
        y_val_true=splits.y_val,
        y_val_probs=val_probs,
        metric="youden_j",
        max_allowed_fpr=0.05,
    )
    print(f"[{model_name} | Seed {seed}] Optimal Validation Threshold tau* = {best_tau:.4f}")

    # Re-evaluate validation metrics at tau*
    val_metrics = compute_nids_metrics(y_true=splits.y_val, y_pred_probs=val_probs, threshold=best_tau)

    # 2. Held-out test evaluation using tau* on all 500,000 flows
    t_test_start = time.time()
    test_loss, test_metrics, _ = evaluate_split(
        model=model,
        features=splits.X_test,
        labels=splits.y_test,
        criterion=criterion,
        device=device,
        threshold=best_tau,
    )
    test_eval_time = time.time() - t_test_start

    # Save portable checkpoint (.cpu() tensors for lab PC and laptop compatibility)
    run_dir = Path(base_output_dir) / f"{model_name}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best_model.pt"

    cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(
        {
            "model_name": model_name,
            "model_family": model_family,
            "seed": seed,
            "input_dim": input_dim,
            "model_kwargs": model_kwargs,
            "state_dict": cpu_state_dict,
            "decision_threshold": float(best_tau),
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val_loss),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "parameter_breakdown": param_breakdown,
        },
        checkpoint_path,
    )

    run_record = {
        "model_name": model_name,
        "model_family": model_family,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": round(float(best_val_loss), 5),
        "optimal_threshold": round(float(best_tau), 4),
        "test_loss": round(float(test_loss), 5),
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
        "training_time_seconds": round(total_train_time, 2),
        "test_eval_time_seconds": round(test_eval_time, 2),
        "parameter_breakdown": param_breakdown,
        "circuit_metadata": circuit_metadata,
        "epoch_history": epoch_histories,
        "checkpoint_path": str(checkpoint_path),
        "device_used": str(device),
    }

    with open(run_dir / "run_metrics.json", "w") as f:
        json.dump(run_record, f, indent=2)

    return run_record


def run_ciciot2023_benchmark_study(
    seeds: Optional[List[int]] = None,
    models: Optional[List[str]] = None,
    output_dir: str = "results/ciciot2023",
    batch_size: int = 1024,
    epochs: int = 25,
    patience: int = 5,
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    """
    Main orchestration routine for CIC-IoT-2023 experimental study.
    """
    if seeds is None:
        seeds = [42, 123, 999]

    all_model_configs: Dict[str, Dict[str, Any]] = {
        "Classical_MLP": {
            "family": "classical",
            "kwargs": {"hidden_dims": [64, 32], "dropout_rate": 0.2, "use_batch_norm": True},
        },
        "Classical_Bottleneck_MLP": {
            "family": "classical_bottleneck",
            "kwargs": {"hidden_dim": 16, "latent_dim": 4},
        },
        "Standard_VQC_L2": {
            "family": "standard_vqc",
            "kwargs": {"n_qubits": 4, "n_layers": 2, "encoder_hidden_dim": 16},
        },
        "Hybrid_Reuploading_L1": {
            "family": "hybrid_reuploading",
            "kwargs": {"n_qubits": 4, "n_layers": 1, "encoder_hidden_dim": 16},
        },
        "Hybrid_Reuploading_L2": {
            "family": "hybrid_reuploading",
            "kwargs": {"n_qubits": 4, "n_layers": 2, "encoder_hidden_dim": 16},
        },
        "Hybrid_Reuploading_L4": {
            "family": "hybrid_reuploading",
            "kwargs": {"n_qubits": 4, "n_layers": 4, "encoder_hidden_dim": 16},
        },
    }

    if models is not None:
        selected_configs = {k: v for k, v in all_model_configs.items() if k in models}
    else:
        selected_configs = all_model_configs

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    controlled_study_dir = out_path / "controlled_study"
    controlled_study_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STARTING CIC-IoT-2023 CONTROLLED BENCHMARK STUDY")
    print(f"Models: {list(selected_configs.keys())}")
    print(f"Seeds: {seeds}")
    print(f"Batch Size: {batch_size} | Max Epochs: {epochs} | Patience: {patience}")
    print("=" * 80)

    # Load preprocessed benchmark (or build if not yet built)
    splits = load_ciciot2023_benchmark(
        cache_dir="data/ciciot2023/processed_parquet",
        scaler_path=str(out_path / "preprocessor.joblib"),
        force_rebuild=force_rebuild,
    )

    train_kwargs = {
        "batch_size": batch_size,
        "lr": 0.001,
        "weight_decay": 1e-4,
        "epochs": epochs,
        "patience": patience,
    }

    all_run_results: Dict[str, List[Dict[str, Any]]] = {name: [] for name in selected_configs}
    total_study_start = time.time()

    for model_name, cfg in selected_configs.items():
        print("\n" + "#" * 60)
        print(f"Benchmarking Architecture: {model_name}")
        print("#" * 60)

        for seed in seeds:
            print(f"\n--- Run: {model_name} | Seed: {seed} ---")
            run_res = train_and_evaluate_ciciot_run(
                model_family=cfg["family"],
                model_name=model_name,
                seed=seed,
                model_kwargs=cfg["kwargs"],
                train_kwargs=train_kwargs,
                splits=splits,
                base_output_dir=str(controlled_study_dir),
            )
            all_run_results[model_name].append(run_res)

    total_study_duration = time.time() - total_study_start

    # Compute summary statistics across random seeds
    summary_stats: Dict[str, Any] = {}
    best_overall_f1 = -1.0
    best_overall_model_info: Optional[Dict[str, Any]] = None

    metric_keys = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "attack_recall",
        "attack_precision",
        "attack_f1",
        "benign_recall",
        "benign_precision",
        "false_positive_rate",
        "roc_auc",
    ]

    for model_name, runs in all_run_results.items():
        if not runs:
            continue

        model_summary: Dict[str, Any] = {
            "model_family": runs[0]["model_family"],
            "parameter_breakdown": runs[0]["parameter_breakdown"],
            "circuit_metadata": runs[0]["circuit_metadata"],
            "n_seeds": len(runs),
            "seeds": [r["seed"] for r in runs],
            "training_time_mean_s": round(float(np.mean([r["training_time_seconds"] for r in runs])), 2),
            "training_time_std_s": round(float(np.std([r["training_time_seconds"] for r in runs])), 2),
            "optimal_threshold_mean": round(float(np.mean([r["optimal_threshold"] for r in runs])), 4),
            "optimal_threshold_std": round(float(np.std([r["optimal_threshold"] for r in runs])), 4),
            "test_metrics": {},
        }

        for k in metric_keys:
            vals = [r["test_metrics"][k] for r in runs if k in r["test_metrics"] and r["test_metrics"][k] is not None]
            if vals:
                model_summary["test_metrics"][k] = {
                    "mean": round(float(np.mean(vals)), 4),
                    "std": round(float(np.std(vals)), 4),
                }

        summary_stats[model_name] = model_summary

        mean_f1 = model_summary["test_metrics"]["macro_f1"]["mean"]
        if mean_f1 > best_overall_f1:
            best_overall_f1 = mean_f1
            # Pick seed with best test F1
            best_run = max(runs, key=lambda r: r["test_metrics"]["macro_f1"])
            best_overall_model_info = {
                "model_name": model_name,
                "seed": best_run["seed"],
                "checkpoint_path": best_run["checkpoint_path"],
                "macro_f1": best_run["test_metrics"]["macro_f1"],
                "accuracy": best_run["test_metrics"]["accuracy"],
                "optimal_threshold": best_run["optimal_threshold"],
            }

    # Save summary statistics JSON
    summary_file = out_path / "summary_statistics.json"
    full_output = {
        "metadata": {
            "dataset": "CIC-IoT-2023",
            "study_duration_seconds": round(total_study_duration, 2),
            "environment": get_environment_metadata(),
            "sample_counts": splits.sample_counts,
            "class_distribution": splits.class_distribution,
        },
        "best_model": best_overall_model_info,
        "models": summary_stats,
    }

    with open(summary_file, "w") as f:
        json.dump(full_output, f, indent=2)

    print(f"\n[CIC-IoT-2023] Summary statistics saved to: {summary_file}")

    # Deploy best model checkpoint to results/ciciot2023/best_model.pt
    if best_overall_model_info is not None:
        import shutil
        src_ckpt = Path(best_overall_model_info["checkpoint_path"])
        dst_ckpt = out_path / "best_model.pt"
        shutil.copyfile(src_ckpt, dst_ckpt)
        print(f"[CIC-IoT-2023] Deployed overall top checkpoint to: {dst_ckpt} (Model: {best_overall_model_info['model_name']})")

    return full_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CIC-IoT-2023 Benchmark Study")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999])
    parser.add_argument("--models", type=str, nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--force_rebuild", action="store_true")
    args = parser.parse_args()

    run_ciciot2023_benchmark_study(
        seeds=args.seeds,
        models=args.models,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        force_rebuild=args.force_rebuild,
    )
