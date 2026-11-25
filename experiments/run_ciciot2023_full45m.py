"""Full-Scale 45,019,243-Flow CIC-IoT-2023 Training and Evaluation Benchmark.

Evaluates 6 architectures across random seeds [42, 123, 999] on 100% of all 63 CSV files:
1. Classical MLP Baseline (64 -> 32)
2. Classical Bottleneck MLP Control (16 -> 4 -> 1)
3. Standard VQC (Single initial encoding E(x), L=2 variational layers)
4. Hybrid Quantum Data Re-uploading (L=1)
5. Hybrid Quantum Data Re-uploading (L=2)
6. Hybrid Quantum Data Re-uploading (L=4)

Hardware Acceleration:
- Classical models utilize CUDA (NVIDIA GeForce RTX 5060 8GB, batch size 8192).
- Quantum circuits utilize PennyLane autograd simulator (batch size 4096).
- Streams from 63 compressed Parquet shards to guarantee deterministic O(1) RAM usage (~1-2 GB).
- BCEWithLogitsLoss(pos_weight=num_benign/num_attack) prevents flood collapse.
- Validates on ~4.5M flows, tunes tau* via Youden's J, and evaluates 100% of ~9M test flows.
- Computes per-category recall across all 33 attack classes plus BenignTraffic.
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

import joblib
import numpy as np
import pandas as pd
import pennylane as qml
import sklearn
import torch
import torch.nn as nn

from src.data.ciciot2023_dataset import (
    CICIOT2023StreamDataset,
    prepare_full_ciciot2023_45m,
)
from src.evaluation.metrics import compute_nids_metrics, optimize_decision_threshold
from src.models.classical import ClassicalBottleneckMLP, ClassicalMLP
from src.models.hybrid import HybridQuantumClassifier, StandardVQCClassifier


def set_seed(seed: int = 42) -> None:
    """Set random seeds across python, numpy, and torch."""
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


def evaluate_sharded_split(
    model: nn.Module,
    shard_paths: List[Path],
    feature_names: List[str],
    scaler: Any,
    criterion: nn.Module,
    device: torch.device,
    batch_size: int = 8192,
    threshold: float = 0.5,
    track_per_class: bool = False,
) -> Tuple[float, Dict[str, Any], np.ndarray, np.ndarray, Optional[Dict[str, float]]]:
    """
    Stream evaluation across all shards in the partition without storing all data in memory.
    Returns: (average_loss, metrics_dict, y_true_all, y_probs_all, per_class_recall)
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    all_y_true: List[np.ndarray] = []
    all_y_probs: List[np.ndarray] = []

    # Category counters if tracking per-attack recall
    cat_correct: Dict[str, int] = {}
    cat_total: Dict[str, int] = {}

    with torch.no_grad():
        for shard_p in shard_paths:
            df = pd.read_parquet(shard_p)
            x_raw = df[feature_names].values.astype(np.float32)
            x_scaled = scaler.transform(x_raw).astype(np.float32)
            y_arr = df["binary_label"].values.astype(np.float32)
            raw_labels = df["Label"].values if "Label" in df.columns else (df["label"].values if "label" in df.columns else None)

            n_samples = len(y_arr)
            shard_logits: List[torch.Tensor] = []

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                x_b = torch.from_numpy(x_scaled[start_idx:end_idx]).to(device)
                y_b = torch.from_numpy(y_arr[start_idx:end_idx]).unsqueeze(1).to(device)

                b_logits = model(x_b)
                b_loss = criterion(b_logits, y_b).item()
                total_loss += b_loss * (end_idx - start_idx)
                shard_logits.append(b_logits.cpu())

            cat_logits = torch.cat(shard_logits, dim=0)
            shard_probs = torch.sigmoid(cat_logits).numpy().ravel()
            shard_preds = (shard_probs >= threshold).astype(np.int64)

            all_y_true.append(y_arr.astype(np.int64))
            all_y_probs.append(shard_probs)
            total_samples += n_samples

            # Per-category recall tracking
            if track_per_class and raw_labels is not None:
                for label_str, pred, true_b in zip(raw_labels, shard_preds, y_arr):
                    lbl = str(label_str).strip()
                    cat_total[lbl] = cat_total.get(lbl, 0) + 1
                    # For benign, correct if pred == 0
                    if true_b == 0 and pred == 0:
                        cat_correct[lbl] = cat_correct.get(lbl, 0) + 1
                    # For attack, correct if pred == 1
                    elif true_b == 1 and pred == 1:
                        cat_correct[lbl] = cat_correct.get(lbl, 0) + 1

    y_true_concat = np.concatenate(all_y_true, axis=0)
    y_probs_concat = np.concatenate(all_y_probs, axis=0)
    avg_loss = total_loss / max(total_samples, 1)

    metrics = compute_nids_metrics(y_true=y_true_concat, y_pred_probs=y_probs_concat, threshold=threshold)

    per_class_recall: Optional[Dict[str, float]] = None
    if track_per_class:
        per_class_recall = {}
        for cat, tot in sorted(cat_total.items()):
            corr = cat_correct.get(cat, 0)
            per_class_recall[cat] = round(corr / max(tot, 1), 4)

    return float(avg_loss), metrics, y_true_concat, y_probs_concat, per_class_recall


def train_and_evaluate_full45m_run(
    model_family: str,
    model_name: str,
    seed: int,
    model_kwargs: Dict[str, Any],
    train_kwargs: Dict[str, Any],
    metadata: Dict[str, Any],
    train_shards: List[Path],
    val_shards: List[Path],
    test_shards: List[Path],
    scaler: Any,
    base_output_dir: str = "results/ciciot2023_full45m/controlled_study",
) -> Dict[str, Any]:
    """
    Execute a single controlled run on the full 45M sharded partition.
    """
    set_seed(seed)
    feature_names = metadata["feature_names"]
    input_dim = len(feature_names)
    pos_weight = metadata["pos_weight"]

    # Device assignment: 100% on GPU for both Classical and Quantum models
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_default_device("cuda")
        batch_size = train_kwargs.get("classical_batch_size", 8192)
        print(f"[{model_name} | Seed {seed}] Running on CUDA GPU: {torch.cuda.get_device_name(0)} (Batch Size: {batch_size})")
    else:
        device = torch.device("cpu")
        torch.set_default_device("cpu")
        batch_size = train_kwargs.get("quantum_batch_size", 4096)
        print(f"[{model_name} | Seed {seed}] Running on CPU PennyLane backprop (Batch Size: {batch_size})")

    lr = train_kwargs.get("lr", 0.001)
    weight_decay = train_kwargs.get("weight_decay", 1e-4)
    epochs = train_kwargs.get("epochs", 15)
    patience = train_kwargs.get("patience", 3)

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

    # Class-weighted loss to counter 97.6% attack skew
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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

        # Streaming training dataset
        train_dataset = CICIOT2023StreamDataset(
            shard_paths=train_shards,
            feature_names=feature_names,
            scaler=scaler,
            batch_size=batch_size,
            shuffle_shards=True,
            random_seed=seed + epoch,
        )

        batch_count = 0
        for x_batch, y_batch in train_dataset:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            bs = len(x_batch)
            train_loss_total += loss.item() * bs
            n_train_samples += bs
            batch_count += 1

            if batch_count % 500 == 0 or batch_count == 1:
                elapsed = time.time() - t_epoch_start
                rate = n_train_samples / max(elapsed, 0.01)
                print(f"    Epoch {epoch:02d} | Step {batch_count:,} ({n_train_samples:,} flows, {rate:,.0f} flows/sec)", flush=True)

        epoch_duration = time.time() - t_epoch_start
        avg_train_loss = train_loss_total / max(n_train_samples, 1)

        # Validation split evaluation across all ~4.5M validation flows
        t_val_start = time.time()
        val_loss, val_metrics_interim, _, _, _ = evaluate_sharded_split(
            model=model,
            shard_paths=val_shards,
            feature_names=feature_names,
            scaler=scaler,
            criterion=criterion,
            device=device,
            batch_size=batch_size * 2,
            threshold=0.5,
            track_per_class=False,
        )
        val_eval_duration = time.time() - t_val_start

        epoch_histories.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 5),
            "val_loss": round(val_loss, 5),
            "val_accuracy": val_metrics_interim["accuracy"],
            "val_macro_f1": val_metrics_interim["macro_f1"],
            "train_duration_s": round(epoch_duration, 2),
            "val_duration_s": round(val_eval_duration, 2),
        })

        print(
            f"  Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {avg_train_loss:.4f} ({epoch_duration:.1f}s) | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val F1: {val_metrics_interim['macro_f1']:.4f} ({val_eval_duration:.1f}s)",
            flush=True,
        )

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
    print(f"[{model_name} | Seed {seed}] Optimizing decision threshold tau* on validation partition...")
    val_loss, _, y_val_true, y_val_probs, _ = evaluate_sharded_split(
        model=model,
        shard_paths=val_shards,
        feature_names=feature_names,
        scaler=scaler,
        criterion=criterion,
        device=device,
        batch_size=batch_size * 2,
        threshold=0.5,
        track_per_class=False,
    )

    best_tau = optimize_decision_threshold(
        y_val_true=y_val_true,
        y_val_probs=y_val_probs,
        metric="youden_j",
        max_allowed_fpr=0.05,
    )
    print(f"[{model_name} | Seed {seed}] Optimal Validation Threshold tau* = {best_tau:.4f}")

    # Re-evaluate validation metrics at tau*
    val_metrics = compute_nids_metrics(y_true=y_val_true, y_pred_probs=y_val_probs, threshold=best_tau)

    # 2. Held-out test evaluation on 100% of test shards (~9,000,000+ flows)
    print(f"[{model_name} | Seed {seed}] Evaluating 100% of held-out test shards (~9M flows) using tau* = {best_tau:.4f}...")
    t_test_start = time.time()
    test_loss, test_metrics, _, _, per_class_recall = evaluate_sharded_split(
        model=model,
        shard_paths=test_shards,
        feature_names=feature_names,
        scaler=scaler,
        criterion=criterion,
        device=device,
        batch_size=batch_size * 2,
        threshold=best_tau,
        track_per_class=True,
    )
    test_eval_time = time.time() - t_test_start

    test_metrics["per_attack_category_recall"] = per_class_recall

    print(
        f"[{model_name} | Seed {seed}] Test Evaluation Complete in {test_eval_time:.1f}s:\n"
        f"  - Accuracy:       {test_metrics['accuracy'] * 100:.2f}%\n"
        f"  - Macro F1:       {test_metrics['macro_f1']:.4f}\n"
        f"  - Attack Recall:  {test_metrics['attack_recall'] * 100:.2f}%\n"
        f"  - Benign Recall:  {test_metrics['benign_recall'] * 100:.2f}%\n"
        f"  - False Alarm:    {test_metrics['false_positive_rate'] * 100:.2f}%\n"
        f"  - ROC-AUC:        {test_metrics.get('roc_auc', 'N/A')}"
    )

    # Save portable CPU-mapped checkpoint
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


def run_ciciot2023_full45m_study(
    seeds: Optional[List[int]] = None,
    models: Optional[List[str]] = None,
    output_dir: str = "results/ciciot2023_full45m",
    classical_batch_size: int = 8192,
    quantum_batch_size: int = 4096,
    epochs: int = 15,
    patience: int = 3,
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    """
    Main orchestrator for the full 45,019,243-flow benchmark study.
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
    print("STARTING FULL-SCALE 45,019,243-FLOW CIC-IoT-2023 BENCHMARK STUDY")
    print(f"Models: {list(selected_configs.keys())}")
    print(f"Seeds: {seeds}")
    print(f"Classical Batch Size: {classical_batch_size} | Quantum Batch Size: {quantum_batch_size}")
    print(f"Max Epochs: {epochs} | Patience: {patience}")
    print("=" * 80)

    # 1. Build or retrieve chunked Parquet shards across all 63 files
    cache_dir = "data/ciciot2023/full45m_parquet"
    scaler_path = out_path / "preprocessor.joblib"
    metadata = prepare_full_ciciot2023_45m(
        data_dir="data/ciciot2023",
        cache_dir=cache_dir,
        scaler_path=str(scaler_path),
        val_ratio=0.10,
        test_ratio=0.20,
        random_seed=42,
        force_rebuild=force_rebuild,
    )

    scaler = joblib.load(scaler_path)
    train_shards = sorted(list((Path(cache_dir) / "train_shards").glob("*.parquet")))
    val_shards = sorted(list((Path(cache_dir) / "val_shards").glob("*.parquet")))
    test_shards = sorted(list((Path(cache_dir) / "test_shards").glob("*.parquet")))

    print(
        f"[CIC-IoT-2023 Full45M] Ready with shards: "
        f"Train={len(train_shards)}, Val={len(val_shards)}, Test={len(test_shards)}"
    )

    train_kwargs = {
        "classical_batch_size": classical_batch_size,
        "quantum_batch_size": quantum_batch_size,
        "lr": 0.001,
        "weight_decay": 1e-4,
        "epochs": epochs,
        "patience": patience,
    }

    all_run_results: Dict[str, List[Dict[str, Any]]] = {name: [] for name in selected_configs}
    total_study_start = time.time()

    for model_name, cfg in selected_configs.items():
        print("\n" + "#" * 70)
        print(f"BENCHMARKING ARCHITECTURE (45M FLOWS): {model_name}")
        print("#" * 70)

        for seed in seeds:
            print(f"\n>>> Running Architecture: {model_name} | Random Seed: {seed} <<<")
            run_res = train_and_evaluate_full45m_run(
                model_family=cfg["family"],
                model_name=model_name,
                seed=seed,
                model_kwargs=cfg["kwargs"],
                train_kwargs=train_kwargs,
                metadata=metadata,
                train_shards=train_shards,
                val_shards=val_shards,
                test_shards=test_shards,
                scaler=scaler,
                base_output_dir=str(controlled_study_dir),
            )
            all_run_results[model_name].append(run_res)

    total_study_duration = time.time() - total_study_start

    # Compute cross-seed summary statistics
    summary_stats: Dict[str, Any] = {}
    best_overall_f1 = -1.0
    best_overall_model_info: Optional[Dict[str, Any]] = None

    metric_keys = [
        "accuracy", "balanced_accuracy", "macro_f1", "attack_recall", "attack_precision",
        "attack_f1", "benign_recall", "benign_precision", "false_positive_rate", "roc_auc"
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
            "test_eval_time_mean_s": round(float(np.mean([r["test_eval_time_seconds"] for r in runs])), 2),
            "optimal_threshold_mean": round(float(np.mean([r["optimal_threshold"] for r in runs])), 4),
            "optimal_threshold_std": round(float(np.std([r["optimal_threshold"] for r in runs])), 4),
            "test_metrics": {},
            "per_attack_category_recall_mean": {},
        }

        for k in metric_keys:
            vals = [r["test_metrics"][k] for r in runs if k in r["test_metrics"] and r["test_metrics"][k] is not None]
            if vals:
                model_summary["test_metrics"][k] = {
                    "mean": round(float(np.mean(vals)), 4),
                    "std": round(float(np.std(vals)), 4),
                }

        # Average per-attack category recall
        all_cats = runs[0]["test_metrics"].get("per_attack_category_recall", {}).keys()
        for cat in all_cats:
            cat_vals = [
                r["test_metrics"]["per_attack_category_recall"][cat]
                for r in runs
                if "per_attack_category_recall" in r["test_metrics"] and cat in r["test_metrics"]["per_attack_category_recall"]
            ]
            if cat_vals:
                model_summary["per_attack_category_recall_mean"][cat] = round(float(np.mean(cat_vals)), 4)

        summary_stats[model_name] = model_summary

        mean_f1 = model_summary["test_metrics"]["macro_f1"]["mean"]
        if mean_f1 > best_overall_f1:
            best_overall_f1 = mean_f1
            best_run = max(runs, key=lambda r: r["test_metrics"]["macro_f1"])
            best_overall_model_info = {
                "model_name": model_name,
                "seed": best_run["seed"],
                "checkpoint_path": best_run["checkpoint_path"],
                "macro_f1": best_run["test_metrics"]["macro_f1"],
                "accuracy": best_run["test_metrics"]["accuracy"],
                "optimal_threshold": best_run["optimal_threshold"],
            }

    summary_file = out_path / "summary_statistics.json"
    full_output = {
        "metadata": {
            "dataset": "CIC-IoT-2023-Full-45M",
            "study_duration_seconds": round(total_study_duration, 2),
            "environment": get_environment_metadata(),
            "total_records": metadata["total_records"],
            "split_counts": metadata["split_counts"],
            "class_distribution": metadata["class_distribution"],
            "pos_weight": metadata["pos_weight"],
        },
        "best_model": best_overall_model_info,
        "models": summary_stats,
    }

    with open(summary_file, "w") as f:
        json.dump(full_output, f, indent=2)

    print(f"\n[CIC-IoT-2023 Full45M] Summary statistics saved to: {summary_file}")

    if best_overall_model_info is not None:
        import shutil
        src_ckpt = Path(best_overall_model_info["checkpoint_path"])
        dst_ckpt = out_path / "best_model.pt"
        shutil.copyfile(src_ckpt, dst_ckpt)
        print(f"[CIC-IoT-2023 Full45M] Deployed top model to: {dst_ckpt}")

    return full_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Full-Scale 45M CIC-IoT-2023 Study")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999])
    parser.add_argument("--models", type=str, nargs="+", default=None)
    parser.add_argument("--classical_batch_size", type=int, default=8192)
    parser.add_argument("--quantum_batch_size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--force_rebuild", action="store_true")
    args = parser.parse_args()

    run_ciciot2023_full45m_study(
        seeds=args.seeds,
        models=args.models,
        classical_batch_size=args.classical_batch_size,
        quantum_batch_size=args.quantum_batch_size,
        epochs=args.epochs,
        patience=args.patience,
        force_rebuild=args.force_rebuild,
    )
    sys.exit(0)
