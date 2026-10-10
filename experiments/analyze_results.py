import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_study_results(study_dir: str = "results/controlled_study") -> Dict[str, List[Dict[str, Any]]]:
    """Load all run results grouped by model_name."""
    base_path = Path(study_dir)
    grouped_runs: Dict[str, List[Dict[str, Any]]] = {}

    for run_dir in sorted(base_path.iterdir()):
        if run_dir.is_dir() and (run_dir / "metrics.json").exists():
            with open(run_dir / "metrics.json", "r", encoding="utf-8") as f:
                metrics = json.load(f)
            with open(run_dir / "config.json", "r", encoding="utf-8") as f:
                config = json.load(f)
            with open(run_dir / "training_history.json", "r", encoding="utf-8") as f:
                history = json.load(f)

            model_name = metrics["model_name"]
            if model_name not in grouped_runs:
                grouped_runs[model_name] = []

            grouped_runs[model_name].append({
                "metrics": metrics,
                "config": config,
                "history": history,
                "dir_name": run_dir.name,
            })

    return grouped_runs


def compute_summary_statistics(grouped_runs: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Compute mean and standard deviation across seeds for each metric."""
    summary: Dict[str, Any] = {}

    for model_name, runs in grouped_runs.items():
        accuracies = [r["metrics"]["test_partition"]["metrics"]["accuracy"] for r in runs]
        roc_aucs = [r["metrics"]["test_partition"]["metrics"]["roc_auc"] for r in runs if r["metrics"]["test_partition"]["metrics"]["roc_auc"] is not None]
        mal_sens = [r["metrics"]["test_partition"]["metrics"]["malignant_sensitivity"] for r in runs]
        mal_spec = [r["metrics"]["test_partition"]["metrics"]["malignant_specificity"] for r in runs]
        ben_f1 = [r["metrics"]["test_partition"]["metrics"]["f1"] for r in runs]
        durations = [r["metrics"]["duration_seconds"] for r in runs]
        best_epochs = [r["metrics"]["best_epoch"] for r in runs]
        test_losses = [r["metrics"]["test_partition"]["loss"] for r in runs]

        cfg = runs[0]["config"]
        params = cfg["model_architecture"]["parameters"]
        circuit_info = cfg["model_architecture"]["circuit_info"]

        summary[model_name] = {
            "model_family": cfg["model_family"],
            "num_runs": len(runs),
            "seeds": [r["metrics"]["random_seed"] for r in runs],
            "parameters": params,
            "circuit_info": circuit_info,
            "test_accuracy": {
                "mean": float(np.mean(accuracies)),
                "std": float(np.std(accuracies)),
                "values": accuracies,
            },
            "test_roc_auc": {
                "mean": float(np.mean(roc_aucs)),
                "std": float(np.std(roc_aucs)),
                "values": roc_aucs,
            },
            "malignant_sensitivity": {
                "mean": float(np.mean(mal_sens)),
                "std": float(np.std(mal_sens)),
                "values": mal_sens,
            },
            "malignant_specificity": {
                "mean": float(np.mean(mal_spec)),
                "std": float(np.std(mal_spec)),
                "values": mal_spec,
            },
            "benign_f1": {
                "mean": float(np.mean(ben_f1)),
                "std": float(np.std(ben_f1)),
                "values": ben_f1,
            },
            "duration_seconds": {
                "mean": float(np.mean(durations)),
                "std": float(np.std(durations)),
                "values": durations,
            },
            "best_epoch": {
                "mean": float(np.mean(best_epochs)),
                "std": float(np.std(best_epochs)),
                "values": best_epochs,
            },
            "test_loss": {
                "mean": float(np.mean(test_losses)),
                "std": float(np.std(test_losses)),
                "values": test_losses,
            },
        }

    return summary


def generate_all_plots(
    grouped_runs: Dict[str, List[Dict[str, Any]]],
    summary: Dict[str, Any],
    plots_dir: str = "results/controlled_study/plots",
) -> List[str]:
    """Generate and save publication-grade analysis plots."""
    out_dir = Path(plots_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_plots: List[str] = []

    # Palette
    colors = {
        "Classical_MLP": "#2b5c8f",
        "Standard_VQC_L2": "#d95f02",
        "Hybrid_Reuploading_L2": "#7570b3",
        "Hybrid_Reuploading_L1": "#1b9e77",
        "Hybrid_Reuploading_L4": "#e7298a",
    }

    # 1. Accuracy and ROC-AUC by Model
    models_to_compare = ["Classical_MLP", "Standard_VQC_L2", "Hybrid_Reuploading_L2"]
    available_models = [m for m in models_to_compare if m in summary]

    if available_models:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        x = np.arange(len(available_models))
        width = 0.35

        acc_means = [summary[m]["test_accuracy"]["mean"] for m in available_models]
        acc_stds = [summary[m]["test_accuracy"]["std"] for m in available_models]
        auc_means = [summary[m]["test_roc_auc"]["mean"] for m in available_models]
        auc_stds = [summary[m]["test_roc_auc"]["std"] for m in available_models]

        rects1 = ax.bar(x - width/2, acc_means, width, yerr=acc_stds, label="Test Accuracy", capsize=5, color="#1f77b4", alpha=0.85)
        rects2 = ax.bar(x + width/2, auc_means, width, yerr=auc_stds, label="Test ROC-AUC", capsize=5, color="#2ca02c", alpha=0.85)

        ax.set_ylabel("Score (Mean ± 1 SD)")
        ax.set_title("Model Comparison: Test Accuracy & ROC-AUC (3 Seeds)")
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", "\n") for m in available_models])
        ax.set_ylim(0.85, 1.02)
        ax.legend(loc="lower right")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        # Value annotations
        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8)
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        p1 = out_dir / "accuracy_roc_auc_by_model.png"
        plt.savefig(p1)
        plt.close()
        saved_plots.append(str(p1))

    # 2. Malignant Sensitivity vs Specificity
    if available_models:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        x = np.arange(len(available_models))
        width = 0.35

        sens_means = [summary[m]["malignant_sensitivity"]["mean"] for m in available_models]
        sens_stds = [summary[m]["malignant_sensitivity"]["std"] for m in available_models]
        spec_means = [summary[m]["malignant_specificity"]["mean"] for m in available_models]
        spec_stds = [summary[m]["malignant_specificity"]["std"] for m in available_models]

        r1 = ax.bar(x - width/2, sens_means, width, yerr=sens_stds, label="Malignant Sensitivity (Recall)", capsize=5, color="#d62728", alpha=0.85)
        r2 = ax.bar(x + width/2, spec_means, width, yerr=spec_stds, label="Malignant Specificity (Benign Recall)", capsize=5, color="#9467bd", alpha=0.85)

        ax.set_ylabel("Score (Mean ± 1 SD)")
        ax.set_title("Clinical Metrics: Malignant Sensitivity vs Specificity")
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", "\n") for m in available_models])
        ax.set_ylim(0.80, 1.05)
        ax.legend(loc="lower right")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        for rect in r1:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8)
        for rect in r2:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        p2 = out_dir / "malignant_sensitivity_specificity.png"
        plt.savefig(p2)
        plt.close()
        saved_plots.append(str(p2))

    # 3. Training Duration
    all_models = list(summary.keys())
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    dur_means = [summary[m]["duration_seconds"]["mean"] for m in all_models]
    dur_stds = [summary[m]["duration_seconds"]["std"] for m in all_models]
    bars = ax.bar(range(len(all_models)), dur_means, yerr=dur_stds, capsize=5, color="#34495e", alpha=0.85)
    ax.set_ylabel("Training Duration (Seconds)")
    ax.set_title("Average Training Duration Across Models (3 Seeds)")
    ax.set_xticks(range(len(all_models)))
    ax.set_xticklabels([m.replace("_", "\n") for m in all_models], fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for rect in bars:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}s", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    p3 = out_dir / "training_time_by_model.png"
    plt.savefig(p3)
    plt.close()
    saved_plots.append(str(p3))

    # 4. Training and Validation Loss Curves
    fig, (ax_train, ax_val) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    for model_name, runs in grouped_runs.items():
        if model_name not in ["Classical_MLP", "Standard_VQC_L2", "Hybrid_Reuploading_L2"]:
            continue
        c = colors.get(model_name, "#333333")
        for i, r in enumerate(runs):
            hist = r["history"]
            epochs = [e["epoch"] for e in hist]
            train_losses = [e["train_loss"] for e in hist]
            val_losses = [e["val_loss"] for e in hist]
            label = model_name if i == 0 else None
            ax_train.plot(epochs, train_losses, color=c, alpha=0.6, label=label, linewidth=1.5)
            ax_val.plot(epochs, val_losses, color=c, alpha=0.6, label=label, linewidth=1.5)

    ax_train.set_title("Training Loss Trajectories")
    ax_train.set_xlabel("Epoch")
    ax_train.set_ylabel("BCE Loss")
    ax_train.grid(True, linestyle="--", alpha=0.5)
    ax_train.legend()

    ax_val.set_title("Validation Loss Trajectories")
    ax_val.set_xlabel("Epoch")
    ax_val.set_ylabel("BCE Loss")
    ax_val.grid(True, linestyle="--", alpha=0.5)
    ax_val.legend()

    plt.tight_layout()
    p4 = out_dir / "loss_curves_all_models.png"
    plt.savefig(p4)
    plt.close()
    saved_plots.append(str(p4))

    # 5. Re-uploading Layer Ablation (L=1, L=2, L=4)
    ablation_models = ["Hybrid_Reuploading_L1", "Hybrid_Reuploading_L2", "Hybrid_Reuploading_L4"]
    if all(m in summary for m in ablation_models):
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        layers = [1, 2, 4]
        accs = [summary[m]["test_accuracy"]["mean"] for m in ablation_models]
        acc_errs = [summary[m]["test_accuracy"]["std"] for m in ablation_models]
        aucs = [summary[m]["test_roc_auc"]["mean"] for m in ablation_models]
        auc_errs = [summary[m]["test_roc_auc"]["std"] for m in ablation_models]
        sens = [summary[m]["malignant_sensitivity"]["mean"] for m in ablation_models]
        sens_errs = [summary[m]["malignant_sensitivity"]["std"] for m in ablation_models]

        ax.errorbar(layers, accs, yerr=acc_errs, fmt="-o", label="Test Accuracy", capsize=5, linewidth=2, color="#1f77b4")
        ax.errorbar(layers, aucs, yerr=auc_errs, fmt="-s", label="Test ROC-AUC", capsize=5, linewidth=2, color="#2ca02c")
        ax.errorbar(layers, sens, yerr=sens_errs, fmt="-^", label="Malignant Sensitivity", capsize=5, linewidth=2, color="#d62728")

        ax.set_xlabel("Number of Quantum Data Re-uploading Layers (L)")
        ax.set_ylabel("Metric Score (Mean ± 1 SD)")
        ax.set_title("Performance vs Number of Re-uploading Layers (L = 1, 2, 4)")
        ax.set_xticks(layers)
        ax.set_ylim(0.85, 1.01)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower right")

        plt.tight_layout()
        p5 = out_dir / "reuploading_layers_ablation.png"
        plt.savefig(p5)
        plt.close()
        saved_plots.append(str(p5))

    # 6. Confusion Matrix Breakdown & False Negative Audit
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=300)
    for ax, m_name in zip(axes, ["Classical_MLP", "Standard_VQC_L2", "Hybrid_Reuploading_L2"]):
        if m_name in grouped_runs:
            # Average raw confusion matrix across seeds
            cm_arrays = [np.array(r["metrics"]["test_partition"]["metrics"]["confusion_matrix"]["raw_matrix"]) for r in grouped_runs[m_name]]
            mean_cm = np.mean(cm_arrays, axis=0)

            im = ax.imshow(mean_cm, interpolation="nearest", cmap=plt.cm.Blues)
            ax.set_title(f"{m_name.replace('_', ' ')}\n(Avg over 3 Seeds)")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Pred Malignant", "Pred Benign"])
            ax.set_yticklabels(["True Malignant", "True Benign"])

            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{mean_cm[i, j]:.1f}", ha="center", va="center",
                            color="white" if mean_cm[i, j] > mean_cm.max()/2 else "black", fontsize=11, fontweight="bold")
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")

    plt.tight_layout()
    p6 = out_dir / "confusion_matrices_summary.png"
    plt.savefig(p6)
    plt.close()
    saved_plots.append(str(p6))

    return saved_plots


def main() -> None:
    grouped = load_study_results("results/controlled_study")
    if not grouped:
        print("No controlled study runs found in results/controlled_study.")
        return

    summary = compute_summary_statistics(grouped)
    summary_file = Path("results/controlled_study/summary_statistics.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plots = generate_all_plots(grouped, summary, "results/controlled_study/plots")
    print(f"Summary statistics saved to {summary_file}")
    print(f"Generated {len(plots)} analysis plots in results/controlled_study/plots/")


if __name__ == "__main__":
    main()
