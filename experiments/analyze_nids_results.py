"""Analysis and Visualization Generator for Controlled NIDS Experiments on UNSW-NB15.

Produces:
1. Aggregate statistical summary (Mean ± Std Dev across seeds) saved to JSON.
2. High-resolution comparison plots:
   - Accuracy and Macro F1 comparison.
   - Attack Recall vs False-Positive Rate (operational trade-off).
   - ROC curves using continuous prediction probabilities.
   - Confusion matrices for all models.
   - Training time by architecture.
   - Data re-uploading layer ablation (L=1, L=2, L=4).
   - Parameter counts and circuit complexity.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc


def load_study_data(base_dir: str = "results/network_intrusion/controlled_study") -> Dict[str, Any]:
    summary_file = Path(base_dir) / "study_results_all_runs.json"
    if not summary_file.exists():
        raise FileNotFoundError(f"Study summary not found at: {summary_file}")
    with open(summary_file, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_aggregate_statistics(runs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Group runs by architecture and compute mean and sample standard deviation."""
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for r in runs:
        m_name = r["model_name"]
        by_model.setdefault(m_name, []).append(r)

    stats: Dict[str, Dict[str, Any]] = {}
    metrics_keys = [
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
        "test_roc_auc",
        "test_attack_recall",
        "test_attack_precision",
        "test_attack_f1",
        "test_benign_recall",
        "test_benign_precision",
        "test_benign_f1",
        "test_false_positive_rate",
        "test_false_negative_rate",
        "duration_seconds",
    ]

    for m_name, m_runs in by_model.items():
        m_stats: Dict[str, Any] = {
            "model_family": m_runs[0]["model_family"],
            "total_parameters": m_runs[0]["total_parameters"],
            "seed_count": len(m_runs),
            "seeds": [r["seed"] for r in m_runs],
        }
        for k in metrics_keys:
            vals = [r[k] for r in m_runs if r.get(k) is not None]
            if vals:
                mean_val = float(np.mean(vals))
                std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                m_stats[k] = {
                    "mean": mean_val,
                    "std": std_val,
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "formatted": f"{mean_val:.4f} ± {std_val:.4f}",
                }
        stats[m_name] = m_stats

    return stats


def generate_plots(
    base_dir: str = "results/network_intrusion/controlled_study",
    plots_dir: str = "results/network_intrusion/plots",
) -> None:
    study_data = load_study_data(base_dir)
    runs = study_data["results"]
    agg_stats = compute_aggregate_statistics(runs)

    p_dir = Path(plots_dir)
    p_dir.mkdir(parents=True, exist_ok=True)

    # Save aggregated statistics JSON
    summary_stats_path = Path("results/network_intrusion/summary_statistics.json")
    summary_stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_stats_path, "w", encoding="utf-8") as f:
        json.dump(agg_stats, f, indent=2)
    print(f"Saved aggregated statistics to: {summary_stats_path}")

    model_names = [
        "Classical_MLP",
        "Standard_VQC_L2",
        "Hybrid_Reuploading_L1",
        "Hybrid_Reuploading_L2",
        "Hybrid_Reuploading_L4",
    ]
    display_names = [
        "Classical MLP",
        "Standard VQC (L=2)",
        "Hybrid Re-up (L=1)",
        "Hybrid Re-up (L=2)",
        "Hybrid Re-up (L=4)",
    ]
    colors = ["#2563EB", "#7C3AED", "#059669", "#D97706", "#DC2626"]

    # 1. Plot: Accuracy & Macro F1
    plt.figure(figsize=(10, 5))
    x = np.arange(len(model_names))
    width = 0.35

    acc_means = [agg_stats[m]["test_accuracy"]["mean"] * 100 for m in model_names]
    acc_stds = [agg_stats[m]["test_accuracy"]["std"] * 100 for m in model_names]
    f1_means = [agg_stats[m]["test_macro_f1"]["mean"] * 100 for m in model_names]
    f1_stds = [agg_stats[m]["test_macro_f1"]["std"] * 100 for m in model_names]

    plt.bar(x - width/2, acc_means, width, yerr=acc_stds, capsize=5, label="Test Accuracy (%)", color="#3B82F6", alpha=0.9)
    plt.bar(x + width/2, f1_means, width, yerr=f1_stds, capsize=5, label="Macro F1 (%)", color="#10B981", alpha=0.9)

    plt.ylabel("Score (%)", fontsize=11)
    plt.title("Test Accuracy and Macro F1 Across Models (Mean ± Std Dev over 3 Seeds)", fontsize=13, fontweight="bold")
    plt.xticks(x, display_names, rotation=15, ha="right", fontsize=10)
    plt.ylim(50, 100)
    plt.legend(frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(p_dir / "accuracy_macro_f1_by_model.png", dpi=300)
    plt.close()

    # 2. Plot: Attack Recall vs False-Positive Rate
    plt.figure(figsize=(10, 5))
    rec_means = [agg_stats[m]["test_attack_recall"]["mean"] * 100 for m in model_names]
    rec_stds = [agg_stats[m]["test_attack_recall"]["std"] * 100 for m in model_names]
    fpr_means = [agg_stats[m]["test_false_positive_rate"]["mean"] * 100 for m in model_names]
    fpr_stds = [agg_stats[m]["test_false_positive_rate"]["std"] * 100 for m in model_names]

    plt.bar(x - width/2, rec_means, width, yerr=rec_stds, capsize=5, label="Attack Recall / Detection Rate (%)", color="#EF4444", alpha=0.9)
    plt.bar(x + width/2, fpr_means, width, yerr=fpr_stds, capsize=5, label="False-Positive Rate (%)", color="#F59E0B", alpha=0.9)

    plt.ylabel("Percentage (%)", fontsize=11)
    plt.title("Attack Recall (Detection Rate) vs False-Positive Rate Across Models", fontsize=13, fontweight="bold")
    plt.xticks(x, display_names, rotation=15, ha="right", fontsize=10)
    plt.ylim(0, 105)
    plt.legend(frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(p_dir / "attack_recall_fpr_by_model.png", dpi=300)
    plt.close()

    # 3. Plot: ROC Curves using actual continuous prediction probabilities (Seed 42)
    plt.figure(figsize=(8, 7))
    for m_name, d_name, color in zip(model_names, display_names, colors):
        run_file = Path(base_dir) / f"{m_name}_seed_42" / "test_predictions.json"
        if run_file.exists():
            with open(run_file, "r", encoding="utf-8") as f:
                p_data = json.load(f)
            y_t = np.array(p_data["y_true"])
            y_p = np.array(p_data["y_prob_attack"])
            fpr, tpr, _ = roc_curve(y_t, y_p)
            r_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{d_name} (AUC = {r_auc:.4f})", color=color, linewidth=2)

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (AUC = 0.50)")
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate (Recall)", fontsize=11)
    plt.title("ROC Curves on Held-Out Test Set (Evaluated with Prediction Probabilities)", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(p_dir / "roc_curves_all_models.png", dpi=300)
    plt.close()

    # 4. Plot: Confusion Matrices (Seed 42)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, m_name, d_name in zip(axes, model_names, display_names):
        m_file = Path(base_dir) / f"{m_name}_seed_42" / "metrics.json"
        if m_file.exists():
            with open(m_file, "r", encoding="utf-8") as f:
                m_data = json.load(f)
            cm = np.array(m_data["test_metrics"]["confusion_matrix"]["raw_matrix"])
            im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="black" if cm[i, j] < cm.max()/2 else "white", fontsize=11, fontweight="bold")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["BENIGN", "ATTACK"])
            ax.set_yticklabels(["BENIGN", "ATTACK"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(d_name, fontsize=10, fontweight="bold")
    plt.suptitle("Confusion Matrices on Held-Out Test Split (Seed 42)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(p_dir / "confusion_matrices_summary.png", dpi=300)
    plt.close()

    # 5. Plot: Training Duration
    plt.figure(figsize=(9, 5))
    dur_means = [agg_stats[m]["duration_seconds"]["mean"] for m in model_names]
    dur_stds = [agg_stats[m]["duration_seconds"]["std"] for m in model_names]
    plt.bar(display_names, dur_means, yerr=dur_stds, capsize=5, color="#6366F1", alpha=0.85)
    plt.ylabel("Wall-Clock Training Time (seconds)", fontsize=11)
    plt.title("Model Training Duration (Seconds, Mean ± Std Dev over 3 Seeds)", fontsize=13, fontweight="bold")
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(p_dir / "training_time_by_model.png", dpi=300)
    plt.close()

    # 6. Plot: Data Re-uploading Layer Ablation (L=1, L=2, L=4)
    reup_layers = [1, 2, 4]
    reup_models = ["Hybrid_Reuploading_L1", "Hybrid_Reuploading_L2", "Hybrid_Reuploading_L4"]
    reup_acc = [agg_stats[m]["test_accuracy"]["mean"] * 100 for m in reup_models]
    reup_f1 = [agg_stats[m]["test_macro_f1"]["mean"] * 100 for m in reup_models]
    reup_recall = [agg_stats[m]["test_attack_recall"]["mean"] * 100 for m in reup_models]
    reup_fpr = [agg_stats[m]["test_false_positive_rate"]["mean"] * 100 for m in reup_models]

    plt.figure(figsize=(9, 5))
    plt.plot(reup_layers, reup_acc, "o-", label="Test Accuracy (%)", color="#2563EB", linewidth=2, markersize=8)
    plt.plot(reup_layers, reup_f1, "s-", label="Macro F1 (%)", color="#10B981", linewidth=2, markersize=8)
    plt.plot(reup_layers, reup_recall, "^-", label="Attack Recall (%)", color="#DC2626", linewidth=2, markersize=8)
    plt.plot(reup_layers, reup_fpr, "d--", label="False-Positive Rate (%)", color="#F59E0B", linewidth=2, markersize=8)
    plt.xlabel("Number of Data Re-uploading Layers (L)", fontsize=11)
    plt.ylabel("Metric Score (%)", fontsize=11)
    plt.title("Data Re-uploading Layer Ablation (L=1, L=2, L=4)", fontsize=13, fontweight="bold")
    plt.xticks(reup_layers, ["L=1 Layer", "L=2 Layers", "L=4 Layers"])
    plt.legend(frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(p_dir / "reuploading_layers_ablation.png", dpi=300)
    plt.close()

    # 7. Plot: Parameter Counts and Circuit Complexity
    plt.figure(figsize=(10, 5))
    param_counts = [agg_stats[m]["total_parameters"] for m in model_names]
    plt.bar(display_names, param_counts, color="#0D9488", alpha=0.85)
    for i, v in enumerate(param_counts):
        plt.text(i, v + max(param_counts)*0.02, f"{v:,}", ha="center", fontweight="bold", fontsize=10)
    plt.ylabel("Total Trainable Parameters", fontsize=11)
    plt.title("Total Trainable Parameters Across Architectures", fontsize=13, fontweight="bold")
    plt.xticks(rotation=15, ha="right")
    plt.ylim(0, max(param_counts) * 1.15)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(p_dir / "parameter_counts_circuit_complexity.png", dpi=300)
    plt.close()

    print(f"All 7 comparison plots successfully generated in: {p_dir}")


if __name__ == "__main__":
    generate_plots()
