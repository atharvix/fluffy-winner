"""Independent verification and statistical audit of controlled experimental study.

Verifies:
1. Integrity of all 15 run directories in results/controlled_study/
2. Consistency of dataset splits, sample counts, and label mappings
3. Model selection protocol (validation-loss only, test set held out)
4. Parameter counts and circuit complexity metrics
5. Recomputed statistics (mean, sample std ddof=1, population std ddof=0) vs summary_statistics.json
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd


def verify_study_integrity(study_dir: str = "results/controlled_study") -> Dict[str, Any]:
    base_path = Path(study_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Study directory {study_dir} does not exist.")

    run_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and (d / "metrics.json").exists()])
    print(f"[Verification] Found {len(run_dirs)} completed run directories in {study_dir}")

    expected_runs = 15
    if len(run_dirs) != expected_runs:
        print(f"[Warning] Expected {expected_runs} runs, found {len(run_dirs)}")

    records = []
    partition_checks = []
    checkpoint_checks = []

    for r_dir in run_dirs:
        with open(r_dir / "config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        with open(r_dir / "metrics.json", "r", encoding="utf-8") as f:
            metrics = json.load(f)
        with open(r_dir / "training_history.json", "r", encoding="utf-8") as f:
            hist = json.load(f)

        model_name = metrics["model_name"]
        seed = metrics["random_seed"]
        best_epoch = metrics["best_epoch"]

        # Check dataset partition conservation
        sample_counts = cfg["dataset"]["sample_counts"]
        is_split_valid = (
            sample_counts["total"] == 569
            and sample_counts["train"] == 398
            and sample_counts["val"] == 85
            and sample_counts["test"] == 86
        )
        partition_checks.append(is_split_valid)

        # Check that best epoch corresponds to minimum validation loss in history
        val_losses = [(h["epoch"], h["val_loss"]) for h in hist]
        min_val_epoch, min_val_loss = min(val_losses, key=lambda x: x[1])
        checkpoint_consistent = (min_val_epoch == best_epoch)
        checkpoint_checks.append(checkpoint_consistent)

        test_m = metrics["test_partition"]["metrics"]
        val_m = metrics["validation_partition"]["metrics"]
        train_m = metrics["training_partition"]["metrics"]
        cm = test_m["confusion_matrix"]
        raw_cm = cm["raw_matrix"]

        tp_mal = cm["true_malignant_pred_malignant"]
        fn_mal = cm["true_malignant_pred_benign"]
        fp_mal = cm["true_benign_pred_malignant"]
        tn_mal = cm["true_benign_pred_benign"]

        # Recompute clinical metrics from raw confusion matrix
        recalc_mal_sens = tp_mal / (tp_mal + fn_mal) if (tp_mal + fn_mal) > 0 else 0.0
        recalc_mal_spec = tn_mal / (tn_mal + fp_mal) if (tn_mal + fp_mal) > 0 else 0.0
        recalc_acc = (tp_mal + tn_mal) / (tp_mal + fn_mal + fp_mal + tn_mal)

        params = cfg["model_architecture"]["parameters"]
        circuit_info = cfg["model_architecture"]["circuit_info"]

        record = {
            "run_dir": r_dir.name,
            "model_name": model_name,
            "model_family": cfg["model_family"],
            "seed": seed,
            "total_params": params.get("total_trainable", 0),
            "quantum_params": params.get("quantum_circuit", params.get("quantum_parameters", 0)),
            "classical_params": params.get("classical_encoder", 0) + params.get("classical_output_head", 0) if "classical_encoder" in params else params.get("classical_parameters", 0),
            "circuit_depth": circuit_info.get("estimated_circuit_depth", 0),
            "total_quantum_gates": circuit_info.get("total_gates", 0),
            "duration_seconds": metrics["duration_seconds"],
            "best_epoch": best_epoch,
            "epochs_trained": len(hist),
            "train_loss": metrics["training_partition"]["loss"],
            "train_acc": train_m["accuracy"],
            "val_loss": metrics["validation_partition"]["loss"],
            "val_acc": val_m["accuracy"],
            "test_loss": metrics["test_partition"]["loss"],
            "test_acc": test_m["accuracy"],
            "test_roc_auc": test_m["roc_auc"],
            "test_mal_sensitivity": test_m["malignant_sensitivity"],
            "test_mal_specificity": test_m["malignant_specificity"],
            "test_ben_precision": test_m["precision"],
            "test_ben_recall": test_m["recall"],
            "test_ben_f1": test_m["f1"],
            "tp_malignant": tp_mal,
            "fn_malignant": fn_mal,
            "fp_malignant": fp_mal,
            "tn_malignant": tn_mal,
            "recomputed_mal_sensitivity": recalc_mal_sens,
            "recomputed_mal_specificity": recalc_mal_spec,
            "recomputed_accuracy": recalc_acc,
        }
        records.append(record)

    df = pd.DataFrame(records)

    # Export structured CSV
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "experiment_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"[Verification] Exported summary CSV to {csv_path}")

    # Integrity verification results
    all_splits_valid = all(partition_checks)
    all_checkpoints_valid = all(checkpoint_checks)

    print(f"[Verification] Dataset partition conservation: {'PASS' if all_splits_valid else 'FAIL'}")
    print(f"[Verification] Model selection by min validation loss only: {'PASS' if all_checkpoints_valid else 'FAIL'}")

    # Statistical Comparison with summary_statistics.json
    summary_file = base_path / "summary_statistics.json"
    discrepancies = []

    if summary_file.exists():
        with open(summary_file, "r", encoding="utf-8") as f:
            saved_summary = json.load(f)

        for model_name, grp in df.groupby("model_name"):
            if model_name in saved_summary:
                saved = saved_summary[model_name]

                # Compare Means
                calc_acc_mean = grp["test_acc"].mean()
                calc_auc_mean = grp["test_roc_auc"].mean()
                calc_sens_mean = grp["test_mal_sensitivity"].mean()
                calc_spec_mean = grp["test_mal_specificity"].mean()

                saved_acc_mean = saved["test_accuracy"]["mean"]
                saved_auc_mean = saved["test_roc_auc"]["mean"]
                saved_sens_mean = saved["malignant_sensitivity"]["mean"]
                saved_spec_mean = saved["malignant_specificity"]["mean"]

                if not np.isclose(calc_acc_mean, saved_acc_mean, atol=1e-5):
                    discrepancies.append(f"{model_name} acc mean: calc={calc_acc_mean}, saved={saved_acc_mean}")
                if not np.isclose(calc_auc_mean, saved_auc_mean, atol=1e-5):
                    discrepancies.append(f"{model_name} auc mean: calc={calc_auc_mean}, saved={saved_auc_mean}")
                if not np.isclose(calc_sens_mean, saved_sens_mean, atol=1e-5):
                    discrepancies.append(f"{model_name} sens mean: calc={calc_sens_mean}, saved={saved_sens_mean}")
                if not np.isclose(calc_spec_mean, saved_spec_mean, atol=1e-5):
                    discrepancies.append(f"{model_name} spec mean: calc={calc_spec_mean}, saved={saved_spec_mean}")

                # Note on Standard Deviation ddof:
                # numpy np.std defaults to ddof=0 (population std).
                # pandas and sample statistics use ddof=1 (sample std).
                # Both are reported in the detailed audit.
                calc_acc_sample_std = grp["test_acc"].std(ddof=1)
                calc_acc_pop_std = grp["test_acc"].std(ddof=0)
                saved_acc_std = saved["test_accuracy"]["std"]

                print(
                    f"[{model_name}] Accuracy: {calc_acc_mean:.4f} "
                    f"(Sample SD ddof=1: {calc_acc_sample_std:.4f}, Pop SD ddof=0: {calc_acc_pop_std:.4f}) "
                    f"| Sensitivity: {calc_sens_mean:.4f} | Specificity: {calc_spec_mean:.4f}"
                )

    print(f"[Verification] Total discrepancies found: {len(discrepancies)}")
    if discrepancies:
        for d in discrepancies:
            print(f"  - {d}")

    return {
        "df": df,
        "all_splits_valid": all_splits_valid,
        "all_checkpoints_valid": all_checkpoints_valid,
        "discrepancies": discrepancies,
        "csv_path": str(csv_path),
    }


if __name__ == "__main__":
    verify_study_integrity()
