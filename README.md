# Breast Cancer Wisconsin Diagnostic Classification Pipeline & Controlled Experimental Study

This repository implements reproducible dataset pipelines, clinical evaluation metrics, classical baseline models (MLP), standard variational quantum classifiers (VQC), and differentiable hybrid data re-uploading quantum classifiers using PennyLane and PyTorch.

---

## Controlled Model Architectures Comparison

| Architecture | Front-End Encoder | Quantum Circuit Structure | Quantum Gates & Depth | Output Head | Total Trainable Parameters |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Classical MLP** | None (direct) | None (Classical) | N/A | Linear(32, 1) | **4,289** (Classical) |
| **Standard VQC ($L=2$)** | Linear(30, 16) $\to$ ReLU $\to$ Linear(16, 4) $\to$ $\pi \cdot \tanh$ (564 params) | Single Encoding $E(x)$ at $t=0$, followed by 2 entangling layers $U_1, U_2$ | 28 gates (depth 7) | Linear(4, 1) (5 params) | **585** (569 Classical, 16 Quantum) |
| **Hybrid Re-uploading ($L=1$)** | Linear(30, 16) $\to$ ReLU $\to$ Linear(16, 4) $\to$ $\pi \cdot \tanh$ (564 params) | 1 layer of interleaved $E(x)$ and $U_1$ | 16 gates (depth 4) | Linear(4, 1) (5 params) | **577** (569 Classical, 8 Quantum) |
| **Hybrid Re-uploading ($L=2$)** | Linear(30, 16) $\to$ ReLU $\to$ Linear(16, 4) $\to$ $\pi \cdot \tanh$ (564 params) | 2 layers of interleaved $E(x)$ and $U_l$ | 32 gates (depth 8) | Linear(4, 1) (5 params) | **585** (569 Classical, 16 Quantum) |
| **Hybrid Re-uploading ($L=4$)** | Linear(30, 16) $\to$ ReLU $\to$ Linear(16, 4) $\to$ $\pi \cdot \tanh$ (564 params) | 4 layers of interleaved $E(x)$ and $U_l$ | 64 gates (depth 16) | Linear(4, 1) (5 params) | **601** (569 Classical, 32 Quantum) |

---

## Dataset & Leakage Prevention Protocol

- **Dataset:** Scikit-learn Wisconsin Breast Cancer Diagnostic dataset (569 samples, 30 continuous features).
- **Label Mapping:** `0` $\rightarrow$ **Malignant** (212 samples), `1` $\rightarrow$ **Benign** (357 samples).
- **Stratified Partition:** Train (70%, 398 samples), Validation (15%, 85 samples), Test (15%, 86 samples).
- **Zero-Leakage Guarantee:** `StandardScaler` is fitted exclusively on the training partition.
- **Model Checkpointing:** All models select checkpoints based strictly on **minimum validation loss**. The test set is evaluated only once on the selected checkpoint.

---

## Summary of Controlled Experimental Results (3 Independent Random Seeds: 42, 123, 999)

*All models evaluated on the held-out test split (86 samples: 32 Malignant, 54 Benign). Metrics reported as Mean $\pm$ 1 Standard Deviation across seeds:*

| Model Architecture | Test Accuracy | Test ROC-AUC | Malignant Sensitivity | Malignant Specificity | Benign F1 | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Classical MLP** | $98.06\% \pm 1.10\%$ | $0.9979 \pm 0.0030$ | $94.79\% \pm 2.95\%$ | $100.0\% \pm 0.00\%$ | $0.9849 \pm 0.0084$ | $2.17\text{s} \pm 0.31\text{s}$ |
| **Standard VQC ($L=2$)** | $96.90\% \pm 1.45\%$ | $0.9934 \pm 0.0064$ | $94.79\% \pm 1.47\%$ | $98.15\% \pm 1.51\%$ | $0.9754 \pm 0.0116$ | $5.71\text{s} \pm 1.17\text{s}$ |
| **Hybrid Re-uploading ($L=1$)** | $95.35\% \pm 2.51\%$ | $0.9861 \pm 0.0109$ | $95.83\% \pm 1.47\%$ | $95.06\% \pm 3.81\%$ | $0.9621 \pm 0.0208$ | $5.56\text{s} \pm 0.79\text{s}$ |
| **Hybrid Re-uploading ($L=2$)** | **$98.45\% \pm 1.45\%$** | **$0.9979 \pm 0.0026$** | **$96.88\% \pm 4.42\%$** | $99.38\% \pm 0.87\%$ | **$0.9879 \pm 0.0112$** | $7.41\text{s} \pm 1.77\text{s}$ |
| **Hybrid Re-uploading ($L=4$)** | $97.67\% \pm 1.90\%$ | $0.9946 \pm 0.0057$ | $97.92\% \pm 1.47\%$ | $97.53\% \pm 2.31\%$ | $0.9812 \pm 0.0154$ | $14.92\text{s} \pm 1.29\text{s}$ |

---

## How to Run

### 1. Run Complete Test Suite
```powershell
.venv\Scripts\pytest.exe -v
```

### 2. Run Controlled Experimental Study (15 runs across 3 seeds)
```powershell
.venv\Scripts\python.exe experiments/run_study.py
```

### 3. Generate Analysis Visualizations & Summary JSON
```powershell
.venv\Scripts\python.exe experiments/analyze_results.py
```

### 4. Launch the Streamlit Web Application
```powershell
.venv\Scripts\streamlit.exe run app.py
```
Provides:
- Batch CSV upload with zero-leakage feature validation and prediction export.
- Single-sample clinical feature inspector with intermediate quantum angle ($\theta_{\text{in}}$) and Pauli-Z expectation ($\langle Z_i \rangle$) state inspection.
- Interactive benchmark gallery and artifact explorer.
- Research disclaimer and architectural documentation.

---

## Artifact Structure

- Original Baseline Results: [`results/`](file:///e:/dl/results) (`config.json`, `training_history.json`, `best_model.pt`, `metrics.json`)
- Original Hybrid Single Run: [`results/hybrid_quantum/`](file:///e:/dl/results/hybrid_quantum)
- Controlled Comparison Matrix: [`results/controlled_study/`](file:///e:/dl/results/controlled_study) (15 distinct per-seed run directories)
- Aggregated Statistical Summary: [`results/controlled_study/summary_statistics.json`](file:///e:/dl/results/controlled_study/summary_statistics.json)
- Analysis Plots: [`results/controlled_study/plots/`](file:///e:/dl/results/controlled_study/plots)
  - `accuracy_roc_auc_by_model.png`
  - `malignant_sensitivity_specificity.png`
  - `training_time_by_model.png`
  - `loss_curves_all_models.png`
  - `reuploading_layers_ablation.png`
  - `confusion_matrices_summary.png`
