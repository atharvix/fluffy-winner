# Hybrid Classical–Quantum Network Intrusion Detection System (NIDS)

A reproducible, research-quality machine learning and quantum computing framework evaluating **Variational Quantum Circuits (VQC)** and **Quantum Data Re-uploading** against classical deep neural networks on the authentic **UNSW-NB15** network intrusion dataset.

Built with **PyTorch**, **PennyLane**, **Scikit-Learn**, and **FastAPI**, with backend inference support for **Google Stitch** frontend integration.

---

## 1. Project Overview & Research Architecture

Network Intrusion Detection Systems (NIDS) face a fundamental trade-off between **Attack Recall** (detecting stealthy cyber intrusions) and **False Positive Rate** (avoiding analyst alert fatigue). This project implements a controlled empirical benchmark comparing:

1. **Classical MLP Baseline:** Over-parameterized classical feedforward network (6,401 trainable parameters).
2. **Standard VQC Baseline ($L=2$):** Variational quantum classifier with a single initial feature encoding stage $E(x)$ at $t=0$, followed by 2 entangling layers (1,113 total parameters: 1,097 classical encoder, 16 quantum).
3. **Hybrid Quantum Data Re-uploading ($L=1$):** Interleaving classical feature re-uploading and 1 variational layer (1,105 parameters: 1,097 classical, 8 quantum).
4. **Hybrid Quantum Data Re-uploading ($L=2$):** Interleaving classical feature re-uploading across 2 variational layers (1,113 parameters: 1,097 classical, 16 quantum).
5. **Hybrid Quantum Data Re-uploading ($L=4$):** Deep 4-layer re-uploading ansatz (1,129 parameters: 1,097 classical, 32 quantum).

```
Raw Network Flow (42 features: 39 numerical, 3 categorical)
                         |
                         v
  Zero-Leakage Preprocessing (ColumnTransformer on Training Split)
                         |
                         v
     Standardized Feature Vector (Dimension: 63)
                         |
      +------------------+------------------+
      | (Classical)                         | (Hybrid Quantum)
      v                                     v
Classical MLP                       Classical Feature Encoder (63 -> 16 -> 4)
Linear(63, 64) -> BatchNorm -> ReLU         |
Linear(64, 32) -> BatchNorm -> ReLU         v
Linear(32, 1)                      4 Bounded Angles in [-pi, pi]
                                            |
                                            v
                                   4-Qubit Quantum Circuit (PennyLane default.qubit)
                                   - Interleaved Data Re-uploading: RY(theta_in)
                                   - Parameterized Rotations: RY(theta_ry), RZ(theta_rz)
                                   - Ring Entanglement: CNOT(0->1), CNOT(1->2), etc.
                                   - Pauli-Z Expectation Values: <Z_i> in [-1, 1]
                                            |
                                            v
                                   Classical Output Head: Linear(4, 1)
```

---

## 2. Dataset Provenance & Zero-Leakage Protocol

- **Dataset:** UNSW-NB15 Network Intrusion Dataset
- **Authors & Institution:** Nour Moustafa and Jill Slay, Australian Centre for Cyber Security (ACCS), UNSW Canberra.
- **Reference:** Moustafa, N., & Slay, J. (2015). *"UNSW-NB15: a comprehensive data set for network intrusion detection systems."* MilCIS IEEE.
- **Partitions:**
  - `data/raw/UNSW_NB15_training-set.csv` (82,332 records, 45 columns, SHA256: `734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559`)
  - `data/raw/UNSW_NB15_testing-set.csv` (175,341 records, 45 columns, SHA256: `bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa`)
- **Target Semantics:**
  - `0` $\rightarrow$ **`BENIGN`** (legitimate network traffic)
  - `1` $\rightarrow$ **`ATTACK`** (malicious network flow across 9 categories)
- **Zero-Leakage Guarantee:**
  - Columns `id`, `label`, and `attack_cat` are completely excluded from model inputs.
  - Preprocessing (`StandardScaler` and `OneHotEncoder`) is fitted **strictly on the training split**.
  - Held-out test set is evaluated only once on checkpoints selected by minimum validation loss.

---

## 3. Summary of Controlled Experimental Results (3 Independent Seeds: 42, 123, 999)

*All models evaluated on the held-out test split (900 samples: 495 Attacks, 405 Benign). Metrics reported as Mean $\pm$ Sample Standard Deviation across seeds:*

| Model Architecture | Test Accuracy | Macro F1 | Test ROC-AUC | Attack Recall (Sensitivity) | False-Positive Rate (FPR) | Trainable Parameters | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Classical MLP** | $93.19\% \pm 0.65\%$ | $0.9313 \pm 0.0065$ | **$0.9844 \pm 0.0008$** | $92.59\% \pm 1.72\%$ | $6.09\% \pm 1.36\%$ | 6,401 | **$27.06\text{s} \pm 11.21\text{s}$** |
| **Standard VQC ($L=2$)** | $93.00\% \pm 0.33\%$ | $0.9295 \pm 0.0033$ | $0.9835 \pm 0.0007$ | $92.26\% \pm 0.76\%$ | $6.09\% \pm 0.57\%$ | 1,113 | $161.42\text{s} \pm 4.30\text{s}$ |
| **Hybrid Re-uploading ($L=1$)** | $92.81\% \pm 0.28\%$ | $0.9275 \pm 0.0027$ | $0.9834 \pm 0.0014$ | **$93.13\% \pm 1.05\%$** | $7.57\% \pm 0.87\%$ | **1,105** | $30.31\text{s} \pm 13.37\text{s}$ |
| **Hybrid Re-uploading ($L=2$)** | **$93.33\% \pm 0.19\%$** | **$0.9329 \pm 0.0019$** | $0.9835 \pm 0.0013$ | $92.12\% \pm 0.93\%$ | **$5.19\% \pm 0.86\%$** | 1,113 | $35.78\text{s} \pm 0.10\text{s}$ |
| **Hybrid Re-uploading ($L=4$)** | $92.41\% \pm 1.00\%$ | $0.9234 \pm 0.0100$ | $0.9824 \pm 0.0017$ | $92.26\% \pm 1.35\%$ | $7.41\% \pm 1.54\%$ | 1,129 | $63.27\text{s} \pm 0.85\text{s}$ |

### Operational Takeaways:
- **Hybrid Re-uploading ($L=2$)** achieves the lowest False-Positive Rate ($5.19\% \pm 0.86\%$), reducing false alerts by ~9,000 per million flows compared to classical models.
- **Hybrid models require 5.75x fewer parameters** than the classical baseline to reach competitive accuracy.
- **Re-uploading speeds up convergence by 4.5x** over standard VQC ($35.8\text{s}$ vs $161.4\text{s}$).

---

## 4. Installation & Environment Setup (Windows PowerShell)

```powershell
# 1. Clone repository and navigate to root
cd E:\dl

# 2. Activate Virtual Environment
$env:CURL_CA_BUNDLE=""
& ".\.venv\Scripts\Activate.ps1"

# 3. Verify Dependencies
pip install -r requirements.txt
```

---

## 5. Execution Commands

### 5.1 Run the Full Test Suite (47 Tests)
```powershell
.venv\Scripts\pytest.exe -v
```
Verifies dataset pipeline, leakage prevention, quantum measurement bounds, gradient backpropagation, API health, and inference consistency.

### 5.2 Run Controlled 5-Model Experimental Study (15 Runs)
```powershell
.venv\Scripts\python.exe experiments/run_nids_study.py
```
Executes all 5 model architectures across 3 random seeds (`42`, `123`, `999`), saves raw metrics, test predictions, and deploys the best checkpoint.

### 5.3 Generate Comparison Visualizations & Summary JSON
```powershell
.venv\Scripts\python.exe experiments/analyze_nids_results.py
```
Generates all 7 publication-quality plots in `results/network_intrusion/plots/` and saves aggregate statistics to `results/network_intrusion/summary_statistics.json`.

### 5.4 Launch FastAPI Inference Service (for Google Stitch & Web UI)
```powershell
.venv\Scripts\uvicorn.exe api.main:app --host 127.0.0.1 --port 8000 --reload
```
Exposes:
- `GET  /api/health` — Service readiness & model status
- `GET  /api/model-info` — Schema, feature counts, class mappings, circuit metadata
- `POST /api/predict` — Validated JSON batch flow inference with quantum angle & expectation inspection
- `POST /api/predict/csv` — CSV file upload inference returning structured JSON or prediction CSV

Interactive documentation:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

---

## 6. Directory & Artifact Structure

```
E:\dl\
|-- api/
|   |-- __init__.py
|   |-- main.py                    # Production FastAPI backend
|-- data/
|   |-- raw/                       # UNSW-NB15 training and testing CSVs
|-- docs/
|   |-- dataset_schema_reference.md# UNSW-NB15 schema & preprocessing reference
|   |-- frontend_integration.md    # Google Stitch & custom UI integration contract
|-- experiments/
|   |-- run_nids_study.py          # Controlled 15-run experiment runner
|   |-- analyze_nids_results.py    # Plot generation & statistical summary
|-- reports/
|   |-- nids_research_report.md    # Comprehensive scientific research report
|   |-- nids_reproducibility_report.md # Environment & artifact reproducibility audit
|   |-- nids_experiment_summary.csv # Machine-readable tabular summary of all 15 runs
|   |-- breast_cancer_research_report.md # Archived original diagnostic report
|-- results/
|   |-- network_intrusion/         # Primary NIDS model artifacts
|   |   |-- best_model.pt          # Deployed best model checkpoint
|   |   |-- preprocessor.joblib    # Pre-fitted ColumnTransformer
|   |   |-- summary_statistics.json# Aggregate metrics across seeds
|   |   |-- controlled_study/      # 15 individual run directories
|   |   |-- plots/                 # 7 high-resolution comparison plots
|   |-- breast_cancer/             # Archived original breast cancer artifacts
|-- src/
|   |-- data/
|   |   |-- nids_dataset.py        # UNSW-NB15 dataset pipeline & zero-leakage splits
|   |   |-- dataset.py             # Breast cancer dataset pipeline (preserved)
|   |-- evaluation/
|   |   |-- metrics.py             # NIDS and clinical metric computation
|   |-- inference/
|   |   |-- nids_predictor.py      # Production NIDS predictor wrapper
|   |-- models/
|   |   |-- classical.py           # Classical MLP baseline
|   |   |-- hybrid.py              # Hybrid Quantum Classifier (re-uploading & VQC)
|   |   |-- quantum.py             # Differentiable 4-qubit PennyLane layers
|-- tests/
|   |-- test_nids_dataset.py       # NIDS pipeline & leakage prevention tests
|   |-- test_nids_metrics.py       # Detection rate, FPR, and metric tests
|   |-- test_nids_inference_and_api.py # Inference & FastAPI backend integration tests
|   |-- test_quantum_and_hybrid.py # Quantum gate bounds and gradient tests
```

---

## 7. Limitations & Scientific Disclaimer

1. **Simulated Execution:** All quantum circuits were executed using PennyLane's `default.qubit` statevector simulator on CPU. No physical NISQ quantum hardware was utilized.
2. **No Quantum Advantage Claim:** Achieving competitive performance with fewer parameters does not demonstrate quantum advantage. Classical gradient boosted decision trees (XGBoost) remain strong baselines for tabular network flow data.
3. **Responsible Use:** Network flow classifiers should operate as decision-support layers alongside signature-based firewalls, protocol validators, and human analyst review.
