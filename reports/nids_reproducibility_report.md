# Reproducibility Report: Hybrid Classical–Quantum NIDS on UNSW-NB15

**Environment Verification & Artifact Provenance**  
**Project:** Hybrid Classical–Quantum NIDS  
**Repository Path:** `E:\dl`  
**Execution Timestamp:** October 2026  

---

## 1. Runtime Environment & Dependencies

| Dependency / Component | Version | Verification Command |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 (AMD64) | `[System.Environment]::OSVersion` |
| **Python Runtime** | Python 3.13.6 | `python --version` |
| **PyTorch** | 2.14.1 | `python -c "import torch; print(torch.__version__)"` |
| **PennyLane** | 0.45.1 | `python -c "import pennylane as qml; print(qml.__version__)"` |
| **PennyLane-Lightning** | 0.45.0 | `python -c "import pennylane_lightning; print(pennylane_lightning.__version__)"` |
| **Scikit-Learn** | 1.9.1 | `python -c "import sklearn; print(sklearn.__version__)"` |
| **Pandas** | 3.0.6 | `python -c "import pandas as pd; print(pd.__version__)"` |
| **NumPy** | 2.5.3 | `python -c "import numpy as np; print(np.__version__)"` |
| **FastAPI** | 0.143.0 | `python -c "import fastapi; print(fastapi.__version__)"` |
| **Pytest** | 9.1.1 | `pytest --version` |

---

## 2. Dataset Provenance & File Integrity

| Partition File | Exact Path | Record Count | Feature Count | SHA-256 Checksum |
| :--- | :--- | :--- | :--- | :--- |
| **Raw Training Set** | `data/raw/UNSW_NB15_training-set.csv` | 82,332 | 45 | `734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559` |
| **Raw Testing Set** | `data/raw/UNSW_NB15_testing-set.csv` | 175,341 | 45 | `bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa` |

---

## 3. Experimental Design & Seed Registry

All 15 controlled experiment runs were generated under strictly fixed seeds:

- **Random Seeds Evaluated:** `42`, `123`, `999`
- **Sample Selection:** Stratified 6,000-flow sample from `UNSW_NB15_training-set.csv` preserving exact class distribution:
  - Total Samples: 6,000 (ATTACK: 3,304, BENIGN: 2,696)
  - Train Partition (70%): 4,200 samples
  - Validation Partition (15%): 900 samples
  - Held-out Test Partition (15%): 900 samples
- **Zero-Leakage Protocol:** ColumnTransformer fitted strictly on the 4,200 training samples. The test set was touched exclusively after checkpoint selection.

---

## 4. Model Checkpoint & Artifact Registry

| Artifact Path | Format | Description | Verification Hash (SHA-256) |
| :--- | :--- | :--- | :--- |
| `results/network_intrusion/preprocessor.joblib` | Joblib | Pre-fitted ColumnTransformer (StandardScaler + OneHotEncoder) | Generated at study start |
| `results/network_intrusion/best_model.pt` | PyTorch | Checkpoint of highest-performing model (`Classical_MLP_seed_42`) | Verified and deployable |
| `results/network_intrusion/controlled_study/Classical_MLP_seed_42/best_model.pt` | PyTorch | Classical MLP checkpoint (Seed 42) | Available |
| `results/network_intrusion/controlled_study/Hybrid_Reuploading_L2_seed_42/best_model.pt` | PyTorch | Hybrid Re-uploading $L=2$ checkpoint (Seed 42) | Available |
| `results/network_intrusion/summary_statistics.json` | JSON | Machine-readable Mean ± Std Dev across seeds | Generated |
| `results/network_intrusion/plots/` | PNG (x7) | 7 high-resolution comparison plots | Complete |

---

## 5. Step-by-Step Reproduction Instructions

Execute in **Windows PowerShell**:

```powershell
# 1. Activate Virtual Environment and configure SSL environment variable
$env:CURL_CA_BUNDLE=""
& "E:\dl\.venv\Scripts\Activate.ps1"

# 2. Run Complete Pytest Suite (47 tests)
.venv\Scripts\pytest.exe -v

# 3. Re-run Controlled Study Runner (15 runs across 5 models and 3 seeds)
.venv\Scripts\python.exe experiments/run_nids_study.py

# 4. Generate Analysis Plots and Summary Statistics
.venv\Scripts\python.exe experiments/analyze_nids_results.py

# 5. Start FastAPI Backend for Inference & UI Integration
.venv\Scripts\uvicorn.exe api.main:app --host 127.0.0.1 --port 8000 --reload
```
