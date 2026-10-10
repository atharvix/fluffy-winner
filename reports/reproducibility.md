# Reproducibility Guide & Verification Protocol

This guide outlines the exact environment configuration, deterministic procedures, and execution commands required to reproduce all findings and verify experimental results in this repository.

---

## 1. Verified Software Environment

The experiments were executed under Windows 11 using Python 3.13 and the following pinned dependencies:

| Package | Version | Purpose |
| :--- | :--- | :--- |
| `python` | `3.13.6` | Runtime Interpreter |
| `torch` | `2.14.1+cpu` | Classical Deep Learning & Reverse-mode Autograd |
| `pennylane` | `0.45.1` | Quantum Circuit Simulation & PyTorch QNode Interface |
| `scikit-learn` | `1.9.1` | Wisconsin Breast Cancer Dataset & Evaluation Metrics |
| `numpy` | `2.5.3` | Array Manipulations & Linear Algebra |
| `pandas` | `3.0.6` | Tabular Data Structuring & CSV Generation |
| `matplotlib` | `3.11.2` | Publication Visualization & Plot Rendering |
| `pytest` | `9.1.1` | Unit & Integration Test Suite Framework |

---

## 2. Determinism & Random Seed Policy

All stochastic operations (partition splitting, parameter initialization, mini-batch shuffling) are governed by explicit integer seeds via [`src.training.train.set_seed`](file:///e:/dl/src/training/train.py#L26-L34):
- `random.seed(seed)`
- `np.random.seed(seed)`
- `torch.manual_seed(seed)`
- `torch.cuda.manual_seed_all(seed)` (if CUDA is present)
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`

The controlled experimental study evaluates 3 fixed seeds: `42`, `123`, `999`.

---

## 3. Step-by-Step Execution Workflow

### Step 1: Execute Complete Test Suite
Run the 21-test pytest suite:
```powershell
.venv\Scripts\pytest.exe -v
```
*Expected Execution Time:* ~6 to 8 seconds.
*Expected Outcome:* 21 passed (100% pass rate).

### Step 2: Run the Controlled Experimental Study
Train and evaluate all 15 model-seed combinations (5 architectures $\times$ 3 seeds):
```powershell
.venv\Scripts\python.exe experiments/run_study.py
```
*Expected Execution Time:* ~100 to 120 seconds on modern CPU.
*Saved Location:* [`results/controlled_study/`](file:///e:/dl/results/controlled_study/)

### Step 3: Generate Publication Plots & Aggregations
Analyze saved experiment artifacts and render plots:
```powershell
.venv\Scripts\python.exe experiments/analyze_results.py
```
*Saved Location:* [`results/controlled_study/plots/`](file:///e:/dl/results/controlled_study/plots/) and [`results/controlled_study/summary_statistics.json`](file:///e:/dl/results/controlled_study/summary_statistics.json)

### Step 4: Run the Independent Verification Audit
Perform automated audit of all run metrics, dataset partition conservation, and confusion matrices:
```powershell
.venv\Scripts\python.exe experiments/verify_study.py
```
*Generated Deliverable:* [`reports/experiment_summary.csv`](file:///e:/dl/reports/experiment_summary.csv)
*Verification Check:* 0 discrepancies, partition conservation PASS, validation-only selection PASS.

---

## 4. Single-Model Command Reference

To execute individual model runs independently:

- **Classical Baseline MLP:**
  ```powershell
  .venv\Scripts\python.exe -m src.training.train --model-type classical --seed 42 --epochs 100 --patience 15 --output-dir results
  ```

- **Hybrid Data Re-uploading VQC ($L=2$):**
  ```powershell
  .venv\Scripts\python.exe -m src.training.train --model-type hybrid --seed 42 --q-layers 2 --epochs 30 --patience 10 --lr 0.01 --output-dir results/hybrid_quantum
  ```
