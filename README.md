# Breast Cancer Wisconsin Diagnostic Classification Pipeline

This repository implements reproducible dataset pipelines, comprehensive clinical evaluation metrics, a classical deep learning baseline (MLP), and a differentiable 4-qubit Quantum Data Re-uploading Classifier using PennyLane and PyTorch.

---

## Dataset Overview & Target-Label Mapping

The dataset is loaded directly via `sklearn.datasets.load_breast_cancer`.

- **Total Samples:** 569 (30 continuous numerical features)
- **Class Labels & Clinical Semantics:**
  - `0`: **Malignant** (212 samples total across dataset)
  - `1`: **Benign** (357 samples total across dataset)
- **Target Label Mapping:** Explicitly declared in [`src.data.dataset.TARGET_LABEL_MAPPING`](file:///e:/dl/src/data/dataset.py#L18-L21). Meaning of class labels is never inferred solely from numeric indices.

---

## Metric Audit & Clinical Conventions

In oncological screening, **Malignant (`0`)** represents the clinical condition of interest:
- **Malignant Sensitivity (Recall on Malignant):** $\frac{\text{TP}_{\text{mal}}}{\text{TP}_{\text{mal}} + \text{FN}_{\text{mal}}} = \frac{\text{CM}_{0,0}}{\text{CM}_{0,0} + \text{CM}_{0,1}}$ (critical for avoiding missed diagnoses).
- **Malignant Specificity:** $\frac{\text{TN}_{\text{mal}}}{\text{TN}_{\text{mal}} + \text{FP}_{\text{mal}}} = \frac{\text{CM}_{1,1}}{\text{CM}_{1,1} + \text{CM}_{1,0}}$ (true negative rate for malignancy / benign recall).
- **Benign Precision & Recall:** Computed with scikit-learn standard positive label `1` (Benign).

---

## Model Architectures

### 1. Classical Baseline ([`ClassicalMLP`](file:///e:/dl/src/models/classical.py#L8-L75))
- **Hidden Layers:** 2 trainable hidden layers (`Linear(30, 64) -> BatchNorm1d(64) -> ReLU -> Dropout(0.2) -> Linear(64, 32) -> BatchNorm1d(32) -> ReLU -> Dropout(0.2)`).
- **Output:** Single linear logit layer trained with `nn.BCEWithLogitsLoss`.

### 2. Differentiable Hybrid Quantum Classifier ([`HybridQuantumClassifier`](file:///e:/dl/src/models/hybrid.py#L71-L157))
- **Classical Feature Encoder:** Maps 30 standardized features to 4 bounded rotation angles in $[-\pi, \pi]$ via `Linear(30, 16) -> ReLU -> Linear(16, 4) -> pi * tanh(...)`.
- **Quantum Circuit Layer ([`QuantumDataReuploadingLayer`](file:///e:/dl/src/models/quantum.py#L32-L121)):**
  - **Wires:** 4 qubits on PennyLane `default.qubit` device.
  - **Re-uploading Architecture:** $L$ layers repeating:
    1. Data encoding: $RY(x_i)$ for qubit $i \in \{0, 1, 2, 3\}$.
    2. Trainable single-qubit rotations: $RY(\theta_{RY}[l, i])$ and $RZ(\theta_{RZ}[l, i])$.
    3. Ring entanglement: CNOT gates with periodic boundary conditions ($0 \to 1, 1 \to 2, 2 \to 3, 3 \to 0$).
  - **Mathematical Structure:** $|\psi(x, \theta)\rangle = U_L(\theta_L) E(x) \cdots U_1(\theta_1) E(x) |0000\rangle$.
  - **Measurement:** 4 Pauli-Z expectation values $\langle Z_i \rangle \in [-1, 1]$.
  - **Differentiation Method:** `diff_method="backprop"` for exact, analytic reverse-mode autograd gradients in a single backward pass.
- **Classical Output Head:** Linear layer `Linear(4, 1)` converting expectation values to 1 binary classification logit.

---

## Running the Code

### 1. Run Complete Test Suite

```powershell
.venv\Scripts\pytest.exe -v
```

Verifies:
- Dataset splits, sample conservation, disjointness, stratification, and zero-leakage preprocessing.
- Model architectures (classical $\ge 2$ hidden layers; quantum 4 qubits with ring topology).
- End-to-end gradient propagation through the quantum circuit into the classical encoder.
- Training loops and artifact preservation.

### 2. Train Classical Baseline

```powershell
.venv\Scripts\python.exe -m src.training.train --model-type classical --seed 42 --epochs 100 --patience 15 --output-dir results
```

### 3. Train Hybrid Quantum-Classical Model

```powershell
.venv\Scripts\python.exe -m src.training.train --model-type hybrid --seed 42 --q-layers 2 --epochs 25 --patience 10 --lr 0.01 --output-dir results/hybrid_quantum
```

---

## Saved Artifacts

- **Classical Baseline:** Saved in [`results/`](file:///e:/dl/results) (`config.json`, `training_history.json`, `best_model.pt`, `metrics.json`).
- **Hybrid Quantum Model:** Saved in [`results/hybrid_quantum/`](file:///e:/dl/results/hybrid_quantum) (`config.json`, `training_history.json`, `best_model.pt`, `metrics.json`).
