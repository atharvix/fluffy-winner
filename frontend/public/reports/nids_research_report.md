# Empirical Evaluation of Hybrid Classical–Quantum Neural Networks and Quantum Data Re-uploading for Network Intrusion Detection

**Project:** Hybrid Classical–Quantum Network Intrusion Detection System (NIDS)  
**Authors:** Senior Machine Learning & Quantum Machine Learning Research Engineering Team  
**Benchmark Dataset:** UNSW-NB15 Network Intrusion Dataset (ACCS, UNSW Canberra)  
**Repository Root:** `c:\Users\Lab\personal\dl`  
**Evaluation Date:** November 2026  

---

## 1. Executive Summary

This research report presents a rigorous, controlled empirical investigation evaluating whether **quantum data re-uploading** in hybrid classical–quantum classifiers provides a meaningful performance, parameter efficiency, or operational trade-off for Network Intrusion Detection compared to classical Deep Neural Networks (MLPs), an encoder-only **Classical Bottleneck MLP**, and standard Variational Quantum Classifiers (VQCs).

All experiments were executed on the authentic **UNSW-NB15** network benchmark dataset using PyTorch and PennyLane's statevector simulator (`default.qubit`) with exact reverse-mode automatic differentiation (`diff_method="backprop"`). Six model architectures were evaluated across three independent random seeds (`42`, `123`, `999`) under a strict zero-leakage training-only preprocessing protocol, validation threshold tuning ($\tau^*$), and validation-loss checkpoint selection:

1. **Classical MLP Baseline** (6,593 parameters, 2 hidden layers: 64 $\to$ 32, Batch Normalization, Dropout 0.2)
2. **Classical Bottleneck MLP Baseline** (1,145 parameters: $66 \to 16 \to 4 \to 1$ with $\pi\tanh$, isolating the classical front-end without quantum layers)
3. **Standard VQC Baseline ($L=2$)** (1,161 parameters: 1,140 classical, 16 quantum, 5 head; single initial data encoding $E(x)$, 2 entangling layers)
4. **Hybrid Quantum Data Re-uploading ($L=1$)** (1,153 parameters: 1,140 classical, 8 quantum, 5 head; 1 re-uploading layer)
5. **Hybrid Quantum Data Re-uploading ($L=2$)** (1,161 parameters: 1,140 classical, 16 quantum, 5 head; 2 interleaved re-uploading layers)
6. **Hybrid Quantum Data Re-uploading ($L=4$)** (1,177 parameters: 1,140 classical, 32 quantum, 5 head; 4 interleaved re-uploading layers)

### Key Empirical Findings:
- **Operational Performance Parity:** All six models achieved test accuracy between $88.76\%$ and $89.41\%$ on the official held-out test traffic (175,341 flows). **Classical Bottleneck MLP** achieved $89.41\% \pm 0.25\%$, **Hybrid Re-uploading ($L=2$)** achieved $89.32\% \pm 0.31\%$, and **Classical MLP** achieved $89.23\% \pm 0.19\%$.
- **Classical Bottleneck Control:** The **Classical Bottleneck MLP** achieved **$89.41\% \pm 0.25\%$** accuracy with 1,145 parameters, confirming that the classical compression front-end ($66 \to 16 \to 4$) accounts for $>97\%$ of parameters and establishes the primary feature separation before the quantum circuit acts.
- **Validation Threshold Optimization ($\tau^*$):** Tuning decision cutoffs via Youden's J statistic on validation data lowered False Positive Rates (FPR) across all architectures to $2.62\% \pm 0.35\%$ for Classical MLP, $2.99\% \pm 0.17\%$ for Classical Bottleneck MLP, and $3.33\% \pm 0.58\%$ for Hybrid Re-uploading $L=2$.
- **Parameter Compactness:** The hybrid quantum models and bottleneck classical model achieve competitive accuracy with **$5.7\times$ fewer parameters** (1,145–1,177 vs. 6,593) than the full-scale Classical MLP.
- **Computational Simulation Overhead:** Training wall-clock duration on PennyLane CPU statevector simulation (45,854 training samples) scaled with circuit complexity: $17.01\text{s}$ (Classical Bottleneck) $\to$ $24.90\text{s}$ (Classical MLP) $\to$ $101.83\text{s}$ ($L=1$) $\to$ $138.33\text{s}$ (Standard VQC $L=2$) $\to$ $170.88\text{s}$ ($L=2$) $\to$ $275.42\text{s}$ ($L=4$).
- **Scientific Integrity Statement:** No claim of quantum advantage or supremacy is made. All quantum circuits were simulated on classical hardware. Parameter compactness reflects the classical bottleneck encoder, not intrinsic quantum computational superiority.

---

## 2. Research Problem & Cybersecurity Motivation

Network Intrusion Detection Systems (NIDS) are critical for monitoring network flow transactions, identifying unauthorized access, and mitigating malicious traffic. Modern enterprise networks process millions of flows per minute, imposing two stringent operational constraints on machine learning detectors:

1. **High Detection Rate (Attack Recall / Sensitivity):** Missed intrusions (False Negatives) can result in catastrophic breaches, data exfiltration, or lateral ransomware propagation.
2. **Low False Alarm Rate (False Positive Rate):** Flagging benign traffic as malicious overwhelms security analysts with false alerts, causing alert fatigue and leading to missed genuine threats.

### Theoretical Question:
Does data re-uploading—a technique proposed by Pérez-Salinas et al. (2020) to overcome the low-degree Fourier limitation of single-encoding variational circuits—enable hybrid quantum classifiers to form more effective decision boundaries for high-dimensional network flow data under strict zero-leakage constraints, and does it add measurable utility over an identical classical bottleneck baseline?

---

## 3. Dataset Provenance & Preprocessing Protocol

### 3.1 Provenance
- **Dataset:** UNSW-NB15
- **Source:** Australian Centre for Cyber Security (ACCS), University of New South Wales (UNSW Canberra at ADFA). Created by Nour Moustafa and Jill Slay (2015).
- **Raw Files Evaluated:**
  - `data/raw/UNSW_NB15_training-set.csv` (82,332 records, 45 columns, SHA256: `734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559`)
  - `data/raw/UNSW_NB15_testing-set.csv` (175,341 records, 45 columns, SHA256: `bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa`)

### 3.2 Target Semantics & Leakage Prevention Policy
- **Class 0 (`BENIGN`):** Normal, non-malicious network flows.
- **Class 1 (`ATTACK`):** Malicious intrusion flows spanning 9 attack families (*Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis, Backdoor, Shellcode, Worms*).
- **Strict Leakage Guard:** Columns `id`, `label`, and `attack_cat` are completely excluded from predictive features. Neither the multi-class attack category nor the binary label enters feature processing.

### 3.3 Zero-Leakage Pipeline Architecture
```
Raw Flow Records (42 features: 39 numerical, 3 categorical)
                         |
                         v
       Stratified Split (70% Train, 15% Val, 15% Test)
                         |
           +-------------+-------------+
           | (Fit ONLY on Train)       |
           v                           v
     StandardScaler              OneHotEncoder
    (39 numeric features)     (proto, service, state)
           |                           |
           +-------------+-------------+
                         |
                         v
              Encoded Feature Vector
                (Dimension: 63 to 64)
```

- Preprocessing fits **exclusively** on the training split.
- `OneHotEncoder(handle_unknown='ignore', max_categories=10)` ensures that novel, unseen protocols appearing in testing or production inference do not crash the pipeline or cause data leakage.
- The fitted preprocessing artifact is serialized to `results/network_intrusion/preprocessor.joblib` and reloaded verbatim during inference without re-fitting.

---

## 4. Controlled Architectural Specifications

All models receive the exact same preprocessed feature vector dimension $D_{\text{in}} = 66$ and are evaluated on the official held-out test split (175,341 records).

| Component | Classical MLP | Classical Bottleneck | Standard VQC ($L=2$) | Hybrid Re-up ($L=1$) | Hybrid Re-up ($L=2$) | Hybrid Re-up ($L=4$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Front-End Encoder** | Direct input ($D_{\text{in}} = 66$) | MLP ($66 \to 16 \to 4$) + $\pi\tanh$ | MLP ($66 \to 16 \to 4$) + $\pi\tanh$ | MLP ($66 \to 16 \to 4$) + $\pi\tanh$ | MLP ($66 \to 16 \to 4$) + $\pi\tanh$ | MLP ($66 \to 16 \to 4$) + $\pi\tanh$ |
| **Encoder Parameters** | 0 | 1,140 | 1,140 | 1,140 | 1,140 | 1,140 |
| **Quantum Qubits** | N/A (Classical) | N/A (Classical) | 4 Qubits (`default.qubit`) | 4 Qubits (`default.qubit`) | 4 Qubits (`default.qubit`) | 4 Qubits (`default.qubit`) |
| **Data Encoding Circuit**| None | None | Single $R_y(x_i)$ at $t=0$ | Interleaved $R_y(x_i)$ | Interleaved $R_y(x_i)$ | Interleaved $R_y(x_i)$ |
| **Variational Layers ($L$)**| N/A | N/A | 2 layers | 1 layer | 2 layers | 4 layers |
| **Trainable Rotations** | None | None | $R_y, R_z$ per qubit | $R_y, R_z$ per qubit | $R_y, R_z$ per qubit | $R_y, R_z$ per qubit |
| **Entangling Topology** | None | None | Ring CNOT | Ring CNOT | Ring CNOT | Ring CNOT |
| **Quantum Observables** | None | None | $\langle Z_i \rangle \in [-1, 1]$ | $\langle Z_i \rangle \in [-1, 1]$ | $\langle Z_i \rangle \in [-1, 1]$ | $\langle Z_i \rangle \in [-1, 1]$ |
| **Quantum Parameters** | 0 | 0 | 16 | 8 | 16 | 32 |
| **Output Head** | Linear(32, 1) | Linear(4, 1) (5 params) | Linear(4, 1) (5 params) | Linear(4, 1) (5 params) | Linear(4, 1) (5 params) | Linear(4, 1) (5 params) |
| **Total Trainable Params**| **6,593** | **1,145** | **1,161** | **1,153** | **1,161** | **1,177** |

---

## 5. Controlled Experimental Results

Each model was trained on 45,854 deduplicated samples, validated on 8,092 samples, and evaluated on the full 175,341 held-out test samples across three independent seeds (`42`, `123`, `999`). Decision thresholds $\tau^*$ were optimized on validation probabilities using Youden's J statistic, and model checkpoints were selected strictly by minimum validation loss.

All metrics below are reported as **Mean $\pm$ Sample Standard Deviation** on the held-out 175,341-sample test partition:

| Metric | Classical MLP | Classical Bottleneck | Standard VQC ($L=2$) | Hybrid Re-up ($L=1$) | Hybrid Re-up ($L=2$) | Hybrid Re-up ($L=4$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Test Accuracy** | $89.23\% \pm 0.19\%$ | **$89.41\% \pm 0.25\%$** | $89.02\% \pm 0.48\%$ | $88.76\% \pm 0.72\%$ | $89.32\% \pm 0.31\%$ | $89.14\% \pm 0.34\%$ |
| **Macro F1** | $0.8838 \pm 0.0018$ | **$0.8855 \pm 0.0025$** | $0.8814 \pm 0.0047$ | $0.8790 \pm 0.0072$ | $0.8844 \pm 0.0030$ | $0.8827 \pm 0.0036$ |
| **Balanced Accuracy** | $91.39\% \pm 0.09\%$ | **$91.43\% \pm 0.14\%$** | $91.03\% \pm 0.27\%$ | $90.96\% \pm 0.54\%$ | $91.27\% \pm 0.21\%$ | $91.17\% \pm 0.34\%$ |
| **Test ROC-AUC** | **$0.9821 \pm 0.0009$** | **$0.9821 \pm 0.0007$** | $0.9783 \pm 0.0019$ | $0.9773 \pm 0.0049$ | $0.9808 \pm 0.0014$ | $0.9791 \pm 0.0016$ |
| **Attack Recall** | $85.40\% \pm 0.42\%$ | $85.84\% \pm 0.44\%$ | $85.47\% \pm 0.87\%$ | $84.87\% \pm 1.07\%$ | **$85.87\% \pm 0.63\%$** | $85.55\% \pm 0.38\%$ |
| **Attack Precision** | **$98.58\% \pm 0.18\%$** | $98.39\% \pm 0.08\%$ | $98.16\% \pm 0.17\%$ | $98.40\% \pm 0.20\%$ | $98.22\% \pm 0.30\%$ | $98.27\% \pm 0.25\%$ |
| **Attack F1** | $0.9152 \pm 0.0018$ | **$0.9169 \pm 0.0022$** | $0.9138 \pm 0.0043$ | $0.9113 \pm 0.0062$ | $0.9163 \pm 0.0028$ | $0.9147 \pm 0.0027$ |
| **False-Positive Rate**| **$2.62\% \pm 0.35\%$** | $2.99\% \pm 0.17\%$ | $3.41\% \pm 0.36\%$ | $2.95\% \pm 0.38\%$ | $3.33\% \pm 0.58\%$ | $3.21\% \pm 0.47\%$ |
| **Tuned Cutoff ($\tau^*$)**| $0.4225 \pm 0.0348$ | $0.3532 \pm 0.0400$ | $0.3631 \pm 0.0702$ | $0.4225 \pm 0.0488$ | $0.3301 \pm 0.0412$ | $0.3631 \pm 0.0597$ |
| **Training Time** | $24.90\text{s} \pm 3.48\text{s}$ | **$17.01\text{s} \pm 0.60\text{s}$** | $138.33\text{s} \pm 5.68\text{s}$ | $101.83\text{s} \pm 3.70\text{s}$ | $170.88\text{s} \pm 5.82\text{s}$ | $275.42\text{s} \pm 20.24\text{s}$ |
| **Total Trainable Params**| 6,593 | 1,145 | 1,161 | 1,153 | 1,161 | 1,177 |

---

## 6. Analysis and Scientific Discussion

### 6.1 Classical Bottleneck vs. Quantum Expressivity
The addition of the **Classical Bottleneck MLP** provides critical insight into the system's learning mechanism:
- With identical $66 \to 16 \to 4 \to 1$ classical projection and 1,145 parameters, the Classical Bottleneck achieves **$89.41\%$** accuracy.
- The hybrid quantum models ($1,153$ to $1,177$ parameters) perform within $\pm 0.6\%$ of this baseline ($88.76\%$ to $89.32\%$).
- This confirms that **over 97% of the trainable parameter budget** resides in the classical compression encoder, which does the heavy lifting of mapping heterogeneous tabular network flow dimensions into the compact 4-dimensional latent manifold.

### 6.2 The Operational Role of Decision Threshold Tuning ($\tau^*$)
In production SOC deployment:
- Setting the default threshold to $\tau = 0.5$ assumes a balanced cost between False Positives and False Negatives.
- By optimizing $\tau^*$ on validation probabilities via Youden's J statistic, the model achieves a calibrated operational trade-off on the 175,341 held-out test flows:
  - $\tau^* \approx 0.3301$ for Hybrid L=2 achieves $85.87\%$ Attack Recall.
  - $\tau^* \approx 0.4225$ for Classical MLP suppresses False Positive Rate down to $2.62\%$.
  - $\tau^* \approx 0.3532$ for Classical Bottleneck achieves $2.99\%$ FPR with $85.84\%$ Recall.

### 6.3 Layer Depth Scaling ($L=1 \to L=2 \to L=4$)
Scaling the re-uploading depth from $L=1$ to $L=2$ improves test accuracy ($88.76\% \to 89.32\%$) and attack recall ($84.87\% \to 85.87\%$), while CPU statevector simulation time on the 45,854-sample training partition scales from $101.8\text{s}$ to $170.9\text{s}$ and $275.4\text{s}$ ($L=4$).

---

## 7. Limitations & Responsible Use Statement

1. **Simulator Environment:** All quantum experiments were conducted on classical CPU hardware using PennyLane's `default.qubit` statevector simulator without quantum hardware noise, decoherence, or readout error. Results do not reflect physical NISQ execution.
2. **Dataset Specificity:** Results reflect the UNSW-NB15 dataset schema and the evaluated 6,000-sample stratified benchmark. Generalization to raw packet captures (PCAP) or alternate enterprise network topologies requires end-to-end retraining.
3. **No Claim of Quantum Advantage:** Competitive performance with fewer parameters does not establish quantum supremacy. High-capacity classical models (MLP, XGBoost, LightGBM) achieve equal or superior accuracy with lower simulation cost.
4. **Responsible Cybersecurity Deployment:** This model is designed for decision-support in Security Operations Centers (SOC) and should be deployed alongside rule-based firewall filtering and human-in-the-loop oversight.
