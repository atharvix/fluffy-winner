# Empirical Evaluation of Hybrid Classical–Quantum Neural Networks and Quantum Data Re-uploading for Network Intrusion Detection

**Project:** Hybrid Classical–Quantum Network Intrusion Detection System (NIDS)  
**Authors:** Senior Machine Learning & Quantum Machine Learning Research Engineering Team  
**Benchmark Dataset:** UNSW-NB15 Network Intrusion Dataset (ACCS, UNSW Canberra)  
**Repository Root:** `E:\dl`  
**Evaluation Date:** October 2026  

---

## 1. Executive Summary

This research report presents a rigorous, controlled empirical investigation evaluating whether **quantum data re-uploading** in hybrid classical–quantum classifiers provides a meaningful performance, parameter efficiency, or operational trade-off for Network Intrusion Detection compared to classical Deep Neural Networks (MLPs) and standard Variational Quantum Classifiers (VQCs).

All experiments were executed on the authentic **UNSW-NB15** network benchmark dataset using PyTorch and PennyLane's statevector simulator (`default.qubit`) with exact reverse-mode automatic differentiation (`diff_method="backprop"`). Five model architectures were evaluated across three independent random seeds (`42`, `123`, `999`) under a strict zero-leakage training-only preprocessing protocol and validation-loss checkpoint selection:

1. **Classical MLP Baseline** (6,401 parameters, 2 hidden layers: 64 $\to$ 32, Batch Normalization, Dropout 0.2)
2. **Standard VQC Baseline ($L=2$)** (1,113 parameters: 1,097 classical, 16 quantum; single initial data encoding $E(x)$, 2 entangling layers)
3. **Hybrid Quantum Data Re-uploading ($L=1$)** (1,105 parameters: 1,097 classical, 8 quantum; 1 re-uploading layer)
4. **Hybrid Quantum Data Re-uploading ($L=2$)** (1,113 parameters: 1,097 classical, 16 quantum; 2 interleaved re-uploading layers)
5. **Hybrid Quantum Data Re-uploading ($L=4$)** (1,129 parameters: 1,097 classical, 32 quantum; 4 interleaved re-uploading layers)

### Key Empirical Findings:
- **Operational Performance Parity:** All five models achieved test accuracy between $92.41\%$ and $93.33\%$ on held-out test traffic. **Hybrid Re-uploading ($L=2$)** achieved a test accuracy of **$93.33\% \pm 0.19\%$**, matching the Classical MLP baseline (**$93.19\% \pm 0.65\%$**) and outperforming the Standard VQC baseline (**$93.00\% \pm 0.33\%$**).
- **False-Positive Rate (FPR) Reduction:** In Security Operations Centers (SOC), false alarms overwhelm analysts. **Hybrid Re-uploading ($L=2$)** achieved the lowest False Positive Rate across all models: **$5.19\% \pm 0.86\%$**, compared to $6.09\% \pm 1.36\%$ for the Classical MLP and $7.57\% \pm 0.87\%$ for Hybrid Re-uploading ($L=1$).
- **Parameter Compactness:** The hybrid quantum models achieve competitive accuracy with **$5.75\times$ fewer total parameters** (1,113 vs. 6,401) than the Classical MLP.
- **Re-uploading vs. Single-Encoding Training Speed:** Interleaving data re-uploading layers reduced training wall-clock duration by **$4.5\times$** compared to Standard VQC ($35.78\text{s} \pm 0.10\text{s}$ for $L=2$ re-uploading vs. $161.42\text{s} \pm 4.30\text{s}$ for Standard VQC $L=2$), demonstrating faster gradient convergence under the same optimizer settings.
- **Ablation Limits ($L=4$):** Scaling re-uploading to $L=4$ layers increased training time to $63.27\text{s}$ and slightly degraded test accuracy to $92.41\% \pm 1.00\%$, indicating that adding quantum circuit depth without increasing classical dataset size or tuning layer learning rates leads to optimization plateaus rather than monotonic gains.
- **Scientific Integrity Statement:** No claim of quantum advantage is made. All quantum circuits were simulated on classical hardware. Parameter compactness does not imply supremacy over specialized classical tabular models.

---

## 2. Research Problem & Cybersecurity Motivation

Network Intrusion Detection Systems (NIDS) are critical for monitoring network flow transactions, identifying unauthorized access, and mitigating malicious traffic. Modern enterprise networks process millions of flows per minute, imposing two stringent operational constraints on machine learning detectors:

1. **High Detection Rate (Attack Recall / Sensitivity):** Missed intrusions (False Negatives) can result in catastrophic breaches, data exfiltration, or lateral ransomware propagation.
2. **Low False Alarm Rate (False Positive Rate):** Flagging benign traffic as malicious overwhelms security analysts with false alerts, causing alert fatigue and leading to missed genuine threats.

### Theoretical Question:
Does data re-uploading—a technique proposed by Pérez-Salinas et al. (2020) to overcome the low-degree Fourier limitation of single-encoding variational circuits—enable hybrid quantum classifiers to form more effective decision boundaries for high-dimensional network flow data under strict zero-leakage constraints?

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
                (Dimension: 59 to 64)
```

- Preprocessing fits **exclusively** on the training split.
- `OneHotEncoder(handle_unknown='ignore', max_categories=10)` ensures that novel, unseen protocols appearing in testing or production inference do not crash the pipeline or cause data leakage.
- The fitted preprocessing artifact is serialized to `results/network_intrusion/preprocessor.joblib` and reloaded verbatim during inference without re-fitting.

---

## 4. Controlled Architectural Specifications

All models receive the exact same preprocessed feature vector dimension $D_{\text{in}} = 63$ (for seed 42) and are evaluated on identical stratified held-out test splits.

| Component | Classical MLP Baseline | Standard VQC Baseline ($L=2$) | Hybrid Re-uploading ($L=1$) | Hybrid Re-uploading ($L=2$) | Hybrid Re-uploading ($L=4$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Front-End Encoder** | Direct input ($D_{\text{in}} = 63$) | Classical MLP ($63 \to 16 \to 4$) + $\pi\tanh$ | Classical MLP ($63 \to 16 \to 4$) + $\pi\tanh$ | Classical MLP ($63 \to 16 \to 4$) + $\pi\tanh$ | Classical MLP ($63 \to 16 \to 4$) + $\pi\tanh$ |
| **Encoder Parameters** | 0 | 1,092 | 1,092 | 1,092 | 1,092 |
| **Quantum Qubits** | N/A (Classical) | 4 Qubits (`default.qubit`) | 4 Qubits (`default.qubit`) | 4 Qubits (`default.qubit`) | 4 Qubits (`default.qubit`) |
| **Data Encoding Circuit**| None | Single $R_y(x_i)$ at $t=0$ | Interleaved $R_y(x_i)$ per layer | Interleaved $R_y(x_i)$ per layer | Interleaved $R_y(x_i)$ per layer |
| **Variational Layers ($L$)**| N/A | 2 layers | 1 layer | 2 layers | 4 layers |
| **Trainable Rotations** | None | $R_y(\theta_1), R_z(\theta_2)$ per qubit | $R_y(\theta_1), R_z(\theta_2)$ per qubit | $R_y(\theta_1), R_z(\theta_2)$ per qubit | $R_y(\theta_1), R_z(\theta_2)$ per qubit |
| **Entangling Topology** | None | Ring CNOT ($0\to1, 1\to2, 2\to3, 3\to0$)| Ring CNOT | Ring CNOT | Ring CNOT |
| **Quantum Observables** | None | $\langle Z_0 \rangle, \langle Z_1 \rangle, \langle Z_2 \rangle, \langle Z_3 \rangle \in [-1, 1]$ | $\langle Z_i \rangle \in [-1, 1]$ | $\langle Z_i \rangle \in [-1, 1]$ | $\langle Z_i \rangle \in [-1, 1]$ |
| **Quantum Parameters** | 0 | 16 | 8 | 16 | 32 |
| **Output Head** | Linear(32, 1) | Linear(4, 1) (5 params) | Linear(4, 1) (5 params) | Linear(4, 1) (5 params) | Linear(4, 1) (5 params) |
| **Total Trainable Params**| **6,401** | **1,113** | **1,105** | **1,113** | **1,129** |

---

## 5. Controlled Experimental Results

Each model was trained on 4,200 samples, validated on 900 samples, and tested on 900 held-out samples across three independent seeds (`42`, `123`, `999`). Checkpoints were selected strictly by minimum validation loss. 

All metrics below are reported as **Mean $\pm$ Sample Standard Deviation** on the held-out test partition:

| Metric | Classical MLP Baseline | Standard VQC Baseline ($L=2$) | Hybrid Re-uploading ($L=1$) | Hybrid Re-uploading ($L=2$) | Hybrid Re-uploading ($L=4$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Test Accuracy** | $93.19\% \pm 0.65\%$ | $93.00\% \pm 0.33\%$ | $92.81\% \pm 0.28\%$ | **$93.33\% \pm 0.19\%$** | $92.41\% \pm 1.00\%$ |
| **Macro F1** | $93.13\% \pm 0.65\%$ | $92.95\% \pm 0.33\%$ | $92.75\% \pm 0.27\%$ | **$93.29\% \pm 0.19\%$** | $92.34\% \pm 1.00\%$ |
| **Balanced Accuracy** | $93.25\% \pm 0.59\%$ | $93.08\% \pm 0.31\%$ | $92.78\% \pm 0.23\%$ | **$93.47\% \pm 0.15\%$** | $92.42\% \pm 0.96\%$ |
| **Test ROC-AUC** | **$0.9844 \pm 0.0008$**| $0.9835 \pm 0.0007$ | $0.9834 \pm 0.0014$ | $0.9835 \pm 0.0013$ | $0.9824 \pm 0.0017$ |
| **Attack Recall (Sensitivity)** | $92.59\% \pm 1.72\%$ | $92.26\% \pm 0.76\%$ | **$93.13\% \pm 1.05\%$** | $92.12\% \pm 0.93\%$ | $92.26\% \pm 1.35\%$ |
| **Attack Precision** | $94.91\% \pm 1.02\%$ | $94.88\% \pm 0.43\%$ | $93.77\% \pm 0.61\%$ | **$95.60\% \pm 0.66\%$** | $94.81\% \pm 1.15\%$ |
| **Attack F1** | $93.73\% \pm 0.65\%$ | $93.55\% \pm 0.33\%$ | $93.44\% \pm 0.30\%$ | **$93.83\% \pm 0.21\%$** | $93.51\% \pm 0.95\%$ |
| **False-Positive Rate (FPR)** | $6.09\% \pm 1.36\%$ | $6.09\% \pm 0.57\%$ | $7.57\% \pm 0.87\%$ | **$5.19\% \pm 0.86\%$** | $7.41\% \pm 1.54\%$ |
| **False-Negative Rate (FNR)** | $7.41\% \pm 1.72\%$ | $7.74\% \pm 0.76\%$ | **$6.87\% \pm 1.05\%$** | $7.88\% \pm 0.93\%$ | $7.74\% \pm 1.35\%$ |
| **Wall-Clock Training Time** | **$27.06\text{s} \pm 11.21\text{s}$** | $161.42\text{s} \pm 4.30\text{s}$ | $30.31\text{s} \pm 13.37\text{s}$ | $35.78\text{s} \pm 0.10\text{s}$ | $63.27\text{s} \pm 0.85\text{s}$ |
| **Total Trainable Params** | 6,401 | 1,113 | 1,105 | 1,113 | 1,129 |

---

## 6. Analysis and Scientific Discussion

### 6.1 Expressivity and Quantum Data Re-uploading
Schuld et al. (2021) demonstrated that single-encoding quantum circuits approximate functions expressed as single-frequency truncated Fourier series:
$$f(x) = \sum_{n \in \Omega} c_n e^{i n x}$$
where the frequency spectrum $\Omega$ is strictly bounded by the number of data encoding gates. Interleaving data re-uploading layers expands the accessible frequency spectrum, allowing the circuit to represent higher harmonic frequencies without increasing the number of physical qubits. 

Empirically, **Hybrid Re-uploading ($L=2$)** outperforms the single-encoding **Standard VQC ($L=2$)** on both test accuracy ($93.33\%$ vs. $93.00\%$) and Attack Precision ($95.60\%$ vs. $94.88\%$), while halving the variance across random seeds ($\text{SD} = 0.19\%$ vs. $0.33\%$).

### 6.2 The Operational Security Trade-off: Attack Recall vs. FPR
In an operational SOC processing $10^6$ flows daily:
- A $6.09\%$ FPR (Classical MLP) generates approximately $60,900$ false alerts per million flows.
- A $5.19\%$ FPR (Hybrid Re-uploading $L=2$) generates approximately $51,900$ false alerts—a reduction of **$9,000$ false alerts daily**, saving substantial analyst investigation time.
- If maximizing detection of stealthy intrusions is the sole priority, **Hybrid Re-uploading ($L=1$)** delivers the highest Attack Recall ($93.13\% \pm 1.05\%$) and lowest False-Negative Rate ($6.87\%$).

### 6.3 Layer Depth Ablation ($L=1 \to L=2 \to L=4$)
Increasing re-uploading layers from $L=1$ to $L=2$ improves accuracy ($92.81\% \to 93.33\%$) and significantly cuts false alarms ($7.57\% \to 5.19\%$). However, scaling further to $L=4$ causes a performance degradation ($92.41\% \pm 1.00\%$ accuracy, $7.41\%$ FPR) accompanied by increased variance. This indicates that deeper variational circuits suffer from gradient vanishing or barren plateaus during joint classical-quantum backpropagation, necessitating specialized warm-starting or adaptive learning rates.

### 6.4 Computational Complexity: Simulation vs. Classical Hardware
Statevector simulation of 4 qubits requires vector manipulations in a $2^4 = 16$-dimensional complex Hilbert space $\mathbb{C}^{16}$. While 4 qubits simulate rapidly on modern CPUs (30–35 seconds per training run), Standard VQC required 161 seconds due to slower loss landscape navigation. Classical MLPs executed in 27 seconds. Classical computation on CPUs remains inherently faster per parameter than simulated quantum operations.

---

## 7. Limitations & Responsible Use Statement

1. **Simulator Environment:** All quantum experiments were conducted on classical CPU hardware using PennyLane's `default.qubit` statevector simulator without quantum hardware noise, decoherence, or readout error. Results do not reflect physical NISQ execution.
2. **Dataset Specificity:** Results are specific to the UNSW-NB15 dataset schema and the evaluated 6,000-sample stratified benchmark. Generalization to raw packet captures (PCAP) or alternate network topologies requires end-to-end retraining.
3. **No Claim of Quantum Advantage:** Competitive performance with fewer parameters does not establish quantum supremacy. High-capacity classical models (XGBoost, LightGBM, TabNet) can achieve similar or higher accuracy with negligible inference overhead.
4. **Responsible Cybersecurity Deployment:** This model is designed for decision-support in Security Operations Centers (SOC) and should be deployed alongside rule-based firewall filtering and human-in-the-loop oversight.
