"""Streamlit Web Interface for Hybrid Quantum-Classical Breast Cancer Classifier.

A differentiable 4-qubit Quantum Data Re-uploading research demonstration using PennyLane & PyTorch.
"""

import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from sklearn.datasets import load_breast_cancer

from src.data.dataset import TARGET_LABEL_MAPPING, prepare_breast_cancer_data
from src.inference.predictor import HybridPredictor, PredictionResult


# Page configuration
st.set_page_config(
    page_title="Hybrid Quantum Classifier | Breast Cancer Wisconsin",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading Hybrid Quantum Model & Preprocessor...")
def get_predictor() -> HybridPredictor:
    """Instantiate and cache the HybridPredictor."""
    return HybridPredictor(checkpoint_path="results/hybrid_quantum/best_model.pt", random_seed=42)


@st.cache_data
def get_sample_test_data() -> Tuple[pd.DataFrame, np.ndarray]:
    """Retrieve samples from the held-out test split for demonstration."""
    splits = prepare_breast_cancer_data(random_seed=42)
    df = pd.DataFrame(splits.X_test_raw, columns=splits.feature_names)
    labels = splits.y_test
    return df, labels


def render_header():
    """Render top branding and research disclaimer."""
    st.title("⚛️ Hybrid Quantum-Classical Diagnostic Classifier")
    st.markdown(
        "**Differentiable 4-Qubit Quantum Data Re-uploading Architecture** • *PennyLane & PyTorch Deep Learning Integration*"
    )

    st.warning(
        "⚠️ **Research & Educational Demonstration Notice:** This application uses a classical CPU-based "
        "quantum statevector simulator (`default.qubit` backend). Predictions are generated for machine learning research "
        "and empirical evaluation purposes only. This system is **not** a certified medical device and must **never** "
        "be used for clinical diagnosis, patient screening, or clinical decision-making."
    )


def render_sidebar(predictor: HybridPredictor) -> str:
    """Render sidebar navigation and model metadata."""
    st.sidebar.image("https://raw.githubusercontent.com/PennyLaneAI/pennylane/master/doc/_static/pennylane_logo.png", width=200)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Module:",
        [
            "🚀 Batch CSV Inference",
            "🔬 Single-Sample Inspector",
            "📊 Benchmark & Study Results",
            "📖 Architectural Documentation",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Configuration")
    cfg = predictor.config
    model_info = cfg.get("model", {})
    st.sidebar.markdown(f"**Architecture:** Hybrid Quantum Classifier")
    st.sidebar.markdown(f"**Qubits:** 4 Wires (`default.qubit`)")
    st.sidebar.markdown(f"**Re-uploading Layers:** {model_info.get('n_layers', 2)}")
    st.sidebar.markdown(f"**Quantum Parameters:** 16 ($RY, RZ$)")
    st.sidebar.markdown(f"**Classical Parameters:** 569 ($30 \\to 16 \\to 4, 4 \\to 1$)")
    st.sidebar.markdown(f"**Total Parameters:** 585")
    st.sidebar.markdown(f"**Diff Method:** `backprop` (Analytic)")
    st.sidebar.markdown("---")
    st.sidebar.caption("E:\\dl • Wisconsin Diagnostic Dataset")

    return page


def render_batch_page(predictor: HybridPredictor):
    """Render batch CSV upload and inference tab."""
    st.header("🚀 Batch CSV Inference")
    st.markdown(
        "Upload a CSV containing the 30 Wisconsin Breast Cancer Diagnostic features. "
        "The application strictly validates columns and applies the training-fitted `StandardScaler` without data leakage."
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Choose a CSV file with 30 continuous features:",
            type=["csv"],
            help="File must contain all 30 expected feature names.",
        )

    with col2:
        st.markdown("**Quick Test Samples**")
        if st.button("📥 Load 15 Samples from Held-Out Test Set", use_container_width=True):
            test_df, test_labels = get_sample_test_data()
            sample_subset = test_df.iloc[:15].copy()
            st.session_state["loaded_sample_df"] = sample_subset
            st.session_state["loaded_sample_labels"] = test_labels[:15]
            st.success("Loaded 15 test samples!")

    # Template download
    template_df = pd.DataFrame(columns=predictor.EXPECTED_FEATURES)
    csv_buffer = io.StringIO()
    template_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📄 Download CSV Feature Template",
        data=csv_buffer.getvalue(),
        file_name="breast_cancer_features_template.csv",
        mime="text/csv",
    )

    df_to_process: Optional[pd.DataFrame] = None

    if uploaded_file is not None:
        try:
            df_to_process = pd.read_csv(uploaded_file)
            st.info(f"Uploaded CSV with {len(df_to_process)} rows and {len(df_to_process.columns)} columns.")
        except Exception as e:
            st.error(f"Error reading uploaded CSV: {str(e)}")
            return
    elif "loaded_sample_df" in st.session_state:
        df_to_process = st.session_state["loaded_sample_df"]

    if df_to_process is not None:
        st.markdown("### Validation & Inference")
        is_valid, errors = predictor.validate_dataframe(df_to_process)

        if not is_valid:
            st.error("❌ CSV Validation Failed:")
            for err in errors:
                st.write(f"- {err}")
            return

        st.success("✅ CSV Validation Passed: All 30 features verified with correct numeric data types.")

        with st.spinner("Executing Quantum Hybrid Inference..."):
            try:
                results: PredictionResult = predictor.predict(df_to_process)
            except Exception as e:
                st.error(f"Inference execution failed: {str(e)}")
                return

        # Build output presentation table
        display_df = pd.DataFrame({
            "Sample_ID": [f"Sample_{i+1:03d}" for i in range(results.sample_count)],
            "Predicted_Diagnosis": results.predicted_class_names,
            "Prob_Malignant": [f"{p*100:.2f}%" for p in results.probabilities_malignant],
            "Prob_Benign": [f"{p*100:.2f}%" for p in results.probabilities_benign],
            "Confidence": [f"{c*100:.2f}%" for c in results.confidences],
            "Raw_Logit": [f"{l:.4f}" for l in results.raw_logits],
        })

        if "loaded_sample_labels" in st.session_state and len(st.session_state["loaded_sample_labels"]) == results.sample_count:
            ground_truth = [TARGET_LABEL_MAPPING[int(lbl)] for lbl in st.session_state["loaded_sample_labels"]]
            display_df["Ground_Truth"] = ground_truth
            display_df["Correct?"] = np.where(display_df["Predicted_Diagnosis"] == display_df["Ground_Truth"], "✅ Yes", "❌ No")

        # Metric summary cards
        m_count = int(np.sum(results.predicted_labels == 0))
        b_count = int(np.sum(results.predicted_labels == 1))
        avg_conf = float(np.mean(results.confidences)) * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Samples", results.sample_count)
        c2.metric("Predicted Malignant (0)", m_count)
        c3.metric("Predicted Benign (1)", b_count)
        c4.metric("Avg Confidence", f"{avg_conf:.1f}%")

        st.markdown("### Inference Results Table")
        st.dataframe(display_df, use_container_width=True)

        # Export predictions to CSV
        export_csv = display_df.to_csv(index=False)
        st.download_button(
            label="💾 Download Predictions as CSV",
            data=export_csv,
            file_name="quantum_hybrid_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_single_sample_page(predictor: HybridPredictor):
    """Render single-sample interactive feature inspector."""
    st.header("🔬 Single-Sample Clinical Feature Inspector")
    st.markdown(
        "Adjust cell nucleus morphometric features to inspect end-to-end model representations, "
        "including intermediate classical encoder angles and measured quantum Pauli-Z expectation values."
    )

    feature_names = predictor.feature_names
    means = predictor.feature_means
    mins = predictor.feature_mins
    maxs = predictor.feature_maxs

    # Presets
    c_p1, c_p2, c_p3 = st.columns([1, 1, 2])
    if c_p1.button("📌 Load Typical Malignant Profile"):
        splits = prepare_breast_cancer_data(random_seed=42)
        mal_means = splits.X_train_raw[splits.y_train == 0].mean(axis=0)
        for idx, f in enumerate(feature_names):
            st.session_state[f"input_{f}"] = float(mal_means[idx])

    if c_p2.button("📌 Load Typical Benign Profile"):
        splits = prepare_breast_cancer_data(random_seed=42)
        ben_means = splits.X_train_raw[splits.y_train == 1].mean(axis=0)
        for idx, f in enumerate(feature_names):
            st.session_state[f"input_{f}"] = float(ben_means[idx])

    # Group features logically
    tab_mean, tab_se, tab_worst = st.tabs(["Mean Cell Characteristics", "Standard Error (SE)", "Worst / Extreme Metrics"])

    inputs: Dict[str, float] = {}

    with tab_mean:
        st.markdown("*Features 1 to 10: Mean measurements across nucleus samples*")
        cols = st.columns(2)
        for i in range(10):
            fname = feature_names[i]
            col_idx = i % 2
            val = st.session_state.get(f"input_{fname}", float(means[i]))
            inputs[fname] = cols[col_idx].number_input(
                f"{fname.capitalize()}",
                min_value=float(mins[i] * 0.5),
                max_value=float(maxs[i] * 1.5),
                value=float(val),
                format="%.4f",
                key=f"input_{fname}",
            )

    with tab_se:
        st.markdown("*Features 11 to 20: Standard error of measurements*")
        cols = st.columns(2)
        for i in range(10, 20):
            fname = feature_names[i]
            col_idx = (i - 10) % 2
            val = st.session_state.get(f"input_{fname}", float(means[i]))
            inputs[fname] = cols[col_idx].number_input(
                f"{fname.capitalize()}",
                min_value=float(mins[i] * 0.5),
                max_value=float(maxs[i] * 1.5),
                value=float(val),
                format="%.4f",
                key=f"input_{fname}",
            )

    with tab_worst:
        st.markdown("*Features 21 to 30: Worst / largest measurements across samples*")
        cols = st.columns(2)
        for i in range(20, 30):
            fname = feature_names[i]
            col_idx = (i - 20) % 2
            val = st.session_state.get(f"input_{fname}", float(means[i]))
            inputs[fname] = cols[col_idx].number_input(
                f"{fname.capitalize()}",
                min_value=float(mins[i] * 0.5),
                max_value=float(maxs[i] * 1.5),
                value=float(val),
                format="%.4f",
                key=f"input_{fname}",
            )

    st.markdown("---")
    if st.button("⚡ Run Hybrid Quantum Inference", use_container_width=True, type="primary"):
        with st.spinner("Computing Quantum Circuit Statevector and Expectation Values..."):
            try:
                res = predictor.predict(inputs)
            except Exception as e:
                st.error(f"Inference error: {str(e)}")
                return

        pred_class = res.predicted_class_names[0]
        prob_mal = res.probabilities_malignant[0]
        prob_ben = res.probabilities_benign[0]
        conf = res.confidences[0]

        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            st.markdown("### Prediction Outcome")
            if pred_class == "malignant":
                st.error(f"### 🔴 Predicted Diagnosis: **MALIGNANT** (Class 0)")
            else:
                st.success(f"### 🟢 Predicted Diagnosis: **BENIGN** (Class 1)")

            st.markdown(f"**Malignant Probability:** `{prob_mal*100:.2f}%`")
            st.markdown(f"**Benign Probability:** `{prob_ben*100:.2f}%`")
            st.markdown(f"**Confidence:** `{conf*100:.2f}%`")
            st.markdown(f"**Raw Model Logit:** `{res.raw_logits[0]:.4f}`")

        with res_col2:
            st.markdown("### Quantum Circuit State Inspection")
            st.markdown("**1. Classical Encoder Output ($\theta_{\\text{in}} \\in [-\\pi, \\pi]^4$):**")
            angles = res.encoder_angles[0]
            st.write({f"Qubit_{i} Angle": f"{angles[i]:.4f} rad" for i in range(4)})

            st.markdown("**2. Measured Pauli-Z Expectation Values ($\langle Z_i \\rangle \\in [-1, 1]$):**")
            expvals = res.quantum_expectation_values[0]
            st.write({f"Qubit_{i} <Z>": f"{expvals[i]:.4f}" for i in range(4)})


def render_benchmark_page():
    """Render verified controlled study comparisons and saved publication plots."""
    st.header("📊 Controlled Benchmark & Study Results")
    st.markdown(
        "Empirical evaluation across 15 runs (5 architectures $\\times$ 3 random seeds: `42`, `123`, `999`) "
        "on the held-out test partition (86 samples: 32 Malignant, 54 Benign)."
    )

    summary_file = Path("results/controlled_study/summary_statistics.json")
    csv_file = Path("reports/experiment_summary.csv")

    if summary_file.exists():
        with open(summary_file, "r") as f:
            summary = json.load(f)

        rows = []
        for m_name, d in summary.items():
            rows.append({
                "Architecture": m_name,
                "Family": d["model_family"],
                "Total Params": d["parameters"].get("total_trainable", 0),
                "Quantum Params": d["parameters"].get("quantum_circuit", 0),
                "Test Accuracy": f"{d['test_accuracy']['mean']*100:.2f}% ± {d['test_accuracy']['std']*100:.2f}%",
                "Test ROC-AUC": f"{d['test_roc_auc']['mean']:.4f} ± {d['test_roc_auc']['std']:.4f}",
                "Malignant Sensitivity": f"{d['malignant_sensitivity']['mean']*100:.2f}% ± {d['malignant_sensitivity']['std']*100:.2f}%",
                "Malignant Specificity": f"{d['malignant_specificity']['mean']*100:.2f}% ± {d['malignant_specificity']['std']*100:.2f}%",
                "Avg Duration": f"{d['duration_seconds']['mean']:.2f}s",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    if csv_file.exists():
        with st.expander("📄 View Full 15-Run Master CSV Table"):
            st.dataframe(pd.read_csv(csv_file), use_container_width=True)

    st.markdown("### Publication Plot Gallery")
    plot_dir = Path("results/controlled_study/plots")
    if plot_dir.exists():
        p_col1, p_col2 = st.columns(2)
        if (plot_dir / "accuracy_roc_auc_by_model.png").exists():
            p_col1.image(str(plot_dir / "accuracy_roc_auc_by_model.png"), caption="Test Accuracy and ROC-AUC by Model")
        if (plot_dir / "malignant_sensitivity_specificity.png").exists():
            p_col2.image(str(plot_dir / "malignant_sensitivity_specificity.png"), caption="Malignant Sensitivity vs Specificity")

        p_col3, p_col4 = st.columns(2)
        if (plot_dir / "training_time_by_model.png").exists():
            p_col3.image(str(plot_dir / "training_time_by_model.png"), caption="Training Duration Comparison (CPU Simulation)")
        if (plot_dir / "loss_curves_all_models.png").exists():
            p_col4.image(str(plot_dir / "loss_curves_all_models.png"), caption="Training & Validation Loss Trajectories")

        p_col5, p_col6 = st.columns(2)
        if (plot_dir / "reuploading_layers_ablation.png").exists():
            p_col5.image(str(plot_dir / "reuploading_layers_ablation.png"), caption="Performance vs Re-uploading Layers (L = 1, 2, 4)")
        if (plot_dir / "confusion_matrices_summary.png").exists():
            p_col6.image(str(plot_dir / "confusion_matrices_summary.png"), caption="Confusion Matrix Summary (Averaged over 3 Seeds)")


def render_about_page():
    """Render architectural documentation and limitations."""
    st.header("📖 Architecture & Scientific Documentation")

    st.markdown(
        r"""
        ### System Architecture
        The Hybrid Quantum-Classical Classifier integrates classical dimensionality reduction with 
        variational quantum data re-uploading:

        1. **Classical Feature Encoder ($30 \to 4$):**
           $$\theta_{\text{in}} = \pi \cdot \tanh\left( W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2 \right)$$
        2. **4-Qubit Quantum Circuit with Data Re-uploading:**
           $$|\psi(x, \theta)\rangle = \left( \prod_{l=L}^1 U_l(\theta_l) E(x) \right) |0000\rangle$$
           - $E(x)$: Pauli-Y rotations $RY(x_i)$ encoding features.
           - $U_l(\theta_l)$: Trainable $RY$ and $RZ$ rotations followed by ring CNOT entanglement $(0\to 1, 1\to 2, 2\to 3, 3\to 0)$.
        3. **Measurement:** Pauli-Z expectation values $\langle Z_i \rangle \in [-1, 1]$.
        4. **Classical Output Head ($4 \to 1$):**
           $$\hat{y}_{\text{logit}} = w^T \langle Z \rangle + b_0$$

        ### Key Scientific Insights & Limitations
        - **Simulator Execution:** All quantum computations are executed classically using PennyLane's exact statevector simulator (`default.qubit`) on CPU. No physical QPU was used.
        - **Parameter Efficiency vs Latency:** The hybrid model achieves $\approx 98.45\%$ accuracy with only 585 total parameters ($86.4\%$ parameter reduction vs Classical MLP), but simulation is $3.4\times$ to $6.8\times$ slower.
        - **No Quantum Advantage Claim:** Both classical and hybrid models operate near the ceiling on this standard tabular dataset.
        """
    )


def main():
    render_header()
    try:
        predictor = get_predictor()
    except Exception as e:
        st.error(f"❌ Failed to load model checkpoint: {str(e)}")
        return

    page = render_sidebar(predictor)

    if page == "🚀 Batch CSV Inference":
        render_batch_page(predictor)
    elif page == "🔬 Single-Sample Inspector":
        render_single_sample_page(predictor)
    elif page == "📊 Benchmark & Study Results":
        render_benchmark_page()
    elif page == "📖 Architectural Documentation":
        render_about_page()


if __name__ == "__main__":
    main()
