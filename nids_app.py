"""Streamlit Web Interface for Hybrid Quantum-Classical Network Intrusion Detection System.

Interactive dashboard providing:
1. Batch CSV flow traffic analysis with zero-leakage preprocessor.
2. Single-flow intrusion inspector with quantum rotation angles and Pauli-Z expectation values.
3. Controlled benchmark comparison gallery (UNSW-NB15).
4. Operational security and research disclaimers.
"""

import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.data.nids_dataset import (
    RAW_INPUT_FEATURES,
    TARGET_LABEL_MAPPING,
    prepare_nids_data,
)
from src.inference.nids_predictor import NIDSPredictor, NIDSPredictionResult


st.set_page_config(
    page_title="Hybrid Quantum NIDS | UNSW-NB15",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading Hybrid Quantum NIDS Model & Preprocessor...")
def get_nids_predictor() -> NIDSPredictor:
    """Instantiate and cache the production NIDSPredictor."""
    return NIDSPredictor(
        checkpoint_path="results/network_intrusion/best_model.pt",
        preprocessor_path="results/network_intrusion/preprocessor.joblib",
    )


@st.cache_data
def get_sample_test_traffic() -> Tuple[pd.DataFrame, np.ndarray]:
    """Retrieve samples from held-out test split for demonstration."""
    splits = prepare_nids_data(sample_size=300, random_seed=42)
    return splits.X_test_raw.copy(), splits.y_test.copy()


def render_header():
    st.title("🛡️ Hybrid Classical–Quantum Network Intrusion Detection System")
    st.markdown(
        "**Variational Quantum Circuits & Quantum Data Re-uploading** • "
        "*Evaluated on the UNSW-NB15 Benchmark Dataset*"
    )
    st.warning(
        "⚠️ **Research Demonstration Notice:** This system simulates 4-qubit variational quantum circuits "
        "on classical CPUs (`default.qubit` statevector simulator). Competitive performance with fewer parameters "
        "does not constitute physical quantum advantage. In production, this model is designed as an operational "
        "decision-support filter alongside traditional firewalls and human analyst oversight."
    )


def render_sidebar(predictor: NIDSPredictor):
    st.sidebar.header("⚙️ System Configuration")
    st.sidebar.markdown(f"**Active Model:** `{predictor.model_name}`")
    st.sidebar.markdown(f"**Model Family:** `{predictor.model_family}`")
    st.sidebar.markdown(f"**Dataset:** `UNSW-NB15`")
    st.sidebar.markdown(f"**Input Features:** `{len(RAW_INPUT_FEATURES)}`")
    st.sidebar.markdown(f"**Quantum Qubits:** `4`")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Decision Threshold")
    threshold = st.sidebar.slider(
        "Attack Alert Threshold (default 0.50)",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Higher thresholds lower false alarms but increase missed attacks.",
    )
    return threshold


def render_single_flow_inspector(predictor: NIDSPredictor, threshold: float):
    st.subheader("🔍 Single-Flow Traffic Inspector")
    st.markdown("Inspect network flow features, intermediate quantum encoding angles $\\theta_i$, and Pauli-Z expectation values $\\langle Z_i \\rangle$.")

    test_df, test_labels = get_sample_test_traffic()
    sample_idx = st.selectbox(
        "Select Sample from Held-Out Test Set:",
        options=range(min(20, len(test_df))),
        format_func=lambda i: f"Sample #{i+1} (Ground Truth: {TARGET_LABEL_MAPPING[int(test_labels[i])]})",
    )

    flow_record = test_df.iloc[sample_idx:sample_idx+1]
    res: NIDSPredictionResult = predictor.predict(flow_record, threshold=threshold)

    pred_label = res.predicted_class_names[0]
    prob_attack = float(res.probabilities_attack[0])
    prob_benign = float(res.probabilities_benign[0])
    confidence = float(res.confidences[0])

    col1, col2, col3 = st.columns(3)
    with col1:
        if pred_label == "ATTACK":
            st.error(f"🚨 **Predicted: ATTACK**\n\nConfidence: {confidence * 100:.1f}%")
        else:
            st.success(f"✅ **Predicted: BENIGN**\n\nConfidence: {confidence * 100:.1f}%")

    with col2:
        st.metric("Attack Probability", f"{prob_attack * 100:.2f}%")
        st.metric("Benign Probability", f"{prob_benign * 100:.2f}%")

    with col3:
        true_label = TARGET_LABEL_MAPPING[int(test_labels[sample_idx])]
        st.metric("Ground Truth Label", true_label)
        match = "MATCH ✅" if pred_label == true_label else "MISMATCH ❌"
        st.metric("Verification", match)

    if res.encoder_angles is not None and res.quantum_expectation_values is not None:
        st.markdown("---")
        st.subheader("⚛️ Intermediate Quantum Circuit State Inspection")
        qcol1, qcol2 = st.columns(2)

        with qcol1:
            st.markdown("**Classical Encoder Bounded Rotation Angles $\\theta_i \\in [-\\pi, \\pi]$:**")
            angles = res.encoder_angles[0]
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar([f"Qubit {i}" for i in range(4)], angles, color="#3B82F6")
            ax.set_ylim(-np.pi - 0.2, np.pi + 0.2)
            ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
            ax.set_ylabel("Angle (radians)")
            st.pyplot(fig)

        with qcol2:
            st.markdown("**Measured Pauli-Z Expectation Values $\\langle Z_i \\rangle \\in [-1, 1]$:**")
            expvals = res.quantum_expectation_values[0]
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar([f"Qubit {i}" for i in range(4)], expvals, color="#10B981")
            ax.set_ylim(-1.1, 1.1)
            ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
            ax.set_ylabel("Expectation Value")
            st.pyplot(fig)


def render_batch_csv_upload(predictor: NIDSPredictor, threshold: float):
    st.subheader("📁 Batch CSV Traffic Analysis")
    st.markdown("Upload a network flow CSV file conforming to the UNSW-NB15 schema.")

    uploaded_file = st.file_uploader("Upload Network Flow CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write(f"Loaded `{len(df)}` records.")
            is_valid, errors = predictor.validate_dataframe(df)
            if not is_valid:
                st.error("Validation Errors:\n" + "\n".join(f"- {e}" for e in errors))
            else:
                with st.spinner("Executing Inference..."):
                    res = predictor.predict(df, threshold=threshold)
                
                attack_count = int(np.sum(res.predicted_labels == 1))
                benign_count = len(df) - attack_count

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Flows", len(df))
                col2.metric("Attacks Detected", f"{attack_count} ({attack_count/len(df)*100:.1f}%)")
                col3.metric("Benign Flows", f"{benign_count} ({benign_count/len(df)*100:.1f}%)")

                out_df = df.copy()
                out_df["predicted_label"] = res.predicted_class_names
                out_df["prob_attack"] = np.round(res.probabilities_attack, 4)
                out_df["confidence"] = np.round(res.confidences, 4)

                st.dataframe(out_df.head(50))

                csv_buffer = io.StringIO()
                out_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    "📥 Download Predictions CSV",
                    csv_buffer.getvalue(),
                    file_name="nids_predictions.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Inference failed: {str(e)}")


def render_benchmark_gallery():
    st.subheader("📊 Controlled Benchmark Study Gallery (UNSW-NB15)")
    st.markdown(
        "Results from the 15-run controlled study across 3 independent random seeds (`42`, `123`, `999`)."
    )

    summary_file = Path("results/network_intrusion/summary_statistics.json")
    if summary_file.exists():
        with open(summary_file, "r", encoding="utf-8") as f:
            stats = json.load(f)

        rows = []
        for m_name, m_data in stats.items():
            rows.append({
                "Architecture": m_name,
                "Parameters": f"{m_data['total_parameters']:,}",
                "Test Accuracy": m_data["test_accuracy"]["formatted"],
                "Macro F1": m_data["test_macro_f1"]["formatted"],
                "Attack Recall": m_data["test_attack_recall"]["formatted"],
                "False-Positive Rate": m_data["test_false_positive_rate"]["formatted"],
                "Training Time (s)": m_data["duration_seconds"]["formatted"],
            })
        st.dataframe(pd.DataFrame(rows))

    plot_dir = Path("results/network_intrusion/plots")
    if plot_dir.exists():
        tabs = st.tabs([
            "Accuracy & F1",
            "Recall vs FPR",
            "ROC Curves",
            "Confusion Matrices",
            "Layer Ablation",
            "Complexity",
        ])
        with tabs[0]:
            p = plot_dir / "accuracy_macro_f1_by_model.png"
            if p.exists(): st.image(str(p))
        with tabs[1]:
            p = plot_dir / "attack_recall_fpr_by_model.png"
            if p.exists(): st.image(str(p))
        with tabs[2]:
            p = plot_dir / "roc_curves_all_models.png"
            if p.exists(): st.image(str(p))
        with tabs[3]:
            p = plot_dir / "confusion_matrices_summary.png"
            if p.exists(): st.image(str(p))
        with tabs[4]:
            p = plot_dir / "reuploading_layers_ablation.png"
            if p.exists(): st.image(str(p))
        with tabs[5]:
            p = plot_dir / "parameter_counts_circuit_complexity.png"
            if p.exists(): st.image(str(p))


def main():
    render_header()
    predictor = get_nids_predictor()
    threshold = render_sidebar(predictor)

    tab1, tab2, tab3 = st.tabs([
        "🔍 Single-Flow Inspector",
        "📁 Batch CSV Analysis",
        "📊 Benchmark Gallery",
    ])

    with tab1:
        render_single_flow_inspector(predictor, threshold)
    with tab2:
        render_batch_csv_upload(predictor, threshold)
    with tab3:
        render_benchmark_gallery()


if __name__ == "__main__":
    main()
