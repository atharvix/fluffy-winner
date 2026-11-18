import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { AlertTriangle, ArrowRight, BarChart2 } from "lucide-react";
import { useTranslation } from "../i18n/useTranslation";
import {
  checkHealth,
  fetchModelInfo,
  fetchLocalBenchmarks,
  type HealthResponse,
  type ModelInfoResponse,
  type AggregateBenchmarkData,
} from "../services/api";
import { QuantumCircuitVisualizer } from "../features/quantum-visualization/QuantumCircuitVisualizer";
import { NetworkPacket3D } from "../features/quantum-visualization/NetworkPacket3D";

export const OverviewPage: React.FC = () => {
  const { t } = useTranslation();
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [benchmarks, setBenchmarks] = useState<AggregateBenchmarkData | null>(null);
  const [activeMetricTab, setActiveMetricTab] = useState<"accuracy" | "recall_fpr" | "params">("accuracy");

  useEffect(() => {
    let isMounted = true;
    const loadData = async () => {
      try {
        const [hData, mData, bData] = await Promise.all([
          checkHealth().catch(() => null),
          fetchModelInfo().catch(() => null),
          fetchLocalBenchmarks().catch(() => null),
        ]);
        if (isMounted) {
          setHealth(hData);
          setModelInfo(mData);
          setBenchmarks(bData);
        }
      } catch {
        // Graceful offline fallback
      }
    };
    loadData();
    return () => {
      isMounted = false;
    };
  }, []);

  const bestModelStats = benchmarks?.["Hybrid_Reuploading_L2"];

  // Five models empirical comparison data
  const comparisonData = [
    {
      name: "Hybrid L=2",
      fullName: "Hybrid Re-uploading (L=2)",
      accuracy: 93.33,
      f1: 93.29,
      recall: 92.12,
      fpr: 5.19,
      params: 1113,
      isDeployed: true,
    },
    {
      name: "MLP Baseline",
      fullName: "Classical PyTorch MLP",
      accuracy: 93.19,
      f1: 93.13,
      recall: 92.59,
      fpr: 6.09,
      params: 6401,
      isDeployed: false,
    },
    {
      name: "Standard VQC",
      fullName: "Standard VQC (L=2)",
      accuracy: 93.00,
      f1: 92.95,
      recall: 92.26,
      fpr: 6.09,
      params: 1113,
      isDeployed: false,
    },
    {
      name: "Hybrid L=1",
      fullName: "Hybrid Re-uploading (L=1)",
      accuracy: 92.81,
      f1: 92.75,
      recall: 93.13,
      fpr: 7.57,
      params: 1105,
      isDeployed: false,
    },
    {
      name: "Hybrid L=4",
      fullName: "Hybrid Re-uploading (L=4)",
      accuracy: 92.41,
      f1: 92.34,
      recall: 92.26,
      fpr: 7.41,
      params: 1129,
      isDeployed: false,
    },
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-6)" }}>
      {/* Hero / Header Section */}
      <section
        className="panel"
        style={{
          borderLeft: "4px solid var(--accent-primary)",
          display: "flex",
          flexDirection: "column",
          gap: "var(--space-3)",
          padding: "var(--space-5)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" }}>
          <span className="badge badge-neutral">UNSW-NB15 BENCHMARK</span>
          <span className="badge badge-neutral">PENNYLANE + PYTORCH</span>
          <span className="badge badge-attack">DECISION-SUPPORT NIDS</span>
          <span className="badge badge-benign">ZERO TARGET LEAKAGE</span>
        </div>

        <h1
          style={{
            fontSize: "var(--text-xl)",
            fontWeight: 700,
            letterSpacing: "-0.02em",
            margin: 0,
            lineHeight: 1.25,
          }}
        >
          {t.overviewTitle}
        </h1>
        <p
          style={{
            fontSize: "var(--text-base)",
            color: "var(--text-secondary)",
            maxWidth: "960px",
            margin: 0,
            lineHeight: "var(--leading-relaxed)",
          }}
        >
          {t.overviewSubtitle}
        </p>

        {/* System Context Strip */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))",
            gap: "var(--space-3)",
            paddingTop: "var(--space-3)",
            borderTop: "1px solid var(--border-subtle)",
          }}
        >
          <div>
            <div style={{ fontSize: "var(--text-2xs)", color: "var(--text-muted)", textTransform: "uppercase" }}>
              {t.activeModelLabel}
            </div>
            <div style={{ fontWeight: 600, fontFamily: "var(--font-mono)", fontSize: "var(--text-sm)", marginTop: "2px" }}>
              {modelInfo?.model_name || "Hybrid_Quantum_NIDS (Deployed)"}
            </div>
          </div>
          <div>
            <div style={{ fontSize: "var(--text-2xs)", color: "var(--text-muted)", textTransform: "uppercase" }}>
              {t.datasetLabel}
            </div>
            <div style={{ fontWeight: 600, fontSize: "var(--text-sm)", marginTop: "2px" }}>
              UNSW-NB15 (42 Raw Features)
            </div>
          </div>
          <div>
            <div style={{ fontSize: "var(--text-2xs)", color: "var(--text-muted)", textTransform: "uppercase" }}>
              FastAPI Liveness
            </div>
            <div
              style={{
                fontWeight: 600,
                fontSize: "var(--text-sm)",
                color: health?.status === "healthy" ? "var(--status-benign)" : "var(--status-attack)",
                marginTop: "2px",
              }}
            >
              {health?.status === "healthy" ? "Online (CPU Simulator)" : "Offline (Local Standalone Mode)"}
            </div>
          </div>
          <div>
            <div style={{ fontSize: "var(--text-2xs)", color: "var(--text-muted)", textTransform: "uppercase" }}>
              Quantum Simulator Device
            </div>
            <div style={{ fontWeight: 600, fontFamily: "var(--font-mono)", fontSize: "var(--text-sm)", marginTop: "2px" }}>
              PennyLane default.qubit (CPU)
            </div>
          </div>
        </div>
      </section>

      {/* Verified Empirical KPI Cards */}
      <section>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "var(--space-3)", flexWrap: "wrap", gap: "var(--space-2)" }}>
          <h2 style={{ fontSize: "var(--text-md)", fontWeight: 600, margin: 0 }}>
            {t.historicalNoticeTitle} (Hybrid Re-uploading L=2)
          </h2>
          <span style={{ fontSize: "var(--text-2xs)", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
            3 Seeds (42, 123, 999) • Held-Out Test Split
          </span>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
            gap: "var(--space-3)",
          }}
        >
          {/* Test Accuracy */}
          <div className="panel" style={{ padding: "var(--space-3)" }}>
            <div style={{ fontSize: "var(--text-2xs)", color: "var(--text-muted)", textTransform: "uppercase" }}>
              {t.kpiTestAccuracy}
            </div>
            <div style={{ fontSize: "var(--text-xl)", fontWeight: 700, margin: "4px 0", color: "var(--accent-primary)" }}>
              {bestModelStats ? bestModelStats.test_accuracy.formatted : "93.33% ± 0.19%"}
            </div>
            <div style={{ fontSize: "var(--text-2xs)", color: "var(--text-secondary)" }}>
              Parity with 6.4k Classical MLP
            </div>
          </div>

          {/* False Positive Rate */}
          <div className="panel" style={{ padding: "var(--space-3)" }}>
            <div style={{ fontSize: "var(--text-2xs)", color: "var(--text-muted)", textTransform: "uppercase" }}>
              {t.kpiFalsePositiveRate}
            </div>
            <div style={{ fontSize: "var(--text-xl)", fontWeight: 700, margin: "4px 0", color: "var(--status-benign)" }}>
              {bestModelStats ? bestModelStats.test_false_positive_rate.formatted : "5.19% ± 0.86%"}
            </div>
            <div style={{ fontSize: "var(--text-2xs)", color: "var(--text-secondary)" }}>
              Lowest false alarm rate across all models
            </div>
          </div>

          {/* Attack Recall */}
          <div className="panel" style={{ padding: "var(--space-3)" }}>
            <div style={{ fontSize: "var(--text-2xs)", color: "var(--text-muted)", textTransform: "uppercase" }}>
              {t.kpiAttackRecall}
            </div>
            <div style={{ fontSize: "var(--text-xl)", fontWeight: 700, margin: "4px 0", color: "var(--status-attack)" }}>
              {bestModelStats ? bestModelStats.test_attack_recall.formatted : "92.12% ± 0.93%"}
            </div>
            <div style={{ fontSize: "var(--text-2xs)", color: "var(--text-secondary)" }}>
              Intrusion Detection Rate
            </div>
          </div>

          {/* Macro F1 */}
          <div className="panel" style={{ padding: "var(--space-3)" }}>
            <div style={{ fontSize: "var(--text-2xs)", color: "var(--text-muted)", textTransform: "uppercase" }}>
              {t.kpiMacroF1}
            </div>
            <div style={{ fontSize: "var(--text-xl)", fontWeight: 700, margin: "4px 0" }}>
              {bestModelStats ? bestModelStats.test_macro_f1.formatted : "0.9329 ± 0.0019"}
            </div>
            <div style={{ fontSize: "var(--text-2xs)", color: "var(--text-secondary)" }}>
              Balanced class F1-Score
            </div>
          </div>

          {/* Parameters */}
          <div className="panel" style={{ padding: "var(--space-3)" }}>
            <div style={{ fontSize: "var(--text-2xs)", color: "var(--text-muted)", textTransform: "uppercase" }}>
              {t.kpiTotalParameters}
            </div>
            <div style={{ fontSize: "var(--text-xl)", fontWeight: 700, margin: "4px 0" }}>
              1,113
            </div>
            <div style={{ fontSize: "var(--text-2xs)", color: "var(--text-secondary)" }}>
              5.75× fewer params vs. MLP (6,401)
            </div>
          </div>
        </div>
      </section>

      {/* Historical Telemetry Disclaimer */}
      <div
        style={{
          padding: "var(--space-3)",
          background: "var(--bg-surface-2)",
          border: "1px solid var(--border-default)",
          borderLeft: "3px solid var(--status-warn)",
          fontSize: "var(--text-xs)",
          color: "var(--text-secondary)",
          display: "flex",
          alignItems: "flex-start",
          gap: "var(--space-2)",
        }}
      >
        <AlertTriangle size={16} color="var(--status-warn)" style={{ flexShrink: 0, marginTop: "2px" }} />
        <div>
          <strong style={{ color: "var(--text-primary)" }}>{t.historicalNoticeTitle}:</strong>{" "}
          {t.historicalNoticeBody}
        </div>
      </div>

      {/* Interactive SVG Benchmark Comparison Charts */}
      <section className="panel" style={{ padding: "var(--space-4)" }}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            flexWrap: "wrap",
            gap: "var(--space-2)",
            marginBottom: "var(--space-4)",
            paddingBottom: "var(--space-2)",
            borderBottom: "1px solid var(--border-subtle)",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <BarChart2 size={16} color="var(--accent-primary)" />
            <span className="panel-title">Controlled Empirical Comparison (5 Architectures)</span>
          </div>

          <div style={{ display: "inline-flex", border: "1px solid var(--border-default)" }}>
            <button
              type="button"
              className={`btn btn-sm ${activeMetricTab === "accuracy" ? "btn-primary" : ""}`}
              onClick={() => setActiveMetricTab("accuracy")}
              style={{ border: "none" }}
            >
              Accuracy & Macro F1
            </button>
            <button
              type="button"
              className={`btn btn-sm ${activeMetricTab === "recall_fpr" ? "btn-primary" : ""}`}
              onClick={() => setActiveMetricTab("recall_fpr")}
              style={{ border: "none" }}
            >
              Recall vs False Alarms
            </button>
            <button
              type="button"
              className={`btn btn-sm ${activeMetricTab === "params" ? "btn-primary" : ""}`}
              onClick={() => setActiveMetricTab("params")}
              style={{ border: "none" }}
            >
              Parameter Footprint
            </button>
          </div>
        </div>

        {/* SVG Chart Display */}
        <div style={{ width: "100%", overflowX: "auto" }}>
          {activeMetricTab === "accuracy" && (
            <div style={{ minWidth: "480px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", color: "var(--text-muted)", marginBottom: "8px" }}>
                <span>Model Architecture</span>
                <span>Accuracy % (Solid Blue) & Macro F1 % (Striped)</span>
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
                {comparisonData.map((d) => (
                  <div key={d.name} style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                    <div style={{ width: "130px", fontSize: "12px", fontWeight: 600, color: d.isDeployed ? "var(--accent-primary)" : "var(--text-primary)" }}>
                      {d.name} {d.isDeployed && "★"}
                    </div>
                    <div style={{ flex: 1, background: "var(--bg-surface-2)", height: "20px", display: "flex", position: "relative" }}>
                      <div
                        style={{
                          width: `${((d.accuracy - 90) / 4) * 100}%`,
                          background: "var(--accent-primary)",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "flex-end",
                          paddingRight: "6px",
                          color: "#ffffff",
                          fontSize: "11px",
                          fontWeight: 700,
                        }}
                      >
                        {d.accuracy.toFixed(2)}%
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {activeMetricTab === "recall_fpr" && (
            <div style={{ minWidth: "480px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", color: "var(--text-muted)", marginBottom: "8px" }}>
                <span>Model Architecture</span>
                <span>Attack Recall % (Red) & False Positive Rate % (Green, lower is better)</span>
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
                {comparisonData.map((d) => (
                  <div key={d.name} style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                    <div style={{ width: "130px", fontSize: "12px", fontWeight: 600 }}>
                      {d.name}
                    </div>
                    <div style={{ flex: 1, display: "flex", gap: "6px" }}>
                      <div style={{ flex: 1, background: "var(--bg-surface-2)", height: "20px" }}>
                        <div
                          style={{
                            width: `${((d.recall - 90) / 4) * 100}%`,
                            background: "var(--status-attack)",
                            height: "100%",
                            display: "flex",
                            alignItems: "center",
                            paddingLeft: "6px",
                            color: "#ffffff",
                            fontSize: "10px",
                            fontWeight: 700,
                          }}
                        >
                          DR: {d.recall.toFixed(1)}%
                        </div>
                      </div>
                      <div style={{ width: "120px", background: "var(--bg-surface-2)", height: "20px" }}>
                        <div
                          style={{
                            width: `${(d.fpr / 10) * 100}%`,
                            background: "var(--status-benign)",
                            height: "100%",
                            display: "flex",
                            alignItems: "center",
                            paddingLeft: "4px",
                            color: "#ffffff",
                            fontSize: "10px",
                            fontWeight: 700,
                          }}
                        >
                          FPR: {d.fpr.toFixed(1)}%
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {activeMetricTab === "params" && (
            <div style={{ minWidth: "480px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", color: "var(--text-muted)", marginBottom: "8px" }}>
                <span>Model Architecture</span>
                <span>Trainable Parameter Count (Fewer is more parameter efficient)</span>
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
                {comparisonData.map((d) => (
                  <div key={d.name} style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                    <div style={{ width: "130px", fontSize: "12px", fontWeight: 600 }}>
                      {d.name}
                    </div>
                    <div style={{ flex: 1, background: "var(--bg-surface-2)", height: "20px" }}>
                      <div
                        style={{
                          width: `${(d.params / 6401) * 100}%`,
                          background: d.params > 2000 ? "var(--border-strong)" : "var(--accent-primary)",
                          height: "100%",
                          display: "flex",
                          alignItems: "center",
                          paddingLeft: "6px",
                          color: "#ffffff",
                          fontSize: "11px",
                          fontWeight: 700,
                        }}
                      >
                        {d.params.toLocaleString()} params
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </section>

      {/* 3D Flow Tensor & Latent Compression Visualizer */}
      <section>
        <NetworkPacket3D />
      </section>

      {/* Interactive 3D Quantum Architecture Preview */}
      <section>
        <div style={{ marginBottom: "var(--space-2)" }}>
          <h2 style={{ fontSize: "var(--text-md)", fontWeight: 600, margin: 0 }}>
            Physical Architecture: Differentiable 4-Qubit Variational Circuit
          </h2>
          <p style={{ fontSize: "var(--text-xs)", color: "var(--text-muted)", margin: "2px 0 0" }}>
            Interleaved data re-uploading expands Fourier expressivity while preserving ring CNOT entanglement topology.
          </p>
        </div>
        <QuantumCircuitVisualizer initialLayers={2} mode="reuploading" />
      </section>

      {/* Compact Methodology Pipeline Diagram */}
      <section className="panel" style={{ padding: "var(--space-4)" }}>
        <div className="panel-header">
          <span className="panel-title">{t.pipelineDiagramTitle}</span>
          <span className="badge badge-neutral">Zero-Leakage Fit</span>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
            gap: "var(--space-3)",
            fontSize: "var(--text-xs)",
          }}
        >
          <div style={{ padding: "var(--space-3)", background: "var(--bg-surface-2)", border: "1px solid var(--border-subtle)" }}>
            <div style={{ fontWeight: 600, color: "var(--text-primary)", marginBottom: "4px" }}>
              1. Raw Flow Ingestion
            </div>
            <p style={{ margin: 0, color: "var(--text-secondary)" }}>
              42 network attributes (39 numeric, 3 categorical). Target labels and flow IDs strictly isolated.
            </p>
          </div>

          <div style={{ padding: "var(--space-3)", background: "var(--bg-surface-2)", border: "1px solid var(--border-subtle)" }}>
            <div style={{ fontWeight: 600, color: "var(--text-primary)", marginBottom: "4px" }}>
              2. Training-Fitted Preprocessing
            </div>
            <p style={{ margin: 0, color: "var(--text-secondary)" }}>
              ColumnTransformer (StandardScaler + OneHotEncoder with handle_unknown='ignore') applied without re-fitting.
            </p>
          </div>

          <div style={{ padding: "var(--space-3)", background: "var(--bg-surface-2)", border: "1px solid var(--border-subtle)" }}>
            <div style={{ fontWeight: 600, color: "var(--text-primary)", marginBottom: "4px" }}>
              3. Classical Encoder Projection
            </div>
            <p style={{ margin: 0, color: "var(--text-secondary)" }}>
              MLP(63 → 16 → 4) followed by π·tanh mapping 63 features to 4 bounded rotation angles θ_i ∈ [-π, π].
            </p>
          </div>

          <div style={{ padding: "var(--space-3)", background: "var(--bg-surface-2)", border: "1px solid var(--border-subtle)" }}>
            <div style={{ fontWeight: 600, color: "var(--text-primary)", marginBottom: "4px" }}>
              4. Quantum Observable & Logit
            </div>
            <p style={{ margin: 0, color: "var(--text-secondary)" }}>
              Pauli-Z expectation values ⟨Z_i⟩ measured in [-1, 1], mapped via Linear(4, 1) head to binary classification logits.
            </p>
          </div>
        </div>

        {/* Action Buttons */}
        <div style={{ display: "flex", gap: "var(--space-3)", marginTop: "var(--space-4)", flexWrap: "wrap" }}>
          <Link to="/traffic" className="btn btn-primary">
            <span>Inspect Batch Traffic</span>
            <ArrowRight size={14} />
          </Link>
          <Link to="/detection" className="btn">
            <span>Launch Detection Lab</span>
          </Link>
          <Link to="/experiments" className="btn">
            <span>View All 15 Experiment Runs</span>
          </Link>
        </div>
      </section>
    </div>
  );
};
