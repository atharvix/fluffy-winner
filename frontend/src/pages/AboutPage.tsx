import React, { useEffect, useState } from "react";
import { fetchHealth, fetchModelInfo, type HealthResponse, type ModelInfoResponse } from "../services/api";
import { 
  Server, 
  Code, 
  Cpu,
  Terminal,
} from "lucide-react";

export const AboutPage: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);

  useEffect(() => {
    let mounted = true;
    Promise.all([fetchHealth(), fetchModelInfo()])
      .then(([h, m]) => {
        if (mounted) {
          setHealth(h);
          setModelInfo(m);
        }
      })
      .catch(() => {});

    return () => {
      mounted = false;
    };
  }, []);

  return (
    <div style={{ display: "flex", flexDirection: "column", minHeight: "100%", width: "100%" }}>
      {/* 1. Technical Masthead */}
      <header className="masthead">
        <div>
          <div className="masthead-eyebrow">
            <Terminal size={12} color="var(--accent-primary)" />
            <span>06 // INSTRUMENT AUDIT & CALIBRATION SPECIFICATION</span>
          </div>
          <h1 className="masthead-title">System Architecture, Endpoints & Test Suite Audit</h1>
        </div>
        <div className="masthead-meta">
          <span>FASTAPI 0.115+</span>
          <span>&bull;</span>
          <span>59 PASSING TESTS</span>
          <span>&bull;</span>
          <span>CUDA 13.0 / RTX 5060</span>
        </div>
      </header>

      {/* 2. Continuous Horizontal Telemetry Strip */}
      <div className="telemetry-strip">
        <div className="telemetry-item">
          <span className="telemetry-label">BACKEND STATUS</span>
          <span className="telemetry-val" style={{ color: health?.status === "healthy" ? "var(--status-benign)" : "var(--status-attack)" }}>
            {health?.status === "healthy" ? "HEALTHY // PORT 8000" : "OFFLINE"}
          </span>
        </div>
        <span className="telemetry-divider">|</span>
        <div className="telemetry-item">
          <span className="telemetry-label">TEST SUITE COVERAGE</span>
          <span className="telemetry-val" style={{ color: "var(--status-benign)" }}>59/59 PYTEST TESTS PASSING</span>
        </div>
        <span className="telemetry-divider">|</span>
        <div className="telemetry-item">
          <span className="telemetry-label">DEPLOYED CHECKPOINT</span>
          <span className="telemetry-val" style={{ color: "var(--accent-quantum)" }}>{modelInfo?.model_name || "Hybrid_Reuploading_L1"}</span>
        </div>
        <span className="telemetry-divider">|</span>
        <div className="telemetry-item">
          <span className="telemetry-label">HOST ACCELERATION</span>
          <span className="telemetry-val">NVIDIA RTX 5060 8GB</span>
        </div>
      </div>

      <div style={{ padding: "1.25rem", display: "flex", flexDirection: "column", gap: "1.25rem" }}>
        {/* Live System Verification Grid */}
        <section
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
            gap: "0.75rem",
          }}
        >
          <div className="panel" style={{ padding: "1rem" }}>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: "10px", color: "var(--text-muted)" }}>BACKEND RUNTIME</div>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.4rem", fontWeight: 700, margin: "4px 0" }}>
              {health ? "FastAPI 0.115+" : "Offline"}
            </div>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: "11px", color: "var(--text-secondary)" }}>
              Python 3.13 / Uvicorn ASGI Server
            </div>
          </div>

          <div className="panel" style={{ padding: "1rem", borderLeft: "3px solid var(--accent-quantum)" }}>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: "10px", color: "var(--text-muted)" }}>DEPLOYED CHECKPOINT</div>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.2rem", fontWeight: 700, margin: "4px 0", color: "var(--accent-quantum)" }}>
              {modelInfo?.model_name || "Hybrid_Reuploading_L1"}
            </div>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: "11px", color: "var(--text-secondary)" }}>
              results/ciciot2023_full45m/best_model.pt
            </div>
          </div>

          <div className="panel" style={{ padding: "1rem", borderLeft: "3px solid var(--accent-primary)" }}>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: "10px", color: "var(--text-muted)" }}>TRAINABLE PARAMETERS</div>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.4rem", fontWeight: 700, margin: "4px 0", color: "var(--accent-primary)" }}>
              {(modelInfo?.parameter_breakdown?.total_trainable || 721).toLocaleString()} PARAMS
            </div>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: "11px", color: "var(--text-secondary)" }}>
              696 Classical Linear + 20 Quantum + 5 Head
            </div>
          </div>

          <div className="panel" style={{ padding: "1rem", borderLeft: "3px solid var(--status-benign)" }}>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: "10px", color: "var(--text-muted)" }}>TEST AUTOMATION SUITE</div>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.4rem", fontWeight: 700, margin: "4px 0", color: "var(--status-benign)" }}>
              59 / 59 PASSED
            </div>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: "11px", color: "var(--text-secondary)" }}>
              Verified deterministic pipeline & zero test leakage
            </div>
          </div>
        </section>

        {/* Tech Stack & REST Endpoints Grid */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.25rem" }}>
          {/* Left: Technology Stack */}
          <section className="panel" style={{ padding: 0, overflow: "hidden" }}>
            <div
              style={{
                padding: "0.75rem 1rem",
                background: "var(--bg-surface-2)",
                borderBottom: "1px solid var(--border-default)",
                fontFamily: "var(--font-heading)",
                fontSize: "13px",
                fontWeight: 700,
              }}
            >
              ENGINEERED HARDWARE & SOFTWARE STACK
            </div>
            <div style={{ padding: "1rem", display: "flex", flexDirection: "column", gap: "0.75rem" }}>
              <div style={{ padding: "0.75rem", background: "var(--bg-code)", border: "1px solid var(--border-subtle)" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "6px", fontWeight: 700, fontFamily: "var(--font-mono)", fontSize: "11px", color: "var(--accent-primary)" }}>
                  <Cpu size={14} />
                  <span>QUANTUM & NEURAL BACKEND</span>
                </div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "11px", color: "var(--text-secondary)", marginTop: "4px", lineHeight: 1.5 }}>
                  PennyLane 0.40+ statevector simulation, PyTorch 2.6 with CUDA 13.0 autograd, Scikit-Learn 1.6+ streaming standard scaler, Joblib atomic serialization.
                </div>
              </div>

              <div style={{ padding: "0.75rem", background: "var(--bg-code)", border: "1px solid var(--border-subtle)" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "6px", fontWeight: 700, fontFamily: "var(--font-mono)", fontSize: "11px", color: "var(--accent-quantum)" }}>
                  <Server size={14} />
                  <span>REST API & INFERENCE PIPELINE</span>
                </div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "11px", color: "var(--text-secondary)", marginTop: "4px", lineHeight: 1.5 }}>
                  FastAPI 0.115 asynchronous JSON/CSV streaming endpoints, Pydantic v2 schema validation, single-turn non-blocking thread execution on Windows Python 3.13.
                </div>
              </div>

              <div style={{ padding: "0.75rem", background: "var(--bg-code)", border: "1px solid var(--border-subtle)" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "6px", fontWeight: 700, fontFamily: "var(--font-mono)", fontSize: "11px", color: "var(--status-benign)" }}>
                  <Code size={14} />
                  <span>OSCILLOSCOPE WORKBENCH UI</span>
                </div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "11px", color: "var(--text-secondary)", marginTop: "4px", lineHeight: 1.5 }}>
                  React 19 + TypeScript 5.8 + Vite, pure CSS custom property design tokens, Keysight ruled-line architecture, 4 high-contrast hardware instrument palettes.
                </div>
              </div>
            </div>
          </section>

          {/* Right: REST Endpoints */}
          <section className="panel" style={{ padding: 0, overflow: "hidden" }}>
            <div
              style={{
                padding: "0.75rem 1rem",
                background: "var(--bg-surface-2)",
                borderBottom: "1px solid var(--border-default)",
                fontFamily: "var(--font-heading)",
                fontSize: "13px",
                fontWeight: 700,
              }}
            >
              REST INTERFACE & TELEMETRY ENDPOINTS
            </div>
            <div style={{ padding: "1rem", display: "flex", flexDirection: "column", gap: "0.5rem" }}>
              <div style={{ padding: "0.5rem 0.75rem", background: "var(--bg-code)", border: "1px solid var(--border-subtle)", fontFamily: "var(--font-mono)", fontSize: "11px" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                  <span className="badge badge-benign">GET</span>
                  <span style={{ color: "var(--text-primary)", fontWeight: 700 }}>/api/health</span>
                </div>
                <div style={{ color: "var(--text-muted)", marginTop: "2px", fontSize: "10px" }}>
                  Returns service heartbeat, backend device, and active model checkpoint state.
                </div>
              </div>

              <div style={{ padding: "0.5rem 0.75rem", background: "var(--bg-code)", border: "1px solid var(--border-subtle)", fontFamily: "var(--font-mono)", fontSize: "11px" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                  <span className="badge badge-benign">GET</span>
                  <span style={{ color: "var(--text-primary)", fontWeight: 700 }}>/api/model-info</span>
                </div>
                <div style={{ color: "var(--text-muted)", marginTop: "2px", fontSize: "10px" }}>
                  Detailed parameter breakdown, ansatz layer depth, and optimal decision threshold &tau;*.
                </div>
              </div>

              <div style={{ padding: "0.5rem 0.75rem", background: "var(--bg-code)", border: "1px solid var(--border-subtle)", fontFamily: "var(--font-mono)", fontSize: "11px" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                  <span className="badge badge-attack">POST</span>
                  <span style={{ color: "var(--text-primary)", fontWeight: 700 }}>/api/predict</span>
                </div>
                <div style={{ color: "var(--text-muted)", marginTop: "2px", fontSize: "10px" }}>
                  Single or batch JSON flow vector prediction returning probabilities and quantum observables.
                </div>
              </div>

              <div style={{ padding: "0.5rem 0.75rem", background: "var(--bg-code)", border: "1px solid var(--border-subtle)", fontFamily: "var(--font-mono)", fontSize: "11px" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                  <span className="badge badge-attack">POST</span>
                  <span style={{ color: "var(--text-primary)", fontWeight: 700 }}>/api/predict-csv</span>
                </div>
                <div style={{ color: "var(--text-muted)", marginTop: "2px", fontSize: "10px" }}>
                  Multipart CSV packet capture batch ingestion with streaming JSON or labeled CSV responses.
                </div>
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
};
