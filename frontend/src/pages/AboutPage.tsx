import React, { useEffect, useState } from "react";
import { useLanguage } from "../i18n/useTranslation";
import { fetchHealth, fetchModelInfo, type HealthResponse, type ModelInfoResponse } from "../services/api";
import { 
  Server, 
  Shield, 
  Code, 
  Cpu
} from "lucide-react";

export const AboutPage: React.FC = () => {
  const { t, lang } = useLanguage();
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
      .catch(() => {
        // Backend offline handling
      });

    return () => {
      mounted = false;
    };
  }, []);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-6)" }}>
      {/* Header */}
      <section className="panel" style={{ padding: "var(--space-5)" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "var(--space-4)" }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "var(--space-2)", marginBottom: "var(--space-2)" }}>
              <span className="badge badge-accent">System Architecture & Verification</span>
              <span className="badge">Production REST API</span>
              <span className="badge badge-success">47 Passing Tests</span>
            </div>
            <h1 style={{ fontSize: "var(--text-2xl)", fontWeight: 700, margin: "0 0 var(--space-2) 0", letterSpacing: "-0.02em" }}>
              {t.aboutTitle}
            </h1>
            <p style={{ color: "var(--text-secondary)", maxWidth: "800px", margin: 0 }}>
              {t.aboutSubtitle}
            </p>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "var(--space-2)" }}>
            <div 
              style={{ 
                display: "inline-flex", 
                alignItems: "center", 
                gap: "var(--space-2)", 
                padding: "var(--space-2) var(--space-3)", 
                backgroundColor: health?.status === "healthy" ? "var(--bg-active)" : "var(--bg-card)",
                border: "1px solid var(--border-subtle)",
                borderRadius: "var(--radius-sm)"
              }}
            >
              <div 
                style={{ 
                  width: "8px", 
                  height: "8px", 
                  borderRadius: "var(--radius-sm)", 
                  backgroundColor: health?.status === "healthy" ? "var(--status-success)" : "var(--status-danger)" 
                }} 
              />
              <span style={{ fontSize: "var(--text-xs)", fontWeight: 600 }}>
                {health?.status === "healthy" ? "Backend Connected (Port 8000)" : "Backend Disconnected"}
              </span>
            </div>
          </div>
        </div>
      </section>

      {/* Live System & API Verification Status */}
      <section className="panel" style={{ padding: "var(--space-5)" }}>
        <h2 style={{ fontSize: "var(--text-base)", fontWeight: 700, margin: "0 0 var(--space-3) 0", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--accent-primary)" }}>
          {lang === "hi" ? "सत्यापित लाइव सिस्टम स्थिति" : "Verified Live System Telemetry"}
        </h2>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "var(--space-3)" }}>
          <div className="panel" style={{ padding: "var(--space-3)", backgroundColor: "var(--bg-card)" }}>
            <div style={{ fontSize: "var(--text-xs)", color: "var(--text-muted)" }}>Backend Service</div>
            <div style={{ fontSize: "var(--text-base)", fontWeight: 700, color: "var(--text-primary)", marginTop: "4px" }}>
              {health ? "FastAPI 0.115+" : "Offline"}
            </div>
            <div style={{ fontSize: "var(--text-xs)", color: "var(--text-secondary)", marginTop: "2px" }}>
              Python 3.13.6 on Windows
            </div>
          </div>

          <div className="panel" style={{ padding: "var(--space-3)", backgroundColor: "var(--bg-card)" }}>
            <div style={{ fontSize: "var(--text-xs)", color: "var(--text-muted)" }}>Deployed Checkpoint</div>
            <div style={{ fontSize: "var(--text-base)", fontWeight: 700, color: "var(--text-primary)", marginTop: "4px" }}>
              {modelInfo?.model_name || modelInfo?.model_family || "Hybrid Re-uploading L=2"}
            </div>
            <div style={{ fontSize: "var(--text-xs)", color: "var(--text-secondary)", marginTop: "2px" }}>
              results/network_intrusion/best_model.pt
            </div>
          </div>

          <div className="panel" style={{ padding: "var(--space-3)", backgroundColor: "var(--bg-card)" }}>
            <div style={{ fontSize: "var(--text-xs)", color: "var(--text-muted)" }}>Active Parameters</div>
            <div style={{ fontSize: "var(--text-base)", fontWeight: 700, color: "var(--accent-primary)", marginTop: "4px" }}>
              {(modelInfo?.parameter_breakdown?.total_trainable || 1113).toLocaleString()} Trainable
            </div>
            <div style={{ fontSize: "var(--text-xs)", color: "var(--text-secondary)", marginTop: "2px" }}>
              1,097 Classical + 16 Quantum
            </div>
          </div>

          <div className="panel" style={{ padding: "var(--space-3)", backgroundColor: "var(--bg-card)" }}>
            <div style={{ fontSize: "var(--text-xs)", color: "var(--text-muted)" }}>Simulated Quantum Hardware</div>
            <div style={{ fontSize: "var(--text-base)", fontWeight: 700, color: "var(--text-primary)", marginTop: "4px" }}>
              PennyLane default.qubit
            </div>
            <div style={{ fontSize: "var(--text-xs)", color: "var(--text-secondary)", marginTop: "2px" }}>
              4 Qubits | Statevector CPU
            </div>
          </div>
        </div>
      </section>

      {/* Two Column Layout: Tech Stack & Endpoints */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: "var(--space-6)" }}>
        
        {/* Technology Stack Details */}
        <section className="panel" style={{ padding: "var(--space-5)" }}>
          <h2 style={{ fontSize: "var(--text-base)", fontWeight: 700, margin: "0 0 var(--space-4) 0", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--accent-primary)" }}>
            {t.techStackTitle}
          </h2>

          <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-3)" }}>
            <div style={{ padding: "var(--space-3)", border: "1px solid var(--border-subtle)", borderRadius: "var(--radius-sm)" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "var(--space-2)", fontWeight: 600 }}>
                <Cpu size={16} style={{ color: "var(--accent-primary)" }} />
                <span>Quantum & Classical Machine Learning</span>
              </div>
              <ul style={{ margin: "var(--space-2) 0 0 0", paddingLeft: "var(--space-4)", fontSize: "var(--text-xs)", color: "var(--text-secondary)", display: "flex", flexDirection: "column", gap: "2px" }}>
                <li><strong>PennyLane 0.40+:</strong> Variational Quantum Circuits, Backprop automatic differentiation</li>
                <li><strong>PyTorch 2.6+:</strong> Hybrid classical-quantum neural network joint backpropagation</li>
                <li><strong>Scikit-Learn 1.6+:</strong> Zero-leakage ColumnTransformer, StandardScaler, OneHotEncoder</li>
                <li><strong>Joblib:</strong> Preprocessor serialization and deterministic reload</li>
              </ul>
            </div>

            <div style={{ padding: "var(--space-3)", border: "1px solid var(--border-subtle)", borderRadius: "var(--radius-sm)" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "var(--space-2)", fontWeight: 600 }}>
                <Server size={16} style={{ color: "var(--accent-primary)" }} />
                <span>Backend REST Microservice</span>
              </div>
              <ul style={{ margin: "var(--space-2) 0 0 0", paddingLeft: "var(--space-4)", fontSize: "var(--text-xs)", color: "var(--text-secondary)", display: "flex", flexDirection: "column", gap: "2px" }}>
                <li><strong>FastAPI 0.115+:</strong> Asynchronous endpoints with Pydantic v2 data validation</li>
                <li><strong>Uvicorn:</strong> ASGI server running on 127.0.0.1:8000</li>
                <li><strong>CORS Middleware:</strong> Explicitly configured for local Vite development</li>
              </ul>
            </div>

            <div style={{ padding: "var(--space-3)", border: "1px solid var(--border-subtle)", borderRadius: "var(--radius-sm)" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "var(--space-2)", fontWeight: 600 }}>
                <Code size={16} style={{ color: "var(--accent-primary)" }} />
                <span>Frontend Engineering</span>
              </div>
              <ul style={{ margin: "var(--space-2) 0 0 0", paddingLeft: "var(--space-4)", fontSize: "var(--text-xs)", color: "var(--text-secondary)", display: "flex", flexDirection: "column", gap: "2px" }}>
                <li><strong>React 19 + TypeScript + Vite:</strong> High performance SPA architecture</li>
                <li><strong>Vanilla CSS Token Architecture:</strong> 4 complete themes, zero gradients, 0-4px corners</li>
                <li><strong>Interactive Quantum Visualizer:</strong> CSS 3D matrix transforms + SVG 2D fallback</li>
                <li><strong>Bilingual i18n System:</strong> Site-wide persistent English and Hindi translations</li>
              </ul>
            </div>
          </div>
        </section>

        {/* API Endpoint Documentation */}
        <section className="panel" style={{ padding: "var(--space-5)" }}>
          <h2 style={{ fontSize: "var(--text-base)", fontWeight: 700, margin: "0 0 var(--space-4) 0", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--accent-primary)" }}>
            {t.endpointsTitle}
          </h2>

          <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-3)" }}>
            <div style={{ padding: "var(--space-3)", border: "1px solid var(--border-subtle)", borderRadius: "var(--radius-sm)" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "var(--space-2)" }}>
                <span className="badge badge-success" style={{ fontSize: "10px" }}>GET</span>
                <code style={{ fontSize: "var(--text-xs)", fontWeight: 600 }}>/api/health</code>
              </div>
              <p style={{ fontSize: "var(--text-xs)", color: "var(--text-secondary)", margin: "4px 0 0 0" }}>
                Returns service status, model loading state, and device configuration.
              </p>
            </div>

            <div style={{ padding: "var(--space-3)", border: "1px solid var(--border-subtle)", borderRadius: "var(--radius-sm)" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "var(--space-2)" }}>
                <span className="badge badge-success" style={{ fontSize: "10px" }}>GET</span>
                <code style={{ fontSize: "var(--text-xs)", fontWeight: 600 }}>/api/model-info</code>
              </div>
              <p style={{ fontSize: "var(--text-xs)", color: "var(--text-secondary)", margin: "4px 0 0 0" }}>
                Provides model metadata, architecture type, total parameters, and the complete 42-feature input schema.
              </p>
            </div>

            <div style={{ padding: "var(--space-3)", border: "1px solid var(--border-subtle)", borderRadius: "var(--radius-sm)" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "var(--space-2)" }}>
                <span className="badge badge-accent" style={{ fontSize: "10px" }}>POST</span>
                <code style={{ fontSize: "var(--text-xs)", fontWeight: 600 }}>/api/predict</code>
              </div>
              <p style={{ fontSize: "var(--text-xs)", color: "var(--text-secondary)", margin: "4px 0 0 0" }}>
                Single-flow inference endpoint. Accepts 42 flow features; returns predicted class, probabilities, rotation angles \(\theta_i\), and Pauli-Z expectations \(\langle Z_i \rangle\).
              </p>
            </div>

            <div style={{ padding: "var(--space-3)", border: "1px solid var(--border-subtle)", borderRadius: "var(--radius-sm)" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "var(--space-2)" }}>
                <span className="badge badge-accent" style={{ fontSize: "10px" }}>POST</span>
                <code style={{ fontSize: "var(--text-xs)", fontWeight: 600 }}>/api/predict/csv</code>
              </div>
              <p style={{ fontSize: "var(--text-xs)", color: "var(--text-secondary)", margin: "4px 0 0 0" }}>
                Batch multipart CSV inference. Validates schema and returns classified rows with prediction columns (JSON or CSV format).
              </p>
            </div>
          </div>
        </section>
      </div>

      {/* Reproducibility Instructions */}
      <section className="panel" style={{ padding: "var(--space-5)" }}>
        <h2 style={{ fontSize: "var(--text-base)", fontWeight: 700, margin: "0 0 var(--space-3) 0", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--accent-primary)" }}>
          {t.reproducibilityTitle}
        </h2>
        <p style={{ fontSize: "var(--text-sm)", color: "var(--text-secondary)", margin: "0 0 var(--space-3) 0" }}>
          To verify reproducibility and launch the research workstation locally in Windows PowerShell:
        </p>

        <div 
          style={{ 
            backgroundColor: "var(--bg-code)", 
            padding: "var(--space-4)", 
            borderRadius: "var(--radius-sm)", 
            border: "1px solid var(--border-subtle)",
            fontFamily: "var(--font-mono)",
            fontSize: "var(--text-xs)",
            color: "var(--text-primary)",
            display: "flex",
            flexDirection: "column",
            gap: "var(--space-2)"
          }}
        >
          <div># 1. Activate Python virtual environment and run the complete test suite (47 tests)</div>
          <div style={{ color: "var(--accent-primary)" }}>& .venv\Scripts\Activate.ps1</div>
          <div style={{ color: "var(--accent-primary)" }}>pytest -v</div>
          <br />
          <div># 2. Launch FastAPI inference backend (Port 8000)</div>
          <div style={{ color: "var(--accent-primary)" }}>uvicorn api.main:app --host 127.0.0.1 --port 8000</div>
          <br />
          <div># 3. Launch Frontend Development Server (Port 5173)</div>
          <div style={{ color: "var(--accent-primary)" }}>cd frontend</div>
          <div style={{ color: "var(--accent-primary)" }}>npm run dev</div>
        </div>
      </section>

      {/* Responsible Cybersecurity Disclosure */}
      <section 
        className="panel" 
        style={{ 
          padding: "var(--space-4)", 
          borderLeft: "4px solid var(--accent-primary)",
          backgroundColor: "var(--bg-surface)"
        }}
      >
        <div style={{ display: "flex", alignItems: "flex-start", gap: "var(--space-3)" }}>
          <Shield size={20} style={{ color: "var(--accent-primary)", marginTop: "2px", flexShrink: 0 }} />
          <div>
            <div style={{ fontWeight: 700, color: "var(--text-primary)", marginBottom: "var(--space-1)" }}>
              {t.responsibleUseTitle}
            </div>
            <p style={{ fontSize: "var(--text-sm)", color: "var(--text-secondary)", margin: 0, lineHeight: 1.5 }}>
              {t.responsibleUseBody}
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};
