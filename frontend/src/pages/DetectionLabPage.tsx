import React, { useState, useEffect } from "react";
import {
  Play,
  RotateCcw,
  Cpu,
  Activity,
  Terminal,
  Crosshair,
} from "lucide-react";
import {
  predictBatch,
  fetchSamplePresets,
  type FlowPreset,
  type SingleFlowRecord,
  type PredictBatchResponse,
} from "../services/api";
import { QuantumCircuitVisualizer } from "../features/quantum-visualization/QuantumCircuitVisualizer";
import { BlochSphere3D } from "../features/quantum-visualization/BlochSphere3D";

// Exact 39 Numerical Features grouped into 4 domain tabs for CIC-IoT-2023 benchmark
const FEATURE_GROUPS = [
  {
    id: "header_rates",
    name: "[HDR / RATE]",
    features: [
      { key: "Header_Length", label: "Header_Length", step: "1" },
      { key: "Protocol Type", label: "Protocol Type", step: "1" },
      { key: "Time_To_Live", label: "Time_To_Live", step: "0.5" },
      { key: "Rate", label: "Rate (pkts/s)", step: "0.1" },
    ],
  },
  {
    id: "tcp_flags",
    name: "[TCP FLAGS]",
    features: [
      { key: "fin_flag_number", label: "fin_flag", step: "0.1" },
      { key: "syn_flag_number", label: "syn_flag", step: "0.1" },
      { key: "rst_flag_number", label: "rst_flag", step: "0.1" },
      { key: "psh_flag_number", label: "psh_flag", step: "0.1" },
      { key: "ack_flag_number", label: "ack_flag", step: "0.1" },
      { key: "ece_flag_number", label: "ece_flag", step: "0.1" },
      { key: "cwr_flag_number", label: "cwr_flag", step: "0.1" },
      { key: "ack_count", label: "ack_count", step: "1" },
      { key: "syn_count", label: "syn_count", step: "1" },
      { key: "fin_count", label: "fin_count", step: "1" },
      { key: "rst_count", label: "rst_count", step: "1" },
    ],
  },
  {
    id: "protocols",
    name: "[PROTOCOLS]",
    features: [
      { key: "HTTP", label: "HTTP", step: "0.1" },
      { key: "HTTPS", label: "HTTPS", step: "0.1" },
      { key: "DNS", label: "DNS", step: "0.1" },
      { key: "Telnet", label: "Telnet", step: "0.1" },
      { key: "SMTP", label: "SMTP", step: "0.1" },
      { key: "SSH", label: "SSH", step: "0.1" },
      { key: "IRC", label: "IRC", step: "0.1" },
      { key: "TCP", label: "TCP", step: "0.1" },
      { key: "UDP", label: "UDP", step: "0.1" },
      { key: "DHCP", label: "DHCP", step: "0.1" },
      { key: "ARP", label: "ARP", step: "0.1" },
      { key: "ICMP", label: "ICMP", step: "0.1" },
      { key: "IGMP", label: "IGMP", step: "0.1" },
      { key: "IPv", label: "IPv", step: "0.1" },
      { key: "LLC", label: "LLC", step: "0.1" },
    ],
  },
  {
    id: "packet_stats",
    name: "[PKT STATS]",
    features: [
      { key: "Tot sum", label: "Tot sum", step: "1" },
      { key: "Min", label: "Min", step: "1" },
      { key: "Max", label: "Max", step: "1" },
      { key: "AVG", label: "AVG", step: "0.1" },
      { key: "Std", label: "Std", step: "0.1" },
      { key: "Tot size", label: "Tot size", step: "0.1" },
      { key: "IAT", label: "IAT (s)", step: "0.0001" },
      { key: "Number", label: "Number", step: "1" },
      { key: "Variance", label: "Variance", step: "0.1" },
    ],
  },
];

export const DetectionLabPage: React.FC = () => {
  const [presets, setPresets] = useState<FlowPreset[]>([]);
  const [selectedPresetIndex, setSelectedPresetIndex] = useState<string>("0");
  const [activeGroup, setActiveGroup] = useState<string>("header_rates");

  // Default flow data initialized to standard CIC-IoT-2023 flow record
  const [formData, setFormData] = useState<Record<string, any>>({
    Header_Length: 28.0,
    "Protocol Type": 6.0,
    Time_To_Live: 63.5,
    Rate: 201.021042,
    fin_flag_number: 0.0,
    syn_flag_number: 0.1,
    rst_flag_number: 0.0,
    psh_flag_number: 0.1,
    ack_flag_number: 0.7,
    ece_flag_number: 0.1,
    cwr_flag_number: 0.1,
    ack_count: 7.0,
    syn_count: 1.0,
    fin_count: 0.0,
    rst_count: 0.0,
    HTTP: 0.1,
    HTTPS: 0.7,
    DNS: 0.0,
    Telnet: 0.0,
    SMTP: 0.0,
    SSH: 0.0,
    IRC: 0.0,
    TCP: 0.8,
    UDP: 0.2,
    DHCP: 0.0,
    ARP: 0.0,
    ICMP: 0.0,
    IGMP: 0.0,
    IPv: 1.0,
    LLC: 1.0,
    "Tot sum": 9434.0,
    Min: 60.0,
    Max: 2962.0,
    AVG: 943.400024,
    Std: 1004.631531,
    "Tot size": 943.400024,
    IAT: 0.005939,
    Number: 10.0,
    Variance: 1009284.5,
  });

  const [threshold, setThreshold] = useState<number>(0.2971);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [result, setResult] = useState<PredictBatchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [inferenceTimeMs, setInferenceTimeMs] = useState<number>(1.42);

  useEffect(() => {
    fetchSamplePresets()
      .then((data) => {
        setPresets(data);
        if (data.length > 0) {
          setSelectedPresetIndex("0");
          setFormData(data[0].record);
        }
      })
      .catch(() => {});
  }, []);

  const handlePresetSelect = (idxStr: string) => {
    setSelectedPresetIndex(idxStr);
    const idx = parseInt(idxStr, 10);
    if (!isNaN(idx) && presets[idx]) {
      setFormData(presets[idx].record);
      setResult(null);
      setError(null);
    }
  };

  const handleInputChange = (field: string, value: any) => {
    setFormData((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  const executePrediction = async () => {
    setIsLoading(true);
    setError(null);
    const startTime = performance.now();

    try {
      const record = { ...(formData as SingleFlowRecord), id: "lab-bench-flow" };
      const res = await predictBatch([record], threshold);
      const elapsed = performance.now() - startTime;
      setInferenceTimeMs(parseFloat(elapsed.toFixed(2)));
      setResult(res);
      setIsLoading(false);
    } catch (err: any) {
      setError(err.message || "Quantum inference execution failed.");
      setIsLoading(false);
    }
  };

  const currentPred = result?.predictions[0];
  const encoderAngles = currentPred?.quantum_inspection?.encoder_angles || [0.42, -0.78, 1.15, -0.31];
  const pauliExpvals = currentPred?.quantum_inspection?.pauli_z_expectations || [0.82, -0.61, 0.45, -0.73];

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", width: "100%" }}>
      {/* 1. Technical Masthead */}
      <header className="masthead">
        <div>
          <div className="masthead-eyebrow">
            <Crosshair size={12} color="var(--accent-primary)" />
            <span>CIC-IoT-2023 // VARIATIONAL QUANTUM NIDS BENCHMARK</span>
          </div>
          <h1 className="masthead-title">39-Feature Qubit Bench & Statevector Oscilloscope</h1>
        </div>
        <div className="masthead-meta">
          <span>PIPELINE: 39-D &rarr; 4-Q REUPLOAD</span>
          <span>&bull;</span>
          <span>OPT-TAU: 0.2971</span>
          <span>&bull;</span>
          <span>ENGINE: PENNYLANE 0.40</span>
        </div>
      </header>

      {/* 2. Continuous Horizontal Telemetry Strip */}
      <div className="telemetry-strip">
        <div className="telemetry-item">
          <span className="telemetry-label">MODEL</span>
          <span className="telemetry-val" style={{ color: "var(--accent-quantum)" }}>HYBRID_REUPLOADING_L1</span>
        </div>
        <span className="telemetry-divider">|</span>
        <div className="telemetry-item">
          <span className="telemetry-label">PRE-TRAINED DATASET</span>
          <span className="telemetry-val">CIC-IoT-2023 (45,019,243 FLOWS)</span>
        </div>
        <span className="telemetry-divider">|</span>
        <div className="telemetry-item">
          <span className="telemetry-label">TRAINABLE PARAMS</span>
          <span className="telemetry-val">721 (-85.2% vs MLP)</span>
        </div>
        <span className="telemetry-divider">|</span>
        <div className="telemetry-item">
          <span className="telemetry-label">LATENT REGISTER</span>
          <span className="telemetry-val" style={{ color: "var(--accent-primary)" }}>4 QUBITS // 1 LAYER</span>
        </div>
        <span className="telemetry-divider">|</span>
        <div className="telemetry-item">
          <span className="telemetry-label">CALIBRATED TAU*</span>
          <span className="telemetry-val" style={{ color: "var(--accent-primary)" }}>{threshold.toFixed(4)}</span>
        </div>
      </div>

      {/* 3. Three-Column Split Workbench Geometry */}
      <div
        style={{
          flex: 1,
          display: "grid",
          gridTemplateColumns: "34% 36% 30%",
          height: "calc(100vh - 120px)",
          overflow: "hidden",
        }}
      >
        {/* COLUMN 1: 39-Feature Register Editor */}
        <section
          className="ruled-border-r"
          style={{
            display: "flex",
            flexDirection: "column",
            height: "100%",
            background: "var(--bg-surface-1)",
            overflow: "hidden",
          }}
        >
          {/* Preset Selector Bar */}
          <div
            style={{
              padding: "0.75rem 1rem",
              background: "var(--bg-surface-2)",
              borderBottom: "1px solid var(--border-default)",
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: "4px",
              }}
            >
              <span
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: "10px",
                  fontWeight: 700,
                  letterSpacing: "0.08em",
                  color: "var(--text-muted)",
                }}
              >
                HELD-OUT CAPTURE PRESETS
              </span>
              <button
                type="button"
                onClick={() => handlePresetSelect("0")}
                style={{
                  background: "transparent",
                  border: "none",
                  color: "var(--text-secondary)",
                  cursor: "pointer",
                  fontSize: "10px",
                  display: "flex",
                  alignItems: "center",
                  gap: "3px",
                  fontFamily: "var(--font-mono)",
                }}
              >
                <RotateCcw size={10} />
                RESET
              </button>
            </div>
            <select
              value={selectedPresetIndex}
              onChange={(e) => handlePresetSelect(e.target.value)}
              className="select-input"
              style={{ fontSize: "11px", fontWeight: 600 }}
            >
              {presets.map((preset, idx) => (
                <option key={idx} value={idx.toString()}>
                  [{preset.attack_cat}] {preset.name}
                </option>
              ))}
            </select>
          </div>

          {/* 4 Domain Group Filter Tabs */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(4, 1fr)",
              background: "var(--bg-surface-1)",
              borderBottom: "1px solid var(--border-default)",
            }}
          >
            {FEATURE_GROUPS.map((grp) => (
              <button
                key={grp.id}
                type="button"
                onClick={() => setActiveGroup(grp.id)}
                style={{
                  padding: "6px 2px",
                  fontSize: "10px",
                  fontFamily: "var(--font-mono)",
                  fontWeight: activeGroup === grp.id ? 700 : 500,
                  background: activeGroup === grp.id ? "var(--bg-surface-3)" : "transparent",
                  color: activeGroup === grp.id ? "var(--accent-primary)" : "var(--text-secondary)",
                  border: "none",
                  borderBottom: activeGroup === grp.id ? "2px solid var(--accent-primary)" : "2px solid transparent",
                  borderRight: "1px solid var(--border-subtle)",
                  cursor: "pointer",
                  textAlign: "center",
                }}
              >
                {grp.name}
              </button>
            ))}
          </div>

          {/* Dense Ruled Spreadsheet Input Rows */}
          <div
            style={{
              flex: 1,
              overflowY: "auto",
              padding: "0.5rem 0.75rem",
              background: "var(--bg-input)",
            }}
          >
            <div style={{ display: "flex", flexDirection: "column", gap: "2px" }}>
              {FEATURE_GROUPS.find((g) => g.id === activeGroup)?.features.map(({ key, label, step }) => (
                <div
                  key={key}
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 110px",
                    alignItems: "center",
                    gap: "8px",
                    padding: "3px 6px",
                    borderBottom: "1px solid var(--border-subtle)",
                    background: "var(--bg-surface-1)",
                  }}
                >
                  <label
                    style={{
                      fontFamily: "var(--font-mono)",
                      fontSize: "11px",
                      color: "var(--text-secondary)",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                    }}
                    title={label}
                  >
                    {label}
                  </label>
                  <input
                    type="number"
                    step={step}
                    value={formData[key] !== undefined ? formData[key] : 0}
                    onChange={(e) => handleInputChange(key, parseFloat(e.target.value) || 0)}
                    style={{
                      padding: "2px 6px",
                      fontSize: "11px",
                      textAlign: "right",
                      fontFamily: "var(--font-mono)",
                      background: "var(--bg-input)",
                      border: "1px solid var(--border-default)",
                      color: "var(--text-primary)",
                    }}
                  />
                </div>
              ))}
            </div>
          </div>

          {/* Pinned Bottom Bar: Threshold Slider & Trigger Button */}
          <div
            style={{
              padding: "0.75rem 1rem",
              background: "var(--bg-surface-2)",
              borderTop: "1px solid var(--border-default)",
              display: "flex",
              flexDirection: "column",
              gap: "8px",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontFamily: "var(--font-mono)", fontSize: "10px", color: "var(--text-muted)" }}>
                DECISION THRESHOLD (TAU*)
              </span>
              <span
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: "11px",
                  fontWeight: 700,
                  color: "var(--accent-primary)",
                }}
              >
                {threshold.toFixed(4)}
              </span>
            </div>
            <input
              type="range"
              min="0.05"
              max="0.95"
              step="0.005"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              style={{
                width: "100%",
                accentColor: "var(--accent-primary)",
                cursor: "pointer",
              }}
            />

            <button
              type="button"
              className="btn btn-primary"
              disabled={isLoading}
              onClick={executePrediction}
              style={{
                width: "100%",
                padding: "8px 12px",
                fontSize: "11px",
                fontWeight: 700,
                letterSpacing: "0.08em",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: "6px",
              }}
            >
              <Play size={12} />
              <span>{isLoading ? "CALCULATING EXPVALS..." : "EXECUTE QUANTUM INFERENCE"}</span>
            </button>
          </div>
        </section>

        {/* COLUMN 2: Quantum Circuit & 3D Bloch Telemetry */}
        <section
          className="ruled-border-r"
          style={{
            display: "flex",
            flexDirection: "column",
            height: "100%",
            background: "var(--bg-surface-1)",
            overflowY: "auto",
          }}
        >
          {/* Architecture Pipeline Banner */}
          <div
            style={{
              padding: "0.6rem 1rem",
              background: "var(--bg-code)",
              borderBottom: "1px solid var(--border-default)",
              fontFamily: "var(--font-mono)",
              fontSize: "10px",
              color: "var(--accent-quantum)",
              display: "flex",
              alignItems: "center",
              gap: "6px",
            }}
          >
            <Cpu size={12} />
            <span>39 RAW &rarr; StandardScaler &rarr; Linear(39&rarr;16&rarr;4) &rarr; 4-Qubit Re-upload</span>
          </div>

          {/* 4-Channel Qubit Telemetry Table */}
          <div style={{ padding: "0.75rem 1rem", borderBottom: "1px solid var(--border-default)" }}>
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: "10px",
                fontWeight: 700,
                letterSpacing: "0.08em",
                color: "var(--text-muted)",
                marginBottom: "6px",
                display: "flex",
                justifyContent: "space-between",
              }}
            >
              <span>4-CHANNEL QUBIT REGISTER OBSERVABLES</span>
              <span style={{ color: "var(--accent-quantum)" }}>&lang;Z_i&rang; &isin; [-1, +1]</span>
            </div>
            <table>
              <thead>
                <tr>
                  <th>CHANNEL</th>
                  <th>&theta;_i (RAD)</th>
                  <th>OBS &lang;Z_i&rang;</th>
                  <th>P(|0&rang;)</th>
                  <th>P(|1&rang;)</th>
                </tr>
              </thead>
              <tbody>
                {[0, 1, 2, 3].map((qIdx) => {
                  const angle = encoderAngles[qIdx] ?? 0;
                  const expval = pauliExpvals[qIdx] ?? 0;
                  const p0 = ((1 + expval) / 2) * 100;
                  const p1 = ((1 - expval) / 2) * 100;
                  return (
                    <tr key={qIdx}>
                      <td style={{ fontFamily: "var(--font-mono)", fontWeight: 700, color: "var(--accent-quantum)" }}>
                        q[{qIdx}]
                      </td>
                      <td>{angle.toFixed(4)}</td>
                      <td style={{ color: expval >= 0 ? "var(--status-benign)" : "var(--status-attack)" }}>
                        {expval >= 0 ? `+${expval.toFixed(4)}` : expval.toFixed(4)}
                      </td>
                      <td>{p0.toFixed(1)}%</td>
                      <td>{p1.toFixed(1)}%</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Interactive 3D Bloch Sphere Component */}
          <div style={{ padding: "0.75rem 1rem", borderBottom: "1px solid var(--border-default)" }}>
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: "10px",
                fontWeight: 700,
                letterSpacing: "0.08em",
                color: "var(--text-muted)",
                marginBottom: "6px",
              }}
            >
              BLOCH SPHERE STATEVECTOR PROJECTION
            </div>
            <BlochSphere3D
              angles={encoderAngles}
              expvals={pauliExpvals}
              initialQubit={0}
              interactive={true}
            />
          </div>

          {/* Quantum Circuit Visualizer */}
          <div style={{ padding: "0.75rem 1rem" }}>
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: "10px",
                fontWeight: 700,
                letterSpacing: "0.08em",
                color: "var(--text-muted)",
                marginBottom: "6px",
              }}
            >
              CIRCUIT TOPOLOGY (DATA RE-UPLOADING L=1)
            </div>
            <QuantumCircuitVisualizer
              initialLayers={1}
              mode="reuploading"
              activeAngles={encoderAngles}
              activeExpvals={pauliExpvals}
            />
          </div>
        </section>

        {/* COLUMN 3: Live Verdict & Probability Gauge */}
        <section
          style={{
            display: "flex",
            flexDirection: "column",
            height: "100%",
            background: "var(--bg-surface-1)",
            padding: "1rem",
            overflowY: "auto",
            gap: "1rem",
          }}
        >
          <div
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: "10px",
              fontWeight: 700,
              letterSpacing: "0.08em",
              color: "var(--text-muted)",
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
            }}
          >
            <span>03 // CLASSIFICATION VERDICT</span>
            <Activity size={12} color="var(--accent-primary)" />
          </div>

          {/* Stark Rectangular Verdict Stamp */}
          {currentPred ? (
            <div
              style={{
                padding: "1rem",
                background:
                  currentPred.predicted_label === "ATTACK"
                    ? "var(--status-attack-bg)"
                    : "var(--status-benign-bg)",
                border:
                  currentPred.predicted_label === "ATTACK"
                    ? "2px solid var(--status-attack)"
                    : "2px solid var(--status-benign)",
                borderRadius: "var(--radius-sm)",
              }}
            >
              <div
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: "10px",
                  letterSpacing: "0.1em",
                  color:
                    currentPred.predicted_label === "ATTACK"
                      ? "var(--status-attack)"
                      : "var(--status-benign)",
                  marginBottom: "4px",
                }}
              >
                DETECTION CLASSIFICATION
              </div>
              <div
                style={{
                  fontFamily: "var(--font-heading)",
                  fontSize: "1.35rem",
                  fontWeight: 800,
                  color:
                    currentPred.predicted_label === "ATTACK"
                      ? "var(--status-attack)"
                      : "var(--status-benign)",
                  lineHeight: 1.1,
                }}
              >
                VERDICT: {currentPred.predicted_label}
              </div>
              <div
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: "11px",
                  color: "var(--text-primary)",
                  marginTop: "6px",
                }}
              >
                CONFIDENCE: {(currentPred.confidence * 100).toFixed(2)}%
              </div>
            </div>
          ) : (
            <div
              style={{
                padding: "1.5rem 1rem",
                background: "var(--bg-surface-2)",
                border: "1px dashed var(--border-default)",
                textAlign: "center",
                fontFamily: "var(--font-mono)",
                fontSize: "11px",
                color: "var(--text-muted)",
              }}
            >
              READY // SELECT PRESET AND CLICK TRIGGER
            </div>
          )}

          {/* Live Probability Distribution Bar with Tau* Tick Marker */}
          <div>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                fontFamily: "var(--font-mono)",
                fontSize: "11px",
                marginBottom: "4px",
              }}
            >
              <span style={{ color: "var(--status-benign)" }}>
                P(BENIGN): {currentPred ? (currentPred.probabilities.BENIGN * 100).toFixed(2) : "0.00"}%
              </span>
              <span style={{ color: "var(--status-attack)" }}>
                P(ATTACK): {currentPred ? (currentPred.probabilities.ATTACK * 100).toFixed(2) : "0.00"}%
              </span>
            </div>

            {/* Gauge with vertical marker */}
            <div
              style={{
                position: "relative",
                height: "14px",
                background: "var(--bg-surface-3)",
                border: "1px solid var(--border-default)",
                overflow: "hidden",
                display: "flex",
              }}
            >
              <div
                style={{
                  width: `${currentPred ? currentPred.probabilities.BENIGN * 100 : 50}%`,
                  background: "var(--status-benign)",
                  transition: "width 200ms ease",
                }}
              />
              <div
                style={{
                  width: `${currentPred ? currentPred.probabilities.ATTACK * 100 : 50}%`,
                  background: "var(--status-attack)",
                  transition: "width 200ms ease",
                }}
              />
              {/* Vertical tick marker for threshold */}
              <div
                style={{
                  position: "absolute",
                  left: `${threshold * 100}%`,
                  top: 0,
                  bottom: 0,
                  width: "2px",
                  background: "var(--text-primary)",
                  boxShadow: "0 0 4px #000000",
                }}
                title={`Threshold tau* = ${threshold.toFixed(4)}`}
              />
            </div>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                fontFamily: "var(--font-mono)",
                fontSize: "9px",
                color: "var(--text-muted)",
                marginTop: "2px",
              }}
            >
              <span>0.00 (BENIGN)</span>
              <span style={{ color: "var(--accent-primary)" }}>&tau;* = {threshold.toFixed(4)}</span>
              <span>1.00 (ATTACK)</span>
            </div>
          </div>

          {/* Diagnostic Logit & Latency Strip */}
          <div
            style={{
              padding: "0.75rem",
              background: "var(--bg-code)",
              border: "1px solid var(--border-default)",
              display: "flex",
              flexDirection: "column",
              gap: "6px",
              fontFamily: "var(--font-mono)",
              fontSize: "11px",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-muted)" }}>RAW LOGIT (z)</span>
              <span style={{ color: "var(--text-primary)", fontWeight: 700 }}>
                {currentPred ? currentPred.raw_logit.toFixed(4) : "--"}
              </span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-muted)" }}>INFERENCE LATENCY</span>
              <span style={{ color: "var(--accent-primary)", fontWeight: 700 }}>
                {inferenceTimeMs} ms
              </span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-muted)" }}>DECISION BOUNDARY</span>
              <span style={{ color: "var(--text-secondary)" }}>
                {threshold.toFixed(4)}
              </span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-muted)" }}>ATTACK MARGIN</span>
              <span
                style={{
                  color:
                    currentPred && currentPred.probabilities.ATTACK >= threshold
                      ? "var(--status-attack)"
                      : "var(--status-benign)",
                  fontWeight: 700,
                }}
              >
                {currentPred
                  ? (currentPred.probabilities.ATTACK - threshold >= 0 ? "+" : "") +
                    (currentPred.probabilities.ATTACK - threshold).toFixed(4)
                  : "--"}
              </span>
            </div>
          </div>

          {/* Log Inspector */}
          <div
            style={{
              flex: 1,
              background: "var(--bg-code)",
              border: "1px solid var(--border-default)",
              padding: "0.75rem",
              fontFamily: "var(--font-mono)",
              fontSize: "10px",
              color: "var(--text-muted)",
              overflowY: "auto",
              display: "flex",
              flexDirection: "column",
              gap: "4px",
            }}
          >
            <div style={{ color: "var(--accent-primary)", display: "flex", alignItems: "center", gap: "4px" }}>
              <Terminal size={10} />
              <span>LOG REGISTER // QUANTUM TELEMETRY</span>
            </div>
            <div>[00.000s] SYSTEM INIT: PennyLane default.qubit simulator active.</div>
            <div>[00.002s] LOADED ANSATZ: 4-Qubit StronglyEntanglingLayers (L=1).</div>
            <div>[00.005s] SCALER: StandardScaler fitted on 45,019,243 train shards.</div>
            {currentPred && (
              <>
                <div style={{ color: "var(--text-primary)" }}>
                  [{inferenceTimeMs}ms] INFERENCE SUCCESS: z = {currentPred.raw_logit.toFixed(4)}, P = {currentPred.probabilities.ATTACK.toFixed(4)}
                </div>
                <div style={{ color: currentPred.predicted_label === "ATTACK" ? "var(--status-attack)" : "var(--status-benign)" }}>
                  [{inferenceTimeMs}ms] VERDICT ASSIGNED: {currentPred.predicted_label} (confidence {(currentPred.confidence * 100).toFixed(1)}%)
                </div>
              </>
            )}
          </div>

          {error && (
            <div
              style={{
                padding: "6px 8px",
                background: "var(--status-attack-bg)",
                border: "1px solid var(--status-attack-border)",
                color: "var(--status-attack)",
                fontSize: "11px",
                fontFamily: "var(--font-mono)",
              }}
            >
              ERROR: {error}
            </div>
          )}
        </section>
      </div>
    </div>
  );
};
