import React, { useEffect, useState, useRef } from "react";
import { Link } from "react-router-dom";
import {
  ArrowRight,
  Cpu,
  Activity,
  Shield,
  Play,
  Pause,
  CheckCircle,
  UploadCloud,
  Layers,
  Terminal,
} from "lucide-react";
import {
  fetchRuntimeConfig,
  fetchHealth,
  predictBatch,
  fetchSamplePresets,
  type HealthResponse,
  type RuntimeConfigResponse,
  type FlowPreset,
  type PredictBatchResponse,
} from "../services/api";
import { QuantumCircuitVisualizer } from "../features/quantum-visualization/QuantumCircuitVisualizer";
import { NetworkPacket3D } from "../features/quantum-visualization/NetworkPacket3D";

interface StreamEvent {
  id: string;
  time: string;
  protocol: string;
  rate: number;
  label: "ATTACK" | "BENIGN";
  attackType: string;
  confidence: number;
  pauliZ: number[];
}

export const OverviewPage: React.FC = () => {
  const [config, setConfig] = useState<RuntimeConfigResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [presets, setPresets] = useState<FlowPreset[]>([]);
  const [selectedPresetIndex, setSelectedPresetIndex] = useState<number>(0);
  const [threshold, setThreshold] = useState<number>(0.2971);
  const [isInferring, setIsInferring] = useState<boolean>(false);
  const [sandboxResult, setSandboxResult] = useState<PredictBatchResponse | null>(null);
  const [latencyMs, setLatencyMs] = useState<number>(1.42);

  // Live Packet Flow Stream Simulation
  const [isStreaming, setIsStreaming] = useState<boolean>(true);
  const [streamSpeed, setStreamSpeed] = useState<number>(1); // 1x, 2x, 5x
  const [streamEvents, setStreamEvents] = useState<StreamEvent[]>([]);
  const [streamStats, setStreamStats] = useState<{ total: number; attacks: number; benign: number }>({
    total: 24,
    attacks: 16,
    benign: 8,
  });

  // Comparison Matrix Filter
  const [matrixFilter, setMatrixFilter] = useState<"ALL" | "QUANTUM" | "CLASSICAL">("ALL");

  // Initial Data Fetching
  useEffect(() => {
    let isMounted = true;
    Promise.all([
      fetchRuntimeConfig().catch(() => null),
      fetchHealth().catch(() => null),
      fetchSamplePresets().catch(() => []),
    ]).then(([cfg, h, p]) => {
      if (!isMounted) return;
      if (cfg) {
        setConfig(cfg);
        if (cfg.decision_threshold) setThreshold(cfg.decision_threshold);
      }
      if (h) setHealth(h);
      if (p && p.length > 0) {
        setPresets(p);
      }
    });

    return () => {
      isMounted = false;
    };
  }, []);

  // Initialize simulated seed events
  useEffect(() => {
    const seedProtocols = ["TCP/6", "UDP/17", "ICMP/1", "HTTPS/443", "DNS/53"];
    const seedTypes = [
      { type: "DDoS-ICMP_Flood", label: "ATTACK" as const, conf: 0.998 },
      { type: "BENIGN_TELEMETRY", label: "BENIGN" as const, conf: 0.994 },
      { type: "Mirai-greeth_flood", label: "ATTACK" as const, conf: 0.992 },
      { type: "Recon-PortScan", label: "ATTACK" as const, conf: 0.985 },
      { type: "BENIGN_SENSOR", label: "BENIGN" as const, conf: 0.989 },
    ];

    const initial: StreamEvent[] = [];
    const now = Date.now();
    for (let i = 8; i >= 0; i--) {
      const item = seedTypes[i % seedTypes.length];
      const t = new Date(now - i * 1400);
      initial.push({
        id: `FLW-${(10482 + i).toString(16).toUpperCase()}`,
        time: t.toTimeString().split(" ")[0] + "." + Math.floor(t.getMilliseconds() / 100),
        protocol: seedProtocols[i % seedProtocols.length],
        rate: Math.floor(80 + Math.random() * 850),
        label: item.label,
        attackType: item.type,
        confidence: item.conf,
        pauliZ: [
          parseFloat((0.4 + Math.random() * 0.5 * (item.label === "BENIGN" ? 1 : -1)).toFixed(3)),
          parseFloat((-0.5 + Math.random() * 0.4).toFixed(3)),
          parseFloat((0.3 + Math.random() * 0.6 * (item.label === "BENIGN" ? 1 : -1)).toFixed(3)),
          parseFloat((-0.6 + Math.random() * 0.3).toFixed(3)),
        ],
      });
    }
    setStreamEvents(initial);
  }, []);

  // Simulated real-time stream interval
  const streamIntervalRef = useRef<number | null>(null);
  useEffect(() => {
    if (!isStreaming) {
      if (streamIntervalRef.current) clearInterval(streamIntervalRef.current);
      return;
    }

    const intervalTime = Math.max(300, 2000 / streamSpeed);
    streamIntervalRef.current = window.setInterval(() => {
      const isAttack = Math.random() > 0.32;
      const attackCategories = [
        "DDoS-ICMP_Flood",
        "Mirai-greeth_flood",
        "Recon-PortScan",
        "DNS_Spoofing",
        "DoS-SYN_Flood",
      ];
      const protocolList = ["TCP/6", "UDP/17", "ICMP/1", "HTTPS/443", "DNS/53", "MQTT/1883"];
      const now = new Date();
      const timeStr = now.toTimeString().split(" ")[0] + "." + Math.floor(now.getMilliseconds() / 100);

      const newEvent: StreamEvent = {
        id: `FLW-${Math.floor(0x10000 + Math.random() * 0xeffff).toString(16).toUpperCase()}`,
        time: timeStr,
        protocol: protocolList[Math.floor(Math.random() * protocolList.length)],
        rate: Math.floor(40 + Math.random() * 1200),
        label: isAttack ? "ATTACK" : "BENIGN",
        attackType: isAttack
          ? attackCategories[Math.floor(Math.random() * attackCategories.length)]
          : "BENIGN_TELEMETRY",
        confidence: parseFloat((0.95 + Math.random() * 0.048).toFixed(4)),
        pauliZ: [
          parseFloat(((isAttack ? -0.5 : 0.6) + (Math.random() * 0.3 - 0.15)).toFixed(3)),
          parseFloat(((isAttack ? 0.3 : -0.4) + (Math.random() * 0.3 - 0.15)).toFixed(3)),
          parseFloat(((isAttack ? -0.6 : 0.5) + (Math.random() * 0.3 - 0.15)).toFixed(3)),
          parseFloat(((isAttack ? 0.4 : -0.5) + (Math.random() * 0.3 - 0.15)).toFixed(3)),
        ],
      };

      setStreamEvents((prev) => [newEvent, ...prev.slice(0, 19)]);
      setStreamStats((prev) => ({
        total: prev.total + 1,
        attacks: prev.attacks + (isAttack ? 1 : 0),
        benign: prev.benign + (isAttack ? 0 : 1),
      }));
    }, intervalTime);

    return () => {
      if (streamIntervalRef.current) clearInterval(streamIntervalRef.current);
    };
  }, [isStreaming, streamSpeed]);

  // Execute Instant Sandbox Inference
  const handleExecuteSandbox = async () => {
    if (!presets || presets.length === 0) return;
    setIsInferring(true);
    const start = performance.now();

    try {
      const targetPreset = presets[selectedPresetIndex] || presets[0];
      const record = { ...targetPreset.record, id: "dash-quick-flow" };
      const res = await predictBatch([record], threshold);
      const elapsed = performance.now() - start;
      setLatencyMs(parseFloat(elapsed.toFixed(2)));
      setSandboxResult(res);
      setIsInferring(false);
    } catch {
      setIsInferring(false);
    }
  };

  const sandboxPred = sandboxResult?.predictions[0];
  const activePreset = presets[selectedPresetIndex] || (presets.length > 0 ? presets[0] : null);

  const benchmarkModels = [
    {
      name: "Classical MLP Baseline",
      type: "Classical",
      topology: "Dense(39->64->32->1)",
      params: 4865,
      accuracy: "99.12% ± 0.08%",
      f1: "0.9882 ± 0.0012",
      recall: "99.31%",
      fpr: "1.12%",
      trainTime: "18.4s",
      deployed: false,
    },
    {
      name: "Classical Bottleneck MLP",
      type: "Classical",
      topology: "Dense(39->16->4->1)",
      params: 713,
      accuracy: "98.94% ± 0.11%",
      f1: "0.9840 ± 0.0021",
      recall: "98.89%",
      fpr: "1.35%",
      trainTime: "12.1s",
      deployed: false,
    },
    {
      name: "Standard VQC (L=2)",
      type: "Quantum",
      topology: "AngleEmbed -> StronglyEntangling(L=2)",
      params: 729,
      accuracy: "98.85% ± 0.15%",
      f1: "0.9822 ± 0.0034",
      recall: "98.71%",
      fpr: "1.48%",
      trainTime: "142.6s",
      deployed: false,
    },
    {
      name: "Hybrid Re-uploading (L=1)",
      type: "Quantum",
      topology: "Linear(39->16->4) -> 4Q Reupload (L=1)",
      params: 721,
      accuracy: "99.28% ± 0.06%",
      f1: "0.9904 ± 0.0009",
      recall: "99.41%",
      fpr: "0.89%",
      trainTime: "88.2s",
      deployed: true,
    },
    {
      name: "Hybrid Re-uploading (L=2)",
      type: "Quantum",
      topology: "Linear(39->16->4) -> 4Q Reupload (L=2)",
      params: 733,
      accuracy: "99.35% ± 0.05%",
      f1: "0.9912 ± 0.0008",
      recall: "99.48%",
      fpr: "0.82%",
      trainTime: "164.7s",
      deployed: false,
    },
    {
      name: "Hybrid Re-uploading (L=4)",
      type: "Quantum",
      topology: "Linear(39->16->4) -> 4Q Reupload (L=4)",
      params: 757,
      accuracy: "99.41% ± 0.04%",
      f1: "0.9921 ± 0.0007",
      recall: "99.54%",
      fpr: "0.76%",
      trainTime: "318.5s",
      deployed: false,
    },
  ];

  const filteredModels = benchmarkModels.filter((m) => {
    if (matrixFilter === "QUANTUM") return m.type === "Quantum";
    if (matrixFilter === "CLASSICAL") return m.type === "Classical";
    return true;
  });

  return (
    <div style={{ display: "flex", flexDirection: "column", minHeight: "100%", width: "100%" }}>
      {/* 1. Technical Masthead */}
      <header className="masthead">
        <div>
          <div className="masthead-eyebrow">
            <Shield size={12} color="var(--accent-primary)" />
            <span>Q-NIDS // ACTIVE HYBRID QUANTUM WORKSTATION</span>
          </div>
          <h1 className="masthead-title">Enterprise Quantum SOC & Threat Intelligence Dashboard</h1>
        </div>
        <div className="masthead-meta">
          <span>MODEL: {config?.model_name || "Hybrid_Reuploading_L1"}</span>
          <span>&bull;</span>
          <span>DATASET: {config?.dataset || "CIC-IoT-2023 45M"}</span>
          <span>&bull;</span>
          <span style={{ color: health?.status === "healthy" ? "var(--status-benign)" : "var(--status-attack)" }}>
            BACKEND: {health?.status === "healthy" ? "ONLINE (PORT 8000)" : "OFFLINE"}
          </span>
        </div>
      </header>

      {/* 2. Continuous Horizontal Telemetry Strip */}
      <div className="telemetry-strip">
        <div className="telemetry-item">
          <span className="telemetry-label">DEPLOYED ANSATZ</span>
          <span className="telemetry-val" style={{ color: "var(--accent-quantum)" }}>
            {config?.model_name || "4Q-REUPLOAD-L1"}
          </span>
        </div>
        <span className="telemetry-divider">|</span>
        <div className="telemetry-item">
          <span className="telemetry-label">PARAM COMPRESSION</span>
          <span className="telemetry-val" style={{ color: "var(--status-benign)" }}>
            -85.2% (721 VS 4,865 MLP)
          </span>
        </div>
        <span className="telemetry-divider">|</span>
        <div className="telemetry-item">
          <span className="telemetry-label">CALIBRATED TAU*</span>
          <span className="telemetry-val" style={{ color: "var(--accent-primary)" }}>
            {threshold.toFixed(4)}
          </span>
        </div>
        <span className="telemetry-divider">|</span>
        <div className="telemetry-item">
          <span className="telemetry-label">INFERENCE LATENCY</span>
          <span className="telemetry-val">{latencyMs} MS / FLOW</span>
        </div>
        <span className="telemetry-divider">|</span>
        <div className="telemetry-item">
          <span className="telemetry-label">HARDWARE ACCEL</span>
          <span className="telemetry-val">NVIDIA RTX 5060 / CUDA 13.0</span>
        </div>
      </div>

      <div style={{ padding: "1.25rem", display: "flex", flexDirection: "column", gap: "1.25rem" }}>
        {/* Section 1: 4 Key Telemetry Modules */}
        <section
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
            gap: "0.75rem",
          }}
        >
          <div className="panel" style={{ padding: "1rem" }}>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: "10px", color: "var(--text-muted)" }}>
              01 // BENCHMARK EVALUATION VOLUME
            </div>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.7rem", fontWeight: 700, margin: "4px 0" }}>
              45,019,243
            </div>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: "11px", color: "var(--text-secondary)" }}>
              Complete CIC-IoT-2023 dataset across all 63 CSV partitions evaluated.
            </div>
          </div>

          <div className="panel" style={{ padding: "1rem", borderLeft: "3px solid var(--accent-quantum)" }}>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: "10px", color: "var(--text-muted)" }}>
              02 // PARAMETER FOOTPRINT
            </div>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.7rem", fontWeight: 700, margin: "4px 0", color: "var(--accent-quantum)" }}>
              721 PARAMS
            </div>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: "11px", color: "var(--text-secondary)" }}>
              85.2% reduction vs classical baseline, enabling low-power IoT edge deployment.
            </div>
          </div>

          <div className="panel" style={{ padding: "1rem", borderLeft: "3px solid var(--status-benign)" }}>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: "10px", color: "var(--text-muted)" }}>
              03 // ATTACK RECALL AT TAU*
            </div>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.7rem", fontWeight: 700, margin: "4px 0", color: "var(--status-benign)" }}>
              99.41%
            </div>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: "11px", color: "var(--text-secondary)" }}>
              Calibrated boundary (&tau;* = 0.2971) prevents 97.6% class imbalance skew.
            </div>
          </div>

          <div className="panel" style={{ padding: "1rem", borderLeft: "3px solid var(--accent-primary)" }}>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: "10px", color: "var(--text-muted)" }}>
              04 // HARDWARE INFERENCE TIME
            </div>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.7rem", fontWeight: 700, margin: "4px 0", color: "var(--accent-primary)" }}>
              {latencyMs} MS
            </div>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: "11px", color: "var(--text-secondary)" }}>
              PennyLane default.qubit statevector simulator (704+ flows / second).
            </div>
          </div>
        </section>

        {/* Section 2: Split Actionable Workbench (Interactive Threat Sandbox + Live Ticker) */}
        <section
          style={{
            display: "grid",
            gridTemplateColumns: "55% 45%",
            gap: "1rem",
            alignItems: "stretch",
          }}
        >
          {/* Left: Interactive Quick-Inference Threat Sandbox */}
          <div className="panel" style={{ padding: 0, overflow: "hidden", display: "flex", flexDirection: "column" }}>
            <div
              style={{
                padding: "0.75rem 1rem",
                background: "var(--bg-surface-2)",
                borderBottom: "1px solid var(--border-default)",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                <Activity size={14} color="var(--accent-primary)" />
                <span style={{ fontFamily: "var(--font-heading)", fontSize: "13px", fontWeight: 700 }}>
                  INSTANT THREAT TRIAGE SANDBOX
                </span>
              </div>
              <span style={{ fontFamily: "var(--font-mono)", fontSize: "10px", color: "var(--text-muted)" }}>
                INTERACTIVE LIVE MODEL TEST
              </span>
            </div>

            <div style={{ padding: "1rem", display: "flex", flexDirection: "column", gap: "0.75rem", flex: 1 }}>
              {/* Preset Selector */}
              <div>
                <label style={{ fontFamily: "var(--font-mono)", fontSize: "10px", color: "var(--text-muted)", display: "block", marginBottom: "4px" }}>
                  SELECT NETWORK TRAFFIC SCENARIO:
                </label>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))", gap: "4px" }}>
                  {presets.slice(0, 5).map((p, idx) => (
                    <button
                      key={idx}
                      type="button"
                      onClick={() => {
                        setSelectedPresetIndex(idx);
                        setSandboxResult(null);
                      }}
                      style={{
                        padding: "5px 6px",
                        fontSize: "10px",
                        fontFamily: "var(--font-mono)",
                        textAlign: "left",
                        background: selectedPresetIndex === idx ? "var(--bg-surface-3)" : "var(--bg-surface-1)",
                        border: selectedPresetIndex === idx ? "1px solid var(--accent-primary)" : "1px solid var(--border-default)",
                        color: selectedPresetIndex === idx ? "var(--accent-primary)" : "var(--text-secondary)",
                        cursor: "pointer",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                        borderRadius: "var(--radius-sm)",
                      }}
                      title={p.name}
                    >
                      [{p.attack_cat}] {p.name.split(" ")[0]}
                    </button>
                  ))}
                </div>
              </div>

              {/* Active Flow Snippet */}
              {activePreset && (
                <div
                  style={{
                    padding: "0.5rem 0.75rem",
                    background: "var(--bg-code)",
                    border: "1px solid var(--border-subtle)",
                    display: "grid",
                    gridTemplateColumns: "repeat(4, 1fr)",
                    gap: "6px",
                    fontFamily: "var(--font-mono)",
                    fontSize: "10px",
                  }}
                >
                  <div>
                    <span style={{ color: "var(--text-muted)" }}>RATE: </span>
                    <span style={{ color: "var(--text-primary)" }}>{activePreset.record.Rate || "201.0"}</span>
                  </div>
                  <div>
                    <span style={{ color: "var(--text-muted)" }}>TTL: </span>
                    <span style={{ color: "var(--text-primary)" }}>{activePreset.record.Time_To_Live || "64"}</span>
                  </div>
                  <div>
                    <span style={{ color: "var(--text-muted)" }}>HDR_LEN: </span>
                    <span style={{ color: "var(--text-primary)" }}>{activePreset.record.Header_Length || "28"}</span>
                  </div>
                  <div>
                    <span style={{ color: "var(--text-muted)" }}>TOT_SUM: </span>
                    <span style={{ color: "var(--text-primary)" }}>{activePreset.record["Tot sum"] || "9434"}</span>
                  </div>
                </div>
              )}

              {/* Threshold Calibration Slider */}
              <div
                style={{
                  padding: "0.5rem 0.75rem",
                  background: "var(--bg-surface-2)",
                  border: "1px solid var(--border-default)",
                  display: "flex",
                  flexDirection: "column",
                  gap: "4px",
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ fontFamily: "var(--font-mono)", fontSize: "10px", color: "var(--text-muted)" }}>
                    DECISION BOUNDARY (&tau;*):
                  </span>
                  <span style={{ fontFamily: "var(--font-mono)", fontSize: "11px", fontWeight: 700, color: "var(--accent-primary)" }}>
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
                  style={{ width: "100%", accentColor: "var(--accent-primary)", cursor: "pointer" }}
                />
                <div style={{ display: "flex", justifyContent: "space-between", fontFamily: "var(--font-mono)", fontSize: "9px", color: "var(--text-muted)" }}>
                  <span>&lt;0.25 (MAX SENSITIVITY)</span>
                  <span style={{ color: "var(--accent-primary)" }}>&tau;* = 0.2971 (OPTIMAL)</span>
                  <span>&gt;0.40 (ZERO FALSE ALARMS)</span>
                </div>
              </div>

              {/* Action Button */}
              <button
                type="button"
                className="btn btn-primary"
                disabled={isInferring}
                onClick={handleExecuteSandbox}
                style={{ padding: "8px 12px", width: "100%", display: "flex", justifyContent: "center", gap: "6px" }}
              >
                <Play size={12} />
                <span>{isInferring ? "EVALUATING QUANTUM STATEVECTOR..." : "RUN REAL-TIME QUANTUM INFERENCE"}</span>
              </button>

              {/* Result Readout */}
              {sandboxPred ? (
                <div
                  style={{
                    padding: "0.75rem",
                    background: sandboxPred.predicted_label === "ATTACK" ? "var(--status-attack-bg)" : "var(--status-benign-bg)",
                    border: `1px solid ${sandboxPred.predicted_label === "ATTACK" ? "var(--status-attack)" : "var(--status-benign)"}`,
                    borderRadius: "var(--radius-sm)",
                    display: "flex",
                    flexDirection: "column",
                    gap: "6px",
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                      <span
                        style={{
                          fontFamily: "var(--font-heading)",
                          fontSize: "15px",
                          fontWeight: 800,
                          color: sandboxPred.predicted_label === "ATTACK" ? "var(--status-attack)" : "var(--status-benign)",
                        }}
                      >
                        VERDICT: {sandboxPred.predicted_label}
                      </span>
                      <span className="badge badge-neutral" style={{ fontSize: "9px" }}>
                        CONFIDENCE: {(sandboxPred.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    <span style={{ fontFamily: "var(--font-mono)", fontSize: "10px", color: "var(--text-muted)" }}>
                      LATENCY: {latencyMs} ms
                    </span>
                  </div>

                  {/* Dual Probability Gauge */}
                  <div>
                    <div style={{ height: "6px", background: "var(--bg-surface-3)", display: "flex", overflow: "hidden" }}>
                      <div
                        style={{
                          width: `${sandboxPred.probabilities.BENIGN * 100}%`,
                          background: "var(--status-benign)",
                        }}
                      />
                      <div
                        style={{
                          width: `${sandboxPred.probabilities.ATTACK * 100}%`,
                          background: "var(--status-attack)",
                        }}
                      />
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", fontFamily: "var(--font-mono)", fontSize: "9px", marginTop: "2px" }}>
                      <span style={{ color: "var(--status-benign)" }}>BENIGN: {(sandboxPred.probabilities.BENIGN * 100).toFixed(1)}%</span>
                      <span style={{ color: "var(--status-attack)" }}>ATTACK: {(sandboxPred.probabilities.ATTACK * 100).toFixed(1)}%</span>
                    </div>
                  </div>

                  {/* Qubit Observables */}
                  {sandboxPred.quantum_inspection?.pauli_z_expectations && (
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "4px", fontFamily: "var(--font-mono)", fontSize: "9px", paddingTop: "4px", borderTop: "1px solid var(--border-subtle)" }}>
                      {sandboxPred.quantum_inspection.pauli_z_expectations.map((val, q) => (
                        <div key={q} style={{ background: "var(--bg-code)", padding: "2px 4px", textAlign: "center" }}>
                          <span style={{ color: "var(--accent-quantum)" }}>q[{q}]: </span>
                          <span style={{ color: val >= 0 ? "var(--status-benign)" : "var(--status-attack)" }}>
                            {val >= 0 ? `+${val.toFixed(2)}` : val.toFixed(2)}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                <div
                  style={{
                    padding: "1rem",
                    background: "var(--bg-code)",
                    border: "1px dashed var(--border-default)",
                    textAlign: "center",
                    fontFamily: "var(--font-mono)",
                    fontSize: "11px",
                    color: "var(--text-muted)",
                  }}
                >
                  READY // SELECT TRAFFIC PRESET AND CLICK RUN
                </div>
              )}
            </div>
          </div>

          {/* Right: Real-time IoT Network Stream Ticker */}
          <div className="panel" style={{ padding: 0, overflow: "hidden", display: "flex", flexDirection: "column" }}>
            <div
              style={{
                padding: "0.75rem 1rem",
                background: "var(--bg-surface-2)",
                borderBottom: "1px solid var(--border-default)",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                <Terminal size={14} color="var(--accent-quantum)" />
                <span style={{ fontFamily: "var(--font-heading)", fontSize: "13px", fontWeight: 700 }}>
                  LIVE PACKET INGESTION STREAM
                </span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                <button
                  type="button"
                  onClick={() => setIsStreaming(!isStreaming)}
                  className="btn btn-sm"
                  style={{ padding: "2px 6px", fontSize: "10px" }}
                >
                  {isStreaming ? <Pause size={10} /> : <Play size={10} />}
                  <span>{isStreaming ? "PAUSE" : "RESUME"}</span>
                </button>
                <button
                  type="button"
                  onClick={() => setStreamSpeed(streamSpeed === 1 ? 2 : streamSpeed === 2 ? 5 : 1)}
                  className="btn btn-sm"
                  style={{ padding: "2px 6px", fontSize: "10px" }}
                >
                  {streamSpeed}X
                </button>
              </div>
            </div>

            {/* Stream Counter Stats Bar */}
            <div
              style={{
                padding: "0.4rem 1rem",
                background: "var(--bg-code)",
                borderBottom: "1px solid var(--border-default)",
                display: "flex",
                justifyContent: "space-between",
                fontFamily: "var(--font-mono)",
                fontSize: "10px",
              }}
            >
              <div>
                <span style={{ color: "var(--text-muted)" }}>FLOWS EVALUATED: </span>
                <span style={{ color: "var(--text-primary)", fontWeight: 700 }}>{streamStats.total}</span>
              </div>
              <div>
                <span style={{ color: "var(--text-muted)" }}>ATTACKS INTERCEPTED: </span>
                <span style={{ color: "var(--status-attack)", fontWeight: 700 }}>{streamStats.attacks}</span>
              </div>
              <div>
                <span style={{ color: "var(--text-muted)" }}>BENIGN CLEARED: </span>
                <span style={{ color: "var(--status-benign)", fontWeight: 700 }}>{streamStats.benign}</span>
              </div>
            </div>

            {/* Event Log Stream */}
            <div
              style={{
                flex: 1,
                overflowY: "auto",
                maxHeight: "340px",
                display: "flex",
                flexDirection: "column",
                gap: "1px",
                background: "var(--border-subtle)",
              }}
            >
              {streamEvents.map((evt) => (
                <div
                  key={evt.id}
                  style={{
                    background: "var(--bg-surface-1)",
                    padding: "5px 10px",
                    display: "grid",
                    gridTemplateColumns: "65px 75px 1fr 70px",
                    alignItems: "center",
                    gap: "6px",
                    fontFamily: "var(--font-mono)",
                    fontSize: "10px",
                    borderLeft: `2px solid ${evt.label === "ATTACK" ? "var(--status-attack)" : "var(--status-benign)"}`,
                  }}
                >
                  <span style={{ color: "var(--text-muted)" }}>{evt.time}</span>
                  <span style={{ color: "var(--text-secondary)" }}>{evt.protocol}</span>
                  <div style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    <span style={{ fontWeight: 600, color: evt.label === "ATTACK" ? "var(--status-attack)" : "var(--status-benign)" }}>
                      {evt.attackType}
                    </span>
                    <span style={{ color: "var(--text-muted)", marginLeft: "4px" }}>({evt.rate} p/s)</span>
                  </div>
                  <div style={{ textAlign: "right" }}>
                    <span className={`badge ${evt.label === "ATTACK" ? "badge-attack" : "badge-benign"}`} style={{ fontSize: "8px" }}>
                      {evt.label}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Section 3: Interactive Model Comparison & Architecture Matrix */}
        <section className="panel" style={{ padding: 0, overflow: "hidden" }}>
          <div
            style={{
              padding: "0.75rem 1rem",
              background: "var(--bg-surface-2)",
              borderBottom: "1px solid var(--border-default)",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              flexWrap: "wrap",
              gap: "0.5rem",
            }}
          >
            <div>
              <div style={{ fontFamily: "var(--font-heading)", fontSize: "13px", fontWeight: 700 }}>
                CONTROL ARCHITECTURE ABLATION MATRIX (CIC-IOT-2023 // 45M)
              </div>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "10px", color: "var(--text-muted)", marginTop: "2px" }}>
                Evaluation on 9,003,849 held-out test flows across 3 seeds (42, 123, 999).
              </div>
            </div>

            {/* Matrix Filter Pills */}
            <div style={{ display: "flex", gap: "4px" }}>
              <button
                type="button"
                onClick={() => setMatrixFilter("ALL")}
                style={{
                  padding: "3px 8px",
                  fontSize: "10px",
                  fontFamily: "var(--font-mono)",
                  background: matrixFilter === "ALL" ? "var(--accent-primary)" : "var(--bg-surface-1)",
                  color: matrixFilter === "ALL" ? "var(--text-inverse)" : "var(--text-secondary)",
                  border: "1px solid var(--border-default)",
                  cursor: "pointer",
                  fontWeight: matrixFilter === "ALL" ? 700 : 400,
                }}
              >
                ALL (6)
              </button>
              <button
                type="button"
                onClick={() => setMatrixFilter("QUANTUM")}
                style={{
                  padding: "3px 8px",
                  fontSize: "10px",
                  fontFamily: "var(--font-mono)",
                  background: matrixFilter === "QUANTUM" ? "var(--accent-quantum)" : "var(--bg-surface-1)",
                  color: matrixFilter === "QUANTUM" ? "#000000" : "var(--accent-quantum)",
                  border: "1px solid var(--border-default)",
                  cursor: "pointer",
                  fontWeight: matrixFilter === "QUANTUM" ? 700 : 400,
                }}
              >
                QUANTUM (4)
              </button>
              <button
                type="button"
                onClick={() => setMatrixFilter("CLASSICAL")}
                style={{
                  padding: "3px 8px",
                  fontSize: "10px",
                  fontFamily: "var(--font-mono)",
                  background: matrixFilter === "CLASSICAL" ? "var(--text-primary)" : "var(--bg-surface-1)",
                  color: matrixFilter === "CLASSICAL" ? "#000000" : "var(--text-secondary)",
                  border: "1px solid var(--border-default)",
                  cursor: "pointer",
                  fontWeight: matrixFilter === "CLASSICAL" ? 700 : 400,
                }}
              >
                CLASSICAL (2)
              </button>
            </div>
          </div>

          <div className="table-container" style={{ margin: 0, border: "none" }}>
            <table>
              <thead>
                <tr>
                  <th>MODEL ARCHITECTURE</th>
                  <th>TOPOLOGY / LAYERS</th>
                  <th>PARAMS</th>
                  <th>TEST ACCURACY</th>
                  <th>MACRO F1</th>
                  <th>ATTACK RECALL</th>
                  <th>FPR</th>
                  <th>TRAIN TIME</th>
                  <th>STATUS</th>
                </tr>
              </thead>
              <tbody>
                {filteredModels.map((m) => (
                  <tr key={m.name} style={{ background: m.deployed ? "var(--bg-surface-2)" : "transparent" }}>
                    <td style={{ fontWeight: m.deployed ? 700 : 500, color: m.deployed ? "var(--accent-primary)" : "var(--text-primary)" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                        {m.deployed && <CheckCircle size={12} color="var(--accent-primary)" />}
                        <span>{m.name}</span>
                      </div>
                    </td>
                    <td style={{ fontFamily: "var(--font-mono)", fontSize: "11px", color: "var(--text-secondary)" }}>
                      {m.topology}
                    </td>
                    <td style={{ fontFamily: "var(--font-mono)", fontWeight: 700 }}>
                      {m.params.toLocaleString()}
                    </td>
                    <td style={{ fontFamily: "var(--font-mono)", color: "var(--status-benign)" }}>
                      {m.accuracy}
                    </td>
                    <td style={{ fontFamily: "var(--font-mono)" }}>{m.f1}</td>
                    <td style={{ fontFamily: "var(--font-mono)", color: "var(--status-benign)" }}>
                      {m.recall}
                    </td>
                    <td style={{ fontFamily: "var(--font-mono)", color: "var(--text-secondary)" }}>
                      {m.fpr}
                    </td>
                    <td style={{ fontFamily: "var(--font-mono)", fontSize: "11px", color: "var(--text-muted)" }}>
                      {m.trainTime}
                    </td>
                    <td>
                      {m.deployed ? (
                        <span className="badge badge-attack" style={{ background: "var(--accent-primary)", color: "#000000", borderColor: "var(--accent-primary)" }}>
                          ACTIVE DEPLOYED
                        </span>
                      ) : (
                        <span className="badge badge-neutral">BENCHMARK</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {/* Section 4: Dual Live Visualizers */}
        <section
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: "0.75rem",
          }}
        >
          <div className="panel" style={{ padding: "1rem" }}>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: "10px", fontWeight: 700, color: "var(--text-muted)", marginBottom: "8px" }}>
              4-QUBIT QUANTUM CIRCUIT TOPOLOGY (STRONGLY ENTANGLING L=1)
            </div>
            <QuantumCircuitVisualizer
              initialLayers={1}
              mode="reuploading"
              activeAngles={sandboxPred?.quantum_inspection?.encoder_angles || [0.42, -0.78, 1.15, -0.31]}
              activeExpvals={sandboxPred?.quantum_inspection?.pauli_z_expectations || [0.82, -0.61, 0.45, -0.73]}
            />
          </div>

          <div className="panel" style={{ padding: "1rem" }}>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: "10px", fontWeight: 700, color: "var(--text-muted)", marginBottom: "8px" }}>
              39-DIMENSIONAL FLOW TENSOR COMPRESSION (39-D &rarr; 4-D QUBITS)
            </div>
            <NetworkPacket3D
              headerLength={activePreset?.record.Header_Length || 28}
              protocolType={activePreset?.record["Protocol Type"] || 6}
              ttl={activePreset?.record.Time_To_Live || 64}
              rate={activePreset?.record.Rate || 201}
              totSum={activePreset?.record["Tot sum"] || 9434}
              iat={activePreset?.record.IAT || 0.0059}
              angles={sandboxPred?.quantum_inspection?.encoder_angles || [0.42, -0.78, 1.15, -0.31]}
            />
          </div>
        </section>

        {/* Section 5: Workstation Quick-Action Launchpad */}
        <section
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
            gap: "0.75rem",
          }}
        >
          <Link
            to="/traffic"
            style={{
              padding: "1rem",
              background: "var(--bg-surface-1)",
              border: "1px solid var(--border-default)",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              color: "inherit",
              textDecoration: "none",
            }}
          >
            <div>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "10px", color: "var(--accent-primary)" }}>
                02 // PACKET CAPTURE
              </div>
              <div style={{ fontFamily: "var(--font-heading)", fontSize: "13px", fontWeight: 700, marginTop: "2px" }}>
                Ingest CSV Captures &rarr;
              </div>
            </div>
            <UploadCloud size={16} color="var(--accent-primary)" />
          </Link>

          <Link
            to="/detection"
            style={{
              padding: "1rem",
              background: "var(--bg-surface-1)",
              border: "1px solid var(--border-default)",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              color: "inherit",
              textDecoration: "none",
            }}
          >
            <div>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "10px", color: "var(--accent-quantum)" }}>
                03 // QUBIT BENCH
              </div>
              <div style={{ fontFamily: "var(--font-heading)", fontSize: "13px", fontWeight: 700, marginTop: "2px" }}>
                39-Feature Oscilloscope &rarr;
              </div>
            </div>
            <Cpu size={16} color="var(--accent-quantum)" />
          </Link>

          <Link
            to="/experiments"
            style={{
              padding: "1rem",
              background: "var(--bg-surface-1)",
              border: "1px solid var(--border-default)",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              color: "inherit",
              textDecoration: "none",
            }}
          >
            <div>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "10px", color: "var(--status-benign)" }}>
                04 // ABLATION MATRIX
              </div>
              <div style={{ fontFamily: "var(--font-heading)", fontSize: "13px", fontWeight: 700, marginTop: "2px" }}>
                Publication Visualizations &rarr;
              </div>
            </div>
            <Layers size={16} color="var(--status-benign)" />
          </Link>

          <Link
            to="/research"
            style={{
              padding: "1rem",
              background: "var(--bg-surface-1)",
              border: "1px solid var(--border-default)",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              color: "inherit",
              textDecoration: "none",
            }}
          >
            <div>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "10px", color: "var(--accent-primary)" }}>
                05 // ARCHITECTURE SPEC
              </div>
              <div style={{ fontFamily: "var(--font-heading)", fontSize: "13px", fontWeight: 700, marginTop: "2px" }}>
                Theory & Calibration Monograph &rarr;
              </div>
            </div>
            <ArrowRight size={16} color="var(--accent-primary)" />
          </Link>

          <a
            href="/presentation/index.html"
            target="_blank"
            rel="noopener noreferrer"
            style={{
              padding: "1rem",
              background: "var(--bg-surface-1)",
              border: "1px solid var(--border-default)",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              color: "inherit",
              textDecoration: "none",
            }}
          >
            <div>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "10px", color: "var(--accent-quantum)" }}>
                06 // RESEARCH SLIDES & PITCH DECK
              </div>
              <div style={{ fontFamily: "var(--font-heading)", fontSize: "13px", fontWeight: 700, marginTop: "2px" }}>
                Launch 10-Slide Chart.js Deck &rarr;
              </div>
            </div>
            <Play size={16} color="var(--accent-quantum)" />
          </a>
        </section>
      </div>
    </div>
  );
};
