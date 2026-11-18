import React, { useState, useEffect } from "react";
import {
  FlaskConical,
  Play,
  RotateCcw,
  Info,
  AlertCircle,
  Sliders,
} from "lucide-react";
import { useTranslation } from "../i18n/useTranslation";
import {
  predictBatch,
  fetchSamplePresets,
  type FlowPreset,
  type SingleFlowRecord,
  type PredictBatchResponse,
} from "../services/api";
import { QuantumCircuitVisualizer } from "../features/quantum-visualization/QuantumCircuitVisualizer";
import { NetworkPacket3D } from "../features/quantum-visualization/NetworkPacket3D";

// Complete 39 Numerical Features grouped into logical network domains
const FEATURE_GROUPS = [
  {
    id: "basic",
    name: "Basic Flow Rates",
    features: [
      { key: "dur", label: "DUR (s)", step: "0.0001", min: 0 },
      { key: "rate", label: "RATE", step: "1", min: 0 },
      { key: "spkts", label: "SPKTS", step: "1", min: 0 },
      { key: "dpkts", label: "DPKTS", step: "1", min: 0 },
      { key: "sbytes", label: "SBYTES", step: "1", min: 0 },
      { key: "dbytes", label: "DBYTES", step: "1", min: 0 },
      { key: "smean", label: "SMEAN", step: "1", min: 0 },
      { key: "dmean", label: "DMEAN", step: "1", min: 0 },
    ],
  },
  {
    id: "load",
    name: "TTL & Load Metrics",
    features: [
      { key: "sttl", label: "STTL", step: "1", min: 0 },
      { key: "dttl", label: "DTTL", step: "1", min: 0 },
      { key: "sload", label: "SLOAD", step: "100", min: 0 },
      { key: "dload", label: "DLOAD", step: "100", min: 0 },
      { key: "sloss", label: "SLOSS", step: "1", min: 0 },
      { key: "dloss", label: "DLOSS", step: "1", min: 0 },
    ],
  },
  {
    id: "jitter",
    name: "Timing & Jitter",
    features: [
      { key: "sinpkt", label: "SINPKT (ms)", step: "0.001", min: 0 },
      { key: "dinpkt", label: "DINPKT (ms)", step: "0.001", min: 0 },
      { key: "sjit", label: "SJIT (ms)", step: "0.001", min: 0 },
      { key: "djit", label: "DJIT (ms)", step: "0.001", min: 0 },
    ],
  },
  {
    id: "tcp",
    name: "TCP Sequence & RTT",
    features: [
      { key: "swin", label: "SWIN", step: "1", min: 0 },
      { key: "dwin", label: "DWIN", step: "1", min: 0 },
      { key: "stcpb", label: "STCPB", step: "1", min: 0 },
      { key: "dtcpb", label: "DTCPB", step: "1", min: 0 },
      { key: "tcprtt", label: "TCPRTT", step: "0.001", min: 0 },
      { key: "synack", label: "SYNACK", step: "0.001", min: 0 },
      { key: "ackdat", label: "ACKDAT", step: "0.001", min: 0 },
    ],
  },
  {
    id: "counts",
    name: "Connection Counts & Rates",
    features: [
      { key: "ct_srv_src", label: "CT_SRV_SRC", step: "1", min: 0 },
      { key: "ct_state_ttl", label: "CT_STATE_TTL", step: "1", min: 0 },
      { key: "ct_dst_ltm", label: "CT_DST_LTM", step: "1", min: 0 },
      { key: "ct_src_dport_ltm", label: "CT_SRC_DPORT", step: "1", min: 0 },
      { key: "ct_dst_sport_ltm", label: "CT_DST_SPORT", step: "1", min: 0 },
      { key: "ct_dst_src_ltm", label: "CT_DST_SRC", step: "1", min: 0 },
      { key: "ct_src_ltm", label: "CT_SRC_LTM", step: "1", min: 0 },
      { key: "ct_srv_dst", label: "CT_SRV_DST", step: "1", min: 0 },
      { key: "trans_depth", label: "TRANS_DEPTH", step: "1", min: 0 },
      { key: "response_body_len", label: "RESP_LEN", step: "1", min: 0 },
      { key: "ct_ftp_cmd", label: "CT_FTP_CMD", step: "1", min: 0 },
      { key: "ct_flw_http_mthd", label: "CT_HTTP_MTHD", step: "1", min: 0 },
    ],
  },
  {
    id: "flags",
    name: "Service Flags",
    features: [
      { key: "is_ftp_login", label: "IS_FTP_LOGIN", step: "1", min: 0, max: 1 },
      { key: "is_sm_ips_ports", label: "IS_SM_IPS_PORTS", step: "1", min: 0, max: 1 },
    ],
  },
];

export const DetectionLabPage: React.FC = () => {
  const { t } = useTranslation();

  const [presets, setPresets] = useState<FlowPreset[]>([]);
  const [selectedPresetIndex, setSelectedPresetIndex] = useState<string>("");
  const [activeGroup, setActiveGroup] = useState<string>("basic");

  // Default flow data initialized to standard benchmark record
  const [formData, setFormData] = useState<Partial<SingleFlowRecord>>({
    dur: 0.000011,
    proto: "udp",
    service: "-",
    state: "INT",
    spkts: 2,
    dpkts: 0,
    sbytes: 496,
    dbytes: 0,
    rate: 90909.09,
    sttl: 254,
    dttl: 0,
    sload: 180363632,
    dload: 0,
    sloss: 0,
    dloss: 0,
    sinpkt: 0.011,
    dinpkt: 0,
    sjit: 0,
    djit: 0,
    swin: 0,
    stcpb: 0,
    dtcpb: 0,
    dwin: 0,
    tcprtt: 0,
    synack: 0,
    ackdat: 0,
    smean: 248,
    dmean: 0,
    trans_depth: 0,
    response_body_len: 0,
    ct_srv_src: 2,
    ct_state_ttl: 2,
    ct_dst_ltm: 1,
    ct_src_dport_ltm: 1,
    ct_dst_sport_ltm: 1,
    ct_dst_src_ltm: 2,
    is_ftp_login: 0,
    ct_ftp_cmd: 0,
    ct_flw_http_mthd: 0,
    ct_src_ltm: 1,
    ct_srv_dst: 2,
    is_sm_ips_ports: 0,
  });

  const [threshold, setThreshold] = useState<number>(0.5);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [result, setResult] = useState<PredictBatchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

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

    try {
      const record = { ...(formData as SingleFlowRecord), id: "lab-flow-01" };
      const res = await predictBatch([record], threshold);
      setResult(res);
      setIsLoading(false);
    } catch (err: any) {
      setError(err.message || "Prediction execution failed.");
      setIsLoading(false);
    }
  };

  const currentPred = result?.predictions[0];
  const encoderAngles = currentPred?.quantum_inspection?.encoder_angles;
  const pauliExpvals = currentPred?.quantum_inspection?.pauli_z_expectations;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-6)" }}>
      {/* Header */}
      <div>
        <h1 style={{ fontSize: "var(--text-xl)", fontWeight: 700, margin: 0 }}>
          {t.labTitle}
        </h1>
        <p style={{ fontSize: "var(--text-sm)", color: "var(--text-secondary)", margin: "4px 0 0" }}>
          {t.labSubtitle}
        </p>
      </div>

      {/* Preset Selector Banner */}
      <section
        className="panel"
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          flexWrap: "wrap",
          gap: "var(--space-3)",
          background: "var(--bg-surface-2)",
          padding: "var(--space-3)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <FlaskConical size={16} color="var(--accent-primary)" />
          <span style={{ fontWeight: 600, fontSize: "var(--text-sm)" }}>
            {t.presetSelectorLabel}:
          </span>
        </div>

        <select
          value={selectedPresetIndex}
          onChange={(e) => handlePresetSelect(e.target.value)}
          className="select-input"
          style={{ maxWidth: "420px", fontWeight: 500 }}
        >
          {presets.map((preset, idx) => (
            <option key={idx} value={idx.toString()}>
              {preset.name} [{preset.attack_cat}]
            </option>
          ))}
        </select>
      </section>

      {/* Prediction Action & Results Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: "var(--space-4)" }}>
        {/* Left: 42-Feature Form */}
        <section className="panel" style={{ display: "flex", flexDirection: "column" }}>
          <div className="panel-header">
            <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
              <Sliders size={14} color="var(--accent-primary)" />
              <span className="panel-title">42-Feature Vector (UNSW-NB15)</span>
            </div>
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => handlePresetSelect("0")}
              title="Reset fields to preset"
            >
              <RotateCcw size={12} />
              <span>{t.reset}</span>
            </button>
          </div>

          {/* Categorical Core Headers (Proto, Service, State) */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(3, 1fr)",
              gap: "var(--space-2)",
              padding: "var(--space-2)",
              background: "var(--bg-surface-2)",
              border: "1px solid var(--border-subtle)",
              marginBottom: "var(--space-3)",
            }}
          >
            {/* Protocol */}
            <div>
              <label style={{ fontSize: "var(--text-2xs)", fontWeight: 700, color: "var(--accent-primary)" }}>
                PROTO (Categorical)
              </label>
              <select
                className="select-input"
                value={formData.proto || "tcp"}
                onChange={(e) => handleInputChange("proto", e.target.value)}
                style={{ marginTop: "2px" }}
              >
                {["tcp", "udp", "unas", "arp", "ospf", "sctp", "any", "gre", "icmp"].map((p) => (
                  <option key={p} value={p}>{p}</option>
                ))}
              </select>
            </div>

            {/* Service */}
            <div>
              <label style={{ fontSize: "var(--text-2xs)", fontWeight: 700, color: "var(--accent-primary)" }}>
                SERVICE (Categorical)
              </label>
              <select
                className="select-input"
                value={formData.service || "-"}
                onChange={(e) => handleInputChange("service", e.target.value)}
                style={{ marginTop: "2px" }}
              >
                {["-", "http", "ftp", "ftp-data", "smtp", "pop3", "dns", "snmp", "ssl", "ssh"].map((s) => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
            </div>

            {/* State */}
            <div>
              <label style={{ fontSize: "var(--text-2xs)", fontWeight: 700, color: "var(--accent-primary)" }}>
                STATE (Categorical)
              </label>
              <select
                className="select-input"
                value={formData.state || "FIN"}
                onChange={(e) => handleInputChange("state", e.target.value)}
                style={{ marginTop: "2px" }}
              >
                {["FIN", "INT", "CON", "REQ", "RST", "ACC", "CLO"].map((st) => (
                  <option key={st} value={st}>{st}</option>
                ))}
              </select>
            </div>
          </div>

          {/* Group Navigation Tabs */}
          <div
            style={{
              display: "flex",
              gap: "4px",
              overflowX: "auto",
              paddingBottom: "4px",
              marginBottom: "var(--space-3)",
              borderBottom: "1px solid var(--border-subtle)",
            }}
          >
            {FEATURE_GROUPS.map((grp) => (
              <button
                key={grp.id}
                type="button"
                className={`btn btn-sm ${activeGroup === grp.id ? "btn-primary" : ""}`}
                onClick={() => setActiveGroup(grp.id)}
                style={{ padding: "3px 7px", fontSize: "11px", whiteSpace: "nowrap" }}
              >
                {grp.name}
              </button>
            ))}
          </div>

          {/* Active Feature Group Inputs */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(130px, 1fr))",
              gap: "var(--space-3)",
              maxHeight: "320px",
              overflowY: "auto",
              paddingRight: "4px",
              flex: 1,
            }}
          >
            {FEATURE_GROUPS.find((g) => g.id === activeGroup)?.features.map(({ key, label, step }) => (
              <div key={key}>
                <label style={{ fontSize: "var(--text-2xs)", fontWeight: 600, color: "var(--text-muted)" }}>
                  {label}
                </label>
                <input
                  type="number"
                  step={step}
                  className="input-text"
                  value={formData[key] !== undefined ? formData[key] : 0}
                  onChange={(e) => handleInputChange(key, parseFloat(e.target.value) || 0)}
                  style={{ marginTop: "2px" }}
                />
              </div>
            ))}
          </div>

          {/* Threshold & Submit Button */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: "var(--space-3)",
              marginTop: "var(--space-4)",
              paddingTop: "var(--space-3)",
              borderTop: "1px solid var(--border-subtle)",
              flexWrap: "wrap",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <span style={{ fontSize: "var(--text-xs)", fontWeight: 600 }}>Threshold: {threshold.toFixed(2)}</span>
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.05"
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                style={{ width: "110px", cursor: "pointer" }}
              />
            </div>

            <button
              type="button"
              className="btn btn-primary"
              disabled={isLoading}
              onClick={executePrediction}
            >
              <Play size={14} />
              <span>{isLoading ? t.loading : t.predictActionBtn}</span>
            </button>
          </div>
        </section>

        {/* Right: Real-time Prediction Diagnostics */}
        <section className="panel" style={{ display: "flex", flexDirection: "column", gap: "var(--space-3)" }}>
          <div className="panel-header">
            <span className="panel-title">{t.predictionResultTitle}</span>
            {currentPred && (
              <span className={`badge ${currentPred.predicted_label === "ATTACK" ? "badge-attack" : "badge-benign"}`}>
                {currentPred.predicted_label}
              </span>
            )}
          </div>

          {currentPred ? (
            <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-4)" }}>
              {/* Classification Outcome Hero */}
              <div
                style={{
                  padding: "var(--space-4)",
                  background: currentPred.predicted_label === "ATTACK" ? "var(--status-attack-bg)" : "var(--status-benign-bg)",
                  border: `1px solid ${currentPred.predicted_label === "ATTACK" ? "var(--status-attack-border)" : "var(--status-benign-border)"}`,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                }}
              >
                <div>
                  <div style={{ fontSize: "var(--text-2xs)", textTransform: "uppercase", color: "var(--text-muted)" }}>
                    {t.predictedClassLabel}
                  </div>
                  <div
                    style={{
                      fontSize: "var(--text-2xl)",
                      fontWeight: 800,
                      color: currentPred.predicted_label === "ATTACK" ? "var(--status-attack)" : "var(--status-benign)",
                    }}
                  >
                    {currentPred.predicted_label}
                  </div>
                </div>

                <div style={{ textAlign: "right" }}>
                  <div style={{ fontSize: "var(--text-2xs)", textTransform: "uppercase", color: "var(--text-muted)" }}>
                    Confidence
                  </div>
                  <div style={{ fontSize: "var(--text-xl)", fontWeight: 700, fontFamily: "var(--font-mono)" }}>
                    {(currentPred.confidence * 100).toFixed(1)}%
                  </div>
                </div>
              </div>

              {/* Class Probability Distribution Meter */}
              <div>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "var(--text-xs)", marginBottom: "4px" }}>
                  <span>{t.attackProbLabel}: <strong>{(currentPred.probabilities.ATTACK * 100).toFixed(2)}%</strong></span>
                  <span>{t.benignProbLabel}: <strong>{(currentPred.probabilities.BENIGN * 100).toFixed(2)}%</strong></span>
                </div>
                <div style={{ height: "10px", width: "100%", background: "var(--status-benign)", borderRadius: "var(--radius-sm)", overflow: "hidden", display: "flex" }}>
                  <div
                    style={{
                      width: `${currentPred.probabilities.ATTACK * 100}%`,
                      background: "var(--status-attack)",
                      transition: "width 250ms ease",
                    }}
                  />
                </div>
              </div>

              {/* Logit & Model Diagnostics */}
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr",
                  gap: "var(--space-2)",
                  fontSize: "var(--text-xs)",
                  fontFamily: "var(--font-mono)",
                  padding: "var(--space-2)",
                  background: "var(--bg-surface-2)",
                  border: "1px solid var(--border-subtle)",
                }}
              >
                <div>Logit Output: {currentPred.raw_logit.toFixed(4)}</div>
                <div>Decision Boundary: {threshold.toFixed(2)}</div>
              </div>

              {/* Quantum State Inspection Badge */}
              {encoderAngles && pauliExpvals ? (
                <div style={{ fontSize: "11px", color: "var(--status-benign)", fontFamily: "var(--font-mono)" }}>
                  ✓ Quantum statevector inspection verified: 4 rotation angles θ and Pauli-Z observables ⟨Z⟩ returned by active model.
                </div>
              ) : (
                <div style={{ fontSize: "11px", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                  • Active checkpoint: Classical MLP baseline. Simulated quantum architecture below demonstrates hybrid encoding.
                </div>
              )}
            </div>
          ) : (
            <div
              style={{
                flex: 1,
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                textAlign: "center",
                color: "var(--text-muted)",
                padding: "var(--space-6)",
              }}
            >
              <Info size={28} style={{ marginBottom: "8px", opacity: 0.6 }} />
              <p style={{ margin: 0, fontSize: "var(--text-sm)" }}>
                Select a flow preset above or customize the features, then click <strong>"{t.predictActionBtn}"</strong>.
              </p>
            </div>
          )}

          {error && (
            <div
              style={{
                padding: "8px 12px",
                background: "var(--status-attack-bg)",
                border: "1px solid var(--status-attack-border)",
                color: "var(--status-attack)",
                fontSize: "var(--text-xs)",
              }}
            >
              {error}
            </div>
          )}
        </section>
      </div>

      {/* 3D Flow Tensor & Latent Compression Visualizer */}
      <section>
        <NetworkPacket3D
          dur={formData.dur}
          proto={formData.proto}
          service={formData.service}
          state={formData.state}
          spkts={formData.spkts}
          dpkts={formData.dpkts}
          sbytes={formData.sbytes}
          dbytes={formData.dbytes}
          angles={encoderAngles || [0.34, -0.89, 1.20, -0.45]}
        />
      </section>

      {/* Live Quantum Circuit State Inspection (Bound to Current Sample) */}
      <section>
        <div style={{ marginBottom: "var(--space-2)" }}>
          <h2 style={{ fontSize: "var(--text-md)", fontWeight: 600, margin: 0 }}>
            {t.quantumInspectionTitle}
          </h2>
          <p style={{ fontSize: "var(--text-xs)", color: "var(--text-muted)", margin: "2px 0 0" }}>
            Real-time projection: The 63 standardized features map into 4 bounded angles θ_i ∈ [-π, π], generating Pauli-Z expectation values ⟨Z_i⟩.
          </p>
        </div>

        <QuantumCircuitVisualizer
          initialLayers={2}
          mode="reuploading"
          activeAngles={encoderAngles}
          activeExpvals={pauliExpvals}
        />
      </section>

      {/* Decision Support Disclaimer */}
      <div
        style={{
          padding: "var(--space-3)",
          background: "var(--bg-surface-2)",
          border: "1px solid var(--border-subtle)",
          fontSize: "var(--text-xs)",
          color: "var(--text-muted)",
          display: "flex",
          alignItems: "center",
          gap: "8px",
        }}
      >
        <AlertCircle size={14} color="var(--status-warn)" style={{ flexShrink: 0 }} />
        <span>{t.decisionSupportDisclaimer}</span>
      </div>
    </div>
  );
};
