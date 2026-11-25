import React, { useState } from "react";
import {
  Upload,
  FileText,
  AlertTriangle,
  CheckCircle,
  Download,
  Play,
  Terminal,
} from "lucide-react";
import { predictCsvFile, type CsvPredictResponse } from "../services/api";

export const TrafficAnalysisPage: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [csvPreviewRows, setCsvPreviewRows] = useState<string[][]>([]);
  const [validationErrors, setValidationErrors] = useState<string[]>([]);
  const [threshold, setThreshold] = useState<number>(0.2971);
  const [isClassifying, setIsClassifying] = useState<boolean>(false);
  const [classificationResult, setClassificationResult] = useState<CsvPredictResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState<boolean>(false);
  const [filterMode, setFilterMode] = useState<"ALL" | "ATTACK" | "BENIGN">("ALL");
  const [throughput, setThroughput] = useState<number | null>(null);

  // Expected 39 raw feature names for CIC-IoT-2023 benchmark
  const EXPECTED_FEATURES = [
    "Header_Length",
    "Protocol Type",
    "Time_To_Live",
    "Rate",
    "fin_flag_number",
    "syn_flag_number",
    "rst_flag_number",
    "psh_flag_number",
    "ack_flag_number",
    "ece_flag_number",
    "cwr_flag_number",
    "ack_count",
    "syn_count",
    "fin_count",
    "rst_count",
    "HTTP",
    "HTTPS",
    "DNS",
    "Telnet",
    "SMTP",
    "SSH",
    "IRC",
    "TCP",
    "UDP",
    "DHCP",
    "ARP",
    "ICMP",
    "IGMP",
    "IPv",
    "LLC",
    "Tot sum",
    "Min",
    "Max",
    "AVG",
    "Std",
    "Tot size",
    "IAT",
    "Number",
    "Variance",
  ];

  const handleFile = (file: File) => {
    if (!file.name.endsWith(".csv")) {
      setValidationErrors(["Uploaded file must have a .csv extension."]);
      return;
    }

    setSelectedFile(file);
    setValidationErrors([]);
    setErrorMessage(null);
    setClassificationResult(null);

    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target?.result as string;
      if (!text) return;

      const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
      if (lines.length < 2) {
        setValidationErrors(["CSV file must contain a header row and at least one record."]);
        return;
      }

      const headers = lines[0].split(",").map((h) => h.trim().replace(/^["']|["']$/g, ""));

      const previewRows = lines.slice(1, 51).map((line) =>
        line.split(",").map((c) => c.trim().replace(/^["']|["']$/g, ""))
      );
      setCsvPreviewRows(previewRows);

      const missing = EXPECTED_FEATURES.filter((f) => !headers.includes(f));
      if (missing.length > 0) {
        setValidationErrors([
          `Missing ${missing.length} mandatory features required by the CIC-IoT-2023 schema: ${missing.slice(0, 6).join(", ")}${missing.length > 6 ? "..." : ""}`,
        ]);
      }
    };
    reader.readAsText(file);
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const loadBundledCapture = async () => {
    try {
      const response = await fetch("/data/ciciot2023_test_500.csv");
      const blob = await response.blob();
      const file = new File([blob], "ciciot2023_test_500.csv", { type: "text/csv" });
      handleFile(file);
    } catch {
      setErrorMessage("Could not load bundled test traffic file.");
    }
  };

  const runClassification = async () => {
    if (!selectedFile || validationErrors.length > 0) return;
    setIsClassifying(true);
    setErrorMessage(null);
    const start = performance.now();

    try {
      const result = (await predictCsvFile(selectedFile, threshold, "json")) as CsvPredictResponse;
      const elapsed = (performance.now() - start) / 1000;
      setThroughput(parseFloat((result.sample_count / elapsed).toFixed(1)));
      setClassificationResult(result);
      setIsClassifying(false);
    } catch (err: any) {
      setErrorMessage(err.message || "Failed to execute batch quantum inference.");
      setIsClassifying(false);
    }
  };

  const downloadPredictedCsv = async () => {
    if (!selectedFile) return;
    try {
      const blob = (await predictCsvFile(selectedFile, threshold, "csv")) as Blob;
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `classified_${selectedFile.name}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err: any) {
      setErrorMessage("Download failed: " + err.message);
    }
  };

  const filteredPreviewRows = csvPreviewRows.filter((_, idx) => {
    if (!classificationResult) return true;
    const pred = classificationResult.predictions[idx];
    if (!pred) return true;
    if (filterMode === "ATTACK") return pred.predicted_label === "ATTACK";
    if (filterMode === "BENIGN") return pred.predicted_label === "BENIGN";
    return true;
  });

  return (
    <div style={{ display: "flex", flexDirection: "column", minHeight: "100%", width: "100%" }}>
      {/* 1. Technical Masthead */}
      <header className="masthead">
        <div>
          <div className="masthead-eyebrow">
            <Terminal size={12} color="var(--accent-primary)" />
            <span>CIC-IoT-2023 // BATCH PACKET CAPTURE & FLOW INGESTION</span>
          </div>
          <h1 className="masthead-title">Network Packet Stream & Batch Inference Station</h1>
        </div>
        <div className="masthead-meta">
          <span>MODEL: 4Q-REUPLOAD-L1</span>
          <span>&bull;</span>
          <span>SCHEMA: 39-D FLOAT32</span>
          <span>&bull;</span>
          <span>CALIBRATED TAU*: {threshold.toFixed(4)}</span>
        </div>
      </header>

      {/* 2. Continuous Telemetry Strip */}
      <div className="telemetry-strip">
        <div className="telemetry-item">
          <span className="telemetry-label">INSPECTION ENGINE</span>
          <span className="telemetry-val" style={{ color: "var(--accent-quantum)" }}>STREAMING CSV BATCH INFER</span>
        </div>
        <span className="telemetry-divider">|</span>
        <div className="telemetry-item">
          <span className="telemetry-label">CALIBRATED TAU*</span>
          <span className="telemetry-val" style={{ color: "var(--accent-primary)" }}>{threshold.toFixed(4)}</span>
        </div>
        <span className="telemetry-divider">|</span>
        <div className="telemetry-item">
          <span className="telemetry-label">LOADED CAPTURE</span>
          <span className="telemetry-val">{selectedFile ? `${selectedFile.name} (${(selectedFile.size / 1024).toFixed(1)} KB)` : "NONE"}</span>
        </div>
        {throughput && (
          <>
            <span className="telemetry-divider">|</span>
            <div className="telemetry-item">
              <span className="telemetry-label">THROUGHPUT</span>
              <span className="telemetry-val" style={{ color: "var(--status-benign)" }}>{throughput} FLOWS / SEC</span>
            </div>
          </>
        )}
      </div>

      <div style={{ padding: "1.25rem", display: "flex", flexDirection: "column", gap: "1rem" }}>
        {/* Upload & Ingest Controls Panel */}
        <section
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          style={{
            border: dragActive ? "1px solid var(--accent-primary)" : "1px solid var(--border-default)",
            background: dragActive ? "var(--bg-surface-2)" : "var(--bg-surface-1)",
            padding: "1rem 1.25rem",
            display: "flex",
            flexDirection: "column",
            gap: "0.75rem",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "1rem" }}>
            <div>
              <div style={{ fontFamily: "var(--font-heading)", fontSize: "14px", fontWeight: 700 }}>
                FLOW RECORD CAPTURE INGESTION
              </div>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "11px", color: "var(--text-muted)", marginTop: "2px" }}>
                Drag and drop a 39-feature CIC-IoT-2023 CSV file or load the official held-out 500-flow capture.
              </div>
            </div>

            <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
              <button
                type="button"
                className="btn btn-primary"
                onClick={loadBundledCapture}
                style={{ fontSize: "11px", padding: "6px 12px" }}
              >
                <FileText size={12} />
                <span>LOAD BUNDLED 500-FLOW CIC-IOT-2023 CAPTURE</span>
              </button>

              <label className="btn" style={{ fontSize: "11px", padding: "6px 12px", cursor: "pointer" }}>
                <Upload size={12} />
                <span>CHOOSE LOCAL CSV</span>
                <input
                  type="file"
                  accept=".csv"
                  style={{ display: "none" }}
                  onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
                />
              </label>
            </div>
          </div>

          {/* Validation Banner if file selected */}
          {selectedFile && (
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                padding: "0.5rem 0.75rem",
                background: "var(--bg-code)",
                border: "1px solid var(--border-default)",
                fontFamily: "var(--font-mono)",
                fontSize: "11px",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <FileText size={12} color="var(--accent-primary)" />
                <span style={{ color: "var(--text-primary)", fontWeight: 700 }}>{selectedFile.name}</span>
                <span style={{ color: "var(--text-muted)" }}>({(selectedFile.size / 1024).toFixed(1)} KB)</span>
              </div>
              {validationErrors.length === 0 ? (
                <span className="badge badge-benign">
                  <CheckCircle size={10} />
                  SCHEMA VALIDATED (39 FEAT)
                </span>
              ) : (
                <span className="badge badge-attack">
                  <AlertTriangle size={10} />
                  VALIDATION FAILED
                </span>
              )}
            </div>
          )}

          {/* Errors */}
          {validationErrors.map((err, idx) => (
            <div
              key={idx}
              style={{
                padding: "6px 10px",
                background: "var(--status-attack-bg)",
                border: "1px solid var(--status-attack-border)",
                color: "var(--status-attack)",
                fontSize: "11px",
                fontFamily: "var(--font-mono)",
              }}
            >
              {err}
            </div>
          ))}

          {/* Controls: Threshold Slider & Action Buttons */}
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              flexWrap: "wrap",
              gap: "1rem",
              paddingTop: "0.75rem",
              borderTop: "1px solid var(--border-subtle)",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
              <span style={{ fontFamily: "var(--font-mono)", fontSize: "11px", color: "var(--text-muted)" }}>
                ALERT THRESHOLD (TAU*):
              </span>
              <span style={{ fontFamily: "var(--font-mono)", fontSize: "11px", fontWeight: 700, color: "var(--accent-primary)" }}>
                {threshold.toFixed(4)}
              </span>
              <input
                type="range"
                min="0.05"
                max="0.95"
                step="0.01"
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                style={{ width: "120px", accentColor: "var(--accent-primary)", cursor: "pointer" }}
              />
            </div>

            <div style={{ display: "flex", gap: "0.5rem" }}>
              <button
                type="button"
                className="btn btn-primary"
                disabled={!selectedFile || validationErrors.length > 0 || isClassifying}
                onClick={runClassification}
                style={{ padding: "6px 14px" }}
              >
                <Play size={12} />
                <span>{isClassifying ? "CLASSIFYING FLOWS..." : "EXECUTE BATCH QUANTUM INFERENCE"}</span>
              </button>

              {classificationResult && (
                <button
                  type="button"
                  className="btn"
                  onClick={downloadPredictedCsv}
                  style={{ padding: "6px 14px" }}
                >
                  <Download size={12} />
                  <span>EXPORT CLASSIFIED CSV</span>
                </button>
              )}
            </div>
          </div>
        </section>

        {/* Aggregate Telemetry Cards */}
        {classificationResult && (
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
              gap: "0.75rem",
            }}
          >
            <div className="panel" style={{ padding: "0.75rem 1rem" }}>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "10px", color: "var(--text-muted)" }}>
                TOTAL FLOWS EVALUATED
              </div>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.4rem", fontWeight: 700 }}>
                {classificationResult.sample_count}
              </div>
            </div>

            <div className="panel" style={{ padding: "0.75rem 1rem", borderLeft: "3px solid var(--status-attack)" }}>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "10px", color: "var(--text-muted)" }}>
                ATTACK FLOWS DETECTED
              </div>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.4rem", fontWeight: 700, color: "var(--status-attack)" }}>
                {classificationResult.summary.attack_count}
                <span style={{ fontSize: "11px", fontWeight: 500, color: "var(--text-muted)", marginLeft: "6px" }}>
                  ({(classificationResult.summary.attack_ratio * 100).toFixed(1)}%)
                </span>
              </div>
            </div>

            <div className="panel" style={{ padding: "0.75rem 1rem", borderLeft: "3px solid var(--status-benign)" }}>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "10px", color: "var(--text-muted)" }}>
                BENIGN FLOWS
              </div>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.4rem", fontWeight: 700, color: "var(--status-benign)" }}>
                {classificationResult.summary.benign_count}
                <span style={{ fontSize: "11px", fontWeight: 500, color: "var(--text-muted)", marginLeft: "6px" }}>
                  ({((classificationResult.summary.benign_count / classificationResult.sample_count) * 100).toFixed(1)}%)
                </span>
              </div>
            </div>

            <div className="panel" style={{ padding: "0.75rem 1rem" }}>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "10px", color: "var(--text-muted)" }}>
                THROUGHPUT RATE
              </div>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.4rem", fontWeight: 700, color: "var(--accent-quantum)" }}>
                {throughput ? `${throughput} /s` : "--"}
              </div>
            </div>
          </div>
        )}

        {/* Monospace Packet Log Table with Row-Level Attack Heat Bars */}
        {selectedFile && (
          <section className="panel" style={{ padding: 0, overflow: "hidden" }}>
            {/* Table Control Header */}
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
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "11px", fontWeight: 700 }}>
                FLOW PACKET TELEMETRY LOG (PREVIEW: {filteredPreviewRows.length} RECORDS)
              </div>

              {/* Filter Pills */}
              <div style={{ display: "flex", gap: "4px" }}>
                <button
                  type="button"
                  onClick={() => setFilterMode("ALL")}
                  style={{
                    padding: "3px 8px",
                    fontSize: "10px",
                    fontFamily: "var(--font-mono)",
                    background: filterMode === "ALL" ? "var(--accent-primary)" : "var(--bg-surface-1)",
                    color: filterMode === "ALL" ? "var(--text-inverse)" : "var(--text-secondary)",
                    border: "1px solid var(--border-default)",
                    cursor: "pointer",
                    fontWeight: filterMode === "ALL" ? 700 : 400,
                  }}
                >
                  ALL
                </button>
                <button
                  type="button"
                  onClick={() => setFilterMode("ATTACK")}
                  style={{
                    padding: "3px 8px",
                    fontSize: "10px",
                    fontFamily: "var(--font-mono)",
                    background: filterMode === "ATTACK" ? "var(--status-attack)" : "var(--bg-surface-1)",
                    color: filterMode === "ATTACK" ? "#ffffff" : "var(--status-attack)",
                    border: "1px solid var(--status-attack-border)",
                    cursor: "pointer",
                    fontWeight: filterMode === "ATTACK" ? 700 : 400,
                  }}
                >
                  ATTACK ONLY
                </button>
                <button
                  type="button"
                  onClick={() => setFilterMode("BENIGN")}
                  style={{
                    padding: "3px 8px",
                    fontSize: "10px",
                    fontFamily: "var(--font-mono)",
                    background: filterMode === "BENIGN" ? "var(--status-benign)" : "var(--bg-surface-1)",
                    color: filterMode === "BENIGN" ? "#ffffff" : "var(--status-benign)",
                    border: "1px solid var(--status-benign-border)",
                    cursor: "pointer",
                    fontWeight: filterMode === "BENIGN" ? 700 : 400,
                  }}
                >
                  BENIGN ONLY
                </button>
              </div>
            </div>

            {/* Scrollable Monospace Table */}
            <div className="table-container" style={{ margin: 0, maxHeight: "420px", border: "none" }}>
              <table>
                <thead>
                  <tr>
                    <th style={{ width: "60px" }}>FLOW</th>
                    {classificationResult && <th style={{ width: "90px" }}>VERDICT</th>}
                    {classificationResult && <th style={{ width: "160px" }}>P(ATTACK) GAUGE</th>}
                    {classificationResult && <th style={{ width: "80px" }}>CONFIDENCE</th>}
                    <th>HDR_LEN</th>
                    <th>PROTO</th>
                    <th>TTL</th>
                    <th>RATE</th>
                    <th>TOT SUM</th>
                    <th>AVG SIZE</th>
                    <th>IAT</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredPreviewRows.map((row, rIdx) => {
                    const pred = classificationResult?.predictions[rIdx];
                    const attackProb = pred ? pred.probabilities.ATTACK : 0;
                    return (
                      <tr key={rIdx}>
                        <td style={{ fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>
                          #{rIdx + 1}
                        </td>
                        {classificationResult && (
                          <td>
                            {pred ? (
                              <span
                                className={`badge ${
                                  pred.predicted_label === "ATTACK" ? "badge-attack" : "badge-benign"
                                }`}
                              >
                                {pred.predicted_label}
                              </span>
                            ) : (
                              "--"
                            )}
                          </td>
                        )}
                        {classificationResult && (
                          <td>
                            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                              <div
                                style={{
                                  flex: 1,
                                  height: "6px",
                                  background: "var(--bg-surface-3)",
                                  overflow: "hidden",
                                  position: "relative",
                                }}
                              >
                                <div
                                  style={{
                                    width: `${attackProb * 100}%`,
                                    height: "100%",
                                    background:
                                      attackProb >= threshold
                                        ? "var(--status-attack)"
                                        : "var(--status-benign)",
                                  }}
                                />
                              </div>
                              <span
                                style={{
                                  fontFamily: "var(--font-mono)",
                                  fontSize: "10px",
                                  width: "42px",
                                  textAlign: "right",
                                }}
                              >
                                {(attackProb * 100).toFixed(1)}%
                              </span>
                            </div>
                          </td>
                        )}
                        {classificationResult && (
                          <td style={{ fontFamily: "var(--font-mono)", fontSize: "11px" }}>
                            {pred ? `${(pred.confidence * 100).toFixed(1)}%` : "--"}
                          </td>
                        )}
                        <td>{row[0] || "--"}</td>
                        <td>{row[1] || "--"}</td>
                        <td>{row[2] || "--"}</td>
                        <td>{row[3] ? parseFloat(row[3]).toFixed(1) : "--"}</td>
                        <td>{row[30] || "--"}</td>
                        <td>{row[33] ? parseFloat(row[33]).toFixed(1) : "--"}</td>
                        <td>{row[36] || "--"}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {errorMessage && (
          <div
            style={{
              padding: "8px 12px",
              background: "var(--status-attack-bg)",
              border: "1px solid var(--status-attack-border)",
              color: "var(--status-attack)",
              fontSize: "11px",
              fontFamily: "var(--font-mono)",
            }}
          >
            {errorMessage}
          </div>
        )}
      </div>
    </div>
  );
};
