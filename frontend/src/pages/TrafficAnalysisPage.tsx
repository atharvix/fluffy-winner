import React, { useState } from "react";
import {
  Upload,
  FileText,
  AlertTriangle,
  CheckCircle,
  Download,
  Play,
  Info,
} from "lucide-react";
import { useTranslation } from "../i18n/useTranslation";
import { predictCsvFile, type CsvPredictResponse } from "../services/api";

export const TrafficAnalysisPage: React.FC = () => {
  const { t } = useTranslation();

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [csvPreviewRows, setCsvPreviewRows] = useState<string[][]>([]);
  const [csvHeaders, setCsvHeaders] = useState<string[]>([]);
  const [validationErrors, setValidationErrors] = useState<string[]>([]);
  const [validationWarnings, setValidationWarnings] = useState<string[]>([]);
  const [threshold, setThreshold] = useState<number>(0.5);
  const [isClassifying, setIsClassifying] = useState<boolean>(false);
  const [classificationResult, setClassificationResult] = useState<CsvPredictResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState<boolean>(false);

  // Expected 42 raw feature names
  const EXPECTED_FEATURES = [
    "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes",
    "rate", "sttl", "dttl", "sload", "dload", "sloss", "dloss", "sinpkt",
    "dinpkt", "sjit", "djit", "swin", "stcpb", "dtcpb", "dwin", "tcprtt",
    "synack", "ackdat", "smean", "dmean", "trans_depth", "response_body_len",
    "ct_srv_src", "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd",
    "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports"
  ];

  const handleFile = (file: File) => {
    if (!file.name.endsWith(".csv")) {
      setValidationErrors(["Uploaded file must have a .csv extension."]);
      return;
    }

    setSelectedFile(file);
    setValidationErrors([]);
    setValidationWarnings([]);
    setErrorMessage(null);
    setClassificationResult(null);

    // Read client-side preview and validate schema
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
      setCsvHeaders(headers);

      const previewRows = lines.slice(1, 11).map((line) =>
        line.split(",").map((c) => c.trim().replace(/^["']|["']$/g, ""))
      );
      setCsvPreviewRows(previewRows);

      // Validate required features
      const missing = EXPECTED_FEATURES.filter((f) => !headers.includes(f));
      if (missing.length > 0) {
        setValidationErrors([
          `Missing ${missing.length} mandatory features required by the UNSW-NB15 schema: ${missing.slice(0, 6).join(", ")}${missing.length > 6 ? "..." : ""}`,
        ]);
      }

      // Check for target label columns present in the file
      if (headers.includes("label") || headers.includes("attack_cat")) {
        setValidationWarnings([
          "Notice: The uploaded CSV contains label/attack_cat columns. These columns are strictly excluded from predictive features to prevent data leakage.",
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

  const loadSampleFile = async () => {
    try {
      const response = await fetch("/data/sample_traffic_test.csv");
      const blob = await response.blob();
      const file = new File([blob], "sample_traffic_test.csv", { type: "text/csv" });
      handleFile(file);
    } catch {
      setErrorMessage("Could not load sample test traffic file.");
    }
  };

  const runClassification = async () => {
    if (!selectedFile || validationErrors.length > 0) return;
    setIsClassifying(true);
    setErrorMessage(null);

    try {
      const result = (await predictCsvFile(selectedFile, threshold, "json")) as CsvPredictResponse;
      setClassificationResult(result);
      setIsClassifying(false);
    } catch (err: any) {
      setErrorMessage(err.message || "Failed to execute batch inference.");
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
      a.download = `predictions_${selectedFile.name}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err: any) {
      setErrorMessage("Download failed: " + err.message);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-6)" }}>
      {/* Header */}
      <div>
        <h1 style={{ fontSize: "var(--text-xl)", fontWeight: 700, margin: 0 }}>
          {t.trafficTitle}
        </h1>
        <p style={{ fontSize: "var(--text-sm)", color: "var(--text-secondary)", margin: "4px 0 0" }}>
          {t.trafficSubtitle}
        </p>
      </div>

      {/* Drag & Drop Upload Zone */}
      <section
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        className="panel"
        style={{
          border: dragActive
            ? "2px dashed var(--accent-primary)"
            : "1px dashed var(--border-strong)",
          background: dragActive ? "var(--bg-surface-2)" : "var(--bg-surface-1)",
          padding: "var(--space-6) var(--space-4)",
          textAlign: "center",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: "var(--space-3)",
          cursor: "pointer",
        }}
      >
        <Upload size={32} color="var(--accent-primary)" />
        <div>
          <div style={{ fontWeight: 600, fontSize: "var(--text-base)" }}>
            {t.dragDropTitle}
          </div>
          <div style={{ fontSize: "var(--text-xs)", color: "var(--text-muted)", marginTop: "4px" }}>
            {t.dragDropSubtitle}
          </div>
        </div>

        <div style={{ display: "flex", gap: "var(--space-3)", flexWrap: "wrap", justifyContent: "center" }}>
          <label className="btn btn-primary" style={{ cursor: "pointer" }}>
            <span>Choose CSV File</span>
            <input
              type="file"
              accept=".csv"
              style={{ display: "none" }}
              onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
            />
          </label>
          <button type="button" className="btn" onClick={loadSampleFile}>
            <FileText size={14} />
            <span>{t.loadSampleBtn}</span>
          </button>
        </div>
      </section>

      {/* Selected File & Pre-inference Validation Summary */}
      {selectedFile && (
        <section className="panel">
          <div className="panel-header">
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <FileText size={16} color="var(--accent-primary)" />
              <span className="panel-title">{selectedFile.name}</span>
              <span style={{ fontSize: "var(--text-xs)", color: "var(--text-muted)" }}>
                ({(selectedFile.size / 1024).toFixed(1)} KB)
              </span>
            </div>
            {validationErrors.length === 0 ? (
              <span className="badge badge-benign">
                <CheckCircle size={10} />
                {t.validationPassed}
              </span>
            ) : (
              <span className="badge badge-attack">
                <AlertTriangle size={10} />
                {t.validationFailed}
              </span>
            )}
          </div>

          {/* Validation Warnings / Errors */}
          {validationErrors.map((err, idx) => (
            <div
              key={idx}
              style={{
                padding: "8px 12px",
                background: "var(--status-attack-bg)",
                border: "1px solid var(--status-attack-border)",
                color: "var(--status-attack)",
                fontSize: "var(--text-xs)",
                marginBottom: "8px",
              }}
            >
              {err}
            </div>
          ))}

          {validationWarnings.map((warn, idx) => (
            <div
              key={idx}
              style={{
                padding: "8px 12px",
                background: "var(--status-warn-bg)",
                border: "1px solid var(--status-warn-border)",
                color: "var(--status-warn)",
                fontSize: "var(--text-xs)",
                marginBottom: "8px",
              }}
            >
              {warn}
            </div>
          ))}

          {/* Threshold Slider and Action Controls */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              flexWrap: "wrap",
              gap: "var(--space-3)",
              margin: "var(--space-4) 0",
              padding: "var(--space-3)",
              background: "var(--bg-surface-2)",
              border: "1px solid var(--border-subtle)",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
              <span style={{ fontSize: "var(--text-xs)", fontWeight: 600 }}>
                Alert Threshold: {threshold.toFixed(2)}
              </span>
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.05"
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                style={{ cursor: "pointer", width: "140px" }}
              />
            </div>

            <div style={{ display: "flex", gap: "var(--space-2)" }}>
              <button
                type="button"
                className="btn btn-primary"
                disabled={validationErrors.length > 0 || isClassifying}
                onClick={runClassification}
              >
                <Play size={14} />
                <span>{isClassifying ? t.loading : t.runBatchBtn}</span>
              </button>

              {classificationResult && (
                <button type="button" className="btn" onClick={downloadPredictedCsv}>
                  <Download size={14} />
                  <span>{t.downloadCsvBtn}</span>
                </button>
              )}
            </div>
          </div>

          {/* Classification Result Summary */}
          {classificationResult && (
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
                gap: "var(--space-3)",
                marginBottom: "var(--space-4)",
              }}
            >
              <div className="panel" style={{ padding: "var(--space-3)" }}>
                <div style={{ fontSize: "var(--text-2xs)", color: "var(--text-muted)" }}>{t.detectedRows}</div>
                <div style={{ fontSize: "var(--text-xl)", fontWeight: 700 }}>
                  {classificationResult.sample_count}
                </div>
              </div>
              <div className="panel" style={{ padding: "var(--space-3)", borderLeft: "3px solid var(--status-attack)" }}>
                <div style={{ fontSize: "var(--text-2xs)", color: "var(--text-muted)" }}>{t.attackDetected}</div>
                <div style={{ fontSize: "var(--text-xl)", fontWeight: 700, color: "var(--status-attack)" }}>
                  {classificationResult.summary.attack_count} ({(classificationResult.summary.attack_ratio * 100).toFixed(1)}%)
                </div>
              </div>
              <div className="panel" style={{ padding: "var(--space-3)", borderLeft: "3px solid var(--status-benign)" }}>
                <div style={{ fontSize: "var(--text-2xs)", color: "var(--text-muted)" }}>{t.benignDetected}</div>
                <div style={{ fontSize: "var(--text-xl)", fontWeight: 700, color: "var(--status-benign)" }}>
                  {classificationResult.summary.benign_count}
                </div>
              </div>
            </div>
          )}

          {/* CSV Data Preview Table */}
          <div style={{ marginTop: "var(--space-3)" }}>
            <div style={{ fontSize: "var(--text-xs)", fontWeight: 600, marginBottom: "4px" }}>
              Preview (First {csvPreviewRows.length} Records)
            </div>
            <div className="table-container" style={{ maxHeight: "280px" }}>
              <table>
                <thead>
                  <tr>
                    {classificationResult && <th>PREDICTION</th>}
                    {csvHeaders.slice(0, 10).map((h, i) => (
                      <th key={i}>{h}</th>
                    ))}
                    {csvHeaders.length > 10 && <th>+{csvHeaders.length - 10} MORE COLS</th>}
                  </tr>
                </thead>
                <tbody>
                  {csvPreviewRows.map((row, rIdx) => {
                    const pred = classificationResult?.predictions[rIdx];
                    return (
                      <tr key={rIdx}>
                        {classificationResult && (
                          <td>
                            {pred ? (
                              <span className={`badge ${pred.predicted_label === "ATTACK" ? "badge-attack" : "badge-benign"}`}>
                                {pred.predicted_label} ({Math.round(pred.confidence * 100)}%)
                              </span>
                            ) : (
                              "-"
                            )}
                          </td>
                        )}
                        {row.slice(0, 10).map((cell, cIdx) => (
                          <td key={cIdx} style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)" }}>
                            {cell}
                          </td>
                        ))}
                        {row.length > 10 && (
                          <td style={{ color: "var(--text-muted)", fontSize: "var(--text-2xs)" }}>...</td>
                        )}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </section>
      )}

      {errorMessage && (
        <div
          style={{
            padding: "var(--space-3)",
            background: "var(--status-attack-bg)",
            border: "1px solid var(--status-attack-border)",
            color: "var(--status-attack)",
            fontSize: "var(--text-xs)",
          }}
        >
          {errorMessage}
        </div>
      )}

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
        <Info size={14} color="var(--accent-primary)" style={{ flexShrink: 0 }} />
        <span>{t.uploadedNotice}</span>
      </div>
    </div>
  );
};
