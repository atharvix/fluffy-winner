import React, { useState } from "react";
import { Eye, RotateCcw, Box, ArrowRight } from "lucide-react";

interface NetworkPacket3DProps {
  headerLength?: number;
  protocolType?: number | string;
  ttl?: number;
  rate?: number;
  totSum?: number;
  iat?: number;
  dur?: number;
  proto?: string;
  service?: string;
  state?: string;
  spkts?: number;
  dpkts?: number;
  sbytes?: number;
  dbytes?: number;
  angles?: number[];
  interactive?: boolean;
}

export const NetworkPacket3D: React.FC<NetworkPacket3DProps> = ({
  headerLength = 28.0,
  protocolType = 6.0,
  ttl = 64.0,
  rate = 201.02,
  totSum = 9434.0,
  iat = 0.0059,
  angles = [0.34, -0.89, 1.20, -0.45],
  interactive = true,
}) => {
  const [is3D, setIs3D] = useState<boolean>(true);
  const [rotX, setRotX] = useState<number>(24);
  const [rotY, setRotY] = useState<number>(-28);

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!is3D || !interactive) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left - rect.width / 2;
    const y = e.clientY - rect.top - rect.height / 2;
    setRotY(Math.max(-45, Math.min(45, (x / rect.width) * 50)));
    setRotX(Math.max(5, Math.min(35, 24 - (y / rect.height) * 30)));
  };

  const resetView = () => {
    setRotX(24);
    setRotY(-28);
  };

  return (
    <div className="panel" style={{ padding: "var(--space-3)", overflow: "hidden" }}>
      {/* Header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          flexWrap: "wrap",
          gap: "var(--space-2)",
          marginBottom: "var(--space-3)",
          paddingBottom: "var(--space-2)",
          borderBottom: "1px solid var(--border-subtle)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <Box size={14} color="var(--accent-primary)" />
          <span style={{ fontSize: "var(--text-xs)", fontWeight: 700, textTransform: "uppercase" }}>
            3D Flow Tensor & Latent Compression
          </span>
          <span className="badge badge-neutral">39 Raw → 39 Scaled → 4 Quantum Angles</span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "var(--space-2)" }}>
          <button
            type="button"
            className="btn btn-sm"
            onClick={() => setIs3D(!is3D)}
            title="Toggle 3D Perspective"
            style={{ padding: "2px 6px" }}
          >
            <Eye size={12} />
            <span>{is3D ? "3D" : "2D"}</span>
          </button>
          {is3D && (
            <button
              type="button"
              className="btn btn-sm"
              onClick={resetView}
              title="Reset 3D Rotation"
              style={{ padding: "2px 6px" }}
            >
              <RotateCcw size={12} />
            </button>
          )}
        </div>
      </div>

      {/* 3D Stage Container */}
      <div
        onMouseMove={handleMouseMove}
        style={{
          perspective: is3D ? "1000px" : "none",
          padding: "var(--space-4)",
          background: "var(--bg-code)",
          border: "1px solid var(--border-subtle)",
          position: "relative",
          cursor: is3D ? "grab" : "default",
          userSelect: "none",
          minHeight: "180px",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          overflow: "hidden",
        }}
      >
        {/* 3D Floating Tensor Elements */}
        <div
          style={{
            transform: is3D
              ? `rotateX(${rotX}deg) rotateY(${rotY}deg) translateZ(10px)`
              : "none",
            transformStyle: "preserve-3d",
            transition: "transform 150ms cubic-bezier(0.1, 0.9, 0.2, 1)",
            display: "flex",
            alignItems: "center",
            gap: "24px",
            flexWrap: "nowrap",
          }}
        >
          {/* Card 1: Raw Flow Header (CIC-IoT-2023) */}
          <div
            style={{
              width: "165px",
              padding: "10px",
              background: "var(--bg-surface-1)",
              border: "1px solid var(--border-default)",
              boxShadow: is3D ? "0 4px 12px rgba(0,0,0,0.3)" : "none",
              transform: is3D ? "translateZ(20px)" : "none",
              fontSize: "11px",
              fontFamily: "var(--font-mono)",
            }}
          >
            <div style={{ color: "var(--accent-primary)", fontWeight: 700, marginBottom: "4px" }}>
              RAW_FLOW_39
            </div>
            <div style={{ color: "var(--text-secondary)" }}>proto_type: {protocolType}</div>
            <div style={{ color: "var(--text-secondary)" }}>ttl: {ttl} | hdr: {headerLength}B</div>
            <div style={{ color: "var(--text-muted)" }}>rate: {typeof rate === 'number' ? rate.toFixed(1) : rate} pkts/s</div>
            <div style={{ color: "var(--text-muted)" }}>tot_sum: {totSum}B</div>
            <div style={{ color: "var(--text-muted)" }}>iat: {typeof iat === 'number' ? iat.toFixed(5) : iat}s</div>
          </div>

          <ArrowRight size={16} color="var(--text-muted)" />

          {/* Card 2: Neural Compression Layer */}
          <div
            style={{
              width: "145px",
              padding: "10px",
              background: "var(--bg-surface-2)",
              border: "1px dashed var(--border-strong)",
              boxShadow: is3D ? "0 4px 10px rgba(0,0,0,0.25)" : "none",
              transform: is3D ? "translateZ(35px)" : "none",
              fontSize: "11px",
              fontFamily: "var(--font-mono)",
              textAlign: "center",
            }}
          >
            <div style={{ color: "var(--text-primary)", fontWeight: 700, marginBottom: "4px" }}>
              MLP_ENCODER
            </div>
            <div style={{ color: "var(--accent-primary)" }}>Linear(39 → 16)</div>
            <div style={{ color: "var(--text-muted)" }}>ReLU Activation</div>
            <div style={{ color: "var(--accent-primary)" }}>Linear(16 → 4)</div>
            <div style={{ color: "var(--status-warn)", fontWeight: 600 }}>π · tanh(·)</div>
          </div>

          <ArrowRight size={16} color="var(--text-muted)" />

          {/* Card 3: 4 Bounded Quantum Angles */}
          <div
            style={{
              width: "165px",
              padding: "10px",
              background: "var(--bg-surface-1)",
              border: "1px solid var(--accent-quantum)",
              boxShadow: is3D ? "0 4px 12px rgba(6,182,212,0.15)" : "none",
              transform: is3D ? "translateZ(50px)" : "none",
              fontSize: "11px",
              fontFamily: "var(--font-mono)",
            }}
          >
              <div style={{ color: "var(--accent-quantum)", fontWeight: 700, marginBottom: "4px" }}>
                ROTATION_ANGLES θ
              </div>
              {angles.map((a, i) => (
                <div key={i} style={{ display: "flex", justifyContent: "space-between", color: "var(--text-primary)" }}>
                  <span>θ_{i}:</span>
                  <span style={{ fontWeight: 600, color: "var(--accent-quantum)" }}>{a !== undefined ? a.toFixed(3) : "0.000"} rad</span>
                </div>
              ))}
            </div>
          </div>

          {is3D && (
            <div
              style={{
                position: "absolute",
                bottom: "4px",
                right: "6px",
                fontSize: "10px",
                fontFamily: "var(--font-mono)",
                color: "var(--text-muted)",
              }}
            >
              Interactive 3D Stage
            </div>
          )}
        </div>
      </div>
      );
};
