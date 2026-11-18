import React, { useState } from "react";
import { Eye, RotateCcw, Info } from "lucide-react";

interface BlochSphere3DProps {
  angles?: number[];
  expvals?: number[];
  initialQubit?: number;
  interactive?: boolean;
}

export const BlochSphere3D: React.FC<BlochSphere3DProps> = ({
  angles = [0.34, -0.89, 1.20, -0.45],
  expvals = [0.81, -0.65, 0.43, -0.78],
  initialQubit = 0,
  interactive = true,
}) => {
  const [selectedQubit, setSelectedQubit] = useState<number>(initialQubit);
  const [is3D, setIs3D] = useState<boolean>(true);
  const [rotX, setRotX] = useState<number>(20);
  const [rotY, setRotY] = useState<number>(-25);

  const angle = angles[selectedQubit] ?? 0;
  // If expvals is provided, Z coordinate is expval, otherwise cos(angle)
  const zVal = expvals[selectedQubit] !== undefined ? expvals[selectedQubit] : Math.cos(angle);
  // Ensure zVal is clamped in [-1, 1]
  const clampedZ = Math.max(-1, Math.min(1, zVal));
  // In pure state: x^2 + y^2 = 1 - z^2
  const rXY = Math.sqrt(Math.max(0, 1 - clampedZ * clampedZ));
  const xVal = rXY * Math.sin(angle);
  const yVal = rXY * Math.cos(angle);

  // Computational probabilities
  const prob0 = ((1 + clampedZ) / 2) * 100;
  const prob1 = ((1 - clampedZ) / 2) * 100;

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!is3D || !interactive) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left - rect.width / 2;
    const y = e.clientY - rect.top - rect.height / 2;
    setRotY(Math.max(-45, Math.min(45, (x / rect.width) * 60)));
    setRotX(Math.max(-10, Math.min(40, 20 - (y / rect.height) * 40)));
  };

  const resetRotation = () => {
    setRotX(20);
    setRotY(-25);
  };

  // 2D orthographic projection constants
  const sphereRadius = 80;
  const cx = 130;
  const cy = 110;

  // 3D vector tip coordinates projected to 2D for fallback / SVG overlay
  const radX = (rotX * Math.PI) / 180;
  const radY = (rotY * Math.PI) / 180;

  // Simple 3D projection matrix:
  // Rotate around Y by radY, then around X by radX
  const px1 = xVal * Math.cos(radY) + yVal * Math.sin(radY);
  const py1 = -xVal * Math.sin(radY) + yVal * Math.cos(radY);
  const pz1 = clampedZ;

  const px2 = px1;
  const pz2 = py1 * Math.sin(radX) + pz1 * Math.cos(radX);

  const vectorScreenX = cx + (is3D ? px2 : xVal) * sphereRadius;
  const vectorScreenY = cy - (is3D ? pz2 : clampedZ) * sphereRadius;

  return (
    <div className="panel" style={{ padding: "var(--space-3)" }}>
      {/* Header & Controls */}
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
          <span style={{ fontSize: "var(--text-xs)", fontWeight: 700, textTransform: "uppercase" }}>
            Bloch Sphere State Inspector
          </span>
          <span className="badge badge-neutral">|ψ⟩ = cos(θ/2)|0⟩ + e^iφ sin(θ/2)|1⟩</span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "var(--space-2)" }}>
          {/* Qubit Selector */}
          <div style={{ display: "inline-flex", border: "1px solid var(--border-default)" }}>
            {[0, 1, 2, 3].map((q) => (
              <button
                key={q}
                type="button"
                className={`btn btn-sm ${selectedQubit === q ? "btn-primary" : ""}`}
                onClick={() => setSelectedQubit(q)}
                style={{
                  border: "none",
                  padding: "2px 8px",
                  fontSize: "var(--text-2xs)",
                  fontFamily: "var(--font-mono)",
                }}
              >
                q_{q}
              </button>
            ))}
          </div>

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
              onClick={resetRotation}
              title="Reset 3D Orientation"
              style={{ padding: "2px 6px" }}
            >
              <RotateCcw size={12} />
            </button>
          )}
        </div>
      </div>

      {/* Main Visualizer Container */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
          gap: "var(--space-4)",
          alignItems: "center",
        }}
      >
        {/* Sphere Display Stage */}
        <div
          onMouseMove={handleMouseMove}
          style={{
            position: "relative",
            minHeight: "220px",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: "var(--bg-code)",
            border: "1px solid var(--border-subtle)",
            cursor: is3D ? "grab" : "default",
            userSelect: "none",
            overflow: "hidden",
          }}
        >
          <svg
            width="260"
            height="220"
            viewBox="0 0 260 220"
            style={{ overflow: "visible" }}
          >
            {/* Sphere Outer Outline */}
            <circle
              cx={cx}
              cy={cy}
              r={sphereRadius}
              fill="none"
              stroke="var(--border-strong)"
              strokeWidth="1.5"
            />

            {/* Equator Ellipse (X-Y plane) */}
            <ellipse
              cx={cx}
              cy={cy}
              rx={sphereRadius}
              ry={is3D ? sphereRadius * Math.sin(radX) : 0}
              fill="none"
              stroke="var(--border-default)"
              strokeDasharray="3 3"
              strokeWidth="1"
            />

            {/* Meridian Ellipse (X-Z plane) */}
            <ellipse
              cx={cx}
              cy={cy}
              rx={is3D ? sphereRadius * Math.sin(radY) : sphereRadius}
              ry={sphereRadius}
              fill="none"
              stroke="var(--border-subtle)"
              strokeDasharray="2 3"
              strokeWidth="1"
            />

            {/* Z-Axis (Vertical: |0⟩ at North, |1⟩ at South) */}
            <line
              x1={cx}
              y1={cy - sphereRadius - 12}
              x2={cx}
              y2={cy + sphereRadius + 12}
              stroke="var(--border-default)"
              strokeWidth="1"
            />
            <text
              x={cx}
              y={cy - sphereRadius - 16}
              textAnchor="middle"
              fill="var(--accent-primary)"
              fontFamily="var(--font-mono)"
              fontSize="11"
              fontWeight="700"
            >
              |0⟩ (+Z)
            </text>
            <text
              x={cx}
              y={cy + sphereRadius + 24}
              textAnchor="middle"
              fill="var(--text-secondary)"
              fontFamily="var(--font-mono)"
              fontSize="11"
              fontWeight="700"
            >
              |1⟩ (-Z)
            </text>

            {/* X-Axis Horizontal */}
            <line
              x1={cx - sphereRadius - 10}
              y1={cy}
              x2={cx + sphereRadius + 10}
              y2={cy}
              stroke="var(--border-subtle)"
              strokeWidth="1"
            />
            <text
              x={cx + sphereRadius + 14}
              y={cy + 3}
              fill="var(--text-muted)"
              fontFamily="var(--font-mono)"
              fontSize="9"
            >
              |+⟩
            </text>

            {/* Center Origin Dot */}
            <circle cx={cx} cy={cy} r="2.5" fill="var(--text-muted)" />

            {/* State Vector Line */}
            <line
              x1={cx}
              y1={cy}
              x2={vectorScreenX}
              y2={vectorScreenY}
              stroke="var(--accent-primary)"
              strokeWidth="2.5"
            />

            {/* Vector Tip (Qubit State Point) */}
            <circle
              cx={vectorScreenX}
              cy={vectorScreenY}
              r="4.5"
              fill="var(--accent-primary)"
              stroke="var(--bg-page)"
              strokeWidth="1.5"
            />

            {/* Projection indicator down to equator */}
            <line
              x1={vectorScreenX}
              y1={vectorScreenY}
              x2={vectorScreenX}
              y2={cy}
              stroke="var(--status-warn)"
              strokeWidth="1"
              strokeDasharray="2 2"
            />
          </svg>

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
              Drag to Tilt
            </div>
          )}
        </div>

        {/* State Telemetry Details */}
        <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-2)" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontSize: "var(--text-xs)", color: "var(--text-muted)" }}>Qubit Wire:</span>
            <span style={{ fontFamily: "var(--font-mono)", fontWeight: 700, color: "var(--accent-primary)" }}>
              |q_{selectedQubit}⟩
            </span>
          </div>

          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontSize: "var(--text-xs)", color: "var(--text-muted)" }}>Latent Rotation θ:</span>
            <span style={{ fontFamily: "var(--font-mono)", fontWeight: 600 }}>
              {angle.toFixed(4)} rad ({( (angle / Math.PI) * 180 ).toFixed(1)}°)
            </span>
          </div>

          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontSize: "var(--text-xs)", color: "var(--text-muted)" }}>Pauli-Z Observable ⟨Z⟩:</span>
            <span
              style={{
                fontFamily: "var(--font-mono)",
                fontWeight: 700,
                color: clampedZ >= 0 ? "var(--status-benign)" : "var(--status-attack)",
              }}
            >
              {clampedZ.toFixed(4)}
            </span>
          </div>

          {/* Basis State Probabilities */}
          <div style={{ marginTop: "var(--space-1)" }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", marginBottom: "3px" }}>
              <span>P(|0⟩): <strong>{prob0.toFixed(1)}%</strong></span>
              <span>P(|1⟩): <strong>{prob1.toFixed(1)}%</strong></span>
            </div>
            <div
              style={{
                height: "6px",
                width: "100%",
                background: "var(--status-attack)",
                borderRadius: "var(--radius-sm)",
                overflow: "hidden",
                display: "flex",
              }}
            >
              <div
                style={{
                  width: `${prob0}%`,
                  background: "var(--accent-primary)",
                  transition: "width 150ms ease",
                }}
              />
            </div>
          </div>

          <div
            style={{
              fontSize: "11px",
              color: "var(--text-secondary)",
              lineHeight: 1.4,
              padding: "6px",
              background: "var(--bg-surface-2)",
              border: "1px solid var(--border-subtle)",
              marginTop: "4px",
            }}
          >
            <Info size={12} style={{ display: "inline", marginRight: "4px", verticalAlign: "middle" }} />
            The state vector points to computational basis projection ⟨Z_{selectedQubit}⟩ mapped to the classical output head.
          </div>
        </div>
      </div>
    </div>
  );
};
