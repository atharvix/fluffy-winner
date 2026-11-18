import React, { useState } from "react";
import { RotateCcw, Eye, Info, Sparkles, Orbit } from "lucide-react";
import { BlochSphere3D } from "./BlochSphere3D";

interface QuantumCircuitVisualizerProps {
  initialLayers?: 1 | 2 | 4;
  mode?: "reuploading" | "standard_vqc";
  activeAngles?: number[];
  activeExpvals?: number[];
  interactive?: boolean;
}

export const QuantumCircuitVisualizer: React.FC<QuantumCircuitVisualizerProps> = ({
  initialLayers = 2,
  mode = "reuploading",
  activeAngles,
  activeExpvals,
  interactive = true,
}) => {
  const [layers, setLayers] = useState<1 | 2 | 4>(initialLayers);
  const [circuitMode, setCircuitMode] = useState<"reuploading" | "standard_vqc">(mode);
  const [is3D, setIs3D] = useState<boolean>(true);
  const [viewTab, setViewTab] = useState<"circuit" | "bloch">("circuit");
  const [rotX, setRotX] = useState<number>(16);
  const [rotY, setRotY] = useState<number>(-10);
  const [hoveredGate, setHoveredGate] = useState<string | null>(null);

  // 4 Qubits: Q0, Q1, Q2, Q3
  const qubits = [0, 1, 2, 3];

  // Gate details for tooltips
  const gateDescriptions: Record<string, { title: string; formula: string; desc: string }> = {
    RY_enc: {
      title: "Data Encoding Gate RY(θ_i)",
      formula: "RY(θ) = exp(-i θ/2 σ_y) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]",
      desc: "Rotates the qubit state around the Y-axis by the classical feature angle θ_i ∈ [-π, π]. In Data Re-uploading, this gate is re-applied before every variational block.",
    },
    RY_var: {
      title: "Trainable Variational Gate RY(w_ry)",
      formula: "RY(w) = exp(-i w/2 σ_y)",
      desc: "Trainable rotation parameter updated during training via PyTorch backpropagation and PennyLane autograd.",
    },
    RZ_var: {
      title: "Trainable Variational Gate RZ(w_rz)",
      formula: "RZ(φ) = exp(-i φ/2 σ_z) = [[exp(-i φ/2), 0], [0, exp(i φ/2)]]",
      desc: "Single-qubit Z-axis rotation parameterizing phase shifts and enriching multi-frequency Fourier basis.",
    },
    CNOT: {
      title: "Ring Entanglement Gate CNOT",
      formula: "|c, t⟩ → |c, t ⊕ c⟩",
      desc: "Closed ring entanglement topology: CNOT(0→1), CNOT(1→2), CNOT(2→3), and CNOT(3→0). Generates full 4-qubit quantum state entanglement.",
    },
    MEAS_Z: {
      title: "Pauli-Z Observable ⟨Z_i⟩",
      formula: "⟨Z_i⟩ = ⟨ψ| σ_z^(i) |ψ⟩ ∈ [-1, 1]",
      desc: "Measures projection onto the computational basis {|0⟩, |1⟩}. The 4 measured values are passed to the classical Linear(4, 1) output head.",
    },
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!is3D || !interactive) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left - rect.width / 2;
    const y = e.clientY - rect.top - rect.height / 2;
    setRotY(Math.max(-25, Math.min(25, (x / rect.width) * 35)));
    setRotX(Math.max(5, Math.min(28, 16 - (y / rect.height) * 20)));
  };

  const resetView = () => {
    setRotX(16);
    setRotY(-10);
  };

  return (
    <div className="quantum-visualizer panel" style={{ overflow: "hidden", padding: "var(--space-3)" }}>
      {/* Header & Controls Toolbar */}
      <div
        className="panel-header"
        style={{
          flexWrap: "wrap",
          gap: "var(--space-2)",
          marginBottom: "var(--space-3)",
          paddingBottom: "var(--space-2)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "var(--space-2)" }}>
          <Sparkles size={16} color="var(--accent-primary)" />
          <span className="panel-title" style={{ fontSize: "var(--text-sm)" }}>
            {circuitMode === "reuploading"
              ? "4-Qubit Data Re-uploading Circuit (PennyLane default.qubit)"
              : "Standard 4-Qubit VQC (Single Encoding at t=0)"}
          </span>
        </div>

        {/* Controls */}
        <div style={{ display: "flex", alignItems: "center", gap: "var(--space-2)", flexWrap: "wrap" }}>
          {/* View Tab (Circuit vs Bloch) */}
          <div style={{ display: "inline-flex", border: "1px solid var(--border-default)" }}>
            <button
              type="button"
              className={`btn btn-sm ${viewTab === "circuit" ? "btn-primary" : ""}`}
              onClick={() => setViewTab("circuit")}
              style={{ border: "none" }}
            >
              Circuit Wireframe
            </button>
            <button
              type="button"
              className={`btn btn-sm ${viewTab === "bloch" ? "btn-primary" : ""}`}
              onClick={() => setViewTab("bloch")}
              style={{ border: "none" }}
            >
              <Orbit size={12} />
              <span>Bloch Sphere</span>
            </button>
          </div>

          {viewTab === "circuit" && (
            <>
              {/* Architecture Mode Switcher */}
              <div style={{ display: "inline-flex", border: "1px solid var(--border-default)" }}>
                <button
                  type="button"
                  className={`btn btn-sm ${circuitMode === "reuploading" ? "btn-primary" : ""}`}
                  onClick={() => setCircuitMode("reuploading")}
                  style={{ border: "none" }}
                >
                  Re-uploading
                </button>
                <button
                  type="button"
                  className={`btn btn-sm ${circuitMode === "standard_vqc" ? "btn-primary" : ""}`}
                  onClick={() => setCircuitMode("standard_vqc")}
                  style={{ border: "none" }}
                >
                  Standard VQC
                </button>
              </div>

              {/* Layer Depth Selector (L) */}
              <div style={{ display: "inline-flex", border: "1px solid var(--border-default)", alignItems: "center" }}>
                <span style={{ fontSize: "var(--text-2xs)", padding: "0 6px", color: "var(--text-muted)" }}>
                  Depth (L):
                </span>
                {([1, 2, 4] as const).map((l) => (
                  <button
                    key={l}
                    type="button"
                    className={`btn btn-sm ${layers === l ? "btn-primary" : ""}`}
                    onClick={() => setLayers(l)}
                    style={{ border: "none", minWidth: "26px", padding: "2px 6px" }}
                  >
                    {l}
                  </button>
                ))}
              </div>

              {/* 3D vs 2D Toggle */}
              <button
                type="button"
                className="btn btn-sm"
                onClick={() => setIs3D(!is3D)}
                title="Toggle 3D Perspective"
              >
                <Eye size={13} />
                {is3D ? "3D Mode" : "2D Flat"}
              </button>

              {is3D && (
                <button
                  type="button"
                  className="btn btn-sm"
                  onClick={resetView}
                  title="Reset 3D Rotation"
                >
                  <RotateCcw size={13} />
                </button>
              )}
            </>
          )}
        </div>
      </div>

      {viewTab === "bloch" ? (
        <BlochSphere3D angles={activeAngles} expvals={activeExpvals} />
      ) : (
        <>
          {/* 3D Circuit Canvas Stage */}
          <div
            onMouseMove={handleMouseMove}
            style={{
              perspective: is3D ? "1100px" : "none",
              padding: "var(--space-5) var(--space-3)",
              background: "var(--bg-code)",
              border: "1px solid var(--border-subtle)",
              position: "relative",
              cursor: is3D ? "grab" : "default",
              userSelect: "none",
              minHeight: "360px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              overflowX: "auto",
              maxWidth: "100%",
            }}
          >
            {/* 3D Transform Container */}
            <div
              style={{
                transform: is3D
                  ? `rotateX(${rotX}deg) rotateY(${rotY}deg) translateZ(10px)`
                  : "none",
                transformStyle: "preserve-3d",
                transition: "transform 140ms cubic-bezier(0.1, 0.9, 0.2, 1)",
                display: "flex",
                flexDirection: "column",
                gap: "26px",
                minWidth: layers === 4 ? "860px" : layers === 2 ? "660px" : "480px",
                position: "relative",
              }}
            >
              {/* Entanglement Ring SVG Overlay */}
              <svg
                style={{
                  position: "absolute",
                  inset: 0,
                  width: "100%",
                  height: "100%",
                  pointerEvents: "none",
                  zIndex: 3,
                }}
              >
                {/* Visual connectors for CNOT ring entanglement will be highlighted when hovered */}
              </svg>

              {/* 4 Qubit Wires */}
              {qubits.map((wire) => {
                const angleVal = activeAngles ? activeAngles[wire] : undefined;
                const expVal = activeExpvals ? activeExpvals[wire] : undefined;

                return (
                  <div
                    key={wire}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      position: "relative",
                      height: "38px",
                    }}
                  >
                    {/* Qubit Wire Label */}
                    <div
                      style={{
                        width: "80px",
                        fontFamily: "var(--font-mono)",
                        fontSize: "var(--text-xs)",
                        fontWeight: 700,
                        color: "var(--accent-primary)",
                        display: "flex",
                        alignItems: "center",
                        gap: "4px",
                        flexShrink: 0,
                      }}
                    >
                      |q_{wire}⟩ = |0⟩
                    </div>

                    {/* Wire Backbone Line */}
                    <div
                      style={{
                        position: "absolute",
                        left: "86px",
                        right: "60px",
                        height: "2px",
                        background: "var(--border-strong)",
                        zIndex: 1,
                      }}
                    />

                    {/* Gates Sequence Stream */}
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        marginLeft: "14px",
                        gap: "10px",
                        zIndex: 2,
                        flex: 1,
                      }}
                    >
                      {/* Standard VQC Single Data Encoding at t=0 */}
                      {circuitMode === "standard_vqc" && (
                        <div
                          onMouseEnter={() => setHoveredGate("RY_enc")}
                          onMouseLeave={() => setHoveredGate(null)}
                          style={{
                            padding: "4px 8px",
                            background: "var(--bg-surface-2)",
                            border: "1px solid var(--accent-primary)",
                            color: "var(--accent-primary)",
                            fontFamily: "var(--font-mono)",
                            fontSize: "var(--text-2xs)",
                            fontWeight: 700,
                            transform: is3D ? "translateZ(14px)" : "none",
                            boxShadow: is3D ? "0 4px 0 var(--border-default)" : "none",
                            cursor: "pointer",
                          }}
                        >
                          RY({angleVal !== undefined ? angleVal.toFixed(2) : `x_${wire}`})
                        </div>
                      )}

                      {/* Interleaved Layers L = 1, 2, 4 */}
                      {Array.from({ length: layers }).map((_, layerIdx) => (
                        <div
                          key={layerIdx}
                          style={{
                            display: "flex",
                            alignItems: "center",
                            gap: "8px",
                            padding: "6px 8px",
                            background: layerIdx % 2 === 0 ? "rgba(255,255,255,0.015)" : "rgba(0,0,0,0.03)",
                            border: "1px dashed var(--border-subtle)",
                            transform: is3D ? `translateZ(${6 + layerIdx * 4}px)` : "none",
                          }}
                        >
                          {/* Data Re-uploading Encoding per layer */}
                          {circuitMode === "reuploading" && (
                            <div
                              onMouseEnter={() => setHoveredGate("RY_enc")}
                              onMouseLeave={() => setHoveredGate(null)}
                              style={{
                                padding: "3px 6px",
                                background: "var(--bg-surface-3)",
                                border: "1px solid var(--accent-primary)",
                                color: "var(--accent-primary)",
                                fontFamily: "var(--font-mono)",
                                fontSize: "var(--text-2xs)",
                                fontWeight: 700,
                                transform: is3D ? "translateZ(8px)" : "none",
                                cursor: "pointer",
                              }}
                            >
                              RY({angleVal !== undefined ? angleVal.toFixed(2) : `x_${wire}`})
                            </div>
                          )}

                          {/* Trainable Rotations: RY and RZ */}
                          <div
                            onMouseEnter={() => setHoveredGate("RY_var")}
                            onMouseLeave={() => setHoveredGate(null)}
                            style={{
                              padding: "3px 6px",
                              background: "var(--bg-surface-2)",
                              border: "1px solid var(--border-strong)",
                              color: "var(--text-primary)",
                              fontFamily: "var(--font-mono)",
                              fontSize: "var(--text-2xs)",
                              fontWeight: 600,
                              cursor: "pointer",
                            }}
                          >
                            RY(θ_{layerIdx},{wire})
                          </div>

                          <div
                            onMouseEnter={() => setHoveredGate("RZ_var")}
                            onMouseLeave={() => setHoveredGate(null)}
                            style={{
                              padding: "3px 6px",
                              background: "var(--bg-surface-2)",
                              border: "1px solid var(--border-strong)",
                              color: "var(--text-primary)",
                              fontFamily: "var(--font-mono)",
                              fontSize: "var(--text-2xs)",
                              fontWeight: 600,
                              cursor: "pointer",
                            }}
                          >
                            RZ(φ_{layerIdx},{wire})
                          </div>

                          {/* Ring Entanglement CNOT Step: wire -> (wire+1)%4 */}
                          <div
                            onMouseEnter={() => setHoveredGate("CNOT")}
                            onMouseLeave={() => setHoveredGate(null)}
                            style={{
                              display: "inline-flex",
                              alignItems: "center",
                              gap: "3px",
                              padding: "2px 5px",
                              background: "var(--bg-surface-1)",
                              border: "1px solid var(--status-attack-border)",
                              color: "var(--status-attack)",
                              fontFamily: "var(--font-mono)",
                              fontSize: "9px",
                              fontWeight: 700,
                              cursor: "pointer",
                            }}
                            title={`Ring Entanglement: CNOT(${wire} → ${(wire + 1) % 4})`}
                          >
                            <span>●</span>
                            <span style={{ fontSize: "8px" }}>→</span>
                            <span>⊕</span>
                            <span style={{ opacity: 0.7 }}>{(wire + 1) % 4}</span>
                          </div>
                        </div>
                      ))}

                      {/* Pauli-Z Measurement */}
                      <div
                        onMouseEnter={() => setHoveredGate("MEAS_Z")}
                        onMouseLeave={() => setHoveredGate(null)}
                        style={{
                          marginLeft: "auto",
                          padding: "4px 8px",
                          background: "var(--status-benign-bg)",
                          border: "1px solid var(--status-benign-border)",
                          color: "var(--status-benign)",
                          fontFamily: "var(--font-mono)",
                          fontSize: "var(--text-2xs)",
                          fontWeight: 700,
                          transform: is3D ? "translateZ(16px)" : "none",
                          cursor: "pointer",
                          whiteSpace: "nowrap",
                        }}
                      >
                        ⟨Z_{wire}⟩ {expVal !== undefined ? `= ${expVal.toFixed(3)}` : ""}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Floating indicator */}
            {is3D && (
              <div
                style={{
                  position: "absolute",
                  bottom: "8px",
                  right: "10px",
                  fontSize: "10px",
                  color: "var(--text-muted)",
                  fontFamily: "var(--font-mono)",
                  background: "var(--bg-surface-1)",
                  padding: "2px 6px",
                  border: "1px solid var(--border-subtle)",
                  pointerEvents: "none",
                }}
              >
                Pointer Tilt Active
              </div>
            )}
          </div>

          {/* Dynamic Unitary Gate Inspector */}
          <div
            style={{
              marginTop: "var(--space-3)",
              padding: "var(--space-3)",
              background: "var(--bg-surface-2)",
              border: "1px solid var(--border-subtle)",
              fontSize: "var(--text-xs)",
              minHeight: "56px",
              display: "flex",
              alignItems: "center",
              gap: "var(--space-3)",
            }}
          >
            <Info size={16} color="var(--accent-primary)" style={{ flexShrink: 0 }} />
            {hoveredGate && gateDescriptions[hoveredGate] ? (
              <div>
                <strong style={{ color: "var(--text-primary)" }}>
                  {gateDescriptions[hoveredGate].title}
                </strong>{" "}
                <code style={{ marginLeft: "6px", color: "var(--accent-primary)" }}>
                  {gateDescriptions[hoveredGate].formula}
                </code>
                <p style={{ margin: "2px 0 0", color: "var(--text-secondary)" }}>
                  {gateDescriptions[hoveredGate].desc}
                </p>
              </div>
            ) : (
              <span style={{ color: "var(--text-muted)" }}>
                Hover over any quantum gate (RY encoding, RY variational, RZ phase, CNOT ring link, or ⟨Z⟩ observable) to inspect mathematical formulation.
              </span>
            )}
          </div>
        </>
      )}
    </div>
  );
};
