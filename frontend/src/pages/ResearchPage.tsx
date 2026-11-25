import React from "react";
import {
  AlertTriangle,
  CheckCircle,
  BookOpen
} from "lucide-react";

export const ResearchPage: React.FC = () => {

  return (
    <div style={{ display: "flex", flexDirection: "column", minHeight: "100%", width: "100%" }}>
      {/* 1. Technical Masthead */}
      <header className="masthead">
        <div>
          <div className="masthead-eyebrow">
            <BookOpen size={12} color="var(--accent-primary)" />
            <span>05 // ARCHITECTURE SPECIFICATION & THEORETICAL FOUNDATION</span>
          </div>
          <h1 className="masthead-title">Variational Quantum Circuit Architectural Specification</h1>
        </div>
        <div className="masthead-meta">
          <a
            href="/presentation/index.html"
            target="_blank"
            rel="noopener noreferrer"
            className="btn btn-primary btn-sm"
            style={{ textDecoration: "none", color: "#000", fontWeight: 700 }}
          >
            OPEN SLIDE DECK [HTML]
          </a>
          <span>&bull;</span>
          <span>FOURIER DEGREE: EXPANDED (L LAYERS)</span>
          <span>&bull;</span>
          <span>DATASET: CIC-IoT-2023 45M</span>
          <span>&bull;</span>
          <span>SIMULATOR: PENNYLANE 0.40</span>
        </div>
      </header>

      {/* 2. Continuous Horizontal Telemetry Strip */}
      <div className="telemetry-strip">
        <div className="telemetry-item">
          <span className="telemetry-label">THEORETICAL FRAMEWORK</span>
          <span className="telemetry-val" style={{ color: "var(--accent-quantum)" }}>FOURIER SPECTRUM RE-UPLOADING</span>
        </div>
        <span className="telemetry-divider">|</span>
        <div className="telemetry-item">
          <span className="telemetry-label">LATENT ANSATZ</span>
          <span className="telemetry-val">4-QUBIT STRONGLY ENTANGLING</span>
        </div>
        <span className="telemetry-divider">|</span>
        <div className="telemetry-item">
          <span className="telemetry-label">OPTIMAL DECISION BOUNDARY</span>
          <span className="telemetry-val" style={{ color: "var(--accent-primary)" }}>&tau;* = 0.2971 (YOUDEN J)</span>
        </div>
        <span className="telemetry-divider">|</span>
        <div className="telemetry-item">
          <span className="telemetry-label">SCIENTIFIC INTEGRITY</span>
          <span className="telemetry-val" style={{ color: "var(--status-benign)" }}>STATEVECTOR CPU/CUDA SIMULATION</span>
        </div>
      </div>

      <div style={{ padding: "1.25rem", display: "flex", flexDirection: "column", gap: "1.25rem" }}>
        {/* Simulator Hardware & Scientific Integrity Banner */}
        <section
          style={{
            padding: "0.75rem 1rem",
            borderLeft: "3px solid var(--accent-primary)",
            background: "var(--bg-surface-2)",
            border: "1px solid var(--border-default)",
            borderLeftWidth: "3px",
            borderLeftColor: "var(--accent-primary)",
            display: "flex",
            alignItems: "flex-start",
            gap: "0.75rem",
          }}
        >
          <AlertTriangle size={16} color="var(--accent-primary)" style={{ flexShrink: 0, marginTop: "2px" }} />
          <div>
            <div style={{ fontFamily: "var(--font-heading)", fontSize: "12px", fontWeight: 700, color: "var(--text-primary)" }}>
              SCIENTIFIC INTEGRITY & QUANTUM SIMULATION NOTICE
            </div>
            <p style={{ fontFamily: "var(--font-mono)", fontSize: "11px", color: "var(--text-secondary)", margin: "4px 0 0", lineHeight: 1.5 }}>
              All quantum models are simulated via PennyLane's <code>default.qubit</code> exact statevector engine. No physical NISQ hardware was utilized, eliminating decoherence noise. This system makes no claim of quantum supremacy or physical quantum speedup over classical backpropagation.
            </p>
          </div>
        </section>

        {/* Two Column Publication Monograph Layout */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.25rem" }}>
          {/* Left Column: Theory & Mathematics */}
          <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
            {/* Section 1 */}
            <article className="panel" style={{ padding: "1rem" }}>
              <div style={{ fontFamily: "var(--font-heading)", fontSize: "13px", fontWeight: 700, color: "var(--accent-primary)", marginBottom: "0.5rem" }}>
                01 // PROBLEM FORMULATION & IOT THREAT LANDSCAPE
              </div>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "11px", color: "var(--text-secondary)", lineHeight: 1.6, display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                <p>
                  Modern Internet of Things (IoT) ecosystems face relentless distributed denial-of-service (DDoS), Mirai botnet propagation, and reconnaissance sweeps. The CIC-IoT-2023 benchmark comprises 45,019,243 recorded network transactions exhibiting extreme class imbalance (~97.6% malicious flood flows vs ~2.4% benign sensor telemetry).
                </p>
                <p>
                  Standard thresholding at \(\tau = 0.50\) leads to catastrophic false alarms or collapsed benign detection. This study calibrates an optimal boundary \(\tau^* = 0.2971\) on training validation splits to maximize Youden's J statistic:
                </p>
                <div style={{ background: "var(--bg-code)", padding: "8px", border: "1px solid var(--border-subtle)", color: "var(--text-primary)" }}>
                  {"J(\u03C4) = Sensitivity(\u03C4) + Specificity(\u03C4) - 1"}
                </div>
              </div>
            </article>

            {/* Section 2 */}
            <article className="panel" style={{ padding: "1rem" }}>
              <div style={{ fontFamily: "var(--font-heading)", fontSize: "13px", fontWeight: 700, color: "var(--accent-primary)", marginBottom: "0.5rem" }}>
                02 // DATA RE-UPLOADING & FOURIER SPECTRUM
              </div>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "11px", color: "var(--text-secondary)", lineHeight: 1.6, display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                <p>
                  Schuld et al. (2021) established that single-encoding quantum circuits U(x) = W &middot; S(x) can only express truncated Fourier series of degree 1. Quantum Data Re-uploading overcomes this limitation by interleaving parameterized unitary rotations W<sub>l</sub>(&theta;) with data encoding operations S<sub>l</sub>(x):
                </p>
                <div style={{ background: "var(--bg-code)", padding: "8px", border: "1px solid var(--border-subtle)", color: "var(--accent-quantum)" }}>
                  {"U(x, \u03B8) = \u220F_{l=1}^L [ W_l(\u03B8) \u00B7 S_l(x) ]"}
                </div>
                <p>
                  Each re-uploading layer expands the accessible harmonic spectrum &Omega;<sub>L</sub> = {"{-L, ..., +L}"}, permitting high-frequency decision boundaries without expanding physical qubit count.
                </p>
              </div>
            </article>

            {/* Section 3 */}
            <article className="panel" style={{ padding: "1rem" }}>
              <div style={{ fontFamily: "var(--font-heading)", fontSize: "13px", fontWeight: 700, color: "var(--accent-primary)", marginBottom: "0.5rem" }}>
                03 // DATA PIPELINE & ZERO-LEAKAGE NORMALIZATION
              </div>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "11px", color: "var(--text-secondary)", lineHeight: 1.6, display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                <p>
                  All 39 numerical features (packet lengths, flow rates, TCP flags, protocol distribution indicators) are preprocessed via Scikit-Learn's <code>StandardScaler</code>. To guarantee zero test leakage, the mean and variance parameters are fitted strictly on training shards.
                </p>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px", marginTop: "4px" }}>
                  <div style={{ background: "var(--bg-code)", padding: "8px", border: "1px solid var(--border-subtle)" }}>
                    <div style={{ color: "var(--text-muted)", fontSize: "10px" }}>INPUT DIMENSION</div>
                    <div style={{ color: "var(--text-primary)", fontWeight: 700 }}>39 RAW FLOW METRICS</div>
                  </div>
                  <div style={{ background: "var(--bg-code)", padding: "8px", border: "1px solid var(--border-subtle)" }}>
                    <div style={{ color: "var(--text-muted)", fontSize: "10px" }}>LATENT REGISTER</div>
                    <div style={{ color: "var(--accent-quantum)", fontWeight: 700 }}>4 PARAMETERIZED QUBITS</div>
                  </div>
                </div>
              </div>
            </article>
          </div>

          {/* Right Column: Empirical Discoveries & Operations */}
          <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
            {/* Section 4 */}
            <article className="panel" style={{ padding: "1rem" }}>
              <div style={{ fontFamily: "var(--font-heading)", fontSize: "13px", fontWeight: 700, color: "var(--accent-primary)", marginBottom: "0.5rem" }}>
                04 // CORE EMPIRICAL DISCOVERIES
              </div>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "11px", color: "var(--text-secondary)", lineHeight: 1.6, display: "flex", flexDirection: "column", gap: "0.6rem" }}>
                <div style={{ display: "flex", gap: "8px" }}>
                  <CheckCircle size={14} color="var(--status-benign)" style={{ flexShrink: 0, marginTop: "2px" }} />
                  <div>
                    <strong style={{ color: "var(--text-primary)" }}>Parameter Parsimony (-85.2%):</strong> Hybrid Re-uploading L=1 matches and slightly outperforms the 4,865-parameter Classical MLP (99.28% vs 99.12% accuracy) with only 721 parameters.
                  </div>
                </div>

                <div style={{ display: "flex", gap: "8px" }}>
                  <CheckCircle size={14} color="var(--status-benign)" style={{ flexShrink: 0, marginTop: "2px" }} />
                  <div>
                    <strong style={{ color: "var(--text-primary)" }}>Bottleneck Control Verification:</strong> The Classical Bottleneck MLP (713 parameters, 98.94% accuracy) demonstrates that classical dimensionality reduction (39 &rarr; 16 &rarr; 4) isolates the principal manifold, while the quantum circuit provides complementary non-linear discrimination.
                  </div>
                </div>

                <div style={{ display: "flex", gap: "8px" }}>
                  <CheckCircle size={14} color="var(--status-benign)" style={{ flexShrink: 0, marginTop: "2px" }} />
                  <div>
                    <strong style={{ color: "var(--text-primary)" }}>False Positive Rate Suppression:</strong> With optimal thresholding \(\tau^* = 0.2971\), False Positive Rate is constrained to 0.89%, providing an operational ~20% reduction in false alarms over classical baseline (1.12%).
                  </div>
                </div>
              </div>
            </article>

            {/* Section 5 */}
            <article className="panel" style={{ padding: "1rem" }}>
              <div style={{ fontFamily: "var(--font-heading)", fontSize: "13px", fontWeight: 700, color: "var(--accent-primary)", marginBottom: "0.5rem" }}>
                05 // SECURITY OPERATIONS CENTER (SOC) WORKFLOW
              </div>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "11px", color: "var(--text-secondary)", lineHeight: 1.6, display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                <p>
                  In high-throughput enterprise monitoring, raw alerts must be filtered by confidence margin. The table below illustrates the classification regime across all 6 architectures:
                </p>

                <div className="table-container" style={{ margin: "4px 0", border: "1px solid var(--border-default)" }}>
                  <table>
                    <thead>
                      <tr>
                        <th>MODEL</th>
                        <th>PARAMS</th>
                        <th>RECALL</th>
                        <th>FPR</th>
                        <th>TAU*</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td style={{ fontWeight: 600 }}>Classical MLP Baseline</td>
                        <td>4,865</td>
                        <td>99.31%</td>
                        <td>1.12%</td>
                        <td>0.3850</td>
                      </tr>
                      <tr>
                        <td style={{ fontWeight: 600 }}>Classical Bottleneck MLP</td>
                        <td>713</td>
                        <td>98.89%</td>
                        <td>1.35%</td>
                        <td>0.3420</td>
                      </tr>
                      <tr style={{ background: "var(--bg-surface-2)" }}>
                        <td style={{ fontWeight: 700, color: "var(--accent-primary)" }}>Hybrid Re-uploading L=1 [ACTIVE]</td>
                        <td style={{ fontWeight: 700 }}>721</td>
                        <td style={{ color: "var(--status-benign)" }}>99.41%</td>
                        <td style={{ color: "var(--status-benign)" }}>0.89%</td>
                        <td>0.2971</td>
                      </tr>
                      <tr>
                        <td style={{ fontWeight: 600 }}>Hybrid Re-uploading L=2</td>
                        <td>733</td>
                        <td>99.48%</td>
                        <td>0.82%</td>
                        <td>0.2910</td>
                      </tr>
                      <tr>
                        <td style={{ fontWeight: 600 }}>Standard VQC L=2</td>
                        <td>729</td>
                        <td>98.71%</td>
                        <td>1.48%</td>
                        <td>0.3250</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </article>

            {/* Section 6 */}
            <article className="panel" style={{ padding: "1rem" }}>
              <div style={{ fontFamily: "var(--font-heading)", fontSize: "13px", fontWeight: 700, color: "var(--status-warn)", marginBottom: "0.5rem" }}>
                06 // RESPONSIBLE USE & LIMITATIONS
              </div>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "11px", color: "var(--text-secondary)", lineHeight: 1.5, display: "flex", flexDirection: "column", gap: "0.4rem" }}>
                <p>
                  <strong>Human-in-the-Loop Requirement:</strong> Predictions generated by this quantum-classical pipeline serve strictly as decision support telemetry for SOC analysts. Automated border gateway protocol (BGP) routing withdrawals or host isolations should never occur without human verification.
                </p>
                <p>
                  <strong>Continuous Adaptation:</strong> While evaluated across 45M flows from 33 distinct attack vectors, zero-day threat variants require periodic streaming retraining to maintain calibrated threshold bounds.
                </p>
              </div>
            </article>
          </div>
        </div>
      </div>
    </div>
  );
};
