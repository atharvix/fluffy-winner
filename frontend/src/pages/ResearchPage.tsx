import React from "react";
import { useLanguage } from "../i18n/useTranslation";
import { 
  Download, 
  AlertTriangle, 
  CheckCircle,
  FileText
} from "lucide-react";

export const ResearchPage: React.FC = () => {
  const { t, lang } = useLanguage();

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-6)" }}>
      {/* Header */}
      <section className="panel" style={{ padding: "var(--space-5)" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "var(--space-4)" }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "var(--space-2)", marginBottom: "var(--space-2)" }}>
              <span className="badge badge-accent">Formal Research Monograph</span>
              <span className="badge">QML & Tabular NIDS</span>
              <span className="badge badge-success">Peer-Review Standard</span>
            </div>
            <h1 style={{ fontSize: "var(--text-2xl)", fontWeight: 700, margin: "0 0 var(--space-2) 0", letterSpacing: "-0.02em" }}>
              {t.researchTitle}
            </h1>
            <p style={{ color: "var(--text-secondary)", maxWidth: "800px", margin: 0 }}>
              {t.researchSubtitle}
            </p>
          </div>

          <div style={{ display: "flex", gap: "var(--space-2)", flexWrap: "wrap" }}>
            <a 
              href="/reports/nids_research_report.md" 
              download="nids_research_report.md"
              className="btn btn-primary btn-sm"
              style={{ textDecoration: "none" }}
            >
              <Download size={14} />
              <span>{t.downloadPaperBtn}</span>
            </a>
            <a 
              href="/data/summary_statistics.json" 
              download="nids_summary_statistics.json"
              className="btn btn-secondary btn-sm"
              style={{ textDecoration: "none" }}
            >
              <FileText size={14} />
              <span>{t.downloadStatsBtn}</span>
            </a>
          </div>
        </div>
      </section>

      {/* Simulator Hardware & Scientific Integrity Disclaimer */}
      <section 
        className="panel" 
        style={{ 
          padding: "var(--space-4)", 
          borderLeft: "4px solid var(--status-warning)",
          backgroundColor: "var(--bg-surface)"
        }}
      >
        <div style={{ display: "flex", alignItems: "flex-start", gap: "var(--space-3)" }}>
          <AlertTriangle size={20} style={{ color: "var(--status-warning)", marginTop: "2px", flexShrink: 0 }} />
          <div>
            <div style={{ fontWeight: 700, color: "var(--text-primary)", marginBottom: "var(--space-1)" }}>
              {t.hardwareDisclaimerTitle}
            </div>
            <p style={{ fontSize: "var(--text-sm)", color: "var(--text-secondary)", margin: 0, lineHeight: 1.5 }}>
              {t.hardwareDisclaimerBody}
            </p>
            <div style={{ marginTop: "var(--space-2)", fontSize: "var(--text-xs)", color: "var(--text-muted)" }}>
              • No physical NISQ hardware utilized • Zero quantum decoherence or readout noise • No claim of quantum advantage or supremacy
            </div>
          </div>
        </div>
      </section>

      {/* Two Column Publication Layout */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: "var(--space-6)" }}>
        
        {/* Left Column: Theoretical Foundations */}
        <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-6)" }}>
          
          {/* Section 1: Problem Formulation */}
          <article className="panel" style={{ padding: "var(--space-5)" }}>
            <h2 style={{ fontSize: "var(--text-base)", fontWeight: 700, margin: "0 0 var(--space-3) 0", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--accent-primary)" }}>
              1. {lang === "hi" ? "समस्या विवरण एवं संदर्भ" : "Problem Statement & Cybersecurity Motivation"}
            </h2>
            <div style={{ fontSize: "var(--text-sm)", color: "var(--text-secondary)", lineHeight: 1.6, display: "flex", flexDirection: "column", gap: "var(--space-3)" }}>
              <p>
                Network Intrusion Detection Systems (NIDS) are critical for enterprise security perimeters, monitoring network flow transactions, identifying unauthorized access, and mitigating malicious traffic. Modern enterprise networks process millions of flows per minute, imposing two stringent operational constraints:
              </p>
              <ul style={{ paddingLeft: "var(--space-4)", display: "flex", flexDirection: "column", gap: "var(--space-1)" }}>
                <li>
                  <strong style={{ color: "var(--text-primary)" }}>High Detection Rate (Attack Recall):</strong> Missed intrusions (False Negatives) allow lateral adversary movement, data exfiltration, or ransomware execution.
                </li>
                <li>
                  <strong style={{ color: "var(--text-primary)" }}>Low False Alarm Rate (False Positive Rate):</strong> Flagging benign transactions overwhelms Security Operations Center (SOC) analysts with alerts, leading to alert fatigue.
                </li>
              </ul>
              <p>
                Tabular network flow datasets exhibit complex, non-linear boundaries. This research formally evaluates whether Variational Quantum Circuits (VQCs) enhanced by <em>Quantum Data Re-uploading</em> provide competitive decision boundary representations with compact parameter footprints.
              </p>
            </div>
          </article>

          {/* Section 2: Fourier Frequency Expressivity */}
          <article className="panel" style={{ padding: "var(--space-5)" }}>
            <h2 style={{ fontSize: "var(--text-base)", fontWeight: 700, margin: "0 0 var(--space-3) 0", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--accent-primary)" }}>
              2. {lang === "hi" ? "क्वांटम डेटा री-अपलोडिंग एवं फूरियर सिद्धांत" : "Quantum Data Re-uploading & Fourier Spectrum"}
            </h2>
            <div style={{ fontSize: "var(--text-sm)", color: "var(--text-secondary)", lineHeight: 1.6, display: "flex", flexDirection: "column", gap: "var(--space-3)" }}>
              <p>
                Schuld et al. (2021) and Vidal et al. proved that standard variational quantum circuits with a single data-encoding unitary gate \(E(x)\) at the beginning of the circuit represent truncated Fourier series of degree 1:
              </p>
              <div 
                style={{ 
                  backgroundColor: "var(--bg-code)", 
                  padding: "var(--space-3)", 
                  borderRadius: "var(--radius-sm)", 
                  border: "1px solid var(--border-subtle)",
                  fontFamily: "var(--font-mono)",
                  fontSize: "var(--text-xs)",
                  color: "var(--text-primary)",
                  textAlign: "center"
                }}
              >
                f(x) = ∑ c_n · e^(i · n · x), where spectrum Ω is strictly bounded by encoding frequency degree.
              </div>
              <p>
                <strong>Quantum Data Re-uploading</strong> (Pérez-Salinas et al., 2020) solves this limitation by interleaving classical feature rotations \(R_y(x_i)\) between parameterized variational ansatz layers \(W(\theta_l)\):
              </p>
              <div 
                style={{ 
                  backgroundColor: "var(--bg-code)", 
                  padding: "var(--space-3)", 
                  borderRadius: "var(--radius-sm)", 
                  border: "1px solid var(--border-subtle)",
                  fontFamily: "var(--font-mono)",
                  fontSize: "var(--text-xs)",
                  color: "var(--text-primary)",
                  textAlign: "center"
                }}
              >
                {"U(x, θ) = ∏_{l=1}^L [ W_l(θ) · S_l(x) ]"}
              </div>
              <p>
                Each re-uploading layer expands the accessible harmonic spectrum \(\Omega_L\), enabling higher-frequency functional representations and complex decision surfaces without requiring physical qubit expansion.
              </p>
            </div>
          </article>

          {/* Section 3: Dataset Provenance & Preprocessing */}
          <article className="panel" style={{ padding: "var(--space-5)" }}>
            <h2 style={{ fontSize: "var(--text-base)", fontWeight: 700, margin: "0 0 var(--space-3) 0", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--accent-primary)" }}>
              3. {lang === "hi" ? "डेटासेट स्रोत एवं शून्य-लीकेज पाइपलाइन" : "Dataset Provenance & Zero-Leakage Pipeline"}
            </h2>
            <div style={{ fontSize: "var(--text-sm)", color: "var(--text-secondary)", lineHeight: 1.6, display: "flex", flexDirection: "column", gap: "var(--space-3)" }}>
              <p>
                The research utilizes the authentic <strong>UNSW-NB15</strong> benchmark created by Nour Moustafa and Jill Slay at the Australian Centre for Cyber Security (ACCS).
              </p>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "var(--space-2)", fontSize: "var(--text-xs)" }}>
                <div className="panel" style={{ padding: "var(--space-2)" }}>
                  <div style={{ fontWeight: 600, color: "var(--text-primary)" }}>Total Evaluated Split</div>
                  <div>4,200 Train / 900 Val / 900 Test</div>
                </div>
                <div className="panel" style={{ padding: "var(--space-2)" }}>
                  <div style={{ fontWeight: 600, color: "var(--text-primary)" }}>Flow Features</div>
                  <div>39 Numeric + 3 Categorical</div>
                </div>
              </div>
              <p>
                <strong>Zero-Leakage Guarantee:</strong> Scikit-Learn's <code>ColumnTransformer</code> is fitted exclusively on the 70% training split. One-Hot Encoding applies <code>handle_unknown='ignore'</code> with <code>max_categories=10</code> for <code>proto</code>, <code>service</code>, and <code>state</code>. Target and category columns (<code>id</code>, <code>label</code>, <code>attack_cat</code>) are completely isolated from feature extraction.
              </p>
            </div>
          </article>
        </div>

        {/* Right Column: Empirical Findings & Operations */}
        <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-6)" }}>
          
          {/* Section 4: Controlled Findings */}
          <article className="panel" style={{ padding: "var(--space-5)" }}>
            <h2 style={{ fontSize: "var(--text-base)", fontWeight: 700, margin: "0 0 var(--space-3) 0", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--accent-primary)" }}>
              4. {lang === "hi" ? "मुख्य अनुभवजन्य निष्कर्ष" : "Core Empirical Findings"}
            </h2>
            <div style={{ fontSize: "var(--text-sm)", color: "var(--text-secondary)", lineHeight: 1.6, display: "flex", flexDirection: "column", gap: "var(--space-3)" }}>
              <div style={{ display: "flex", alignItems: "flex-start", gap: "var(--space-2)" }}>
                <CheckCircle size={16} style={{ color: "var(--status-success)", marginTop: "2px", flexShrink: 0 }} />
                <div>
                  <strong style={{ color: "var(--text-primary)" }}>Accuracy Parity at 82.6% Parameter Reduction:</strong> Hybrid Re-uploading L=2 achieved 93.33% ± 0.19% test accuracy with 1,113 parameters, matching the 6,401-parameter Classical MLP (93.19% ± 0.65%).
                </div>
              </div>

              <div style={{ display: "flex", alignItems: "flex-start", gap: "var(--space-2)" }}>
                <CheckCircle size={16} style={{ color: "var(--status-success)", marginTop: "2px", flexShrink: 0 }} />
                <div>
                  <strong style={{ color: "var(--text-primary)" }}>Operational Noise Reduction in SOC:</strong> Hybrid L=2 achieved the lowest False Positive Rate (5.19% ± 0.86%), preventing approximately 9,000 false alarms per million processed flows compared to Classical MLP (6.09%).
                </div>
              </div>

              <div style={{ display: "flex", alignItems: "flex-start", gap: "var(--space-2)" }}>
                <CheckCircle size={16} style={{ color: "var(--status-success)", marginTop: "2px", flexShrink: 0 }} />
                <div>
                  <strong style={{ color: "var(--text-primary)" }}>4.5× Faster Loss Convergence:</strong> Interleaving re-uploading reduced training duration from 161.4s (Standard VQC L=2) to 35.8s (Hybrid Re-uploading L=2) on PennyLane CPU statevector simulation.
                </div>
              </div>

              <div style={{ display: "flex", alignItems: "flex-start", gap: "var(--space-2)" }}>
                <CheckCircle size={16} style={{ color: "var(--status-warning)", marginTop: "2px", flexShrink: 0 }} />
                <div>
                  <strong style={{ color: "var(--text-primary)" }}>Depth Ablation Limit (L=4):</strong> Scaling circuit depth to L=4 increased variance and reduced test accuracy (92.41% ± 1.00%), demonstrating that overparameterizing variational circuits without regularized optimization impairs generalization.
                </div>
              </div>
            </div>
          </article>

          {/* Section 5: Operational Relevance in SOC */}
          <article className="panel" style={{ padding: "var(--space-5)" }}>
            <h2 style={{ fontSize: "var(--text-base)", fontWeight: 700, margin: "0 0 var(--space-3) 0", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--accent-primary)" }}>
              5. {lang === "hi" ? "सुरक्षा संचालन (SOC) प्रासंगिकता" : "Operational SOC Deployment Context"}
            </h2>
            <div style={{ fontSize: "var(--text-sm)", color: "var(--text-secondary)", lineHeight: 1.6, display: "flex", flexDirection: "column", gap: "var(--space-3)" }}>
              <p>
                In production network monitoring, the trade-off between sensitivity and alert volume is critical:
              </p>
              
              <div className="table-scroll-container">
                <table className="table" style={{ fontSize: "var(--text-xs)" }}>
                  <thead>
                    <tr>
                      <th>Model Architecture</th>
                      <th>Attack Recall</th>
                      <th>False Positive Rate</th>
                      <th>Operational Profile</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td style={{ fontWeight: 600 }}>Hybrid Re-uploading L=2</td>
                      <td style={{ color: "var(--status-success)" }}>92.12%</td>
                      <td style={{ color: "var(--status-success)", fontWeight: 700 }}>5.19% (Lowest)</td>
                      <td>Balanced SOC triage; minimized analyst alert fatigue</td>
                    </tr>
                    <tr>
                      <td style={{ fontWeight: 600 }}>Hybrid Re-uploading L=1</td>
                      <td style={{ color: "var(--status-success)", fontWeight: 700 }}>93.13% (Highest)</td>
                      <td style={{ color: "var(--status-warning)" }}>7.57%</td>
                      <td>High-sensitivity boundary; catches stealthy exploits</td>
                    </tr>
                    <tr>
                      <td style={{ fontWeight: 600 }}>Classical MLP Baseline</td>
                      <td>92.59%</td>
                      <td>6.09%</td>
                      <td>Traditional baseline; high parameter footprint (6,401)</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </article>

          {/* Section 6: Responsible Use & Limitations */}
          <article className="panel" style={{ padding: "var(--space-5)" }}>
            <h2 style={{ fontSize: "var(--text-base)", fontWeight: 700, margin: "0 0 var(--space-3) 0", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--status-warning)" }}>
              6. {lang === "hi" ? "सीमाएं एवं उत्तरदायी उपयोग" : "Limitations & Responsible Use Policy"}
            </h2>
            <div style={{ fontSize: "var(--text-sm)", color: "var(--text-secondary)", lineHeight: 1.6, display: "flex", flexDirection: "column", gap: "var(--space-2)" }}>
              <p>
                <strong>Decision-Support Only:</strong> Predictions from this system serve as prioritized telemetry for human Tier-1/Tier-2 SOC analysts. They must not trigger automated, irreversible border gateway route withdrawals or port quarentines without analyst verification.
              </p>
              <p>
                <strong>Evaluation Boundary:</strong> Experiments reflect the UNSW-NB15 dataset distribution. Deployment in live software-defined networks (SDN) requires ongoing retraining against zero-day packet flow dynamics.
              </p>
            </div>
          </article>
        </div>
      </div>
    </div>
  );
};
