import React, { useState } from "react";
import { useLanguage } from "../i18n/useTranslation";
import { 
  Info, 
  Download, 
  CheckCircle, 
  TrendingUp, 
  Sliders, 
  Maximize2 
} from "lucide-react";

interface ModelStat {
  name: string;
  key: string;
  family: string;
  params: number;
  accuracy: string;
  balancedAcc: string;
  f1: string;
  attackRecall: string;
  fpr: string;
  rocAuc: string;
  cpuTrainingDuration: string;
  isDeployed?: boolean;
}

const EXPERIMENT_MODELS: ModelStat[] = [
  {
    name: "Hybrid Re-uploading (L=2)",
    key: "Hybrid_Reuploading_L2",
    family: "Hybrid Quantum",
    params: 1113,
    accuracy: "93.33% ± 0.19%",
    balancedAcc: "93.47% ± 0.15%",
    f1: "0.9329 ± 0.0019",
    attackRecall: "92.12% ± 0.93%",
    fpr: "5.19% ± 0.86%",
    rocAuc: "0.9835 ± 0.0013",
    cpuTrainingDuration: "35.78s ± 0.10s",
    isDeployed: true,
  },
  {
    name: "Classical MLP Baseline",
    key: "Classical_MLP",
    family: "Classical",
    params: 6401,
    accuracy: "93.19% ± 0.65%",
    balancedAcc: "93.25% ± 0.59%",
    f1: "0.9313 ± 0.0065",
    attackRecall: "92.59% ± 1.72%",
    fpr: "6.09% ± 1.36%",
    rocAuc: "0.9844 ± 0.0008",
    cpuTrainingDuration: "27.06s ± 11.21s",
    isDeployed: false,
  },
  {
    name: "Standard VQC (L=2)",
    key: "Standard_VQC_L2",
    family: "Variational Quantum",
    params: 1113,
    accuracy: "93.00% ± 0.33%",
    balancedAcc: "93.08% ± 0.31%",
    f1: "0.9295 ± 0.0033",
    attackRecall: "92.26% ± 0.76%",
    fpr: "6.09% ± 0.57%",
    rocAuc: "0.9835 ± 0.0007",
    cpuTrainingDuration: "161.42s ± 4.30s",
    isDeployed: false,
  },
  {
    name: "Hybrid Re-uploading (L=1)",
    key: "Hybrid_Reuploading_L1",
    family: "Hybrid Quantum",
    params: 1105,
    accuracy: "92.81% ± 0.28%",
    balancedAcc: "92.78% ± 0.23%",
    f1: "0.9275 ± 0.0027",
    attackRecall: "93.13% ± 1.05%",
    fpr: "7.57% ± 0.87%",
    rocAuc: "0.9834 ± 0.0014",
    cpuTrainingDuration: "30.31s ± 13.37s",
    isDeployed: false,
  },
  {
    name: "Hybrid Re-uploading (L=4)",
    key: "Hybrid_Reuploading_L4",
    family: "Hybrid Quantum",
    params: 1129,
    accuracy: "92.41% ± 1.00%",
    balancedAcc: "92.42% ± 0.96%",
    f1: "0.9234 ± 0.0100",
    attackRecall: "92.26% ± 1.35%",
    fpr: "7.41% ± 0.74%",
    rocAuc: "0.9824 ± 0.0017",
    cpuTrainingDuration: "63.25s ± 0.81s",
    isDeployed: false,
  },
];

const PLOTS = [
  {
    id: "acc_f1",
    title: "Accuracy & Macro F1 by Architecture",
    description: "Evaluated across seeds 42, 123, 999. Hybrid L=2 achieves 93.33% accuracy, slightly outperforming Classical MLP with 82.6% fewer parameters.",
    src: "/plots/accuracy_macro_f1_by_model.png",
    category: "Performance"
  },
  {
    id: "recall_fpr",
    title: "Attack Detection Rate vs. False Positive Rate",
    description: "Hybrid Re-uploading L=2 achieves the lowest False Positive Rate (5.19%), providing crucial operational noise reduction for SOC analysts.",
    src: "/plots/attack_recall_fpr_by_model.png",
    category: "Operational"
  },
  {
    id: "roc_curves",
    title: "ROC Curves & AUC Benchmark",
    description: "Mean ROC curves across test folds. All architectures demonstrate high separability (ROC-AUC > 0.982) on the preprocessed 42-feature manifold.",
    src: "/plots/roc_curves_all_models.png",
    category: "Diagnostic"
  },
  {
    id: "confusion_matrices",
    title: "Aggregated Test Confusion Matrices",
    description: "Hold-out test classification distribution across benign and attack traffic for each architecture under seed 42.",
    src: "/plots/confusion_matrices_summary.png",
    category: "Diagnostic"
  },
  {
    id: "params_complexity",
    title: "Parameter Efficiency & Complexity",
    description: "Demonstrates parameter economy: Hybrid models operate with 1,105 to 1,129 parameters compared to 6,401 parameters for classical MLP.",
    src: "/plots/parameter_counts_circuit_complexity.png",
    category: "Complexity"
  },
  {
    id: "layer_ablation",
    title: "Quantum Re-uploading Layer Depth Ablation (L=1, 2, 4)",
    description: "Empirical ablation of re-uploading depth L. Shows L=2 achieves optimal regularization balance; L=4 exhibits slight overparameterization on small simulated circuits.",
    src: "/plots/reuploading_layers_ablation.png",
    category: "Ablation"
  },
  {
    id: "training_time",
    title: "CPU Statevector Simulation Duration",
    description: "PennyLane default.qubit CPU simulator runtime per training session. Illustrates simulator scaling across layers and classical versus quantum training overhead.",
    src: "/plots/training_time_by_model.png",
    category: "Simulation"
  }
];

export const ExperimentsPage: React.FC = () => {
  const { t, lang } = useLanguage();
  const [activeCategory, setActiveCategory] = useState<string>("All");
  const [selectedPlot, setSelectedPlot] = useState<typeof PLOTS[0] | null>(null);

  const categories = ["All", "Performance", "Operational", "Diagnostic", "Complexity", "Ablation", "Simulation"];

  const filteredPlots = activeCategory === "All" 
    ? PLOTS 
    : PLOTS.filter(p => p.category === activeCategory);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-6)" }}>
      {/* Page Header */}
      <section className="panel" style={{ padding: "var(--space-5)" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "var(--space-4)" }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "var(--space-2)", marginBottom: "var(--space-2)" }}>
              <span className="badge badge-accent">5 Models × 3 Seeds</span>
              <span className="badge">UNSW-NB15 Benchmark</span>
              <span className="badge badge-success">Empirical Artifacts</span>
            </div>
            <h1 style={{ fontSize: "var(--text-2xl)", fontWeight: 700, margin: "0 0 var(--space-2) 0", letterSpacing: "-0.02em" }}>
              {t.experimentsTitle}
            </h1>
            <p style={{ color: "var(--text-secondary)", maxWidth: "800px", margin: 0 }}>
              {t.experimentsSubtitle}
            </p>
          </div>

          <div style={{ display: "flex", gap: "var(--space-2)", flexWrap: "wrap" }}>
            <a 
              href="/data/summary_statistics.json" 
              download="nids_summary_statistics.json"
              className="btn btn-secondary btn-sm"
              style={{ textDecoration: "none" }}
            >
              <Download size={14} />
              <span>{lang === "hi" ? "सांख्यिकी JSON डाउनलोड" : "Summary JSON"}</span>
            </a>
            <a 
              href="/data/sample_traffic_test.csv" 
              download="sample_traffic_test.csv"
              className="btn btn-secondary btn-sm"
              style={{ textDecoration: "none" }}
            >
              <Download size={14} />
              <span>{lang === "hi" ? "टेस्ट डेटा CSV डाउनलोड" : "Test Split CSV"}</span>
            </a>
          </div>
        </div>
      </section>

      {/* Controlled Methodology Protocol Note */}
      <section className="panel" style={{ padding: "var(--space-4)", borderLeft: "4px solid var(--accent-primary)" }}>
        <div style={{ display: "flex", alignItems: "flex-start", gap: "var(--space-3)" }}>
          <Info size={18} style={{ color: "var(--accent-primary)", marginTop: "2px", flexShrink: 0 }} />
          <div>
            <div style={{ fontWeight: 600, color: "var(--text-primary)", marginBottom: "var(--space-1)" }}>
              {t.methodologyNoteTitle}
            </div>
            <div style={{ fontSize: "var(--text-sm)", color: "var(--text-secondary)", lineHeight: 1.5 }}>
              {t.methodologyNoteBody}
            </div>
            <div style={{ marginTop: "var(--space-2)", fontSize: "var(--text-xs)", color: "var(--text-muted)", display: "flex", gap: "var(--space-4)", flexWrap: "wrap" }}>
              <span>• Stratification: 70% Train, 15% Validation, 15% Held-out Test</span>
              <span>• Fixed Seeds: 42, 123, 999</span>
              <span>• Zero Data Leakage: Preprocessor fitted strictly on Train</span>
              <span>• Simulator: PennyLane default.qubit CPU statevector</span>
            </div>
          </div>
        </div>
      </section>

      {/* Model Benchmark Comparison Table */}
      <section className="panel" style={{ padding: "var(--space-5)" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "var(--space-4)", flexWrap: "wrap", gap: "var(--space-2)" }}>
          <div>
            <h2 style={{ fontSize: "var(--text-lg)", fontWeight: 600, margin: 0 }}>
              {lang === "hi" ? "नियंत्रित मॉडल तुलना तालिका" : "Controlled Model Comparison Table"}
            </h2>
            <p style={{ fontSize: "var(--text-xs)", color: "var(--text-muted)", margin: "4px 0 0 0" }}>
              {lang === "hi" ? "3 बीजों (42, 123, 999) पर माध्य ± मानक विचलन" : "Mean ± Standard Deviation across 3 independent random seeds"}
            </p>
          </div>
          <span className="badge">N = 15 Runs Completed</span>
        </div>

        <div className="table-scroll-container">
          <table className="table">
            <thead>
              <tr>
                <th style={{ minWidth: "180px" }}>{t.architectureCol}</th>
                <th>Type</th>
                <th>{t.paramsCol}</th>
                <th>{t.accuracyCol}</th>
                <th>Balanced Acc</th>
                <th>{t.f1Col}</th>
                <th>{t.recallCol}</th>
                <th>{t.fprCol}</th>
                <th>ROC-AUC</th>
                <th>{t.trainingTimeCol}</th>
              </tr>
            </thead>
            <tbody>
              {EXPERIMENT_MODELS.map((m) => (
                <tr 
                  key={m.key} 
                  style={m.isDeployed ? { backgroundColor: "var(--bg-active)" } : undefined}
                >
                  <td style={{ fontWeight: 600, color: "var(--text-primary)" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "var(--space-2)" }}>
                      {m.isDeployed && <CheckCircle size={14} style={{ color: "var(--status-success)" }} />}
                      <span>{m.name}</span>
                      {m.isDeployed && (
                        <span className="badge badge-success" style={{ fontSize: "10px", padding: "1px 4px" }}>
                          Deployed
                        </span>
                      )}
                    </div>
                  </td>
                  <td>
                    <span className="badge" style={{ fontSize: "11px" }}>{m.family}</span>
                  </td>
                  <td style={{ fontFamily: "var(--font-mono)", fontWeight: 600 }}>
                    {m.params.toLocaleString()}
                  </td>
                  <td style={{ fontFamily: "var(--font-mono)", fontWeight: 600, color: "var(--accent-primary)" }}>
                    {m.accuracy}
                  </td>
                  <td style={{ fontFamily: "var(--font-mono)" }}>
                    {m.balancedAcc}
                  </td>
                  <td style={{ fontFamily: "var(--font-mono)" }}>
                    {m.f1}
                  </td>
                  <td style={{ fontFamily: "var(--font-mono)", color: "var(--status-success)" }}>
                    {m.attackRecall}
                  </td>
                  <td style={{ fontFamily: "var(--font-mono)", color: m.isDeployed ? "var(--status-success)" : "inherit" }}>
                    {m.fpr}
                  </td>
                  <td style={{ fontFamily: "var(--font-mono)" }}>
                    {m.rocAuc}
                  </td>
                  <td style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-xs)", color: "var(--text-secondary)" }}>
                    {m.cpuTrainingDuration}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div style={{ marginTop: "var(--space-3)", fontSize: "var(--text-xs)", color: "var(--text-muted)", display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: "var(--space-2)" }}>
          <span>* Training time reflects classical CPU statevector simulation (PennyLane default.qubit), not quantum hardware clock cycles.</span>
          <span>Deployed checkpoint: results/network_intrusion/best_model.pt</span>
        </div>
      </section>

      {/* Verified Research Plots Grid */}
      <section className="panel" style={{ padding: "var(--space-5)" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "var(--space-4)", flexWrap: "wrap", gap: "var(--space-2)" }}>
          <div>
            <h2 style={{ fontSize: "var(--text-lg)", fontWeight: 600, margin: 0 }}>
              {lang === "hi" ? "सत्यापित अनुसंधान आलेख एवं चार्ट" : "Verified Empirical Research Visualizations"}
            </h2>
            <p style={{ fontSize: "var(--text-xs)", color: "var(--text-muted)", margin: "4px 0 0 0" }}>
              {lang === "hi" ? "परिणाम निर्देशिका results/network_intrusion/plots/ से वास्तविक आलेख" : "Authentic publication figures directly loaded from repository plot artifacts"}
            </p>
          </div>

          {/* Category Filter */}
          <div style={{ display: "flex", gap: "var(--space-1)", flexWrap: "wrap" }}>
            {categories.map((cat) => (
              <button
                key={cat}
                className={`btn btn-sm ${activeCategory === cat ? "btn-primary" : "btn-secondary"}`}
                onClick={() => setActiveCategory(cat)}
                style={{ fontSize: "11px", padding: "3px 8px" }}
              >
                {cat}
              </button>
            ))}
          </div>
        </div>

        {/* Plots Grid */}
        <div 
          style={{ 
            display: "grid", 
            gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", 
            gap: "var(--space-4)" 
          }}
        >
          {filteredPlots.map((plot) => (
            <div 
              key={plot.id} 
              className="panel" 
              style={{ 
                padding: "var(--space-3)", 
                backgroundColor: "var(--bg-card)",
                display: "flex",
                flexDirection: "column",
                gap: "var(--space-2)"
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: "var(--space-2)" }}>
                <div>
                  <span className="badge" style={{ fontSize: "10px", marginBottom: "4px" }}>{plot.category}</span>
                  <h3 style={{ fontSize: "var(--text-sm)", fontWeight: 600, margin: 0 }}>
                    {plot.title}
                  </h3>
                </div>
                <button
                  className="btn btn-secondary btn-sm"
                  onClick={() => setSelectedPlot(plot)}
                  title="Enlarge plot"
                  style={{ padding: "4px" }}
                  aria-label="Enlarge plot"
                >
                  <Maximize2 size={13} />
                </button>
              </div>

              <div 
                style={{ 
                  backgroundColor: "#ffffff", 
                  border: "1px solid var(--border-subtle)", 
                  borderRadius: "var(--radius-sm)",
                  overflow: "hidden",
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  minHeight: "220px"
                }}
                onClick={() => setSelectedPlot(plot)}
              >
                <img 
                  src={plot.src} 
                  alt={plot.title}
                  style={{ 
                    width: "100%", 
                    height: "auto", 
                    display: "block",
                    objectFit: "contain",
                    maxHeight: "280px"
                  }} 
                  loading="lazy"
                />
              </div>

              <p style={{ fontSize: "var(--text-xs)", color: "var(--text-secondary)", margin: 0, lineHeight: 1.4 }}>
                {plot.description}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* Modal for full size plot view */}
      {selectedPlot && (
        <div 
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: "rgba(0, 0, 0, 0.75)",
            zIndex: 1000,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            padding: "var(--space-4)"
          }}
          onClick={() => setSelectedPlot(null)}
        >
          <div 
            className="panel"
            style={{
              maxWidth: "960px",
              width: "100%",
              maxHeight: "90vh",
              overflowY: "auto",
              padding: "var(--space-4)",
              backgroundColor: "var(--bg-surface)",
              boxShadow: "0 10px 25px rgba(0, 0, 0, 0.5)"
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "var(--space-3)" }}>
              <div>
                <span className="badge badge-accent">{selectedPlot.category}</span>
                <h3 style={{ fontSize: "var(--text-lg)", fontWeight: 700, margin: "4px 0 0 0" }}>
                  {selectedPlot.title}
                </h3>
              </div>
              <button 
                className="btn btn-secondary btn-sm"
                onClick={() => setSelectedPlot(null)}
              >
                {t.close}
              </button>
            </div>

            <div style={{ backgroundColor: "#ffffff", padding: "var(--space-2)", border: "1px solid var(--border-subtle)", borderRadius: "var(--radius-sm)", marginBottom: "var(--space-3)" }}>
              <img 
                src={selectedPlot.src} 
                alt={selectedPlot.title}
                style={{ width: "100%", height: "auto", display: "block" }} 
              />
            </div>

            <p style={{ fontSize: "var(--text-sm)", color: "var(--text-secondary)", margin: 0 }}>
              {selectedPlot.description}
            </p>
          </div>
        </div>
      )}

      {/* Key Research Insights Cards */}
      <section style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "var(--space-4)" }}>
        <div className="panel" style={{ padding: "var(--space-4)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "var(--space-2)", marginBottom: "var(--space-2)", color: "var(--accent-primary)" }}>
            <TrendingUp size={16} />
            <h3 style={{ fontSize: "var(--text-sm)", fontWeight: 600, margin: 0 }}>
              {lang === "hi" ? "पैरामीटर दक्षता" : "Parameter Efficiency"}
            </h3>
          </div>
          <p style={{ fontSize: "var(--text-xs)", color: "var(--text-secondary)", margin: 0, lineHeight: 1.5 }}>
            Hybrid Re-uploading L=2 uses <strong>1,113 parameters</strong>—an <strong>82.6% reduction</strong> compared to Classical MLP (6,401 parameters)—while matching 93.33% vs 93.19% test accuracy.
          </p>
        </div>

        <div className="panel" style={{ padding: "var(--space-4)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "var(--space-2)", marginBottom: "var(--space-2)", color: "var(--status-success)" }}>
            <CheckCircle size={16} />
            <h3 style={{ fontSize: "var(--text-sm)", fontWeight: 600, margin: 0 }}>
              {lang === "hi" ? "मिथ्या सकारात्मकता नियंत्रण" : "FPR Noise Reduction"}
            </h3>
          </div>
          <p style={{ fontSize: "var(--text-xs)", color: "var(--text-secondary)", margin: 0, lineHeight: 1.5 }}>
            Hybrid L=2 achieved the lowest False Positive Rate (<strong>5.19%</strong>) among all evaluated architectures, reducing alert fatigue for SOC operators by ~15% relative to Classical MLP (6.09%).
          </p>
        </div>

        <div className="panel" style={{ padding: "var(--space-4)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "var(--space-2)", marginBottom: "var(--space-2)", color: "var(--status-warning)" }}>
            <Sliders size={16} />
            <h3 style={{ fontSize: "var(--text-sm)", fontWeight: 600, margin: 0 }}>
              {lang === "hi" ? "री-अपलोडिंग डेप्थ एब्लेशन" : "Depth Ablation Saturation"}
            </h3>
          </div>
          <p style={{ fontSize: "var(--text-xs)", color: "var(--text-secondary)", margin: 0, lineHeight: 1.5 }}>
            Ablation reveals performance peaks at L=2 (93.33%). Increasing depth to L=4 introduces slight parameter redundancy and increased variance (92.41% ± 1.00%) on the 4-qubit simulated ansatz.
          </p>
        </div>
      </section>
    </div>
  );
};
