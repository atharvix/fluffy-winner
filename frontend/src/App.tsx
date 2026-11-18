import React from "react";
import { BrowserRouter, Routes, Route, Navigate, Link } from "react-router-dom";
import { ThemeProvider } from "./app/ThemeContext";
import { LanguageProvider } from "./i18n/LanguageContext";
import { useLanguage } from "./i18n/useTranslation";
import { Navbar } from "./components/Navbar";
import { OverviewPage } from "./pages/OverviewPage";
import { TrafficAnalysisPage } from "./pages/TrafficAnalysisPage";
import { DetectionLabPage } from "./pages/DetectionLabPage";
import { ExperimentsPage } from "./pages/ExperimentsPage";
import { ResearchPage } from "./pages/ResearchPage";
import { AboutPage } from "./pages/AboutPage";

const AppFooter: React.FC = () => {
  const { t, lang } = useLanguage();

  return (
    <footer 
      style={{
        marginTop: "auto",
        borderTop: "1px solid var(--border-subtle)",
        backgroundColor: "var(--bg-surface)",
        padding: "var(--space-6) var(--space-4)",
        fontSize: "var(--text-xs)",
        color: "var(--text-secondary)"
      }}
    >
      <div 
        style={{
          maxWidth: "1440px",
          margin: "0 auto",
          display: "flex",
          flexDirection: "column",
          gap: "var(--space-4)"
        }}
      >
        <div 
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-start",
            flexWrap: "wrap",
            gap: "var(--space-4)"
          }}
        >
          {/* Brand & Mission */}
          <div style={{ maxWidth: "420px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "var(--space-2)", marginBottom: "var(--space-2)" }}>
              <div 
                style={{ 
                  width: "18px", 
                  height: "18px", 
                  backgroundColor: "var(--accent-primary)", 
                  color: "#ffffff",
                  display: "flex", 
                  alignItems: "center", 
                  justifyContent: "center", 
                  fontWeight: 900,
                  fontSize: "11px",
                  borderRadius: "2px"
                }}
              >
                Q
              </div>
              <span style={{ fontWeight: 700, color: "var(--text-primary)", fontSize: "var(--text-sm)" }}>
                Q-NIDS Research Workstation
              </span>
            </div>
            <p style={{ margin: 0, lineHeight: 1.5, color: "var(--text-secondary)" }}>
              {lang === "hi" 
                ? "हाइब्रिड क्लासिकल-क्वांटम नेटवर्क घुसपैठ पहचान प्रणाली। UNSW-NB15 बेंचमार्क और पेनीलेन सिमुलेशन पर आधारित।"
                : "Hybrid Classical–Quantum Network Intrusion Detection System evaluating Variational Quantum Circuits and Data Re-uploading on the UNSW-NB15 benchmark."}
            </p>
          </div>

          {/* Quick Navigation Links */}
          <div style={{ display: "flex", gap: "var(--space-6)", flexWrap: "wrap" }}>
            <div>
              <div style={{ fontWeight: 600, color: "var(--text-primary)", marginBottom: "var(--space-2)" }}>
                {lang === "hi" ? "कार्यक्षेत्र" : "Workstation"}
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-1)" }}>
                <Link to="/" style={{ color: "var(--text-secondary)" }}>{t.navOverview}</Link>
                <Link to="/traffic" style={{ color: "var(--text-secondary)" }}>{t.navTraffic}</Link>
                <Link to="/detection" style={{ color: "var(--text-secondary)" }}>{t.navDetection}</Link>
              </div>
            </div>

            <div>
              <div style={{ fontWeight: 600, color: "var(--text-primary)", marginBottom: "var(--space-2)" }}>
                {lang === "hi" ? "अनुसंधान" : "Empirical Study"}
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-1)" }}>
                <Link to="/experiments" style={{ color: "var(--text-secondary)" }}>{t.navExperiments}</Link>
                <Link to="/research" style={{ color: "var(--text-secondary)" }}>{t.navResearch}</Link>
                <Link to="/about" style={{ color: "var(--text-secondary)" }}>{t.navAbout}</Link>
              </div>
            </div>

            <div>
              <div style={{ fontWeight: 600, color: "var(--text-primary)", marginBottom: "var(--space-2)" }}>
                {lang === "hi" ? "दस्तावेज़ एवं कलाकृतियां" : "Artifacts & Reports"}
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-1)" }}>
                <a href="/reports/nids_research_report.md" download style={{ color: "var(--text-secondary)" }}>
                  Research Report (.md)
                </a>
                <a href="/data/summary_statistics.json" download style={{ color: "var(--text-secondary)" }}>
                  Summary Stats (.json)
                </a>
                <a href="/data/sample_traffic_test.csv" download style={{ color: "var(--text-secondary)" }}>
                  Test Split CSV (.csv)
                </a>
              </div>
            </div>
          </div>
        </div>

        {/* Bottom Disclaimer & Environment */}
        <div 
          style={{
            borderTop: "1px solid var(--border-subtle)",
            paddingTop: "var(--space-3)",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            flexWrap: "wrap",
            gap: "var(--space-2)",
            color: "var(--text-muted)",
            fontSize: "11px"
          }}
        >
          <div>
            PennyLane CPU statevector simulation (default.qubit) • No physical quantum hardware • Decision support only
          </div>
          <div style={{ display: "flex", gap: "var(--space-3)", alignItems: "center" }}>
            <span>PyTorch 2.6</span>
            <span>•</span>
            <span>PennyLane 0.40</span>
            <span>•</span>
            <span>FastAPI 0.115</span>
            <span>•</span>
            <span>React 19 + TypeScript</span>
          </div>
        </div>
      </div>
    </footer>
  );
};

export const AppContent: React.FC = () => {
  return (
    <div 
      style={{ 
        minHeight: "100vh", 
        display: "flex", 
        flexDirection: "column",
        backgroundColor: "var(--bg-page)",
        color: "var(--text-primary)",
        width: "100%",
        overflowX: "hidden"
      }}
    >
      <Navbar />

      <main 
        style={{ 
          flex: 1, 
          width: "100%", 
          maxWidth: "1440px", 
          margin: "0 auto", 
          padding: "var(--space-5) var(--space-4)",
          boxSizing: "border-box",
          minWidth: 0
        }}
      >
        <Routes>
          <Route path="/" element={<OverviewPage />} />
          <Route path="/traffic" element={<TrafficAnalysisPage />} />
          <Route path="/detection" element={<DetectionLabPage />} />
          <Route path="/experiments" element={<ExperimentsPage />} />
          <Route path="/research" element={<ResearchPage />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>

      <AppFooter />
    </div>
  );
};

export default function App() {
  return (
    <ThemeProvider>
      <LanguageProvider>
        <BrowserRouter>
          <AppContent />
        </BrowserRouter>
      </LanguageProvider>
    </ThemeProvider>
  );
}
