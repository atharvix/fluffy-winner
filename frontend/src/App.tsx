import React from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { ThemeProvider } from "./app/ThemeContext";
import { LanguageProvider } from "./i18n/LanguageContext";
import { InstrumentRail } from "./components/InstrumentRail";
import { OverviewPage } from "./pages/OverviewPage";
import { TrafficAnalysisPage } from "./pages/TrafficAnalysisPage";
import { DetectionLabPage } from "./pages/DetectionLabPage";
import { ExperimentsPage } from "./pages/ExperimentsPage";
import { ResearchPage } from "./pages/ResearchPage";
import { AboutPage } from "./pages/AboutPage";

export const AppContent: React.FC = () => {
  return (
    <div
      style={{
        width: "100vw",
        height: "100vh",
        display: "flex",
        flexDirection: "row",
        overflow: "hidden",
        backgroundColor: "var(--bg-page)",
        color: "var(--text-primary)",
      }}
    >
      <InstrumentRail />

      <main className="workbench-canvas">
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
