import React, { useState, useEffect } from "react";
import { NavLink, useLocation } from "react-router-dom";
import { useTranslation } from "../i18n/useTranslation";
import { type ThemeName } from "../app/themeContextInstance";
import { useTheme } from "../app/useTheme";
import { checkHealth, fetchRuntimeConfig, type RuntimeConfigResponse } from "../services/api";

export const InstrumentRail: React.FC = () => {
  const { language, setLanguage } = useTranslation();
  const { theme, setTheme } = useTheme();
  const location = useLocation();

  const [healthStatus, setHealthStatus] = useState<"healthy" | "degraded" | "offline">("healthy");
  const [config, setConfig] = useState<RuntimeConfigResponse | null>(null);

  useEffect(() => {
    let isMounted = true;
    const fetchStatus = async () => {
      try {
        const [hData, cData] = await Promise.all([
          checkHealth().catch(() => null),
          fetchRuntimeConfig().catch(() => null),
        ]);
        if (isMounted) {
          if (hData) setHealthStatus(hData.status);
          if (cData) setConfig(cData);
        }
      } catch {
        if (isMounted) {
          setHealthStatus("offline");
        }
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 15000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const navItems = [
    { to: "/", index: "01", label: "TELEMETRY OVERVIEW" },
    { to: "/traffic", index: "02", label: "PACKET CAPTURE (CSV)" },
    { to: "/detection", index: "03", label: "QUBIT BENCH (39-FEAT)" },
    { to: "/experiments", index: "04", label: "ABLATION MATRIX (45M)" },
    { to: "/research", index: "05", label: "ARCHITECTURE SPEC" },
    { to: "/about", index: "06", label: "AUDIT & CALIBRATION" },
  ];

  const themes: Array<{ id: ThemeName; label: string }> = [
    { id: "dark-research", label: "GRAPHITE" },
    { id: "light-research", label: "PAPER" },
    { id: "high-contrast", label: "AAA" },
    { id: "cyberpunk", label: "SCOPE" },
  ];

  return (
    <aside
      style={{
        width: "var(--rail-width)",
        minWidth: "var(--rail-width)",
        maxWidth: "var(--rail-width)",
        height: "100vh",
        background: "var(--bg-surface-1)",
        borderRight: "1px solid var(--border-default)",
        display: "flex",
        flexDirection: "column",
        justifyContent: "space-between",
        userSelect: "none",
        zIndex: 40,
      }}
    >
      {/* Top Header / Hardware Brand */}
      <div>
        <div
          style={{
            padding: "1rem",
            borderBottom: "1px solid var(--border-default)",
            background: "var(--bg-surface-2)",
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              marginBottom: "4px",
            }}
          >
            <span
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: "10px",
                fontWeight: 700,
                letterSpacing: "0.14em",
                color: "var(--accent-primary)",
              }}
            >
              KEYSIGHT // Q-NIDS
            </span>
            <span
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: "9px",
                color: "var(--text-muted)",
                padding: "1px 4px",
                border: "1px solid var(--border-default)",
                background: "var(--bg-input)",
              }}
            >
              REV 2.5
            </span>
          </div>
          <div
            style={{
              fontFamily: "var(--font-heading)",
              fontSize: "13px",
              fontWeight: 700,
              color: "var(--text-primary)",
              letterSpacing: "-0.01em",
              lineHeight: 1.2,
            }}
          >
            QUANTUM OSCILLOSCOPE
          </div>
          <div
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: "10px",
              color: "var(--text-secondary)",
              marginTop: "2px",
            }}
          >
            CIC-IoT-2023 BENCH // 45M
          </div>
        </div>

        {/* Numbered Navigation Items */}
        <nav style={{ padding: "0.75rem 0" }}>
          {navItems.map((item) => {
            const isActive = location.pathname === item.to;
            return (
              <NavLink
                key={item.to}
                to={item.to}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "10px",
                  padding: "8px 14px",
                  fontSize: "11px",
                  fontFamily: "var(--font-mono)",
                  textDecoration: "none",
                  borderLeft: isActive
                    ? "3px solid var(--accent-primary)"
                    : "3px solid transparent",
                  background: isActive ? "var(--bg-surface-2)" : "transparent",
                  color: isActive ? "var(--text-primary)" : "var(--text-secondary)",
                  fontWeight: isActive ? 600 : 400,
                  transition: "background var(--transition-fast), color var(--transition-fast)",
                }}
              >
                <span
                  style={{
                    color: isActive ? "var(--accent-primary)" : "var(--text-muted)",
                    fontSize: "10px",
                    fontWeight: 700,
                  }}
                >
                  {item.index}
                </span>
                <span
                  style={{
                    letterSpacing: "0.04em",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                  }}
                >
                  {item.label}
                </span>
              </NavLink>
            );
          })}
        </nav>
      </div>

      {/* Middle Controls: Theme & Language */}
      <div
        style={{
          padding: "0.75rem 1rem",
          borderTop: "1px solid var(--border-default)",
          borderBottom: "1px solid var(--border-default)",
          background: "var(--bg-page)",
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: "6px",
          }}
        >
          <span
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: "9px",
              color: "var(--text-muted)",
              textTransform: "uppercase",
              letterSpacing: "0.06em",
            }}
          >
            PALETTE MODE
          </span>
          <div style={{ display: "flex", gap: "2px" }}>
            <button
              onClick={() => setLanguage(language === "en" ? "hi" : "en")}
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: "9px",
                padding: "2px 5px",
                background: "var(--bg-surface-2)",
                border: "1px solid var(--border-default)",
                color: "var(--accent-primary)",
                cursor: "pointer",
              }}
              title="Toggle Language"
            >
              {language.toUpperCase()}
            </button>
          </div>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "4px" }}>
          {themes.map((tItem) => (
            <button
              key={tItem.id}
              onClick={() => setTheme(tItem.id)}
              style={{
                padding: "3px 4px",
                fontSize: "9px",
                fontFamily: "var(--font-mono)",
                background: theme === tItem.id ? "var(--accent-primary)" : "var(--bg-surface-2)",
                color: theme === tItem.id ? "var(--text-inverse)" : "var(--text-secondary)",
                border: "1px solid " + (theme === tItem.id ? "var(--accent-primary)" : "var(--border-default)"),
                cursor: "pointer",
                fontWeight: theme === tItem.id ? 700 : 400,
              }}
            >
              {tItem.label}
            </button>
          ))}
        </div>
      </div>

      {/* Bottom Hardware Status Readout Panel */}
      <div
        style={{
          padding: "1rem",
          background: "var(--bg-code)",
          borderTop: "1px solid var(--border-default)",
          fontFamily: "var(--font-mono)",
          fontSize: "10px",
          lineHeight: 1.45,
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "6px",
            marginBottom: "8px",
            paddingBottom: "6px",
            borderBottom: "1px solid var(--border-subtle)",
          }}
        >
          <span
            style={{
              display: "inline-block",
              width: "7px",
              height: "7px",
              borderRadius: "50%",
              backgroundColor: healthStatus === "offline" ? "var(--status-attack)" : "var(--status-benign)",
              boxShadow: healthStatus === "offline" ? "none" : "0 0 6px var(--status-benign)",
            }}
          />
          <span
            style={{
              fontWeight: 700,
              letterSpacing: "0.08em",
              color: healthStatus === "offline" ? "var(--status-attack)" : "var(--status-benign)",
            }}
          >
            {healthStatus === "offline" ? "BACKEND: OFFLINE" : "SYSTEM: ONLINE"}
          </span>
        </div>

        <div style={{ color: "var(--text-muted)", display: "flex", flexDirection: "column", gap: "3px" }}>
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <span>HOST</span>
            <span style={{ color: "var(--text-primary)" }}>127.0.0.1:8000</span>
          </div>
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <span>ANSATZ</span>
            <span style={{ color: "var(--accent-quantum)" }}>{config?.model_name || "4Q-REUPLOAD-L1"}</span>
          </div>
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <span>TRAINED</span>
            <span style={{ color: "var(--text-primary)" }}>45,019,243</span>
          </div>
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <span>PARAMS</span>
            <span style={{ color: "var(--text-primary)" }}>721 (-85.2%)</span>
          </div>
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <span>TAU*</span>
            <span style={{ color: "var(--accent-primary)" }}>{config?.decision_threshold ? config.decision_threshold.toFixed(4) : "0.2971"}</span>
          </div>
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <span>DEVICE</span>
            <span style={{ color: "var(--text-secondary)" }}>RTX 5060 / CUDA</span>
          </div>
        </div>
      </div>
    </aside>
  );
};
