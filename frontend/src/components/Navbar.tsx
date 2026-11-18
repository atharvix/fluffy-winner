import React, { useState, useEffect } from "react";
import { Link, useLocation } from "react-router-dom";
import {
  Shield,
  Activity,
  Layers,
  FlaskConical,
  BookOpen,
  HelpCircle,
  Sun,
  Moon,
  Contrast,
  Zap,
  Globe,
  Menu,
  X,
  Server,
} from "lucide-react";
import { useTranslation } from "../i18n/useTranslation";
import { type ThemeName } from "../app/themeContextInstance";
import { useTheme } from "../app/useTheme";
import { checkHealth, type HealthResponse } from "../services/api";

export const Navbar: React.FC = () => {
  const { t, language, setLanguage } = useTranslation();
  const { theme, setTheme } = useTheme();
  const location = useLocation();

  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthStatus, setHealthStatus] = useState<"healthy" | "degraded" | "offline">("offline");

  useEffect(() => {
    let isMounted = true;
    const fetchHealthStatus = async () => {
      try {
        const data = await checkHealth();
        if (isMounted) {
          setHealth(data);
          setHealthStatus(data.status);
        }
      } catch {
        if (isMounted) {
          setHealthStatus("offline");
        }
      }
    };

    fetchHealthStatus();
    const interval = setInterval(fetchHealthStatus, 15000); // 15s heartbeat
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const navLinks = [
    { to: "/", label: t.navOverview, icon: Activity },
    { to: "/traffic", label: t.navTraffic, icon: Shield },
    { to: "/detection", label: t.navDetection, icon: FlaskConical },
    { to: "/experiments", label: t.navExperiments, icon: Layers },
    { to: "/research", label: t.navResearch, icon: BookOpen },
    { to: "/about", label: t.navAbout, icon: HelpCircle },
  ];

  const themes: Array<{ id: ThemeName; label: string; icon: React.FC<{ size?: number }> }> = [
    { id: "light-research", label: t.themeLight, icon: Sun },
    { id: "dark-research", label: t.themeDark, icon: Moon },
    { id: "high-contrast", label: t.themeHighContrast, icon: Contrast },
    { id: "cyberpunk", label: t.themeCyberpunk, icon: Zap },
  ];

  return (
    <header
      style={{
        height: "var(--nav-height)",
        background: "var(--bg-surface-1)",
        borderBottom: "1px solid var(--border-default)",
        position: "sticky",
        top: 0,
        zIndex: 50,
        width: "100%",
      }}
    >
      <div
        className="container"
        style={{
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: "var(--space-3)",
        }}
      >
        {/* Brand mark */}
        <Link
          to="/"
          style={{
            display: "flex",
            alignItems: "center",
            gap: "8px",
            textDecoration: "none",
            color: "var(--text-primary)",
            fontWeight: 700,
            fontSize: "var(--text-base)",
            letterSpacing: "-0.02em",
          }}
        >
          <div
            style={{
              padding: "4px 6px",
              background: "var(--accent-primary)",
              color: "var(--text-inverse)",
              fontFamily: "var(--font-mono)",
              fontSize: "var(--text-xs)",
              fontWeight: 800,
              borderRadius: "var(--radius-sm)",
            }}
          >
            Q
          </div>
          <span>Q-NIDS</span>
          <span
            style={{
              fontSize: "10px",
              padding: "1px 5px",
              background: "var(--bg-surface-3)",
              color: "var(--text-secondary)",
              fontFamily: "var(--font-mono)",
              borderRadius: "var(--radius-sm)",
              border: "1px solid var(--border-subtle)",
              display: "none",
            }}
            className="d-md-inline"
          >
            v1.0
          </span>
        </Link>

        {/* Desktop Nav Items */}
        <nav
          className="desktop-nav"
          style={{
            display: "none",
            alignItems: "center",
            gap: "4px",
          }}
        >
          {navLinks.map((link) => {
            const isActive = location.pathname === link.to;
            const Icon = link.icon;
            return (
              <Link
                key={link.to}
                to={link.to}
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "6px",
                  padding: "6px 10px",
                  fontSize: "var(--text-sm)",
                  fontWeight: isActive ? 600 : 500,
                  color: isActive ? "var(--accent-primary)" : "var(--text-secondary)",
                  background: isActive ? "var(--bg-surface-2)" : "transparent",
                  borderBottom: isActive ? "2px solid var(--accent-primary)" : "2px solid transparent",
                  textDecoration: "none",
                  borderRadius: "var(--radius-sm) var(--radius-sm) 0 0",
                  transition: "background var(--transition-fast), color var(--transition-fast)",
                  whiteSpace: "nowrap",
                }}
              >
                <Icon size={14} />
                <span>{link.label}</span>
              </Link>
            );
          })}
        </nav>

        {/* Actions (Status, Theme, Lang, Mobile Menu) */}
        <div style={{ display: "flex", alignItems: "center", gap: "var(--space-2)" }}>
          {/* Live Backend Indicator */}
          <div
            title={`Backend Status: ${healthStatus} (${health?.selected_model || "Offline"})`}
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "5px",
              padding: "3px 7px",
              background:
                healthStatus === "healthy"
                  ? "var(--status-benign-bg)"
                  : healthStatus === "degraded"
                  ? "var(--status-warn-bg)"
                  : "var(--status-attack-bg)",
              color:
                healthStatus === "healthy"
                  ? "var(--status-benign)"
                  : healthStatus === "degraded"
                  ? "var(--status-warn)"
                  : "var(--status-attack)",
              border: `1px solid ${
                healthStatus === "healthy"
                  ? "var(--status-benign-border)"
                  : healthStatus === "degraded"
                  ? "var(--status-warn-border)"
                  : "var(--status-attack-border)"
              }`,
              fontSize: "var(--text-2xs)",
              fontFamily: "var(--font-mono)",
              fontWeight: 700,
              borderRadius: "var(--radius-sm)",
            }}
          >
            <Server size={11} />
            <span style={{ display: "none" }} className="d-sm-inline">
              {healthStatus === "healthy"
                ? t.statusBackendReady
                : healthStatus === "degraded"
                ? t.statusDegraded
                : t.statusBackendOffline}
            </span>
          </div>

          {/* Theme Selector Dropdown */}
          <select
            value={theme}
            onChange={(e) => setTheme(e.target.value as ThemeName)}
            aria-label="Select Color Theme"
            style={{
              padding: "4px 8px",
              fontSize: "var(--text-xs)",
              background: "var(--bg-surface-2)",
              color: "var(--text-primary)",
              border: "1px solid var(--border-default)",
              cursor: "pointer",
            }}
          >
            {themes.map((th) => (
              <option key={th.id} value={th.id}>
                {th.label}
              </option>
            ))}
          </select>

          {/* Language Switcher */}
          <button
            type="button"
            className="btn btn-sm"
            onClick={() => setLanguage(language === "en" ? "hi" : "en")}
            title="Toggle Language / भाषा बदलें"
            style={{ padding: "4px 7px", fontFamily: "var(--font-mono)" }}
          >
            <Globe size={13} />
            <span>{language.toUpperCase()}</span>
          </button>

          {/* Mobile Menu Hamburger */}
          <button
            type="button"
            className="btn btn-sm mobile-menu-btn"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            aria-label="Toggle navigation menu"
            style={{ display: "flex", padding: "6px" }}
          >
            {mobileMenuOpen ? <X size={16} /> : <Menu size={16} />}
          </button>
        </div>
      </div>

      {/* Mobile Drawer Navigation (Clean vertical menu, NO horizontal scrolling) */}
      {mobileMenuOpen && (
        <div
          style={{
            position: "absolute",
            top: "var(--nav-height)",
            left: 0,
            right: 0,
            background: "var(--bg-surface-1)",
            borderBottom: "2px solid var(--border-default)",
            padding: "var(--space-3) var(--space-4)",
            display: "flex",
            flexDirection: "column",
            gap: "var(--space-2)",
            boxShadow: "0 8px 16px rgba(0,0,0,0.25)",
            zIndex: 49,
          }}
        >
          {navLinks.map((link) => {
            const isActive = location.pathname === link.to;
            const Icon = link.icon;
            return (
              <Link
                key={link.to}
                to={link.to}
                onClick={() => setMobileMenuOpen(false)}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "10px",
                  padding: "10px 12px",
                  fontSize: "var(--text-base)",
                  fontWeight: isActive ? 600 : 500,
                  color: isActive ? "var(--accent-primary)" : "var(--text-primary)",
                  background: isActive ? "var(--bg-surface-2)" : "transparent",
                  borderLeft: isActive ? "3px solid var(--accent-primary)" : "3px solid transparent",
                  textDecoration: "none",
                  borderRadius: "var(--radius-sm)",
                }}
              >
                <Icon size={16} />
                <span>{link.label}</span>
              </Link>
            );
          })}
        </div>
      )}

      {/* Responsive helper styles */}
      <style>{`
        @media (min-width: 992px) {
          .desktop-nav {
            display: flex !important;
          }
          .mobile-menu-btn {
            display: none !important;
          }
        }
        @media (min-width: 640px) {
          .d-sm-inline {
            display: inline !important;
          }
        }
        @media (min-width: 768px) {
          .d-md-inline {
            display: inline !important;
          }
        }
      `}</style>
    </header>
  );
};
