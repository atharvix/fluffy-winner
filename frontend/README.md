# Q-NIDS Research Workstation (Frontend)

Production-grade research and network-security dashboard for the **Hybrid Classical–Quantum Network Intrusion Detection System (NIDS)** evaluating Variational Quantum Circuits and Data Re-uploading on the **UNSW-NB15** benchmark.

---

## 1. Architecture Overview

- **Framework:** React 19 + TypeScript + Vite
- **Styling Architecture:** Vanilla CSS design tokens (`tokens.css`, `themes.css`, `global.css`)
- **Visual Design Rules:**
  - Strictly **ZERO gradients** across the entire UI.
  - Strictly rectangular controls (0–4 px radius for inputs, buttons, badges; 4–6 px for panels; **NO pill shapes**).
  - No permanent background animations or distracting neon overloads.
  - Zero horizontal overflow across all viewports (`320px`, `375px`, `768px`, `1024px`, `1440px`).
- **3D Visualization:** Lightweight CSS 3D matrix transforms and SVG rendering for 4-qubit quantum circuits, ring entanglement topology, and interactive Bloch sphere statevector inspection (with instant 2D flat mode fallback).
- **Internationalization:** Site-wide bilingual English (`en`) and Hindi (`hi`) translations with persistent local storage.

---

## 2. Four Working Themes

Switchable via the top navigation bar and persisted in `localStorage`:

1. **Light Research (`data-theme="light-research"`):** Clean off-white background (`#f8f9fa`), charcoal text (`#111827`), slate accents, and fine border separators.
2. **Dark Research (`data-theme="dark-research"`):** Deep graphite background (`#101216`), light grey typography, and restrained technical blue accents.
3. **High Contrast (`data-theme="high-contrast"`):** Pure black surfaces (`#000000`), stark white text/borders, and bright cyan/yellow focus indicators targeting WCAG 2.2 AA.
4. **Cyberpunk (`data-theme="cyberpunk"`):** Near-black surfaces with flat electric cyan, magenta, and violet accents (strictly no gradients).

---

## 3. Application Routes

| Route | Page Component | Description |
| :--- | :--- | :--- |
| `/` | `OverviewPage` | System summary, real-time FastAPI health, deployed model metadata, verified KPI metrics, SVG benchmark comparison, and 3D quantum circuit preview. |
| `/traffic` | `TrafficAnalysisPage` | Drag-and-drop CSV traffic inspector, 42-feature schema validation, batch inference via `/api/predict/csv`, and classified CSV download. |
| `/detection` | `DetectionLabPage` | Single-flow prediction laboratory with all 42 flow features, 5 real UNSW-NB15 presets, real-time inference via `/api/predict`, and quantum statevector inspection. |
| `/experiments` | `ExperimentsPage` | Controlled 5-model × 3-seed empirical study table, parameter efficiency comparison, and high-resolution empirical plot gallery. |
| `/research` | `ResearchPage` | Publication-style monograph on Quantum Data Re-uploading, Fourier frequency expressivity, dataset provenance, SOC operational context, and hardware disclaimers. |
| `/about` | `AboutPage` | System telemetry, verified dependencies (PennyLane, PyTorch, FastAPI), OpenAPI endpoint specifications, and reproducibility protocol. |

---

## 4. Launching the System (Windows PowerShell)

### Step 1: Start the FastAPI Backend
```powershell
# From the repository root (D:\dl)
$env:CURL_CA_BUNDLE=""
.venv\Scripts\uvicorn.exe api.main:app --host 127.0.0.1 --port 8000 --reload
```
API endpoints will be live at:
- Service root: `http://127.0.0.1:8000`
- Swagger UI: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/api/health`

### Step 2: Start the Frontend Application
```powershell
# In a separate terminal, navigate to frontend/
cd frontend
npm run dev
```
The frontend workstation will start at `http://localhost:5173`.

---

## 5. Verification & Testing

### Python Backend & Model Tests (47 tests)
```powershell
.venv\Scripts\pytest.exe -v
```

### Frontend Typecheck, Lint, and Build
```powershell
cd frontend
npm run lint    # Oxlint verification (0 errors, 0 warnings)
npm run build   # TypeScript typecheck (tsc -b) & Vite production build
```
