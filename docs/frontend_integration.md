# Frontend Integration Guide: Google Stitch & Custom UI

This specification documents the contract between the **Hybrid Classical–Quantum Network Intrusion Detection (NIDS) Inference Backend** and frontend applications (such as interfaces designed in **Google Stitch**, React, Vue, or Next.js).

---

## 1. Local Development Architecture

```
+------------------------------------+           +-------------------------------------+
|   Google Stitch / Custom Frontend  |           |         FastAPI Backend             |
|   (e.g., http://localhost:3000)    | --------> |    (http://127.0.0.1:8000)          |
+------------------------------------+           +-------------------------------------+
                                                                   |
                                                                   v
                                                 +-------------------------------------+
                                                 |      NIDSPredictor Pipeline         |
                                                 |  - ColumnTransformer (pre-fitted)   |
                                                 |  - PyTorch Hybrid Quantum Model     |
                                                 +-------------------------------------+
```

- **API Base URL:** `http://127.0.0.1:8000`
- **Supported Methods:** `GET`, `POST`, `OPTIONS`
- **Authentication:** None (Local Development Environment)
- **CORS Allowed Origins:**
  - `http://localhost:3000`
  - `http://localhost:5173`
  - `http://localhost:8000`
  - `http://localhost:8080`
  - `http://127.0.0.1:3000`
  - `http://127.0.0.1:5173`

---

## 2. API Endpoints

### 2.1 Health & Readiness: `GET /api/health`

Use this endpoint on frontend mount to verify backend liveness and ensure the neural/quantum model is loaded.

#### Response Example (`200 OK`)
```json
{
  "status": "healthy",
  "service": "Hybrid Classical-Quantum NIDS Inference Service",
  "model_loaded": true,
  "selected_model": "Hybrid_Reuploading_L2",
  "device": "cpu",
  "uptime_seconds": 128.45,
  "timestamp": "2026-10-12T15:30:00Z",
  "version": "1.0.0"
}
```

#### UI State Handling:
- If `model_loaded === false` or `status === "degraded"`: Display an alert banner: *"Model backend is initializing. Please wait..."*
- If request times out or returns network error: Display offline screen: *"Cannot connect to inference service at http://127.0.0.1:8000"*.

---

### 2.2 Model Metadata: `GET /api/model-info`

Retrieves the active model architecture, parameter count, circuit complexity, and ordered feature schema.

#### Response Example (`200 OK`)
```json
{
  "model_name": "Hybrid_Reuploading_L2",
  "model_family": "hybrid_reuploading",
  "dataset": "UNSW-NB15",
  "dataset_provenance": "Australian Centre for Cyber Security (ACCS), UNSW Canberra. Authors: Nour Moustafa and Jill Slay (2015).",
  "input_feature_count": 42,
  "raw_features": ["dur", "proto", "service", "state", "spkts", "..."],
  "numerical_features": ["dur", "spkts", "dpkts", "..."],
  "categorical_features": ["proto", "service", "state"],
  "class_label_mapping": {
    "0": "BENIGN",
    "1": "ATTACK"
  },
  "positive_class": "ATTACK (1)",
  "negative_class": "BENIGN (0)",
  "decision_threshold": 0.5,
  "parameter_breakdown": {
    "classical_encoder": 1092,
    "quantum_circuit": 16,
    "classical_output_head": 5,
    "total_trainable": 1113
  },
  "circuit_metadata": {
    "type": "DataReuploadingVQC",
    "n_qubits": 4,
    "n_layers": 2,
    "num_parameters": 16,
    "encoding_gates_per_layer": 4,
    "trainable_gates_per_layer": 8,
    "cnot_gates_per_layer": 4,
    "total_gates": 32,
    "estimated_circuit_depth": 8
  }
}
```

---

### 2.3 JSON Batch & Single-Record Inference: `POST /api/predict`

Accepts an array of network flow feature objects and returns detection predictions along with quantum circuit inspection.

#### Request Headers
```http
Content-Type: application/json
```

#### Request Payload
```json
{
  "records": [
    {
      "id": "flow-001",
      "dur": 0.000011,
      "proto": "udp",
      "service": "-",
      "state": "INT",
      "spkts": 2,
      "dpkts": 0,
      "sbytes": 496,
      "dbytes": 0,
      "rate": 90909.09,
      "sttl": 254,
      "dttl": 0,
      "sload": 180363632.0,
      "dload": 0.0,
      "sloss": 0,
      "dloss": 0,
      "sinpkt": 0.011,
      "dinpkt": 0.0,
      "sjit": 0.0,
      "djit": 0.0,
      "swin": 0,
      "stcpb": 0,
      "dtcpb": 0,
      "dwin": 0,
      "tcprtt": 0.0,
      "synack": 0.0,
      "ackdat": 0.0,
      "smean": 248,
      "dmean": 0,
      "trans_depth": 0,
      "response_body_len": 0,
      "ct_srv_src": 2,
      "ct_state_ttl": 2,
      "ct_dst_ltm": 1,
      "ct_src_dport_ltm": 1,
      "ct_dst_sport_ltm": 1,
      "ct_dst_src_ltm": 2,
      "is_ftp_login": 0,
      "ct_ftp_cmd": 0,
      "ct_flw_http_mthd": 0,
      "ct_src_ltm": 1,
      "ct_srv_dst": 2,
      "is_sm_ips_ports": 0
    }
  ],
  "threshold": 0.5
}
```

#### Response Example (`200 OK`)
```json
{
  "status": "success",
  "sample_count": 1,
  "summary": {
    "attack_count": 0,
    "benign_count": 1,
    "attack_ratio": 0.0
  },
  "decision_threshold": 0.5,
  "predictions": [
    {
      "record_id": "flow-001",
      "index": 0,
      "predicted_label": "BENIGN",
      "predicted_class_id": 0,
      "confidence": 0.9632,
      "probabilities": {
        "ATTACK": 0.0368,
        "BENIGN": 0.9632
      },
      "raw_logit": -3.2641,
      "quantum_inspection": {
        "encoder_angles": [0.3412, -0.8912, 1.2045, -0.4518],
        "pauli_z_expectations": [0.8124, -0.6512, 0.4312, -0.7819]
      }
    }
  ]
}
```

---

### 2.4 CSV File Upload Inference: `POST /api/predict/csv`

Accepts a multipart CSV file upload containing UNSW-NB15 flow features.

#### Query Parameters:
- `threshold` (float, default `0.5`): Decision boundary for `ATTACK`.
- `format` (string, `json` or `csv`, default `json`): Output response format.

#### JSON Mode Response Example
```json
{
  "status": "success",
  "filename": "test_traffic.csv",
  "sample_count": 50,
  "summary": {
    "attack_count": 22,
    "benign_count": 28,
    "attack_ratio": 0.44
  },
  "decision_threshold": 0.5,
  "predictions": [
    {
      "index": 0,
      "predicted_label": "ATTACK",
      "confidence": 0.9781,
      "probabilities": {
        "ATTACK": 0.9781,
        "BENIGN": 0.0219
      }
    }
  ]
}
```

---

## 3. UI Visualization Guidelines for Google Stitch

1. **Classification Badges:**
   - `ATTACK`: Display in crimson red badge (`#DC2626` / `#F87171`) with alert icon.
   - `BENIGN`: Display in emerald green badge (`#059669` / `#34D399`) with check icon.
2. **Confidence Gauge:**
   - Display a progress ring or horizontal bar showing `confidence * 100%`.
3. **Quantum Circuit Inspector:**
   - Render 4 gauge needles or polar dials showing the classical encoder rotation angles $\theta_0, \theta_1, \theta_2, \theta_3 \in [-\pi, \pi]$.
   - Render 4 bar meters for the Pauli-Z expectation values $\langle Z_i \rangle \in [-1, 1]$ representing simulated qubit spin measurements.
4. **Historical Experiment Metrics vs Live Inference:**
   - Clearly label summary benchmark tables (e.g., *Classical MLP 93.7% Accuracy*, *Hybrid Quantum 93.0% Accuracy*) as **Controlled Benchmark Results on Held-Out Test Set (UNSW-NB15)**.
   - Do NOT present historical test metrics as real-time server stream stats.
5. **No Synthetic Detections:**
   - When the user uploads a CSV file or submits records, compute and show the actual output from the backend.
   - Never invent mock alerts when no data is uploaded.

---

## 4. Starting the Backend Service

To start the API service locally:

```powershell
# In Windows PowerShell from repository root:
$env:CURL_CA_BUNDLE=""
.venv\Scripts\uvicorn.exe api.main:app --host 127.0.0.1 --port 8000 --reload
```

Once running, interactive OpenAPI documentation is automatically available at:
- **Swagger UI:** `http://127.0.0.1:8000/docs`
- **ReDoc:** `http://127.0.0.1:8000/redoc`
