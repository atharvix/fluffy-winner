export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

export interface HealthResponse {
  status: "healthy" | "degraded" | "offline";
  service: string;
  model_loaded: boolean;
  selected_model: string;
  device: string;
  uptime_seconds: number;
  timestamp: string;
  version: string;
}

export interface ModelInfoResponse {
  model_name: string;
  model_family: string;
  dataset: string;
  dataset_provenance: string;
  input_feature_count: number;
  raw_features: string[];
  numerical_features: string[];
  categorical_features: string[];
  class_label_mapping: Record<string, string>;
  positive_class: string;
  negative_class: string;
  decision_threshold: number;
  parameter_breakdown: {
    classical_encoder?: number;
    quantum_circuit?: number;
    classical_output_head?: number;
    total_trainable: number;
    feature_extractor?: number;
    output_layer?: number;
  };
  circuit_metadata: {
    type: string;
    n_qubits?: number;
    n_layers?: number;
    num_parameters?: number;
    encoding_gates_per_layer?: number;
    trainable_gates_per_layer?: number;
    cnot_gates_per_layer?: number;
    total_gates?: number;
    estimated_circuit_depth?: number;
  };
  environment?: Record<string, string>;
}

export interface SingleFlowRecord {
  id?: string | number;
  dur: number;
  proto: string;
  service: string;
  state: string;
  spkts: number;
  dpkts: number;
  sbytes: number;
  dbytes: number;
  rate: number;
  sttl: number;
  dttl: number;
  sload: number;
  dload: number;
  sloss: number;
  dloss: number;
  sinpkt: number;
  dinpkt: number;
  sjit: number;
  djit: number;
  swin: number;
  stcpb: number;
  dtcpb: number;
  dwin: number;
  tcprtt: number;
  synack: number;
  ackdat: number;
  smean: number;
  dmean: number;
  trans_depth: number;
  response_body_len: number;
  ct_srv_src: number;
  ct_state_ttl: number;
  ct_dst_ltm: number;
  ct_src_dport_ltm: number;
  ct_dst_sport_ltm: number;
  ct_dst_src_ltm: number;
  is_ftp_login: number;
  ct_ftp_cmd: number;
  ct_flw_http_mthd: number;
  ct_src_ltm: number;
  ct_srv_dst: number;
  is_sm_ips_ports: number;
  [key: string]: any;
}

export interface PredictionItem {
  record_id: string | number;
  index: number;
  predicted_label: "ATTACK" | "BENIGN";
  predicted_class_id: number;
  confidence: number;
  probabilities: {
    ATTACK: number;
    BENIGN: number;
  };
  raw_logit: number;
  quantum_inspection?: {
    encoder_angles?: number[];
    pauli_z_expectations?: number[];
  };
}

export interface PredictBatchResponse {
  status: string;
  sample_count: number;
  summary: {
    attack_count: number;
    benign_count: number;
    attack_ratio: number;
  };
  decision_threshold: number;
  predictions: PredictionItem[];
}

export interface CsvPredictResponse {
  status: string;
  filename: string;
  sample_count: number;
  summary: {
    attack_count: number;
    benign_count: number;
    attack_ratio: number;
  };
  decision_threshold: number;
  predictions: Array<{
    index: number;
    predicted_label: "ATTACK" | "BENIGN";
    confidence: number;
    probabilities: {
      ATTACK: number;
      BENIGN: number;
    };
  }>;
}

export interface FlowPreset {
  name: string;
  true_label: number;
  attack_cat: string;
  record: SingleFlowRecord;
}

export interface ModelBenchmarkStats {
  model_family: string;
  total_parameters: number;
  seed_count: number;
  seeds: number[];
  test_accuracy: { mean: number; std: number; min: number; max: number; formatted: string };
  test_balanced_accuracy: { mean: number; std: number; formatted: string };
  test_macro_f1: { mean: number; std: number; formatted: string };
  test_roc_auc: { mean: number; std: number; formatted: string };
  test_attack_recall: { mean: number; std: number; formatted: string };
  test_attack_precision: { mean: number; std: number; formatted: string };
  test_attack_f1: { mean: number; std: number; formatted: string };
  test_benign_recall: { mean: number; std: number; formatted: string };
  test_benign_precision: { mean: number; std: number; formatted: string };
  test_benign_f1: { mean: number; std: number; formatted: string };
  test_false_positive_rate: { mean: number; std: number; formatted: string };
  test_false_negative_rate: { mean: number; std: number; formatted: string };
  duration_seconds: { mean: number; std: number; formatted: string };
}

export type AggregateBenchmarkData = Record<string, ModelBenchmarkStats>;

// API Client Functions

export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE_URL}/api/health`, {
    headers: { Accept: "application/json" },
  });
  if (!res.ok) {
    throw new Error(`Health check failed with status ${res.status}`);
  }
  return res.json();
}

export const fetchHealth = checkHealth;


export async function fetchModelInfo(): Promise<ModelInfoResponse> {
  const res = await fetch(`${API_BASE_URL}/api/model-info`, {
    headers: { Accept: "application/json" },
  });
  if (!res.ok) {
    throw new Error(`Failed to fetch model info: ${res.statusText}`);
  }
  return res.json();
}

export async function predictBatch(
  records: SingleFlowRecord[],
  threshold: number = 0.5
): Promise<PredictBatchResponse> {
  const res = await fetch(`${API_BASE_URL}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ records, threshold }),
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorData.detail || `Prediction error (${res.status})`);
  }
  return res.json();
}

export async function predictCsvFile(
  file: File,
  threshold: number = 0.5,
  format: "json" | "csv" = "json"
): Promise<CsvPredictResponse | Blob> {
  const formData = new FormData();
  formData.append("file", file);

  const url = `${API_BASE_URL}/api/predict/csv?threshold=${threshold}&format=${format}`;
  const res = await fetch(url, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const errText = await res.text();
    let message = errText;
    try {
      const errJson = JSON.parse(errText);
      message = errJson.detail || errText;
    } catch {
      // Retain raw error text
    }
    throw new Error(message || `CSV inference failed with HTTP ${res.status}`);
  }

  if (format === "csv") {
    return res.blob();
  }
  return res.json();
}

export async function fetchLocalBenchmarks(): Promise<AggregateBenchmarkData> {
  const res = await fetch("/data/summary_statistics.json");
  if (!res.ok) {
    throw new Error("Unable to load summary statistics from repository artifacts.");
  }
  return res.json();
}

export async function fetchSamplePresets(): Promise<FlowPreset[]> {
  const res = await fetch("/data/sample_presets.json");
  if (!res.ok) {
    throw new Error("Unable to load flow presets.");
  }
  return res.json();
}
