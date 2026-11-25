"""Inference engine and input validator for CIC-IoT-2023 & UNSW-NB15 NIDS models."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import torch

from src.models.classical import ClassicalBottleneckMLP, ClassicalMLP
from src.models.hybrid import HybridQuantumClassifier, StandardVQCClassifier

CICIOT_FEATURES: List[str] = ["Header_Length", "Protocol Type", "Time_To_Live", "Rate", "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number", "ack_flag_number", "ece_flag_number", "cwr_flag_number", "ack_count", "syn_count", "fin_count", "rst_count", "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP", "ARP", "ICMP", "IGMP", "IPv", "LLC", "Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "IAT", "Number", "Variance"]
TARGET_LABEL_MAPPING: Dict[int, str] = {0: "BENIGN", 1: "ATTACK"}


@dataclass
class NIDSPredictionResult:
    """Container for batch and single-record NIDS inference outputs."""
    probabilities_attack: np.ndarray
    probabilities_benign: np.ndarray
    predicted_labels: np.ndarray
    predicted_class_names: List[str]
    confidences: np.ndarray
    encoder_angles: Optional[np.ndarray]
    quantum_expectation_values: Optional[np.ndarray]
    raw_logits: np.ndarray
    sample_count: int


class NIDSPredictor:
    """Production inference engine for Classical & Hybrid Quantum NIDS (CIC-IoT-2023 & UNSW-NB15)."""

    EXPECTED_FEATURES: List[str] = CICIOT_FEATURES
    NUMERICAL_FEATURES: List[str] = CICIOT_FEATURES
    CATEGORICAL_FEATURES: List[str] = []
    MAX_BATCH_ROWS: int = 10000

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        preprocessor_path: Optional[str] = None,
    ) -> None:
        self.device = torch.device("cpu")
        self.checkpoint_path = self._resolve_checkpoint(checkpoint_path)
        self.preprocessor_path = self._resolve_preprocessor(preprocessor_path)
        self.preprocessor = joblib.load(self.preprocessor_path)
        
        # Auto-detect feature schema from preprocessor type
        if hasattr(self.preprocessor, "transformers_"):
            from src.data.nids_dataset import RAW_INPUT_FEATURES, RAW_NUMERICAL_FEATURES, RAW_CATEGORICAL_FEATURES
            self.EXPECTED_FEATURES = list(RAW_INPUT_FEATURES)
            self.NUMERICAL_FEATURES = list(RAW_NUMERICAL_FEATURES)
            self.CATEGORICAL_FEATURES = list(RAW_CATEGORICAL_FEATURES)
            self.dataset_name = "UNSW-NB15"
        else:
            self.EXPECTED_FEATURES = list(CICIOT_FEATURES)
            self.NUMERICAL_FEATURES = list(CICIOT_FEATURES)
            self.CATEGORICAL_FEATURES = []
            self.dataset_name = "CIC-IoT-2023 (45,019,243 Flows)"

        self.model, self.config, self.decision_threshold = self._load_model_from_checkpoint(self.checkpoint_path)
        self.model_name = self.config.get("model_name", "Hybrid_Reuploading_L1")
        self.model_family = self.config.get("model_family", "hybrid_reuploading")

    def _resolve_checkpoint(self, path: Optional[str]) -> Path:
        candidates = [
            path,
            "results/ciciot2023_full45m/best_quantum_model.pt",
            "results/ciciot2023_full45m/controlled_study/Hybrid_Reuploading_L1_seed42/best_model.pt",
            "results/ciciot2023_full45m/best_model.pt",
            "results/network_intrusion/best_model.pt",
        ]
        for c in candidates:
            if c is not None and Path(c).exists():
                return Path(c)
        raise FileNotFoundError(f"Could not locate checkpoint in {candidates}")

    def _resolve_preprocessor(self, path: Optional[str]) -> Path:
        candidates = [
            path,
            "results/ciciot2023_full45m/preprocessor.joblib",
            "results/network_intrusion/preprocessor.joblib",
        ]
        for c in candidates:
            if c is not None and Path(c).exists():
                return Path(c)
        raise FileNotFoundError(f"Could not locate preprocessor in {candidates}")

    def _load_model_from_checkpoint(self, checkpoint_path: Path) -> Tuple[torch.nn.Module, Dict[str, Any], float]:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {})
        if not config:
            config = {
                "model_name": checkpoint.get("model_name", "Hybrid_Reuploading_L1"),
                "model_family": checkpoint.get("model_family", "hybrid_reuploading"),
                "model": checkpoint.get("model_kwargs", {}),
            }
        model_family = checkpoint.get("model_family", config.get("model_family", "hybrid_reuploading"))
        model_kwargs = checkpoint.get("model_kwargs", config.get("model", {}))
        input_dim = int(checkpoint.get("input_dim", model_kwargs.get("input_dim", 39)))
        decision_threshold = float(checkpoint.get("decision_threshold", 0.2971))

        if model_family == "classical":
            model = ClassicalMLP(
                input_dim=input_dim,
                hidden_dims=model_kwargs.get("hidden_dims", [64, 32]),
                dropout_rate=model_kwargs.get("dropout_rate", 0.2),
                use_batch_norm=model_kwargs.get("use_batch_norm", True),
            )
        elif model_family == "classical_bottleneck":
            model = ClassicalBottleneckMLP(
                input_dim=input_dim,
                hidden_dim=model_kwargs.get("hidden_dim", 16),
                latent_dim=model_kwargs.get("latent_dim", 4),
            )
        elif model_family == "standard_vqc":
            model = StandardVQCClassifier(
                input_dim=input_dim,
                n_qubits=model_kwargs.get("n_qubits", 4),
                n_layers=model_kwargs.get("n_layers", 2),
                encoder_hidden_dim=model_kwargs.get("encoder_hidden_dim", 16),
            )
        else:
            model = HybridQuantumClassifier(
                input_dim=input_dim,
                n_qubits=model_kwargs.get("n_qubits", 4),
                n_layers=model_kwargs.get("n_layers", 1),
                encoder_hidden_dim=model_kwargs.get("encoder_hidden_dim", 16),
            )

        state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict"))
        model.load_state_dict(state_dict)
        model.eval()
        return model, config, decision_threshold

    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        if len(df) == 0:
            return False, ["Input dataset is empty."]
        if len(df) > self.MAX_BATCH_ROWS:
            return False, [f"Batch size {len(df)} exceeds limit of {self.MAX_BATCH_ROWS} rows."]
        missing = [c for c in self.EXPECTED_FEATURES if c not in df.columns]
        if missing:
            errors.append(f"Missing {len(missing)} required features: {missing[:6]}")
        
        # Check null values if testing UNSW pipeline
        if hasattr(self.preprocessor, "transformers_"):
            present_cols = [c for c in self.EXPECTED_FEATURES if c in df.columns]
            if df[present_cols].isnull().any().any():
                errors.append("DataFrame contains null values.")
            for col in self.NUMERICAL_FEATURES:
                if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                    errors.append(f"Column '{col}' must be numeric.")

        return len(errors) == 0, errors

    def predict(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
        threshold: Optional[float] = None,
    ) -> NIDSPredictionResult:
        eff_threshold = float(threshold) if threshold is not None else self.decision_threshold
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        is_valid, errors = self.validate_dataframe(df)
        if not is_valid:
            raise ValueError("Input validation failed: " + "; ".join(errors))

        if hasattr(self.preprocessor, "transformers_"):
            scaled = self.preprocessor.transform(df[self.EXPECTED_FEATURES])
            if hasattr(scaled, "toarray"):
                scaled = scaled.toarray()
            scaled = scaled.astype(np.float32)
        else:
            x_raw = df[self.EXPECTED_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)
            scaled = self.preprocessor.transform(x_raw).astype(np.float32)
        x_tensor = torch.tensor(scaled, dtype=torch.float32, device=self.device)

        self.model.eval()
        angles: Optional[np.ndarray] = None
        expvals: Optional[np.ndarray] = None
        with torch.no_grad():
            if hasattr(self.model, "encode_features"):
                angles = self.model.encode_features(x_tensor).cpu().numpy()
            if hasattr(self.model, "quantum_features"):
                expvals = self.model.quantum_features(x_tensor).cpu().numpy()
            logits_tensor = self.model(x_tensor)
            probs_attack_tensor = torch.sigmoid(logits_tensor)

        logits = logits_tensor.cpu().numpy().ravel()
        probs_attack = probs_attack_tensor.cpu().numpy().ravel()
        probs_benign = 1.0 - probs_attack
        predicted_labels = (probs_attack >= eff_threshold).astype(int)
        predicted_names = [TARGET_LABEL_MAPPING[int(lbl)] for lbl in predicted_labels]
        confidences = np.maximum(probs_attack, probs_benign)

        return NIDSPredictionResult(
            probabilities_attack=probs_attack,
            probabilities_benign=probs_benign,
            predicted_labels=predicted_labels,
            predicted_class_names=predicted_names,
            confidences=confidences,
            encoder_angles=angles,
            quantum_expectation_values=expvals,
            raw_logits=logits,
            sample_count=len(df),
        )
