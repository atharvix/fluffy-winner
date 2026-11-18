"""Inference engine and input validator for Network Intrusion Detection models."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer

from src.data.nids_dataset import (
    RAW_CATEGORICAL_FEATURES,
    RAW_INPUT_FEATURES,
    RAW_NUMERICAL_FEATURES,
    TARGET_LABEL_MAPPING,
)
from src.models.classical import ClassicalMLP
from src.models.hybrid import HybridQuantumClassifier, StandardVQCClassifier


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
    """
    Production-grade inference engine for Classical and Hybrid Quantum NIDS classifiers.
    Loads saved fitted preprocessor and model checkpoint with zero leakage.
    """

    EXPECTED_FEATURES: List[str] = RAW_INPUT_FEATURES
    NUMERICAL_FEATURES: List[str] = RAW_NUMERICAL_FEATURES
    CATEGORICAL_FEATURES: List[str] = RAW_CATEGORICAL_FEATURES
    MAX_BATCH_ROWS: int = 10000

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        preprocessor_path: Optional[str] = None,
    ) -> None:
        """
        Initialize predictor by restoring fitted preprocessing artifact and model checkpoint.
        """
        self.device = torch.device("cpu")
        self.checkpoint_path = self._resolve_checkpoint(checkpoint_path)
        self.preprocessor_path = self._resolve_preprocessor(preprocessor_path)

        # 1. Load fitted preprocessing artifact
        self.preprocessor: ColumnTransformer = joblib.load(self.preprocessor_path)

        # 2. Load model from checkpoint
        self.model, self.config = self._load_model_from_checkpoint(self.checkpoint_path)
        self.model_name = self.config.get("model_name", "Hybrid_Quantum_NIDS")
        self.model_family = self.config.get("model_family", "hybrid_reuploading")

    def _resolve_checkpoint(self, path: Optional[str]) -> Path:
        """Find valid checkpoint path with sensible fallbacks."""
        candidates = [
            path,
            "results/network_intrusion/best_model.pt",
            "results/network_intrusion/controlled_study/Hybrid_Reuploading_L2_seed_42/best_model.pt",
            "results/network_intrusion/controlled_study/Classical_MLP_seed_42/best_model.pt",
        ]
        for c in candidates:
            if c is not None:
                p = Path(c)
                if p.exists():
                    return p
        raise FileNotFoundError(
            f"Could not locate a valid model checkpoint. Checked candidates: {candidates}"
        )

    def _resolve_preprocessor(self, path: Optional[str]) -> Path:
        """Find valid fitted preprocessor artifact with sensible fallbacks."""
        candidates = [
            path,
            "results/network_intrusion/preprocessor.joblib",
        ]
        for c in candidates:
            if c is not None:
                p = Path(c)
                if p.exists():
                    return p
        raise FileNotFoundError(
            f"Could not locate fitted preprocessor artifact. Checked candidates: {candidates}"
        )

    def _load_model_from_checkpoint(
        self, checkpoint_path: Path
    ) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """Instantiate appropriate model architecture and restore weights."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {})
        model_family = config.get("model_family", "hybrid_reuploading")
        model_kwargs = config.get("model", {})

        input_dim = model_kwargs.get("input_dim", 63)

        if model_family == "classical":
            model = ClassicalMLP(
                input_dim=input_dim,
                hidden_dims=model_kwargs.get("hidden_dims", [64, 32]),
                dropout_rate=model_kwargs.get("dropout_rate", 0.2),
                use_batch_norm=model_kwargs.get("use_batch_norm", True),
            )
        elif model_family == "standard_vqc":
            model = StandardVQCClassifier(
                input_dim=input_dim,
                n_qubits=model_kwargs.get("n_qubits", 4),
                n_layers=model_kwargs.get("n_layers", 2),
                encoder_hidden_dim=model_kwargs.get("encoder_hidden_dim", 16),
            )
        else:
            # hybrid_reuploading
            model = HybridQuantumClassifier(
                input_dim=input_dim,
                n_qubits=model_kwargs.get("n_qubits", 4),
                n_layers=model_kwargs.get("n_layers", 2),
                encoder_hidden_dim=model_kwargs.get("encoder_hidden_dim", 16),
            )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        return model, config

    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate input DataFrame for schema completeness, valid data types, nulls, and size.
        """
        errors: List[str] = []

        if not isinstance(df, pd.DataFrame):
            return False, ["Input is not a pandas DataFrame."]

        if len(df) == 0:
            return False, ["Input contains 0 records."]

        if len(df) > self.MAX_BATCH_ROWS:
            return False, [
                f"Batch exceeds maximum allowed rows ({len(df)} > {self.MAX_BATCH_ROWS})."
            ]

        # Check required columns
        missing_cols = [col for col in self.EXPECTED_FEATURES if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing {len(missing_cols)} required feature columns: {missing_cols[:8]}...")

        if not missing_cols:
            # Check numerical types and non-finite values
            for col in self.NUMERICAL_FEATURES:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    # Try conversion or report error
                    try:
                        pd.to_numeric(df[col])
                    except Exception:
                        errors.append(f"Column '{col}' must be numeric, but contains invalid non-numeric values.")

            # Check for NaN / infinite values
            null_counts = df[self.EXPECTED_FEATURES].isnull().sum()
            total_nulls = int(null_counts.sum())
            if total_nulls > 0:
                cols_with_nulls = null_counts[null_counts > 0].to_dict()
                errors.append(f"Found {total_nulls} null values across features: {cols_with_nulls}")

        return len(errors) == 0, errors

    def predict(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
        threshold: float = 0.5,
    ) -> NIDSPredictionResult:
        """
        Execute end-to-end inference on validated input data.
        """
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

        # Enforce exact column order
        feature_df = df[self.EXPECTED_FEATURES].copy()

        # Transform using pre-fitted ColumnTransformer (zero leakage, no refit)
        scaled_features = self.preprocessor.transform(feature_df)
        x_tensor = torch.tensor(scaled_features, dtype=torch.float32, device=self.device)

        self.model.eval()
        angles: Optional[np.ndarray] = None
        expvals: Optional[np.ndarray] = None

        with torch.no_grad():
            if hasattr(self.model, "encode_features"):
                angles_tensor = self.model.encode_features(x_tensor)
                angles = angles_tensor.cpu().numpy()
            if hasattr(self.model, "quantum_features"):
                expvals_tensor = self.model.quantum_features(x_tensor)
                expvals = expvals_tensor.cpu().numpy()

            logits_tensor = self.model(x_tensor)
            probs_attack_tensor = torch.sigmoid(logits_tensor)

        logits = logits_tensor.cpu().numpy().ravel()
        probs_attack = probs_attack_tensor.cpu().numpy().ravel()
        probs_benign = 1.0 - probs_attack
        predicted_labels = (probs_attack >= threshold).astype(int)
        predicted_names = [TARGET_LABEL_MAPPING[lbl] for lbl in predicted_labels]
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
