"""Inference engine and input validator for the Hybrid Quantum Classifier."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

from src.data.dataset import TARGET_LABEL_MAPPING, prepare_breast_cancer_data
from src.models.hybrid import HybridQuantumClassifier


@dataclass
class PredictionResult:
    """Container for batch and single-sample inference results."""
    probabilities_benign: np.ndarray
    probabilities_malignant: np.ndarray
    predicted_labels: np.ndarray
    predicted_class_names: List[str]
    confidences: np.ndarray
    encoder_angles: np.ndarray
    quantum_expectation_values: np.ndarray
    raw_logits: np.ndarray
    sample_count: int


class HybridPredictor:
    """
    Production-grade inference wrapper for the trained Hybrid Quantum-Classical Classifier.
    """

    EXPECTED_FEATURES: List[str] = list(load_breast_cancer().feature_names)

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        random_seed: int = 42,
    ) -> None:
        """
        Initialize predictor by loading checkpoint and pre-fitted training scaler.

        Args:
            checkpoint_path: Path to saved PyTorch checkpoint (best_model.pt).
            random_seed: Random seed used to recover exact training scaler fit.
        """
        self.device = torch.device("cpu")
        self.checkpoint_path = self._resolve_checkpoint(checkpoint_path)
        
        # 1. Load exact fitted preprocessing scaler from training partition (zero leakage)
        splits = prepare_breast_cancer_data(random_seed=random_seed)
        self.scaler: StandardScaler = splits.scaler
        self.feature_names: List[str] = splits.feature_names
        self.feature_means: np.ndarray = splits.X_train_raw.mean(axis=0)
        self.feature_stds: np.ndarray = splits.X_train_raw.std(axis=0)
        self.feature_mins: np.ndarray = splits.X_train_raw.min(axis=0)
        self.feature_maxs: np.ndarray = splits.X_train_raw.max(axis=0)

        # 2. Load model from checkpoint
        self.model, self.config = self._load_model_from_checkpoint(self.checkpoint_path)

    def _resolve_checkpoint(self, path: Optional[str]) -> Path:
        """Find valid checkpoint path with sensible fallbacks."""
        candidates = [
            path,
            "results/hybrid_quantum/best_model.pt",
            "results/controlled_study/Hybrid_Reuploading_L2_seed_42/best_model.pt",
            "results/best_model.pt",
        ]
        for c in candidates:
            if c is not None:
                p = Path(c)
                if p.exists():
                    return p
        raise FileNotFoundError(
            f"Could not locate a valid model checkpoint. Checked candidates: {candidates}"
        )

    def _load_model_from_checkpoint(
        self, checkpoint_path: Path
    ) -> Tuple[HybridQuantumClassifier, Dict[str, Any]]:
        """Instantiate architecture and restore weights from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {})

        model_kwargs = config.get("model", {})
        input_dim = model_kwargs.get("input_dim", 30)
        n_qubits = model_kwargs.get("n_qubits", 4)
        n_layers = model_kwargs.get("n_layers", 2)
        encoder_hidden_dim = model_kwargs.get("encoder_hidden_dim", 16)

        model = HybridQuantumClassifier(
            input_dim=input_dim,
            n_qubits=n_qubits,
            n_layers=n_layers,
            encoder_hidden_dim=encoder_hidden_dim,
        ).to(self.device)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, config

    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate uploaded DataFrame against expected 30 features, numeric types, and nulls.
        """
        errors: List[str] = []

        if not isinstance(df, pd.DataFrame):
            return False, ["Input is not a pandas DataFrame."]

        if len(df) == 0:
            return False, ["The uploaded file contains 0 rows of data."]

        # Check column presence and order
        missing_cols = [col for col in self.EXPECTED_FEATURES if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing {len(missing_cols)} required feature columns: {missing_cols[:5]}...")

        if not missing_cols:
            # Check for non-numeric values
            for col in self.EXPECTED_FEATURES:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    errors.append(f"Column '{col}' must be numeric, but found type '{df[col].dtype}'.")

            # Check for null / NaN values
            null_counts = df[self.EXPECTED_FEATURES].isnull().sum()
            total_nulls = int(null_counts.sum())
            if total_nulls > 0:
                cols_with_nulls = null_counts[null_counts > 0].to_dict()
                errors.append(f"Found {total_nulls} missing/null values across columns: {cols_with_nulls}")

        return len(errors) == 0, errors

    def predict(self, data: Union[pd.DataFrame, np.ndarray, Dict[str, float]]) -> PredictionResult:
        """
        Execute end-to-end inference on validated data using existing fitted scaler.
        """
        if isinstance(data, dict):
            # Single sample dictionary
            missing = [f for f in self.EXPECTED_FEATURES if f not in data]
            if missing:
                raise ValueError(f"Missing required features: {missing}")
            arr = np.array([[data[f] for f in self.EXPECTED_FEATURES]], dtype=np.float32)
        elif isinstance(data, pd.DataFrame):
            is_valid, errors = self.validate_dataframe(data)
            if not is_valid:
                raise ValueError("DataFrame validation failed: " + "; ".join(errors))
            arr = data[self.EXPECTED_FEATURES].to_numpy(dtype=np.float32)
        elif isinstance(data, np.ndarray):
            if data.ndim != 2 or data.shape[1] != len(self.EXPECTED_FEATURES):
                raise ValueError(
                    f"Expected 2D numpy array with shape (N, {len(self.EXPECTED_FEATURES)}), got {data.shape}"
                )
            if not np.isfinite(data).all():
                raise ValueError("Input array contains NaN or infinite values.")
            arr = data.astype(np.float32)
        else:
            raise TypeError(f"Unsupported input type: {type(data)}")

        # Apply existing fitted scaler (NEVER fit on inference inputs)
        scaled_arr = self.scaler.transform(arr)
        x_tensor = torch.tensor(scaled_arr, dtype=torch.float32, device=self.device)

        self.model.eval()
        with torch.no_grad():
            angles_tensor = self.model.encode_features(x_tensor)
            expvals_tensor = self.model.quantum_features(x_tensor)
            logits_tensor = self.model(x_tensor)
            probs_benign_tensor = torch.sigmoid(logits_tensor)

        logits = logits_tensor.cpu().numpy().ravel()
        probs_benign = probs_benign_tensor.cpu().numpy().ravel()
        probs_malignant = 1.0 - probs_benign
        predicted_labels = (probs_benign >= 0.5).astype(int)
        predicted_names = [TARGET_LABEL_MAPPING[lbl] for lbl in predicted_labels]
        confidences = np.maximum(probs_benign, probs_malignant)
        angles = angles_tensor.cpu().numpy()
        expvals = expvals_tensor.cpu().numpy()

        return PredictionResult(
            probabilities_benign=probs_benign,
            probabilities_malignant=probs_malignant,
            predicted_labels=predicted_labels,
            predicted_class_names=predicted_names,
            confidences=confidences,
            encoder_angles=angles,
            quantum_expectation_values=expvals,
            raw_logits=logits,
            sample_count=len(arr),
        )
