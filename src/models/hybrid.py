"""Hybrid Classical-Quantum Classifier combining PyTorch neural networks and PennyLane QNodes."""

from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn

from src.models.quantum import QuantumDataReuploadingLayer


class ClassicalFeatureEncoder(nn.Module):
    """
    Encodes 30 standardized input features into 4 bounded quantum rotation angles in [-pi, pi].
    """

    def __init__(
        self,
        input_dim: int = 30,
        latent_dim: int = 4,
        hidden_dim: int = 16,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map input features to bounded rotation angles in [-pi, pi].

        Args:
            x: Standardized features tensor of shape (batch_size, 30).

        Returns:
            Angles tensor of shape (batch_size, 4) with values in [-pi, pi].
        """
        raw = self.net(x)
        # Bounding to [-pi, pi] via tanh activation
        angles = torch.pi * torch.tanh(raw)
        return angles


class ClassicalOutputHead(nn.Module):
    """
    Maps 4 measured Pauli-Z expectation values into a single binary classification logit.
    """

    def __init__(self, in_features: int = 4) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Pauli-Z expectation values of shape (batch_size, 4).

        Returns:
            Logits tensor of shape (batch_size, 1).
        """
        return self.linear(x)


class HybridQuantumClassifier(nn.Module):
    """
    End-to-end differentiable Hybrid Quantum-Classical Classifier.

    Architecture:
        x (30 features)
          -> Classical Feature Encoder
          -> 4 bounded angles in [-pi, pi]
          -> 4-Qubit Data Re-uploading Circuit (PennyLane)
          -> 4 Pauli-Z expectation values in [-1, 1]
          -> Classical Output Head
          -> 1 Binary Classification Logit
    """

    def __init__(
        self,
        input_dim: int = 30,
        n_qubits: int = 4,
        n_layers: int = 2,
        encoder_hidden_dim: int = 16,
    ) -> None:
        """
        Args:
            input_dim: Number of input features (default 30).
            n_qubits: Number of quantum wires / qubits (default 4).
            n_layers: Number of quantum data re-uploading layers (default 2).
            encoder_hidden_dim: Hidden dimension for classical encoder.
        """
        super().__init__()

        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # 1. Classical Feature Encoder: 30 -> 4 bounded angles
        self.encoder = ClassicalFeatureEncoder(
            input_dim=input_dim,
            latent_dim=n_qubits,
            hidden_dim=encoder_hidden_dim,
        )

        # 2. Quantum Data Re-uploading Circuit: 4 angles -> 4 Pauli-Z expectation values
        self.quantum_layer = QuantumDataReuploadingLayer(
            n_qubits=n_qubits,
            n_layers=n_layers,
        )

        # 3. Classical Output Head: 4 expectation values -> 1 logit
        self.output_head = ClassicalOutputHead(in_features=n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        End-to-end forward pass producing binary classification logits.

        Args:
            x: Tensor of standardized input features, shape (batch_size, 30).

        Returns:
            Logits tensor of shape (batch_size, 1).
        """
        angles = self.encoder(x)
        expvals = self.quantum_layer(angles)
        logits = self.output_head(expvals)
        return logits

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        """Helper to inspect classical encoder angles in [-pi, pi]."""
        return self.encoder(x)

    def quantum_features(self, x: torch.Tensor) -> torch.Tensor:
        """Helper to inspect measured quantum expectation values in [-1, 1]."""
        angles = self.encoder(x)
        return self.quantum_layer(angles)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute predicted probabilities for class 1 (benign)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
        return probs

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Compute binary class predictions given threshold."""
        probs = self.predict_proba(x)
        return (probs >= threshold).long()
