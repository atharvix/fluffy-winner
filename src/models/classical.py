"""Classical PyTorch Multilayer Perceptron baseline model."""

from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn


class ClassicalBottleneckMLP(nn.Module):
    """
    Parameter-matched classical bottleneck baseline isolating quantum circuit utility.

    Architecture:
        x (input_dim)
          -> Linear(input_dim, hidden_dim) [default: 63 -> 16]
          -> ReLU()
          -> Linear(hidden_dim, latent_dim) [default: 16 -> 4]
          -> pi * tanh() (bounded latent angles in [-pi, pi])
          -> Linear(latent_dim, 1) [default: 4 -> 1 logit]

    Parameter count for input_dim=63, hidden_dim=16, latent_dim=4:
        Linear(63, 16): 63 * 16 + 16 = 1,024
        Linear(16, 4):  16 * 4 + 4   =    68
        Linear(4, 1):    4 * 1 + 1   =     5
        Total: 1,097 trainable parameters (strictly matches the classical components of Hybrid models)
    """

    def __init__(
        self,
        input_dim: int = 63,
        hidden_dim: int = 16,
        latent_dim: int = 4,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.output_head = nn.Linear(latent_dim, 1)

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        """Map input features to bounded latent representation in [-pi, pi]."""
        raw = self.encoder(x)
        return torch.pi * torch.tanh(raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass producing raw logits."""
        latent = self.encode_features(x)
        logits = self.output_head(latent)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute predicted probabilities for class 1 (ATTACK)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
        return probs

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Compute binary class predictions given threshold."""
        probs = self.predict_proba(x)
        return (probs >= threshold).long()

    @property
    def parameter_breakdown(self) -> Dict[str, int]:
        """Detailed breakdown of trainable parameters."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        output_params = sum(p.numel() for p in self.output_head.parameters() if p.requires_grad)
        return {
            "classical_encoder": encoder_params,
            "quantum_circuit": 0,
            "classical_output_head": output_params,
            "total_trainable": encoder_params + output_params,
        }


class ClassicalMLP(nn.Module):
    """
    Classical Multilayer Perceptron for binary classification.
    
    Contains two or more trainable hidden layers followed by a single logit output.
    """

    def __init__(
        self,
        input_dim: int = 30,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True,
    ) -> None:
        """
        Args:
            input_dim: Number of input features (default 30 for Breast Cancer dataset).
            hidden_dims: List of integers specifying hidden layer units (minimum 2 layers).
                         Defaults to [64, 32] -> 2 hidden layers.
            dropout_rate: Dropout probability for regularization.
            use_batch_norm: Whether to apply batch normalization between linear layers and activations.
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        if len(hidden_dims) < 2:
            raise ValueError(
                f"ClassicalMLP requires at least 2 hidden layers, but got {len(hidden_dims)}: {hidden_dims}"
            )

        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        layers: List[nn.Module] = []
        prev_dim = input_dim

        for i, h_dim in enumerate(self.hidden_dims):
            linear_layer = nn.Linear(prev_dim, h_dim)
            layers.append(linear_layer)
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
                
            layers.append(nn.ReLU())
            
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))
                
            prev_dim = h_dim

        # Final output layer: single logit output for binary classification
        self.feature_extractor = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, 1)

    @property
    def num_hidden_layers(self) -> int:
        """Return the number of hidden layers."""
        return len(self.hidden_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing raw logits.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Logits tensor of shape (batch_size, 1)
        """
        features = self.feature_extractor(x)
        logits = self.output_layer(features)
        return logits

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

    @property
    def parameter_breakdown(self) -> dict:
        """Detailed breakdown of trainable parameters."""
        extractor_params = sum(p.numel() for p in self.feature_extractor.parameters() if p.requires_grad)
        output_params = sum(p.numel() for p in self.output_layer.parameters() if p.requires_grad)
        return {
            "feature_extractor": extractor_params,
            "output_layer": output_params,
            "quantum_circuit": 0,
            "total_trainable": extractor_params + output_params,
        }

