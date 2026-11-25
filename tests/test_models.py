"""Tests for Classical MLP model architecture and inference methods."""

import pytest
import torch
import torch.nn as nn

from src.models.classical import ClassicalBottleneckMLP, ClassicalMLP


def test_model_hidden_layers_requirement():
    """Verify that ClassicalMLP enforces at least two trainable hidden layers."""
    # Valid configurations with 2, 3, and 4 hidden layers
    model_2 = ClassicalMLP(input_dim=30, hidden_dims=[64, 32])
    assert model_2.num_hidden_layers == 2

    model_3 = ClassicalMLP(input_dim=30, hidden_dims=[128, 64, 32])
    assert model_3.num_hidden_layers == 3

    # Count linear layers: hidden_dims + 1 output layer
    linear_layers = [m for m in model_2.modules() if isinstance(m, nn.Linear)]
    assert len(linear_layers) == 3  # (30->64, 64->32, 32->1)

    # Invalid configuration with < 2 hidden layers should raise ValueError
    with pytest.raises(ValueError):
        ClassicalMLP(input_dim=30, hidden_dims=[64])

    with pytest.raises(ValueError):
        ClassicalMLP(input_dim=30, hidden_dims=[])


def test_model_forward_pass_and_predictions():
    """Verify forward pass output shape, predict_proba range [0, 1], and binary predictions."""
    batch_size = 8
    input_dim = 30
    model = ClassicalMLP(input_dim=input_dim, hidden_dims=[64, 32], dropout_rate=0.0)
    model.eval()

    dummy_input = torch.randn(batch_size, input_dim)

    # Test raw logits
    logits = model(dummy_input)
    assert logits.shape == (batch_size, 1)

    # Test probabilities
    probs = model.predict_proba(dummy_input)
    assert probs.shape == (batch_size, 1)
    assert torch.all(probs >= 0.0)
    assert torch.all(probs <= 1.0)

    # Test binary predictions
    preds = model.predict(dummy_input, threshold=0.5)
    assert preds.shape == (batch_size, 1)
    unique_preds = torch.unique(preds).tolist()
    for p in unique_preds:
        assert p in [0, 1]


def test_classical_bottleneck_mlp():
    """Verify ClassicalBottleneckMLP architecture, parameter count, and bounded latent features."""
    input_dim = 63
    model = ClassicalBottleneckMLP(input_dim=input_dim, hidden_dim=16, latent_dim=4)
    model.eval()

    # Exact parameter breakdown check: 1,024 + 68 + 5 = 1,097
    breakdown = model.parameter_breakdown
    assert breakdown["classical_encoder"] == 1092
    assert breakdown["quantum_circuit"] == 0
    assert breakdown["classical_output_head"] == 5
    assert breakdown["total_trainable"] == 1097

    x = torch.randn(10, input_dim)

    # Test encode_features bounded in [-pi, pi]
    latent = model.encode_features(x)
    assert latent.shape == (10, 4)
    assert torch.all(latent >= -torch.pi - 1e-5)
    assert torch.all(latent <= torch.pi + 1e-5)

    # Test forward pass logits
    logits = model(x)
    assert logits.shape == (10, 1)

    # Test predict_proba and predict
    probs = model.predict_proba(x)
    assert probs.shape == (10, 1)
    assert torch.all(probs >= 0.0)
    assert torch.all(probs <= 1.0)

    preds = model.predict(x, threshold=0.5)
    assert preds.shape == (10, 1)
    assert set(torch.unique(preds).tolist()).issubset({0, 1})
