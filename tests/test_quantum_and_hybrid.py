import math
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from src.models.hybrid import (
    ClassicalFeatureEncoder,
    ClassicalOutputHead,
    HybridQuantumClassifier,
)
from src.models.quantum import QuantumDataReuploadingLayer


def test_quantum_circuit_batch_input_and_output_shape():
    """Verify that the quantum circuit accepts a batch of 4D feature vectors and produces finite (B, 4) expectation values."""
    batch_size = 12
    n_qubits = 4
    n_layers = 2

    qlayer = QuantumDataReuploadingLayer(n_qubits=n_qubits, n_layers=n_layers)
    x = torch.randn(batch_size, n_qubits)

    out = qlayer(x)

    # Verify tensor shape
    assert out.shape == (batch_size, n_qubits), f"Expected shape ({batch_size}, {n_qubits}), got {out.shape}"
    # Verify all outputs are finite numbers (no NaN or inf)
    assert torch.isfinite(out).all().item(), "Quantum circuit produced non-finite values"


def test_quantum_measurements_range_within_minus_one_to_one():
    """Verify that measured Pauli-Z expectation values remain within [-1.0, 1.0] within numerical tolerance."""
    qlayer = QuantumDataReuploadingLayer(n_qubits=4, n_layers=3)
    # Test across a variety of angles including extreme values
    test_inputs = torch.cat([
        torch.zeros(4, 4),
        torch.ones(4, 4) * math.pi,
        torch.ones(4, 4) * -math.pi,
        torch.randn(32, 4) * 2 * math.pi,
    ], dim=0)

    out = qlayer(test_inputs)

    min_val = out.min().item()
    max_val = out.max().item()
    tolerance = 1e-5

    assert min_val >= -1.0 - tolerance, f"Minimum expectation value {min_val} is below -1.0"
    assert max_val <= 1.0 + tolerance, f"Maximum expectation value {max_val} is above 1.0"


def test_reuploading_layer_count_changes_configuration():
    """Verify that changing re-uploading layer count updates parameter shapes and model configuration."""
    layer_2 = QuantumDataReuploadingLayer(n_qubits=4, n_layers=2)
    layer_4 = QuantumDataReuploadingLayer(n_qubits=4, n_layers=4)

    assert layer_2.weights_ry.shape == (2, 4)
    assert layer_2.weights_rz.shape == (2, 4)
    assert layer_2.num_parameters == 16

    assert layer_4.weights_ry.shape == (4, 4)
    assert layer_4.weights_rz.shape == (4, 4)
    assert layer_4.num_parameters == 32

    # Verification that invalid configs raise ValueError
    with pytest.raises(ValueError):
        QuantumDataReuploadingLayer(n_qubits=3, n_layers=2)  # must be 4 qubits
    with pytest.raises(ValueError):
        QuantumDataReuploadingLayer(n_qubits=4, n_layers=0)  # layers must be >= 1


def test_hybrid_model_returns_one_logit_per_sample():
    """Verify that HybridQuantumClassifier maps 30-dim inputs to 1 binary-classification logit per sample."""
    batch_size = 16
    input_dim = 30
    model = HybridQuantumClassifier(input_dim=input_dim, n_qubits=4, n_layers=2)

    x = torch.randn(batch_size, input_dim)

    # 1. Forward output (logits)
    logits = model(x)
    assert logits.shape == (batch_size, 1), f"Expected shape ({batch_size}, 1), got {logits.shape}"
    assert torch.isfinite(logits).all().item()

    # 2. Predicted probabilities
    probs = model.predict_proba(x)
    assert probs.shape == (batch_size, 1)
    assert (probs >= 0.0).all().item() and (probs <= 1.0).all().item()

    # 3. Binary predictions
    preds = model.predict(x, threshold=0.5)
    assert preds.shape == (batch_size, 1)
    unique_preds = torch.unique(preds).tolist()
    for p in unique_preds:
        assert p in [0, 1]

    # 4. Feature encoder intermediate angles in [-pi, pi]
    angles = model.encode_features(x)
    assert angles.shape == (batch_size, 4)
    assert angles.min().item() >= -math.pi - 1e-5
    assert angles.max().item() <= math.pi + 1e-5

    # 5. Quantum expectation values in [-1, 1]
    q_expvals = model.quantum_features(x)
    assert q_expvals.shape == (batch_size, 4)
    assert q_expvals.min().item() >= -1.0 - 1e-5
    assert q_expvals.max().item() <= 1.0 + 1e-5


def test_gradients_propagate_end_to_end_through_quantum_and_classical():
    """Verify that gradients propagate from the loss through output head, quantum circuit, and into classical encoder."""
    torch.manual_seed(42)
    model = HybridQuantumClassifier(input_dim=30, n_qubits=4, n_layers=2)

    x = torch.randn(8, 30)
    y = torch.tensor([[1.0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0]])

    criterion = nn.BCEWithLogitsLoss()
    logits = model(x)
    loss = criterion(logits, y)

    # Backward pass
    loss.backward()

    # 1. Output head gradients
    head_weight_grad = model.output_head.linear.weight.grad
    assert head_weight_grad is not None, "Output head weight gradient is None"
    assert torch.isfinite(head_weight_grad).all().item(), "Output head weight gradient has non-finite values"
    assert not torch.all(head_weight_grad == 0.0), "Output head weight gradient is completely zero"

    # 2. Quantum circuit parameters gradients
    q_ry_grad = model.quantum_layer.weights_ry.grad
    q_rz_grad = model.quantum_layer.weights_rz.grad
    assert q_ry_grad is not None, "Quantum RY parameter gradient is None"
    assert q_rz_grad is not None, "Quantum RZ parameter gradient is None"
    assert torch.isfinite(q_ry_grad).all().item(), "Quantum RY gradient contains non-finite values"
    assert torch.isfinite(q_rz_grad).all().item(), "Quantum RZ gradient contains non-finite values"
    assert not torch.all(q_ry_grad == 0.0), "Quantum RY gradient is completely zero"
    assert not torch.all(q_rz_grad == 0.0), "Quantum RZ gradient is completely zero"

    # 3. Classical encoder gradients (ensures gradients flowed through the quantum circuit)
    first_encoder_layer_grad = model.encoder.net[0].weight.grad
    assert first_encoder_layer_grad is not None, "First encoder layer gradient is None"
    assert torch.isfinite(first_encoder_layer_grad).all().item(), "Encoder layer gradient contains non-finite values"
    assert not torch.all(first_encoder_layer_grad == 0.0), "Encoder layer gradient is completely zero"


def test_hybrid_training_short_smoke_run():
    """Verify that train_hybrid executes a mini-run, tracks loss/metrics, and produces valid checkpoints."""
    import tempfile
    from src.training.train import train_hybrid

    with tempfile.TemporaryDirectory() as tmp_dir:
        res = train_hybrid(
            random_seed=42,
            epochs=2,
            patience=2,
            n_qubits=4,
            n_layers=2,
            batch_size=64,
            output_dir=tmp_dir,
        )

        assert res["metrics"]["best_epoch"] >= 1
        assert "test_partition" in res["metrics"]
        assert "malignant_sensitivity" in res["metrics"]["test_partition"]["metrics"]
        assert "malignant_specificity" in res["metrics"]["test_partition"]["metrics"]
        assert Path(res["checkpoint_path"]).exists()
