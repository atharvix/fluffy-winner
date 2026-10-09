"""Differentiable 4-qubit Quantum Data Re-uploading layer using PennyLane and PyTorch.

Mathematical Structure:
    |psi(x, theta)> = U_L(theta_L) E(x) ... U_1(theta_1) E(x) |0000>

Circuit Specification per re-uploading layer l:
    1. Data Encoding: RY(x_i) on each wire i in {0, 1, 2, 3}.
    2. Trainable Single-Qubit Rotations: RY(theta_ry[l, i]) followed by RZ(theta_rz[l, i]) on each wire i.
    3. Entangling Gates: CNOT in ring topology:
       - CNOT(0, 1)
       - CNOT(1, 2)
       - CNOT(2, 3)
       - CNOT(3, 0)
    4. Measurement: Pauli-Z expectation value on each qubit <Z_i> in [-1, 1].

Differentiation Method:
    We use `diff_method="backprop"` with the `"default.qubit"` simulator.
    `diff_method="backprop"` integrates directly with PyTorch's reverse-mode automatic differentiation
    (autograd) via PennyLane's pure Python/PyTorch statevector simulator backend. This computes exact
    analytic gradients in a single backward pass without requiring multiple 2*N parameter-shift circuit
    evaluations, ensuring fast, deterministic, CPU-compatible training and gradient propagation.
"""

from typing import List, Tuple
import pennylane as qml
import torch
import torch.nn as nn


class QuantumDataReuploadingLayer(nn.Module):
    """
    Four-qubit variational quantum circuit module with data re-uploading and ring entanglement.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
    ) -> None:
        """
        Args:
            n_qubits: Number of qubits (fixed at 4 for this architecture).
            n_layers: Number of data re-uploading layers L (default 2).
        """
        super().__init__()

        if n_qubits != 4:
            raise ValueError(f"This architecture requires exactly 4 qubits, but got {n_qubits}.")
        if n_layers < 1:
            raise ValueError(f"Number of layers must be >= 1, got {n_layers}.")

        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Trainable parameters for each layer and qubit
        # Initialized with small normal weights to break symmetry
        self.weights_ry = nn.Parameter(
            torch.randn(n_layers, n_qubits, dtype=torch.float32) * 0.1
        )
        self.weights_rz = nn.Parameter(
            torch.randn(n_layers, n_qubits, dtype=torch.float32) * 0.1
        )

        # Initialize PennyLane device and QNode with PyTorch backpropagation
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self._qnode = self._build_qnode()

    def _build_qnode(self):
        """Construct the differentiable vectorized QNode."""
        n_qubits = self.n_qubits
        n_layers = self.n_layers

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs: torch.Tensor, w_ry: torch.Tensor, w_rz: torch.Tensor):
            """
            Args:
                inputs: (batch_size, n_qubits) batch of 4 bounded feature angles.
                w_ry: (n_layers, n_qubits) trainable RY parameters.
                w_rz: (n_layers, n_qubits) trainable RZ parameters.
            """
            for l in range(n_layers):
                # 1. Data re-uploading encoding: RY(x_i)
                for i in range(n_qubits):
                    qml.RY(inputs[:, i], wires=i)

                # 2. Trainable single-qubit rotations: RY and RZ
                for i in range(n_qubits):
                    qml.RY(w_ry[l, i], wires=i)
                    qml.RZ(w_rz[l, i], wires=i)

                # 3. Ring entanglement: CNOT(0->1), CNOT(1->2), CNOT(2->3), CNOT(3->0)
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])

            # 4. Measure Pauli-Z expectation values for each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        return circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the quantum circuit.

        Args:
            x: Input tensor of shape (batch_size, 4) containing bounded encoding angles.

        Returns:
            Tensor of Pauli-Z expectation values of shape (batch_size, 4), values in [-1, 1].
        """
        if x.ndim != 2 or x.shape[1] != self.n_qubits:
            raise ValueError(
                f"Expected input shape (batch_size, {self.n_qubits}), but got shape {tuple(x.shape)}."
            )

        # Execute QNode and stack measured expectation values across qubits
        expvals = self._qnode(x, self.weights_ry, self.weights_rz)
        # expvals is a list of 4 tensors each of shape (batch_size,)
        out = torch.stack(expvals, dim=1).to(dtype=x.dtype)
        return out

    @property
    def num_parameters(self) -> int:
        """Total number of trainable quantum parameters (2 * n_layers * n_qubits)."""
        return 2 * self.n_layers * self.n_qubits
