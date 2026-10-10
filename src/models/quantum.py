"""Differentiable 4-qubit Quantum layers using PennyLane and PyTorch.

1. QuantumDataReuploadingLayer:
   Mathematical Structure:
       |psi(x, theta)> = U_L(theta_L) E(x) ... U_1(theta_1) E(x) |0000>
   Per re-uploading layer l:
       - Data Encoding: RY(x_i) on each wire i in {0, 1, 2, 3}.
       - Trainable Single-Qubit Rotations: RY(theta_ry[l, i]) and RZ(theta_rz[l, i]) on wire i.
       - Entangling Gates: CNOT in ring topology (0->1, 1->2, 2->3, 3->0).
       - Measurement: Pauli-Z expectation value on each qubit <Z_i> in [-1, 1].

2. StandardVQCLayer:
   Mathematical Structure:
       |psi(x, theta)> = U_L(theta_L) ... U_1(theta_1) E(x) |0000>
   Single encoding at t=0:
       - Data Encoding: RY(x_i) on wire i in {0, 1, 2, 3} (applied once only).
   Followed by L variational layers:
       - Trainable Single-Qubit Rotations: RY(theta_ry[l, i]) and RZ(theta_rz[l, i]) on wire i.
       - Entangling Gates: CNOT in ring topology (0->1, 1->2, 2->3, 3->0).
       - Measurement: Pauli-Z expectation value on each qubit <Z_i> in [-1, 1].

Differentiation Method:
    `diff_method="backprop"` with `"default.qubit"` simulator.
    Computes exact analytic gradients via PyTorch autograd in a single backward pass.
"""

from typing import Any, Dict, List, Tuple
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
        self.weights_ry = nn.Parameter(
            torch.randn(n_layers, n_qubits, dtype=torch.float32) * 0.1
        )
        self.weights_rz = nn.Parameter(
            torch.randn(n_layers, n_qubits, dtype=torch.float32) * 0.1
        )

        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self._qnode = self._build_qnode()

    def _build_qnode(self):
        """Construct the differentiable vectorized QNode."""
        n_qubits = self.n_qubits
        n_layers = self.n_layers

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs: torch.Tensor, w_ry: torch.Tensor, w_rz: torch.Tensor):
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
        Args:
            x: Input tensor of shape (batch_size, 4) containing bounded encoding angles.

        Returns:
            Tensor of Pauli-Z expectation values of shape (batch_size, 4), values in [-1, 1].
        """
        if x.ndim != 2 or x.shape[1] != self.n_qubits:
            raise ValueError(
                f"Expected input shape (batch_size, {self.n_qubits}), but got shape {tuple(x.shape)}."
            )

        expvals = self._qnode(x, self.weights_ry, self.weights_rz)
        out = torch.stack(expvals, dim=1).to(dtype=x.dtype)
        return out

    @property
    def num_parameters(self) -> int:
        """Total number of trainable quantum parameters (2 * n_layers * n_qubits)."""
        return 2 * self.n_layers * self.n_qubits

    @property
    def circuit_info(self) -> Dict[str, Any]:
        """Return circuit complexity metrics."""
        return {
            "type": "DataReuploadingVQC",
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "num_parameters": self.num_parameters,
            "encoding_gates_per_layer": self.n_qubits,
            "trainable_gates_per_layer": 2 * self.n_qubits,
            "cnot_gates_per_layer": self.n_qubits,
            "total_gates": self.n_layers * (4 * self.n_qubits),
            "estimated_circuit_depth": self.n_layers * 4,
        }


class StandardVQCLayer(nn.Module):
    """
    Four-qubit standard Variational Quantum Classifier (single initial data encoding, followed by L entangling layers).
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
    ) -> None:
        """
        Args:
            n_qubits: Number of qubits (fixed at 4).
            n_layers: Number of variational entangling layers L (default 2).
        """
        super().__init__()

        if n_qubits != 4:
            raise ValueError(f"This architecture requires exactly 4 qubits, but got {n_qubits}.")
        if n_layers < 1:
            raise ValueError(f"Number of layers must be >= 1, got {n_layers}.")

        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Trainable parameters for each layer and qubit
        self.weights_ry = nn.Parameter(
            torch.randn(n_layers, n_qubits, dtype=torch.float32) * 0.1
        )
        self.weights_rz = nn.Parameter(
            torch.randn(n_layers, n_qubits, dtype=torch.float32) * 0.1
        )

        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self._qnode = self._build_qnode()

    def _build_qnode(self):
        """Construct the differentiable vectorized QNode for standard VQC."""
        n_qubits = self.n_qubits
        n_layers = self.n_layers

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs: torch.Tensor, w_ry: torch.Tensor, w_rz: torch.Tensor):
            # 1. Single data encoding at t=0
            for i in range(n_qubits):
                qml.RY(inputs[:, i], wires=i)

            # 2. Variational layers
            for l in range(n_layers):
                # Trainable rotations: RY and RZ
                for i in range(n_qubits):
                    qml.RY(w_ry[l, i], wires=i)
                    qml.RZ(w_rz[l, i], wires=i)

                # Ring entanglement: CNOT(0->1), CNOT(1->2), CNOT(2->3), CNOT(3->0)
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])

            # 3. Measure Pauli-Z expectation values for each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        return circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, 4) containing bounded encoding angles.

        Returns:
            Tensor of Pauli-Z expectation values of shape (batch_size, 4), values in [-1, 1].
        """
        if x.ndim != 2 or x.shape[1] != self.n_qubits:
            raise ValueError(
                f"Expected input shape (batch_size, {self.n_qubits}), but got shape {tuple(x.shape)}."
            )

        expvals = self._qnode(x, self.weights_ry, self.weights_rz)
        out = torch.stack(expvals, dim=1).to(dtype=x.dtype)
        return out

    @property
    def num_parameters(self) -> int:
        """Total number of trainable quantum parameters (2 * n_layers * n_qubits)."""
        return 2 * self.n_layers * self.n_qubits

    @property
    def circuit_info(self) -> Dict[str, Any]:
        """Return circuit complexity metrics."""
        return {
            "type": "StandardVQC",
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "num_parameters": self.num_parameters,
            "encoding_gates_total": self.n_qubits,
            "trainable_gates_total": 2 * self.n_layers * self.n_qubits,
            "cnot_gates_total": self.n_layers * self.n_qubits,
            "total_gates": self.n_qubits + self.n_layers * (3 * self.n_qubits),
            "estimated_circuit_depth": 1 + self.n_layers * 3,
        }
