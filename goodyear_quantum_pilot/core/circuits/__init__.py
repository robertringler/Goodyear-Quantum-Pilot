"""Quantum circuit primitives for tire materials simulation.

Provides specialized quantum circuits optimized for:
- Polymer Hamiltonian simulation
- Crosslink dynamics
- Material property calculation
- Molecular energy estimation
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np


class GateType(Enum):
    """Enumeration of supported quantum gates."""

    # Single-qubit gates
    IDENTITY = auto()
    PAULI_X = auto()
    PAULI_Y = auto()
    PAULI_Z = auto()
    HADAMARD = auto()
    S_GATE = auto()
    T_GATE = auto()
    RX = auto()
    RY = auto()
    RZ = auto()
    U3 = auto()

    # Two-qubit gates
    CNOT = auto()
    CZ = auto()
    SWAP = auto()
    ISWAP = auto()
    CRX = auto()
    CRY = auto()
    CRZ = auto()

    # Multi-qubit gates
    TOFFOLI = auto()
    FREDKIN = auto()

    # Specialized gates for materials
    POLYMER_COUPLING = auto()
    CROSSLINK_GATE = auto()
    ENTANGLEMENT_GATE = auto()


@dataclass
class QuantumGate:
    """Representation of a quantum gate operation.

    Attributes:
        gate_type: Type of quantum gate
        qubits: Target qubit indices
        params: Gate parameters (angles, etc.)
        label: Optional gate label for visualization
    """

    gate_type: GateType
    qubits: tuple[int, ...]
    params: dict[str, float] = field(default_factory=dict)
    label: str | None = None

    @property
    def num_qubits(self) -> int:
        """Number of qubits this gate acts on."""
        return len(self.qubits)

    @property
    def is_parameterized(self) -> bool:
        """Check if gate has variational parameters."""
        return len(self.params) > 0

    def bind_params(self, values: dict[str, float]) -> QuantumGate:
        """Bind parameter values to create concrete gate.

        Args:
            values: Dictionary mapping parameter names to values

        Returns:
            New gate with bound parameters
        """
        new_params = {}
        for name, value in self.params.items():
            if isinstance(value, str) and value in values:
                new_params[name] = values[value]
            else:
                new_params[name] = value

        return QuantumGate(
            gate_type=self.gate_type,
            qubits=self.qubits,
            params=new_params,
            label=self.label,
        )

    def to_matrix(self) -> np.ndarray:
        """Get matrix representation of the gate.

        Returns:
            Complex numpy array representing the gate
        """
        if self.gate_type == GateType.IDENTITY:
            return np.eye(2, dtype=complex)

        if self.gate_type == GateType.PAULI_X:
            return np.array([[0, 1], [1, 0]], dtype=complex)

        if self.gate_type == GateType.PAULI_Y:
            return np.array([[0, -1j], [1j, 0]], dtype=complex)

        if self.gate_type == GateType.PAULI_Z:
            return np.array([[1, 0], [0, -1]], dtype=complex)

        if self.gate_type == GateType.HADAMARD:
            return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

        if self.gate_type == GateType.S_GATE:
            return np.array([[1, 0], [0, 1j]], dtype=complex)

        if self.gate_type == GateType.T_GATE:
            return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

        if self.gate_type == GateType.RX:
            theta = self.params.get("theta", 0)
            c, s = np.cos(theta / 2), np.sin(theta / 2)
            return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)

        if self.gate_type == GateType.RY:
            theta = self.params.get("theta", 0)
            c, s = np.cos(theta / 2), np.sin(theta / 2)
            return np.array([[c, -s], [s, c]], dtype=complex)

        if self.gate_type == GateType.RZ:
            theta = self.params.get("theta", 0)
            return np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)])

        if self.gate_type == GateType.CNOT:
            return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)

        if self.gate_type == GateType.CZ:
            return np.diag([1, 1, 1, -1]).astype(complex)

        if self.gate_type == GateType.SWAP:
            return np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)

        # Default to identity
        dim = 2**self.num_qubits
        return np.eye(dim, dtype=complex)


@dataclass
class ParameterizedGate(QuantumGate):
    """Variational quantum gate with trainable parameters.

    Used in VQE and QAOA circuits for optimization.
    """

    param_names: list[str] = field(default_factory=list)
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

    def get_gradient(self, param_name: str) -> QuantumGate:
        """Get the parameter shift rule gradient circuit.

        Args:
            param_name: Name of parameter to differentiate

        Returns:
            Gradient circuit using parameter shift rule
        """
        if param_name not in self.param_names:
            raise ValueError(f"Unknown parameter: {param_name}")

        # Parameter shift: ∂f/∂θ = (f(θ+π/2) - f(θ-π/2)) / 2
        shift = np.pi / 2

        return self  # Simplified; actual would return shifted circuits


@dataclass
class CircuitLayer:
    """A layer of parallel quantum gates.

    Gates in a layer can be executed simultaneously if they
    act on disjoint sets of qubits.
    """

    gates: list[QuantumGate] = field(default_factory=list)

    def add_gate(self, gate: QuantumGate) -> None:
        """Add a gate to this layer.

        Args:
            gate: Gate to add

        Raises:
            ValueError: If gate overlaps with existing gates
        """
        used_qubits = set()
        for existing in self.gates:
            used_qubits.update(existing.qubits)

        if any(q in used_qubits for q in gate.qubits):
            raise ValueError("Gate overlaps with existing gates in layer")

        self.gates.append(gate)

    @property
    def qubits_used(self) -> set[int]:
        """Set of qubits used in this layer."""
        qubits = set()
        for gate in self.gates:
            qubits.update(gate.qubits)
        return qubits


class QuantumCircuit:
    """Quantum circuit for materials simulation.

    Provides methods to construct, manipulate, and analyze quantum
    circuits optimized for polymer and elastomer simulation.

    Attributes:
        num_qubits: Number of qubits in the circuit
        name: Circuit name for identification
    """

    def __init__(
        self,
        num_qubits: int,
        name: str = "circuit",
    ) -> None:
        """Initialize quantum circuit.

        Args:
            num_qubits: Number of qubits
            name: Circuit identifier
        """
        self.num_qubits = num_qubits
        self.name = name
        self._gates: list[dict[str, Any]] = []
        self._parameters: dict[str, float] = {}
        self._measurements: list[int] = []

    @property
    def gates(self) -> list[dict[str, Any]]:
        """List of gates in the circuit."""
        return self._gates.copy()

    @property
    def depth(self) -> int:
        """Circuit depth (number of layers)."""
        if not self._gates:
            return 0

        # Count sequential gate layers
        qubit_depths = [0] * self.num_qubits
        for gate in self._gates:
            qubits = gate.get("qubits", [0])
            max_depth = max(qubit_depths[q] for q in qubits)
            for q in qubits:
                qubit_depths[q] = max_depth + 1

        return max(qubit_depths)

    @property
    def gate_count(self) -> int:
        """Total number of gates."""
        return len(self._gates)

    @property
    def parameters(self) -> dict[str, float]:
        """Circuit parameters."""
        return self._parameters.copy()

    # Single-qubit gates

    def h(self, qubit: int) -> QuantumCircuit:
        """Apply Hadamard gate.

        Args:
            qubit: Target qubit

        Returns:
            Self for method chaining
        """
        self._gates.append(
            {
                "type": "H",
                "qubits": [qubit],
                "params": {},
            }
        )
        return self

    def x(self, qubit: int) -> QuantumCircuit:
        """Apply Pauli-X (NOT) gate."""
        self._gates.append(
            {
                "type": "X",
                "qubits": [qubit],
                "params": {},
            }
        )
        return self

    def y(self, qubit: int) -> QuantumCircuit:
        """Apply Pauli-Y gate."""
        self._gates.append(
            {
                "type": "Y",
                "qubits": [qubit],
                "params": {},
            }
        )
        return self

    def z(self, qubit: int) -> QuantumCircuit:
        """Apply Pauli-Z gate."""
        self._gates.append(
            {
                "type": "Z",
                "qubits": [qubit],
                "params": {},
            }
        )
        return self

    def s(self, qubit: int) -> QuantumCircuit:
        """Apply S (phase) gate."""
        self._gates.append(
            {
                "type": "S",
                "qubits": [qubit],
                "params": {},
            }
        )
        return self

    def t(self, qubit: int) -> QuantumCircuit:
        """Apply T gate."""
        self._gates.append(
            {
                "type": "T",
                "qubits": [qubit],
                "params": {},
            }
        )
        return self

    def rx(self, qubit: int, theta: float) -> QuantumCircuit:
        """Apply RX rotation gate.

        Args:
            qubit: Target qubit
            theta: Rotation angle in radians
        """
        self._gates.append(
            {
                "type": "RX",
                "qubits": [qubit],
                "params": {"theta": theta},
            }
        )
        return self

    def ry(self, qubit: int, theta: float) -> QuantumCircuit:
        """Apply RY rotation gate."""
        self._gates.append(
            {
                "type": "RY",
                "qubits": [qubit],
                "params": {"theta": theta},
            }
        )
        return self

    def rz(self, qubit: int, theta: float) -> QuantumCircuit:
        """Apply RZ rotation gate."""
        self._gates.append(
            {
                "type": "RZ",
                "qubits": [qubit],
                "params": {"theta": theta},
            }
        )
        return self

    # Two-qubit gates

    def cx(self, control: int, target: int) -> QuantumCircuit:
        """Apply CNOT (controlled-X) gate.

        Args:
            control: Control qubit
            target: Target qubit
        """
        self._gates.append(
            {
                "type": "CNOT",
                "qubits": [control, target],
                "params": {},
            }
        )
        return self

    def cnot(self, control: int, target: int) -> QuantumCircuit:
        """Alias for cx gate."""
        return self.cx(control, target)

    def cz(self, control: int, target: int) -> QuantumCircuit:
        """Apply controlled-Z gate."""
        self._gates.append(
            {
                "type": "CZ",
                "qubits": [control, target],
                "params": {},
            }
        )
        return self

    def swap(self, qubit1: int, qubit2: int) -> QuantumCircuit:
        """Apply SWAP gate."""
        self._gates.append(
            {
                "type": "SWAP",
                "qubits": [qubit1, qubit2],
                "params": {},
            }
        )
        return self

    def crx(self, control: int, target: int, theta: float) -> QuantumCircuit:
        """Apply controlled-RX gate."""
        self._gates.append(
            {
                "type": "CRX",
                "qubits": [control, target],
                "params": {"theta": theta},
            }
        )
        return self

    def cry(self, control: int, target: int, theta: float) -> QuantumCircuit:
        """Apply controlled-RY gate."""
        self._gates.append(
            {
                "type": "CRY",
                "qubits": [control, target],
                "params": {"theta": theta},
            }
        )
        return self

    def crz(self, control: int, target: int, theta: float) -> QuantumCircuit:
        """Apply controlled-RZ gate."""
        self._gates.append(
            {
                "type": "CRZ",
                "qubits": [control, target],
                "params": {"theta": theta},
            }
        )
        return self

    # Multi-qubit gates

    def ccx(self, control1: int, control2: int, target: int) -> QuantumCircuit:
        """Apply Toffoli (CCX) gate."""
        self._gates.append(
            {
                "type": "CCX",
                "qubits": [control1, control2, target],
                "params": {},
            }
        )
        return self

    def toffoli(self, control1: int, control2: int, target: int) -> QuantumCircuit:
        """Alias for ccx gate."""
        return self.ccx(control1, control2, target)

    # Materials-specific gates

    def polymer_coupling(
        self,
        qubits: list[int],
        coupling_strength: float,
    ) -> QuantumCircuit:
        """Apply polymer coupling gate for molecular simulation.

        Simulates coupling between polymer chain segments.

        Args:
            qubits: Qubits representing coupled segments
            coupling_strength: Coupling interaction strength (eV)
        """
        self._gates.append(
            {
                "type": "POLYMER_COUPLING",
                "qubits": qubits,
                "params": {"J": coupling_strength},
            }
        )
        return self

    def crosslink_gate(
        self,
        qubit1: int,
        qubit2: int,
        crosslink_type: str = "sulfur",
    ) -> QuantumCircuit:
        """Apply crosslink formation gate.

        Simulates the quantum dynamics of crosslink formation
        between polymer chains.

        Args:
            qubit1: First chain qubit
            qubit2: Second chain qubit
            crosslink_type: Type of crosslink (sulfur, peroxide, etc.)
        """
        # Map crosslink type to coupling parameter
        coupling_map = {
            "sulfur": 0.35,
            "peroxide": 0.42,
            "carbon_carbon": 0.58,
            "silane": 0.28,
            "quantum_tunneling": 0.15,
        }

        coupling = coupling_map.get(crosslink_type, 0.35)

        self._gates.append(
            {
                "type": "CROSSLINK",
                "qubits": [qubit1, qubit2],
                "params": {"coupling": coupling, "type": crosslink_type},
            }
        )
        return self

    def entanglement_layer(self, qubits: list[int] | None = None) -> QuantumCircuit:
        """Apply entanglement layer across qubits.

        Creates GHZ-like entanglement useful for simulating
        collective polymer behavior.

        Args:
            qubits: Qubits to entangle (default: all)
        """
        qubits = qubits or list(range(self.num_qubits))

        if len(qubits) < 2:
            return self

        # Create entanglement: H on first, then CNOT cascade
        self.h(qubits[0])
        for i in range(len(qubits) - 1):
            self.cx(qubits[i], qubits[i + 1])

        return self

    # Measurement

    def measure(self, qubit: int) -> QuantumCircuit:
        """Add measurement to a qubit.

        Args:
            qubit: Qubit to measure
        """
        if qubit not in self._measurements:
            self._measurements.append(qubit)
        return self

    def measure_all(self) -> QuantumCircuit:
        """Add measurement to all qubits."""
        self._measurements = list(range(self.num_qubits))
        return self

    # Circuit composition

    def compose(self, other: QuantumCircuit) -> QuantumCircuit:
        """Compose this circuit with another.

        Args:
            other: Circuit to append

        Returns:
            New combined circuit
        """
        if other.num_qubits != self.num_qubits:
            raise ValueError("Circuit qubit counts must match")

        new_circuit = QuantumCircuit(
            self.num_qubits,
            f"{self.name}_{other.name}",
        )
        new_circuit._gates = self._gates + other._gates
        new_circuit._parameters = {**self._parameters, **other._parameters}

        return new_circuit

    def inverse(self) -> QuantumCircuit:
        """Get the inverse (adjoint) circuit.

        Returns:
            Circuit that undoes this circuit's operation
        """
        inv = QuantumCircuit(self.num_qubits, f"{self.name}_inv")

        # Reverse gates and invert each
        for gate in reversed(self._gates):
            # Invert rotation angles
            if gate["type"] in ("RX", "RY", "RZ", "CRX", "CRY", "CRZ"):
                inv_gate = gate.copy()
                inv_gate["params"] = {"theta": -gate["params"]["theta"]}
                inv._gates.append(inv_gate)
            else:
                # Self-inverse gates
                inv._gates.append(gate.copy())

        return inv

    # Utility methods

    def bind_parameters(self, values: dict[str, float]) -> QuantumCircuit:
        """Bind parameter values to create concrete circuit.

        Args:
            values: Parameter name to value mapping

        Returns:
            New circuit with bound parameters
        """
        bound = QuantumCircuit(self.num_qubits, self.name)

        for gate in self._gates:
            new_gate = gate.copy()
            new_params = {}
            for key, val in gate.get("params", {}).items():
                if isinstance(val, str) and val in values:
                    new_params[key] = values[val]
                else:
                    new_params[key] = val
            new_gate["params"] = new_params
            bound._gates.append(new_gate)

        return bound

    def to_dict(self) -> dict[str, Any]:
        """Serialize circuit to dictionary.

        Returns:
            Dictionary representation of circuit
        """
        return {
            "name": self.name,
            "num_qubits": self.num_qubits,
            "gates": self._gates,
            "parameters": self._parameters,
            "measurements": self._measurements,
            "depth": self.depth,
            "gate_count": self.gate_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuantumCircuit:
        """Deserialize circuit from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            Reconstructed circuit
        """
        circuit = cls(data["num_qubits"], data.get("name", "circuit"))
        circuit._gates = data.get("gates", [])
        circuit._parameters = data.get("parameters", {})
        circuit._measurements = data.get("measurements", [])
        return circuit

    def __repr__(self) -> str:
        return f"QuantumCircuit({self.num_qubits}, depth={self.depth}, gates={self.gate_count})"


class CircuitOptimizer:
    """Optimizer for quantum circuits.

    Provides circuit optimization passes for:
    - Gate fusion
    - Redundancy elimination
    - Layout optimization
    - Depth reduction
    """

    def __init__(self, optimization_level: int = 2) -> None:
        """Initialize optimizer.

        Args:
            optimization_level: Optimization aggressiveness (0-3)
        """
        self.optimization_level = optimization_level

    def optimize(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Optimize a quantum circuit.

        Args:
            circuit: Circuit to optimize

        Returns:
            Optimized circuit
        """
        optimized = circuit

        if self.optimization_level >= 1:
            optimized = self._remove_identity_gates(optimized)

        if self.optimization_level >= 2:
            optimized = self._fuse_rotations(optimized)

        if self.optimization_level >= 3:
            optimized = self._cancel_adjacent_gates(optimized)

        return optimized

    def _remove_identity_gates(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Remove identity gates from circuit."""
        new_circuit = QuantumCircuit(circuit.num_qubits, circuit.name)

        for gate in circuit._gates:
            if gate["type"] != "I":
                new_circuit._gates.append(gate)

        return new_circuit

    def _fuse_rotations(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Fuse consecutive rotation gates on same qubit."""
        new_circuit = QuantumCircuit(circuit.num_qubits, circuit.name)
        new_circuit._gates = circuit._gates.copy()

        # Simplified implementation
        return new_circuit

    def _cancel_adjacent_gates(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Cancel adjacent inverse gates."""
        new_circuit = QuantumCircuit(circuit.num_qubits, circuit.name)

        # Self-inverse gates: X, Y, Z, H, CNOT
        self_inverse = {"X", "Y", "Z", "H", "CNOT", "CZ", "SWAP"}

        i = 0
        gates = circuit._gates
        while i < len(gates):
            if i + 1 < len(gates):
                g1, g2 = gates[i], gates[i + 1]
                if (
                    g1["type"] == g2["type"]
                    and g1["qubits"] == g2["qubits"]
                    and g1["type"] in self_inverse
                ):
                    # Cancel adjacent self-inverse gates
                    i += 2
                    continue

            new_circuit._gates.append(gates[i])
            i += 1

        return new_circuit


# Convenience functions for circuit construction


def create_bell_pair(qubit1: int = 0, qubit2: int = 1) -> QuantumCircuit:
    """Create a Bell pair (maximally entangled state).

    Args:
        qubit1: First qubit index
        qubit2: Second qubit index

    Returns:
        Circuit that creates |Φ+⟩ = (|00⟩ + |11⟩)/√2
    """
    circuit = QuantumCircuit(max(qubit1, qubit2) + 1, "bell_pair")
    circuit.h(qubit1).cx(qubit1, qubit2)
    return circuit


def create_ghz_state(num_qubits: int) -> QuantumCircuit:
    """Create a GHZ (Greenberger-Horne-Zeilinger) state.

    Args:
        num_qubits: Number of qubits in GHZ state

    Returns:
        Circuit that creates (|00...0⟩ + |11...1⟩)/√2
    """
    circuit = QuantumCircuit(num_qubits, "ghz_state")
    circuit.h(0)
    for i in range(num_qubits - 1):
        circuit.cx(i, i + 1)
    return circuit


def create_qft(num_qubits: int) -> QuantumCircuit:
    """Create Quantum Fourier Transform circuit.

    Args:
        num_qubits: Number of qubits

    Returns:
        QFT circuit
    """
    circuit = QuantumCircuit(num_qubits, "qft")

    for i in range(num_qubits):
        circuit.h(i)
        for j in range(i + 1, num_qubits):
            # Controlled phase rotation
            angle = np.pi / (2 ** (j - i))
            circuit.crz(j, i, angle)

    # Swap qubits
    for i in range(num_qubits // 2):
        circuit.swap(i, num_qubits - 1 - i)

    return circuit


def create_ansatz_hardware_efficient(
    num_qubits: int,
    depth: int,
    entanglement: str = "linear",
) -> QuantumCircuit:
    """Create hardware-efficient variational ansatz.

    Args:
        num_qubits: Number of qubits
        depth: Number of ansatz layers
        entanglement: Entanglement pattern ("linear", "circular", "full")

    Returns:
        Parameterized ansatz circuit
    """
    circuit = QuantumCircuit(num_qubits, "hw_efficient_ansatz")

    param_count = 0

    for layer in range(depth):
        # Rotation layer
        for q in range(num_qubits):
            circuit.ry(q, f"theta_{param_count}")
            param_count += 1
            circuit.rz(q, f"phi_{param_count}")
            param_count += 1

        # Entanglement layer
        if entanglement == "linear":
            for q in range(num_qubits - 1):
                circuit.cx(q, q + 1)
        elif entanglement == "circular":
            for q in range(num_qubits - 1):
                circuit.cx(q, q + 1)
            if num_qubits > 2:
                circuit.cx(num_qubits - 1, 0)
        elif entanglement == "full":
            for q1 in range(num_qubits):
                for q2 in range(q1 + 1, num_qubits):
                    circuit.cx(q1, q2)

    return circuit
