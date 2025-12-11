"""Core quantum computing infrastructure for Goodyear Quantum Pilot.

This module provides the foundational quantum computing primitives including:
- Quantum backend abstraction for multiple hardware providers
- Quantum circuit construction and manipulation
- Quantum state management
- Tensor network simulation
"""

from goodyear_quantum_pilot.core.backends import (
    QuantumBackend,
    QiskitBackend,
    BraketBackend,
    IonQBackend,
    QuEraBackend,
    SimulatorBackend,
    BackendConfig,
)

from goodyear_quantum_pilot.core.circuits import (
    QuantumCircuit,
    QuantumGate,
    ParameterizedGate,
    CircuitLayer,
    CircuitOptimizer,
)

from goodyear_quantum_pilot.core.state import (
    QuantumState,
    StateVector,
    DensityMatrix,
    MixedState,
    StateTomography,
)

from goodyear_quantum_pilot.core.tensor_networks import (
    TensorNetwork,
    TensorNode,
    TensorContraction,
    MPSState,
    PEPSState,
    TTNState,
)

__all__ = [
    # Backends
    "QuantumBackend",
    "QiskitBackend",
    "BraketBackend",
    "IonQBackend",
    "QuEraBackend",
    "SimulatorBackend",
    "BackendConfig",
    # Circuits
    "QuantumCircuit",
    "QuantumGate",
    "ParameterizedGate",
    "CircuitLayer",
    "CircuitOptimizer",
    # State
    "QuantumState",
    "StateVector",
    "DensityMatrix",
    "MixedState",
    "StateTomography",
    # Tensor Networks
    "TensorNetwork",
    "TensorNode",
    "TensorContraction",
    "MPSState",
    "PEPSState",
    "TTNState",
]
