"""Goodyear Quantum Pilot Platform.

Quantum-Accelerated Materials Science & Tire Simulation Platform.

This package provides:
- 100+ quantum-engineered materials with complete Hamiltonian specifications
- Full tire lifecycle simulation from polymerization to end-of-life
- Hybrid quantum-classical optimization for material discovery
- Real-time safety prediction using quantum Monte Carlo methods
- Patent-protected innovations (Patents #81-#100)

Example:
    >>> from goodyear_quantum_pilot import TireSimulator, MaterialsLibrary
    >>> materials = MaterialsLibrary.load_category("quantum_engineered")
    >>> simulator = TireSimulator(material=materials["QESBR-7"])
    >>> results = simulator.run_lifecycle(stages=["factory", "vehicle"])
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "Goodyear Quantum Computing Division"
__license__ = "Proprietary"

# Core imports
from goodyear_quantum_pilot.core import (
    QuantumBackend,
    QuantumCircuit,
    QuantumState,
    TensorNetwork,
)

# Materials imports
from goodyear_quantum_pilot.materials import (
    Material,
    MaterialsLibrary,
    Elastomer,
    NaturalRubber,
    QuantumEngineeredMaterial,
    SelfHealingPolymer,
    NanoArchitecture,
)

# Algorithm imports
from goodyear_quantum_pilot.algorithms import (
    VQEPolymer,
    QAOATire,
    QuantumTunneling,
    QuantumMonteCarlo,
    EntanglementSolver,
    RareEventPredictor,
    LiouvilleEvolution,
)

# Simulation imports
from goodyear_quantum_pilot.simulation import (
    TireSimulator,
    FactorySimulation,
    ShippingSimulation,
    VehicleSimulation,
    EnvironmentSimulation,
    CatastrophicSimulation,
)

# Optimization imports
from goodyear_quantum_pilot.optimization import (
    QuantumOptimizer,
    MaterialOptimizer,
    CompoundOptimizer,
)

# Benchmark imports
from goodyear_quantum_pilot.benchmarks import (
    MaterialBenchmark,
    AlgorithmBenchmark,
    SimulationBenchmark,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__license__",
    # Core
    "QuantumBackend",
    "QuantumCircuit",
    "QuantumState",
    "TensorNetwork",
    # Materials
    "Material",
    "MaterialsLibrary",
    "Elastomer",
    "NaturalRubber",
    "QuantumEngineeredMaterial",
    "SelfHealingPolymer",
    "NanoArchitecture",
    # Algorithms
    "VQEPolymer",
    "QAOATire",
    "QuantumTunneling",
    "QuantumMonteCarlo",
    "EntanglementSolver",
    "RareEventPredictor",
    "LiouvilleEvolution",
    # Simulation
    "TireSimulator",
    "FactorySimulation",
    "ShippingSimulation",
    "VehicleSimulation",
    "EnvironmentSimulation",
    "CatastrophicSimulation",
    # Optimization
    "QuantumOptimizer",
    "MaterialOptimizer",
    "CompoundOptimizer",
    # Benchmarks
    "MaterialBenchmark",
    "AlgorithmBenchmark",
    "SimulationBenchmark",
]
