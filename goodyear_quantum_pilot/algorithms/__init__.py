"""Quantum Algorithms for Tire Material Simulation.

This module provides ultra-advanced quantum algorithms specifically designed
for polymer chain simulation, material optimization, and tire performance
prediction. All algorithms are designed for NISQ-era quantum hardware with
hybrid classical-quantum optimization loops.

Modules:
    vqe: Variational Quantum Eigensolver for polymer Hamiltonians
    qaoa: Quantum Approximate Optimization for material blending
    monte_carlo: Quantum Monte Carlo for rare-event simulation
    tunneling: Quantum tunneling dynamics for crosslink lifetime
    stress: Quantum stress cracking and failure prediction
    entanglement: Entanglement-enhanced material sensing
    dynamics: Real-time quantum dynamics for polymer motion
    error_mitigation: Error mitigation strategies for NISQ devices

Example:
    >>> from goodyear_quantum_pilot.algorithms import PolymerVQE, TireQAOA
    >>>
    >>> # Initialize VQE for polymer ground state
    >>> vqe = PolymerVQE(
    ...     polymer_chain=sbr_chain,
    ...     backend="ibm_brisbane",
    ...     ansatz="hardware_efficient",
    ... )
    >>> ground_state = vqe.run(shots=10000)
    >>>
    >>> # Optimize material blend with QAOA
    >>> qaoa = TireQAOA(
    ...     materials=[sbr, nbr, carbon_black],
    ...     constraints=tire_constraints,
    ...     depth=3,
    ... )
    >>> optimal_blend = qaoa.optimize()
"""

from __future__ import annotations

__all__ = [
    # VQE algorithms
    "PolymerVQE",
    "MultiReferenceVQE",
    "AdaptiveVQE",
    "VQEConfig",
    # QAOA algorithms
    "TireQAOA",
    "BlendingQAOA",
    "ConstraintQAOA",
    "QAOAConfig",
    # Monte Carlo algorithms
    "QuantumMonteCarlo",
    "DiffusionMonteCarlo",
    "PathIntegralMC",
    "RareEventMC",
    # Tunneling dynamics
    "TunnelingSimulator",
    "CrosslinkTunneling",
    "ProtonTunneling",
    # Stress and failure
    "QuantumStressPredictor",
    "CrackPropagation",
    "FatigueAnalyzer",
    # Entanglement
    "EntanglementSensor",
    "QuantumCorrelator",
    # Dynamics
    "RealTimeDynamics",
    "TrotterEvolution",
    "VariationalDynamics",
    # Error mitigation
    "ZNEMitigation",
    "PECMitigation",
    "ReadoutMitigation",
]


# Version and metadata
__version__ = "1.0.0"
__author__ = "Goodyear Quantum Research"
