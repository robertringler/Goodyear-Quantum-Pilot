"""End-to-End Tire Simulation Codebase.

This module provides comprehensive tire simulation capabilities
covering the entire tire lifecycle from manufacturing to end-of-life.

Simulation Domains:
    - Factory: Curing, vulcanization, quality control
    - Shipping: Storage aging, transport stress
    - On-Vehicle: Rolling dynamics, thermal, wear
    - Environmental: UV, ozone, temperature cycling
    - Catastrophic: Blowout, impact, sudden failure

Architecture:
    The simulation is built on a multi-physics coupling framework
    that integrates:
    - Quantum material properties (from algorithms module)
    - Finite element structural analysis
    - Computational fluid dynamics (thermal)
    - Molecular dynamics (polymer behavior)
    - Monte Carlo (rare events)

Example:
    >>> from goodyear_quantum_pilot.simulation import TireSimulator
    >>> 
    >>> # Create simulator with quantum materials
    >>> sim = TireSimulator(
    ...     tire_model="P225/60R16",
    ...     compound=compound_spec,
    ...     backend="gpu",
    ... )
    >>> 
    >>> # Run full lifecycle simulation
    >>> results = sim.run_lifecycle(
    ...     factory_params=factory_config,
    ...     service_profile=usage_profile,
    ...     duration_years=5,
    ... )
"""

from __future__ import annotations

# Core simulation
from .core import (
    TireSimulator,
    SimulationConfig,
    TireGeometry,
    LoadCase,
    OperatingCondition,
    PhysicsEngine,
    MaterialIntegrator,
)

# Factory simulation
from .factory import (
    FactorySimulator,
    CuringSimulator,
    MoldFlowSimulator,
    DefectPredictor,
    QualityController,
)

# Shipping simulation
from .shipping import (
    ShippingSimulator,
    VibrationSimulator,
    TemperatureSimulator as ShippingTempSimulator,
    HumiditySimulator as ShippingHumiditySimulator,
    ShockSimulator,
    CompressionSimulator,
    TransportCondition,
    TransportMode,
    DegradationState,
)

# On-vehicle simulation
from .on_vehicle import (
    OnVehicleSimulator,
    RollingDynamics,
    ThermalModel,
    WearSimulator,
    TractionModel,
    TireForces,
    WearState,
    ThermalState,
    RoadSurface,
)

# Environmental simulation
from .environment import (
    EnvironmentSimulator,
    UVDegradation,
    OzoneDegradation,
    ThermalOxidation,
    HumidityDegradation,
    ChemicalDegradation,
    EnvironmentalConditions,
    MaterialState,
    ClimateZone,
)

# Catastrophic failure simulation
from .catastrophic import (
    CatastrophicSimulator,
    BlowoutSimulator,
    TreadSeparationSimulator,
    ImpactDamageSimulator,
    RunFlatSimulator,
    FailureMode,
    FailureEvent,
    SeverityLevel,
    TireStructure,
    DamageState as CatastrophicDamageState,
)


__all__ = [
    # Core simulation
    "TireSimulator",
    "SimulationConfig",
    "TireGeometry",
    "LoadCase",
    "OperatingCondition",
    "PhysicsEngine",
    "MaterialIntegrator",
    # Factory simulation
    "FactorySimulator",
    "CuringSimulator",
    "MoldFlowSimulator",
    "DefectPredictor",
    "QualityController",
    # Shipping simulation
    "ShippingSimulator",
    "VibrationSimulator",
    "ShippingTempSimulator",
    "ShippingHumiditySimulator",
    "ShockSimulator",
    "CompressionSimulator",
    "TransportCondition",
    "TransportMode",
    "DegradationState",
    # On-vehicle simulation
    "OnVehicleSimulator",
    "RollingDynamics",
    "ThermalModel",
    "WearSimulator",
    "TractionModel",
    "TireForces",
    "WearState",
    "ThermalState",
    "RoadSurface",
    # Environmental simulation
    "EnvironmentSimulator",
    "UVDegradation",
    "OzoneDegradation",
    "ThermalOxidation",
    "HumidityDegradation",
    "ChemicalDegradation",
    "EnvironmentalConditions",
    "MaterialState",
    "ClimateZone",
    # Catastrophic failure simulation
    "CatastrophicSimulator",
    "BlowoutSimulator",
    "TreadSeparationSimulator",
    "ImpactDamageSimulator",
    "RunFlatSimulator",
    "FailureMode",
    "FailureEvent",
    "SeverityLevel",
    "TireStructure",
    "CatastrophicDamageState",
]


__version__ = "1.0.0"
__author__ = "Goodyear Quantum Research"
