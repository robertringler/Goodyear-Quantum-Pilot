"""Quantum Materials Library for Tire Simulation.

Comprehensive library of 100+ materials including:
- Synthetic elastomers (SBR, NBR, EPDM, etc.)
- Natural rubbers and derivatives
- Quantum-engineered chain variants
- Nanoarchitecture materials
- Self-healing polymer systems
- Zero-wear entangled lattices
"""

from goodyear_quantum_pilot.materials.base import (
    Material,
    MaterialCategory,
    MaterialProperty,
    PropertyType,
)
from goodyear_quantum_pilot.materials.database import (
    MaterialDatabase,
    MaterialsLibrary,
    load_material,
    search_materials,
)
from goodyear_quantum_pilot.materials.elastomers import (
    BR,
    CR,
    ELASTOMER_CATALOG,
    EPDM,
    FKM,
    HNBR,
    IIR,
    NBR,
    SBR,
    Elastomer,
    SyntheticElastomer,
)
from goodyear_quantum_pilot.materials.nanoarchitectures import (
    NANO_CATALOG,
    AdaptiveCrosslink,
    NanoArchitecture,
    SelfAssemblingPolymer,
)
from goodyear_quantum_pilot.materials.quantum_engineered import (
    QUANTUM_MATERIAL_CATALOG,
    EntangledLattice,
    QuantumEngineeredMaterial,
    QuantumEnhancedSBR,
    TunnelingCrosslink,
)
from goodyear_quantum_pilot.materials.rubbers import (
    RUBBER_CATALOG,
    SMR,
    SVR,
    TSR,
    NaturalRubber,
)
from goodyear_quantum_pilot.materials.self_healing import (
    HEALING_CATALOG,
    IntrinsicHealing,
    MicrocapsuleHealing,
    SelfHealingPolymer,
    VascularHealing,
)

__all__ = [
    # Base classes
    "Material",
    "MaterialProperty",
    "PropertyType",
    "MaterialCategory",
    # Elastomers
    "Elastomer",
    "SyntheticElastomer",
    "SBR",
    "NBR",
    "EPDM",
    "CR",
    "BR",
    "IIR",
    "HNBR",
    "FKM",
    "ELASTOMER_CATALOG",
    # Rubbers
    "NaturalRubber",
    "SMR",
    "SVR",
    "TSR",
    "RUBBER_CATALOG",
    # Quantum Engineered
    "QuantumEngineeredMaterial",
    "QuantumEnhancedSBR",
    "TunnelingCrosslink",
    "EntangledLattice",
    "QUANTUM_MATERIAL_CATALOG",
    # Nanoarchitectures
    "NanoArchitecture",
    "SelfAssemblingPolymer",
    "AdaptiveCrosslink",
    "NANO_CATALOG",
    # Self-healing
    "SelfHealingPolymer",
    "MicrocapsuleHealing",
    "VascularHealing",
    "IntrinsicHealing",
    "HEALING_CATALOG",
    # Database
    "MaterialsLibrary",
    "MaterialDatabase",
    "load_material",
    "search_materials",
]
