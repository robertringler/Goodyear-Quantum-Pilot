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
    MaterialProperty,
    PropertyType,
    MaterialCategory,
)

from goodyear_quantum_pilot.materials.elastomers import (
    Elastomer,
    SyntheticElastomer,
    SBR,
    NBR,
    EPDM,
    CR,
    BR,
    IIR,
    HNBR,
    FKM,
    ELASTOMER_CATALOG,
)

from goodyear_quantum_pilot.materials.rubbers import (
    NaturalRubber,
    SMR,
    SVR,
    TSR,
    RUBBER_CATALOG,
)

from goodyear_quantum_pilot.materials.quantum_engineered import (
    QuantumEngineeredMaterial,
    QuantumEnhancedSBR,
    TunnelingCrosslink,
    EntangledLattice,
    QUANTUM_MATERIAL_CATALOG,
)

from goodyear_quantum_pilot.materials.nanoarchitectures import (
    NanoArchitecture,
    SelfAssemblingPolymer,
    AdaptiveCrosslink,
    NANO_CATALOG,
)

from goodyear_quantum_pilot.materials.self_healing import (
    SelfHealingPolymer,
    MicrocapsuleHealing,
    VascularHealing,
    IntrinsicHealing,
    HEALING_CATALOG,
)

from goodyear_quantum_pilot.materials.database import (
    MaterialsLibrary,
    MaterialDatabase,
    load_material,
    search_materials,
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
