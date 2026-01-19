"""Quantum-Engineered Materials Database.

Contains 20 advanced quantum-engineered materials including:
- Quantum-Enhanced SBR with tunneling crosslinks
- Entangled lattice elastomers
- Quantum-coherent polymer chains
- Tunneling-stabilized crosslink networks
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from goodyear_quantum_pilot.materials.base import (
    EconomicFactors,
    EnvironmentalReactivity,
    HamiltonianParameters,
    Material,
    MaterialCategory,
    MechanicalBehavior,
    MolecularSpecification,
    PropertyType,
    ThermalBehavior,
    WearModel,
)


@dataclass
class QuantumCrosslinkParameters:
    """Parameters for quantum-enhanced crosslinks.

    Attributes:
        tunneling_probability: Probability of quantum tunneling
        coherence_time: Quantum coherence time (μs)
        entanglement_density: Entangled pairs per nm³
        barrier_height: Potential barrier height (eV)
        barrier_width: Potential barrier width (nm)
        coupling_strength: Crosslink coupling strength (eV)
    """

    tunneling_probability: float = 0.15
    coherence_time: float = 100.0  # μs
    entanglement_density: float = 0.5  # pairs/nm³
    barrier_height: float = 0.3  # eV
    barrier_width: float = 0.2  # nm
    coupling_strength: float = 0.05  # eV

    def calculate_tunneling_rate(self, temperature: float = 298.0) -> float:
        """Calculate quantum tunneling rate.

        Uses WKB approximation for tunneling through barrier.

        Args:
            temperature: Temperature (K)

        Returns:
            Tunneling rate (1/s)
        """
        # Physical constants
        hbar = 1.054e-34  # J·s
        m_e = 9.109e-31  # kg (electron mass)
        eV_to_J = 1.602e-19
        nm_to_m = 1e-9

        # Convert units
        V = self.barrier_height * eV_to_J
        a = self.barrier_width * nm_to_m

        # Effective mass (assume 0.1 * electron mass for polymer)
        m_eff = 0.1 * m_e

        # WKB transmission coefficient
        kappa = np.sqrt(2 * m_eff * V) / hbar
        T = np.exp(-2 * kappa * a)

        # Attempt frequency (thermal)
        k_B = 1.38e-23  # J/K
        nu_0 = k_B * temperature / (2 * np.pi * hbar)

        return nu_0 * T * self.tunneling_probability


class QuantumEngineeredMaterial(Material):
    """Quantum-engineered tire material.

    Extends base Material with quantum-specific properties and
    simulation capabilities.
    """

    quantum_crosslinks: QuantumCrosslinkParameters = field(
        default_factory=QuantumCrosslinkParameters
    )

    def __init__(
        self,
        material_id: str,
        name: str,
        quantum_params: QuantumCrosslinkParameters | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize quantum-engineered material."""
        super().__init__(
            material_id=material_id,
            name=name,
            category=MaterialCategory.QUANTUM_ENGINEERED,
            **kwargs,
        )
        self.quantum_crosslinks = quantum_params or QuantumCrosslinkParameters()
        self.patents = ["Patent #81", "Patent #82"]  # Associated patents

    def get_quantum_enhancement_factor(self) -> float:
        """Calculate overall quantum enhancement factor.

        Returns:
            Enhancement factor (>1 means improvement)
        """
        tunneling = 1 + self.quantum_crosslinks.tunneling_probability
        coherence = 1 + np.log10(max(1, self.quantum_crosslinks.coherence_time)) / 10
        entanglement = 1 + self.quantum_crosslinks.entanglement_density

        return tunneling * coherence * entanglement

    def predict_enhanced_wear_life(self, base_life: float) -> float:
        """Predict wear life with quantum enhancement.

        Args:
            base_life: Base material wear life

        Returns:
            Enhanced wear life
        """
        enhancement = self.get_quantum_enhancement_factor()
        return base_life * enhancement

    def get_crosslink_stability(self, temperature: float = 373.0) -> float:
        """Calculate crosslink stability at temperature.

        Args:
            temperature: Temperature (K)

        Returns:
            Stability factor (0-1)
        """
        # Quantum tunneling stabilization
        tunneling_rate = self.quantum_crosslinks.calculate_tunneling_rate(temperature)

        # Higher tunneling rate = faster relaxation = lower stability
        # But quantum effects can also stabilize through entanglement
        entanglement_stabilization = self.quantum_crosslinks.entanglement_density * 0.1

        stability = 0.8 + entanglement_stabilization - tunneling_rate * 1e-8

        return np.clip(stability, 0.0, 1.0)


def create_quantum_sbr(
    variant: str,
    styrene: float,
    vinyl: float,
    tunneling_prob: float,
    coherence: float,
    entanglement: float,
) -> QuantumEngineeredMaterial:
    """Create a quantum-enhanced SBR material.

    Args:
        variant: Material variant ID
        styrene: Styrene content (%)
        vinyl: Vinyl content (%)
        tunneling_prob: Quantum tunneling probability
        coherence: Coherence time (μs)
        entanglement: Entanglement density (pairs/nm³)

    Returns:
        Quantum-engineered SBR material
    """
    material = QuantumEngineeredMaterial(
        material_id=f"QESBR-{variant}",
        name=f"Quantum-Enhanced SBR {variant}",
        description=(
            f"Quantum-engineered SBR with {styrene}% styrene, {vinyl}% vinyl. "
            f"Features tunneling crosslinks (P={tunneling_prob:.2f}) and "
            f"entanglement stabilization (ρ={entanglement:.2f} pairs/nm³)."
        ),
        quantum_params=QuantumCrosslinkParameters(
            tunneling_probability=tunneling_prob,
            coherence_time=coherence,
            entanglement_density=entanglement,
        ),
    )

    # Base SBR molecular structure
    material.molecular = MolecularSpecification(
        formula="(C8H8)n(C4H6)m[Q-link]",
        molecular_weight=104.15 * styrene / 100 + 54.09 * (1 - styrene / 100),
        smiles="C=Cc1ccccc1.C=CC=C.[Q]",
        monomer_units=2000,
        chain_length=85.0 + 0.5 * vinyl,
        crosslink_density=150.0 + 50 * entanglement,  # Enhanced crosslinking
        vulcanization_system="quantum_tunneling",
    )

    # Enhanced Hamiltonian with quantum crosslink terms
    num_orb = 12 + int(styrene / 10)
    one_body = np.random.randn(num_orb, num_orb) * 0.1 - 0.5 * np.eye(num_orb)

    # Add tunneling terms to off-diagonal
    for i in range(num_orb - 1):
        tunneling_coupling = tunneling_prob * 0.1
        one_body[i, i + 1] += tunneling_coupling
        one_body[i + 1, i] += tunneling_coupling

    material.hamiltonian = HamiltonianParameters(
        one_body=one_body,
        two_body=np.random.randn(num_orb, num_orb, num_orb, num_orb) * 0.01,
        nuclear_repulsion=18.0 + styrene * 0.2 + entanglement * 2,
        num_orbitals=num_orb,
        num_electrons=2 * num_orb - 2,
        basis_set="cc-pVTZ",  # Higher basis for quantum accuracy
        spin_multiplicity=1,
    )

    # Enhanced mechanical behavior
    quantum_factor = 1 + tunneling_prob + 0.5 * entanglement

    material.mechanical = MechanicalBehavior(
        model_type="ogden",
        c10=0.5 * quantum_factor + 0.01 * styrene,
        c01=0.12 * quantum_factor,
        ogden_mu=[0.75 * quantum_factor, 0.015, -0.012],
        ogden_alpha=[1.3, 5.0, -2.0],
        viscosity=1000 + 30 * styrene,
        relaxation_time=0.06,  # Faster relaxation
    )

    # Improved thermal stability
    material.thermal = ThermalBehavior(
        glass_transition=208.0 + 1.2 * styrene - 5 * entanglement,  # Lower Tg
        melting_point=458.0,
        degradation_temp=543.0 + 20 * entanglement,  # Higher stability
        thermal_conductivity=0.16 + 0.01 * entanglement,
        specific_heat=1900 + 5 * styrene,
        heat_buildup_rate=0.010 / quantum_factor,  # Reduced heat buildup
    )

    # Dramatically improved wear
    material.wear = WearModel(
        abrasion_index=150.0 * quantum_factor,
        wear_rate=5e-9 / quantum_factor,
        fatigue_exponent=2.0,  # Improved fatigue
        paris_c=5e-16 / quantum_factor,
        paris_m=3.5,
        endurance_limit=80.0 * quantum_factor,
        tearing_energy=45.0 * quantum_factor,
    )

    # Improved environmental resistance
    material.environmental = EnvironmentalReactivity(
        ozone_resistance=100.0 * quantum_factor,
        uv_stability=500.0,
        oxidation_rate=1e-5 / quantum_factor,
    )

    # Higher cost due to quantum processing
    material.economic = EconomicFactors(
        raw_material_cost=8.50 + 2 * entanglement,
        processing_cost=5.00 + tunneling_prob * 10,
        energy_intensity=80.0,
        manufacturability_score=60.0,  # More difficult
        sustainability_score=65.0,
    )

    # Set properties
    material.set_property(
        PropertyType.TENSILE_STRENGTH, 28.0 * quantum_factor, "MPa", quantum_enhanced=True
    )
    material.set_property(PropertyType.ELONGATION, 600.0, "%", quantum_enhanced=True)
    material.set_property(PropertyType.HARDNESS, 55.0, "Shore A")
    material.set_property(
        PropertyType.RESILIENCE, 70.0 * quantum_factor, "%", quantum_enhanced=True
    )
    material.set_property(
        PropertyType.ABRASION_RESISTANCE, 150.0 * quantum_factor, "index", quantum_enhanced=True
    )
    material.set_property(
        PropertyType.FATIGUE_LIFE, 1e7 * quantum_factor, "cycles", quantum_enhanced=True
    )
    material.set_property(PropertyType.QUANTUM_COHERENCE, coherence, "μs", quantum_enhanced=True)
    material.set_property(
        PropertyType.TUNNELING_PROBABILITY, tunneling_prob, "dimensionless", quantum_enhanced=True
    )
    material.set_property(
        PropertyType.ENTANGLEMENT_DENSITY, entanglement, "pairs/nm³", quantum_enhanced=True
    )

    material.patents = ["Patent #81", "Patent #82", "Patent #83"]

    return material


def create_tunneling_crosslink_material(
    variant: str,
    base_polymer: str,
    barrier_height: float,
    barrier_width: float,
) -> QuantumEngineeredMaterial:
    """Create material with quantum tunneling crosslinks.

    Args:
        variant: Material variant ID
        base_polymer: Base polymer type
        barrier_height: Tunneling barrier height (eV)
        barrier_width: Tunneling barrier width (nm)

    Returns:
        Material with tunneling crosslinks
    """
    # Calculate tunneling probability from WKB
    hbar_eV_nm = 0.197  # ℏc in eV·nm
    m_eff_eV = 0.5e6 / (3e8) ** 2  # Effective mass in eV/c²

    kappa = np.sqrt(2 * m_eff_eV * barrier_height) / hbar_eV_nm
    tunneling_prob = np.exp(-2 * kappa * barrier_width)
    tunneling_prob = min(0.5, tunneling_prob)  # Cap at 50%

    material = QuantumEngineeredMaterial(
        material_id=f"QTC-{variant}",
        name=f"Quantum Tunneling Crosslink {variant}",
        description=(
            f"Polymer with quantum tunneling crosslinks. "
            f"Barrier: {barrier_height:.2f} eV × {barrier_width:.2f} nm. "
            f"Tunneling probability: {tunneling_prob:.4f}"
        ),
        quantum_params=QuantumCrosslinkParameters(
            tunneling_probability=tunneling_prob,
            coherence_time=50.0,
            barrier_height=barrier_height,
            barrier_width=barrier_width,
            coupling_strength=0.1,
        ),
    )

    material.patents = ["Patent #84", "Patent #85"]

    # Set properties based on tunneling enhancement
    enhancement = 1 + 5 * tunneling_prob  # Up to 3.5x with max tunneling

    material.set_property(PropertyType.CROSSLINK_STABILITY, 0.95, "fraction", quantum_enhanced=True)
    material.set_property(
        PropertyType.TENSILE_STRENGTH, 25.0 * enhancement, "MPa", quantum_enhanced=True
    )
    material.set_property(
        PropertyType.FATIGUE_LIFE, 5e6 * enhancement, "cycles", quantum_enhanced=True
    )

    return material


def create_entangled_lattice_material(
    variant: str,
    lattice_type: str,
    entanglement_pairs: int,
) -> QuantumEngineeredMaterial:
    """Create material with entangled lattice structure.

    Args:
        variant: Material variant ID
        lattice_type: Type of entanglement lattice
        entanglement_pairs: Number of entangled pairs per unit cell

    Returns:
        Entangled lattice material
    """
    entanglement_density = entanglement_pairs / 10.0  # pairs/nm³

    material = QuantumEngineeredMaterial(
        material_id=f"QEL-{variant}",
        name=f"Quantum Entangled Lattice {variant}",
        description=(
            f"Elastomer with {lattice_type} entanglement lattice. "
            f"{entanglement_pairs} entangled pairs per unit cell. "
            f"Provides non-local stress distribution."
        ),
        quantum_params=QuantumCrosslinkParameters(
            tunneling_probability=0.1,
            coherence_time=200.0,  # Long coherence for entanglement
            entanglement_density=entanglement_density,
        ),
    )

    material.patents = ["Patent #86", "Patent #87", "Patent #88"]

    # Entanglement provides unique properties
    nonlocal_factor = 1 + 0.2 * entanglement_density

    material.set_property(
        PropertyType.ENTANGLEMENT_DENSITY, entanglement_density, "pairs/nm³", quantum_enhanced=True
    )
    material.set_property(
        PropertyType.TENSILE_STRENGTH, 30.0 * nonlocal_factor, "MPa", quantum_enhanced=True
    )
    material.set_property(
        PropertyType.TEAR_STRENGTH, 100.0 * nonlocal_factor, "kN/m", quantum_enhanced=True
    )

    # Non-local stress distribution improves fatigue
    material.set_property(PropertyType.FATIGUE_LIFE, 1e8, "cycles", quantum_enhanced=True)

    return material


# Build Quantum Material Catalog - 20 materials

QUANTUM_MATERIAL_CATALOG: dict[str, QuantumEngineeredMaterial] = {}

# Quantum-Enhanced SBR variants (7 materials)
_qesbr_configs = [
    ("1", 25.0, 30.0, 0.10, 50.0, 0.3),
    ("2", 25.0, 45.0, 0.15, 75.0, 0.5),
    ("3", 30.0, 50.0, 0.20, 100.0, 0.7),
    ("4", 35.0, 55.0, 0.25, 150.0, 1.0),
    ("5", 35.0, 63.0, 0.30, 200.0, 1.2),
    ("6", 40.0, 65.0, 0.35, 250.0, 1.5),
    ("7", 45.0, 70.0, 0.40, 300.0, 2.0),  # Ultra-high performance
]

for variant, styrene, vinyl, tunnel, coherence, entangle in _qesbr_configs:
    key = f"QESBR-{variant}"
    QUANTUM_MATERIAL_CATALOG[key] = create_quantum_sbr(
        variant, styrene, vinyl, tunnel, coherence, entangle
    )

# Tunneling Crosslink materials (5 materials)
_qtc_configs = [
    ("A1", "SBR", 0.2, 0.15),
    ("A2", "SBR", 0.25, 0.12),
    ("B1", "NR", 0.18, 0.18),
    ("B2", "NR", 0.22, 0.14),
    ("C1", "BR", 0.15, 0.20),
]

for variant, base, height, width in _qtc_configs:
    key = f"QTC-{variant}"
    QUANTUM_MATERIAL_CATALOG[key] = create_tunneling_crosslink_material(
        variant, base, height, width
    )

# Entangled Lattice materials (5 materials)
_qel_configs = [
    ("1", "cubic", 4),
    ("2", "hexagonal", 6),
    ("3", "tetrahedral", 4),
    ("4", "BCC", 8),
    ("5", "FCC", 12),
]

for variant, lattice, pairs in _qel_configs:
    key = f"QEL-{variant}"
    QUANTUM_MATERIAL_CATALOG[key] = create_entangled_lattice_material(variant, lattice, pairs)


# Hybrid quantum materials (3 materials)
def create_hybrid_quantum_material(variant: str, features: list[str]) -> QuantumEngineeredMaterial:
    """Create hybrid material combining multiple quantum features."""
    material = QuantumEngineeredMaterial(
        material_id=f"QHM-{variant}",
        name=f"Quantum Hybrid Material {variant}",
        description=f"Hybrid material with: {', '.join(features)}",
        quantum_params=QuantumCrosslinkParameters(
            tunneling_probability=0.25,
            coherence_time=150.0,
            entanglement_density=1.0,
        ),
    )

    material.patents = ["Patent #89", "Patent #90"]

    # Superior properties from hybrid approach
    material.set_property(PropertyType.TENSILE_STRENGTH, 35.0, "MPa", quantum_enhanced=True)
    material.set_property(PropertyType.FATIGUE_LIFE, 5e7, "cycles", quantum_enhanced=True)
    material.set_property(PropertyType.ABRASION_RESISTANCE, 250.0, "index", quantum_enhanced=True)

    return material


QUANTUM_MATERIAL_CATALOG["QHM-1"] = create_hybrid_quantum_material(
    "1", ["tunneling", "entanglement", "coherent_chains"]
)
QUANTUM_MATERIAL_CATALOG["QHM-2"] = create_hybrid_quantum_material(
    "2", ["high_coherence", "adaptive_crosslinks", "self_repair"]
)
QUANTUM_MATERIAL_CATALOG["QHM-3"] = create_hybrid_quantum_material(
    "3", ["zero_wear_lattice", "quantum_healing", "extreme_stability"]
)

# Aliases for common access
QuantumEnhancedSBR = QUANTUM_MATERIAL_CATALOG["QESBR-7"]
TunnelingCrosslink = QUANTUM_MATERIAL_CATALOG["QTC-A1"]
EntangledLattice = QUANTUM_MATERIAL_CATALOG["QEL-5"]
