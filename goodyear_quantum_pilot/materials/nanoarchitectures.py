"""Nanoarchitecture Materials Database.

Contains 15 advanced nanoarchitecture materials including:
- Self-assembling polymer networks
- Adaptive crosslink systems
- Hierarchical nanostructures
- Dynamic covalent networks
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
    MaterialProperty,
    MechanicalBehavior,
    MolecularSpecification,
    PropertyType,
    ThermalBehavior,
    WearModel,
)


@dataclass
class NanostructureParameters:
    """Parameters for nanoarchitecture structures.
    
    Attributes:
        structure_type: Type of nanostructure
        domain_size: Average domain size (nm)
        ordering_parameter: Degree of structural ordering (0-1)
        responsiveness: Environmental responsiveness (0-1)
        self_assembly_rate: Self-assembly rate constant (1/s)
    """
    
    structure_type: str = "block_copolymer"
    domain_size: float = 20.0  # nm
    ordering_parameter: float = 0.8  # 0-1
    responsiveness: float = 0.5  # 0-1
    self_assembly_rate: float = 0.01  # 1/s
    
    def get_effective_modulus_enhancement(self) -> float:
        """Calculate modulus enhancement from nanostructure."""
        # Higher ordering gives better reinforcement
        return 1.0 + 0.5 * self.ordering_parameter


class NanoArchitecture(Material):
    """Nanoarchitecture tire material.
    
    Advanced materials with designed nanoscale structures for
    enhanced performance through hierarchical organization.
    """
    
    nanostructure: NanostructureParameters = field(
        default_factory=NanostructureParameters
    )
    
    def __init__(
        self,
        material_id: str,
        name: str,
        nano_params: NanostructureParameters | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize nanoarchitecture material."""
        super().__init__(
            material_id=material_id,
            name=name,
            category=MaterialCategory.NANOARCHITECTURE,
            **kwargs,
        )
        self.nanostructure = nano_params or NanostructureParameters()
        self.patents = ["Patent #91", "Patent #92"]
    
    def predict_stress_response(
        self,
        strain: float,
        strain_rate: float = 0.01,
    ) -> float:
        """Predict stress response accounting for nanostructure.
        
        Args:
            strain: Applied strain
            strain_rate: Strain rate (1/s)
            
        Returns:
            Stress (MPa)
        """
        # Base hyperelastic response
        G = self.mechanical.get_shear_modulus()
        
        # Nanostructure enhancement
        enhancement = self.nanostructure.get_effective_modulus_enhancement()
        
        # Strain rate effect (adaptive materials can respond to rate)
        rate_factor = 1 + 0.1 * self.nanostructure.responsiveness * np.log10(max(0.001, strain_rate) / 0.01)
        
        # Mooney-Rivlin type response
        stretch = 1 + strain
        stress = 2 * G * enhancement * rate_factor * (stretch - 1/stretch**2)
        
        return stress


def create_self_assembling_polymer(
    variant: str,
    assembly_type: str,
    domain_size: float,
    ordering: float,
) -> NanoArchitecture:
    """Create a self-assembling polymer material.
    
    Args:
        variant: Material variant ID
        assembly_type: Type of self-assembly (block, star, dendritic)
        domain_size: Characteristic domain size (nm)
        ordering: Ordering parameter (0-1)
        
    Returns:
        Self-assembling polymer material
    """
    material = NanoArchitecture(
        material_id=f"SAP-{variant}",
        name=f"Self-Assembling Polymer {variant}",
        description=(
            f"Self-assembling {assembly_type} polymer with "
            f"{domain_size:.1f} nm domains and {ordering:.0%} ordering."
        ),
        nano_params=NanostructureParameters(
            structure_type=assembly_type,
            domain_size=domain_size,
            ordering_parameter=ordering,
            self_assembly_rate=0.02,
        ),
    )
    
    # Molecular specification
    material.molecular = MolecularSpecification(
        formula="(A)n-(B)m block",
        molecular_weight=150000,  # High MW for self-assembly
        monomer_units=1500,
        chain_length=100.0,
        crosslink_density=80.0,  # Physical crosslinks from domains
    )
    
    # Enhanced Hamiltonian
    num_orb = 14
    material.hamiltonian = HamiltonianParameters(
        one_body=np.random.randn(num_orb, num_orb) * 0.08 - 0.4 * np.eye(num_orb),
        two_body=np.random.randn(num_orb, num_orb, num_orb, num_orb) * 0.006,
        nuclear_repulsion=20.0,
        num_orbitals=num_orb,
        num_electrons=24,
        basis_set="def2-SVP",
    )
    
    # Mechanical behavior enhanced by nanostructure
    modulus_factor = 1 + 0.5 * ordering
    
    material.mechanical = MechanicalBehavior(
        model_type="ogden",
        c10=0.6 * modulus_factor,
        c01=0.15 * modulus_factor,
        ogden_mu=[0.8, 0.02, -0.015],
        ogden_alpha=[1.2, 4.5, -2.2],
        viscosity=1500,
        relaxation_time=0.1,
    )
    
    material.thermal = ThermalBehavior(
        glass_transition=220.0 - 10 * ordering,  # Better low temp
        degradation_temp=533.0,
        thermal_conductivity=0.18,
        heat_buildup_rate=0.012,
    )
    
    # Improved wear from reinforcing domains
    material.wear = WearModel(
        abrasion_index=140.0 * modulus_factor,
        wear_rate=6e-9 / modulus_factor,
        fatigue_exponent=2.2,
        tearing_energy=50.0 * modulus_factor,
    )
    
    material.environmental = EnvironmentalReactivity(
        ozone_resistance=150.0,
        uv_stability=400.0,
    )
    
    material.economic = EconomicFactors(
        raw_material_cost=6.50,
        processing_cost=3.00,
        manufacturability_score=65.0,
        sustainability_score=60.0,
    )
    
    # Properties
    material.set_property(PropertyType.TENSILE_STRENGTH, 30.0 * modulus_factor, "MPa")
    material.set_property(PropertyType.ELONGATION, 550.0, "%")
    material.set_property(PropertyType.HARDNESS, 60.0, "Shore A")
    material.set_property(PropertyType.RESILIENCE, 65.0, "%")
    material.set_property(PropertyType.ABRASION_RESISTANCE, 140.0 * modulus_factor, "index")
    
    material.patents = ["Patent #91"]
    
    return material


def create_adaptive_crosslink_material(
    variant: str,
    crosslink_type: str,
    responsiveness: float,
) -> NanoArchitecture:
    """Create material with adaptive crosslinks.
    
    Args:
        variant: Material variant ID
        crosslink_type: Type of adaptive crosslink
        responsiveness: Environmental responsiveness (0-1)
        
    Returns:
        Adaptive crosslink material
    """
    material = NanoArchitecture(
        material_id=f"ACL-{variant}",
        name=f"Adaptive Crosslink {variant}",
        description=(
            f"Material with {crosslink_type} adaptive crosslinks. "
            f"Responsiveness: {responsiveness:.0%}. "
            "Crosslinks can reform after stress or damage."
        ),
        nano_params=NanostructureParameters(
            structure_type="dynamic_network",
            responsiveness=responsiveness,
        ),
    )
    
    material.molecular = MolecularSpecification(
        formula="Polymer-[Dynamic-Link]",
        crosslink_density=150.0,  # High initial crosslinking
        vulcanization_system="dynamic_covalent",
    )
    
    # Adaptive behavior
    material.mechanical = MechanicalBehavior(
        model_type="mooney_rivlin",
        c10=0.45,
        c01=0.12,
        viscosity=2000,  # Higher viscosity for stress relaxation
        relaxation_time=0.5 * responsiveness,  # Faster with responsiveness
    )
    
    # Self-repair capability affects wear
    repair_factor = 1 + responsiveness
    
    material.wear = WearModel(
        abrasion_index=120.0 * repair_factor,
        fatigue_exponent=2.0,  # Better fatigue from repair
        endurance_limit=70.0 * repair_factor,
    )
    
    material.set_property(PropertyType.TENSILE_STRENGTH, 22.0, "MPa")
    material.set_property(PropertyType.FATIGUE_LIFE, 2e7 * repair_factor, "cycles")
    
    material.patents = ["Patent #92", "Patent #93"]
    
    return material


def create_hierarchical_structure_material(
    variant: str,
    levels: int,
    primary_size: float,
) -> NanoArchitecture:
    """Create material with hierarchical nanostructure.
    
    Args:
        variant: Material variant ID  
        levels: Number of hierarchical levels
        primary_size: Primary structure size (nm)
        
    Returns:
        Hierarchical structure material
    """
    material = NanoArchitecture(
        material_id=f"HNS-{variant}",
        name=f"Hierarchical Nanostructure {variant}",
        description=(
            f"Material with {levels}-level hierarchical structure. "
            f"Primary features at {primary_size:.0f} nm scale."
        ),
        nano_params=NanostructureParameters(
            structure_type="hierarchical",
            domain_size=primary_size,
            ordering_parameter=0.9,  # High ordering
        ),
    )
    
    # Enhancement increases with hierarchy levels
    hierarchy_factor = 1 + 0.2 * levels
    
    material.mechanical = MechanicalBehavior(
        c10=0.55 * hierarchy_factor,
        c01=0.14 * hierarchy_factor,
    )
    
    material.wear = WearModel(
        abrasion_index=160.0 * hierarchy_factor,
        tearing_energy=55.0 * hierarchy_factor,
    )
    
    material.set_property(PropertyType.TENSILE_STRENGTH, 28.0 * hierarchy_factor, "MPa")
    material.set_property(PropertyType.TEAR_STRENGTH, 90.0 * hierarchy_factor, "kN/m")
    
    material.patents = ["Patent #94"]
    
    return material


def create_nanocomposite_material(
    variant: str,
    filler_type: str,
    filler_loading: float,
    aspect_ratio: float,
) -> NanoArchitecture:
    """Create nanocomposite material.
    
    Args:
        variant: Material variant ID
        filler_type: Type of nanofiller
        filler_loading: Filler loading (vol%)
        aspect_ratio: Filler aspect ratio
        
    Returns:
        Nanocomposite material
    """
    material = NanoArchitecture(
        material_id=f"NCP-{variant}",
        name=f"Nanocomposite {variant}",
        description=(
            f"Nanocomposite with {filler_type} at {filler_loading:.1f} vol%. "
            f"Aspect ratio: {aspect_ratio:.0f}."
        ),
        nano_params=NanostructureParameters(
            structure_type="nanocomposite",
            domain_size=aspect_ratio,  # Use for filler length
            ordering_parameter=0.7,
        ),
    )
    
    # Halpin-Tsai type reinforcement
    # E_c/E_m = (1 + ξηφ)/(1 - ηφ)
    E_f = 1000.0  # GPa for graphene/CNT
    E_m = 0.001  # GPa for rubber
    xi = 2 * aspect_ratio
    eta = (E_f/E_m - 1) / (E_f/E_m + xi)
    phi = filler_loading / 100
    
    reinforcement = (1 + xi * eta * phi) / (1 - eta * phi)
    reinforcement = min(reinforcement, 10.0)  # Cap at 10x
    
    material.mechanical = MechanicalBehavior(
        c10=0.4 * reinforcement,
        c01=0.1 * reinforcement,
    )
    
    material.thermal = ThermalBehavior(
        glass_transition=218.0,
        thermal_conductivity=0.15 + 0.5 * phi,  # Enhanced thermal
    )
    
    material.wear = WearModel(
        abrasion_index=180.0 * np.sqrt(reinforcement),
    )
    
    material.set_property(PropertyType.TENSILE_STRENGTH, 20.0 * reinforcement, "MPa")
    material.set_property(PropertyType.MODULUS, 5.0 * reinforcement, "MPa")
    
    material.patents = ["Patent #95"]
    
    return material


# Build Nanoarchitecture Catalog - 15 materials

NANO_CATALOG: dict[str, NanoArchitecture] = {}

# Self-Assembling Polymers (4 materials)
_sap_configs = [
    ("1", "block_copolymer", 15.0, 0.75),
    ("2", "block_copolymer", 25.0, 0.85),
    ("3", "star_polymer", 10.0, 0.70),
    ("4", "dendritic", 8.0, 0.90),
]

for variant, assembly, size, ordering in _sap_configs:
    key = f"SAP-{variant}"
    NANO_CATALOG[key] = create_self_assembling_polymer(variant, assembly, size, ordering)

# Adaptive Crosslink materials (4 materials)
_acl_configs = [
    ("1", "disulfide_exchange", 0.5),
    ("2", "imine_exchange", 0.7),
    ("3", "boronate_ester", 0.8),
    ("4", "metal_ligand", 0.9),
]

for variant, crosslink, resp in _acl_configs:
    key = f"ACL-{variant}"
    NANO_CATALOG[key] = create_adaptive_crosslink_material(variant, crosslink, resp)

# Hierarchical structures (3 materials)
_hns_configs = [
    ("1", 2, 50.0),
    ("2", 3, 30.0),
    ("3", 4, 20.0),
]

for variant, levels, size in _hns_configs:
    key = f"HNS-{variant}"
    NANO_CATALOG[key] = create_hierarchical_structure_material(variant, levels, size)

# Nanocomposites (4 materials)
_ncp_configs = [
    ("GNP", "graphene_nanoplatelet", 2.0, 1000),
    ("CNT", "carbon_nanotube", 1.0, 500),
    ("NSi", "nanosilica", 15.0, 1),
    ("NCl", "nanoclay", 5.0, 100),
]

for variant, filler, loading, ar in _ncp_configs:
    key = f"NCP-{variant}"
    NANO_CATALOG[key] = create_nanocomposite_material(variant, filler, loading, ar)

# Aliases
SelfAssemblingPolymer = NANO_CATALOG["SAP-2"]
AdaptiveCrosslink = NANO_CATALOG["ACL-4"]
