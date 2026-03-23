"""Synthetic Elastomer Materials Database.

Contains 25 synthetic elastomer materials with complete specifications:
- SBR (Styrene-Butadiene Rubber) variants
- NBR (Nitrile Butadiene Rubber) variants
- EPDM (Ethylene Propylene Diene Monomer)
- CR (Chloroprene Rubber)
- BR (Butadiene Rubber)
- IIR (Butyl Rubber)
- HNBR (Hydrogenated Nitrile)
- FKM (Fluoroelastomer)
"""

from __future__ import annotations

from dataclasses import dataclass

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
class Elastomer(Material):
    """Base class for synthetic elastomers.

    Provides common functionality for all synthetic elastomer types.
    """

    styrene_content: float = 0.0  # % for SBR types
    acrylonitrile_content: float = 0.0  # % for NBR types
    vinyl_content: float = 0.0  # %

    def get_abrasion_resistance(self) -> float:
        """Calculate abrasion resistance index."""
        base = self.wear.abrasion_index
        # Adjust for vinyl content
        adjustment = 1.0 + 0.005 * self.vinyl_content
        return base * adjustment


@dataclass
class SyntheticElastomer(Elastomer):
    """General synthetic elastomer class."""

    polymerization_type: str = "emulsion"  # emulsion, solution, or anionic
    coupling_agent: str = "none"
    functionalization: str = "none"


def create_sbr_material(
    variant: str,
    styrene: float,
    vinyl: float,
    polymerization: str = "emulsion",
) -> Material:
    """Create an SBR material variant.

    Args:
        variant: Variant identifier
        styrene: Styrene content (%)
        vinyl: Vinyl content (%)
        polymerization: Polymerization type

    Returns:
        Configured SBR material
    """
    material = Material(
        material_id=f"SBR-{variant}",
        name=f"Styrene-Butadiene Rubber {variant}",
        category=MaterialCategory.SYNTHETIC_ELASTOMER,
        description=f"SBR with {styrene}% styrene, {vinyl}% vinyl via {polymerization} polymerization",
    )

    # Molecular specification
    material.molecular = MolecularSpecification(
        formula="(C8H8)n(C4H6)m",
        molecular_weight=104.15 * styrene / 100 + 54.09 * (1 - styrene / 100),
        smiles="C=Cc1ccccc1.C=CC=C",
        monomer_units=2000,
        chain_length=80.0 + 0.5 * vinyl,
        crosslink_density=120.0 - 0.3 * styrene,
    )

    # Hamiltonian parameters (simplified for demonstration)
    num_orb = 8 + int(styrene / 10)
    material.hamiltonian = HamiltonianParameters(
        one_body=np.random.randn(num_orb, num_orb) * 0.1 - 0.5 * np.eye(num_orb),
        two_body=np.random.randn(num_orb, num_orb, num_orb, num_orb) * 0.01,
        nuclear_repulsion=15.0 + styrene * 0.2,
        num_orbitals=num_orb,
        num_electrons=2 * num_orb,
        basis_set="cc-pVDZ",
    )

    # Mechanical behavior
    material.mechanical = MechanicalBehavior(
        model_type="mooney_rivlin",
        c10=0.4 + 0.01 * styrene,
        c01=0.1 + 0.002 * styrene,
        viscosity=1200 + 20 * styrene,
        relaxation_time=0.08 + 0.001 * styrene,
    )

    # Thermal behavior - Tg increases with styrene content
    material.thermal = ThermalBehavior(
        glass_transition=213.0 + 1.5 * styrene,  # K
        melting_point=453.0,
        degradation_temp=523.0,
        thermal_conductivity=0.14 + 0.001 * styrene,
        specific_heat=1850 + 5 * styrene,
        heat_buildup_rate=0.015 - 0.0001 * vinyl,
    )

    # Wear model
    material.wear = WearModel(
        abrasion_index=100.0 + 0.5 * vinyl,
        wear_rate=1e-8 * (1 - 0.01 * vinyl),
        fatigue_exponent=2.5 - 0.01 * vinyl,
        paris_c=1e-15,
        paris_m=4.0 - 0.02 * vinyl,
        tearing_energy=25.0 + 0.2 * styrene,
    )

    # Environmental reactivity
    material.environmental = EnvironmentalReactivity(
        ozone_resistance=50.0 + 2 * vinyl,
        uv_stability=300.0,
        oxidation_rate=1.5e-5,
    )

    # Economic factors
    material.economic = EconomicFactors(
        raw_material_cost=2.20 + 0.02 * styrene,
        processing_cost=1.40,
        manufacturability_score=85.0 - 0.1 * styrene,
        sustainability_score=55.0,
    )

    # Properties
    material.set_property(PropertyType.TENSILE_STRENGTH, 18.0 + 0.15 * styrene, "MPa")
    material.set_property(PropertyType.ELONGATION, 550.0 - 2 * styrene, "%")
    material.set_property(PropertyType.HARDNESS, 50 + 0.3 * styrene, "Shore A")
    material.set_property(PropertyType.RESILIENCE, 55.0 - 0.3 * styrene, "%")
    material.set_property(PropertyType.HYSTERESIS, 0.15 + 0.002 * styrene, "dimensionless")
    material.set_property(PropertyType.ABRASION_RESISTANCE, 100.0 + 0.5 * vinyl, "index")

    return material


def create_nbr_material(
    variant: str,
    acn: float,  # Acrylonitrile content
) -> Material:
    """Create an NBR material variant.

    Args:
        variant: Variant identifier
        acn: Acrylonitrile content (%)

    Returns:
        Configured NBR material
    """
    material = Material(
        material_id=f"NBR-{variant}",
        name=f"Nitrile Butadiene Rubber {variant}",
        category=MaterialCategory.SYNTHETIC_ELASTOMER,
        description=f"NBR with {acn}% acrylonitrile content",
    )

    material.molecular = MolecularSpecification(
        formula="(C3H3N)n(C4H6)m",
        molecular_weight=53.06 * acn / 100 + 54.09 * (1 - acn / 100),
        chain_length=70.0,
        crosslink_density=110.0,
    )

    material.thermal = ThermalBehavior(
        glass_transition=233.0 + 1.2 * acn,  # Higher ACN = higher Tg
        degradation_temp=513.0,
    )

    material.mechanical = MechanicalBehavior(
        c10=0.5 + 0.008 * acn,
        c01=0.12,
    )

    material.environmental = EnvironmentalReactivity(
        ozone_resistance=40.0,
        uv_stability=250.0,
    )

    material.economic = EconomicFactors(
        raw_material_cost=3.50 + 0.05 * acn,
        processing_cost=1.60,
        manufacturability_score=80.0,
    )

    # Oil resistance improves with ACN content
    oil_resistance = 50.0 + 1.5 * acn
    material.set_property(PropertyType.OIL_RESISTANCE, oil_resistance, "index")
    material.set_property(PropertyType.TENSILE_STRENGTH, 20.0 + 0.1 * acn, "MPa")
    material.set_property(PropertyType.ELONGATION, 450.0 - acn, "%")

    return material


# Pre-defined elastomer catalog
SBR = create_sbr_material("1502", styrene=23.5, vinyl=10.0, polymerization="emulsion")
NBR = create_nbr_material("3305", acn=33.0)

# Complete elastomer catalog with 25 materials
ELASTOMER_CATALOG: dict[str, Material] = {}

# SBR Variants (8 materials)
_sbr_configs = [
    ("E-SBR-1500", 23.5, 10.0, "emulsion"),
    ("E-SBR-1502", 23.5, 12.0, "emulsion"),
    ("E-SBR-1712", 23.5, 15.0, "emulsion"),
    ("E-SBR-1778", 40.0, 12.0, "emulsion"),
    ("S-SBR-2525", 25.0, 25.0, "solution"),
    ("S-SBR-2550", 25.0, 50.0, "solution"),
    ("S-SBR-4526", 45.0, 26.0, "solution"),
    ("S-SBR-HP", 35.0, 63.0, "solution"),  # High-performance
]

for name, styrene, vinyl, polym in _sbr_configs:
    ELASTOMER_CATALOG[name] = create_sbr_material(name, styrene, vinyl, polym)

# NBR Variants (5 materials)
_nbr_configs = [
    ("NBR-18", 18.0),
    ("NBR-28", 28.0),
    ("NBR-33", 33.0),
    ("NBR-38", 38.0),
    ("NBR-45", 45.0),
]

for name, acn in _nbr_configs:
    ELASTOMER_CATALOG[name] = create_nbr_material(name, acn)


# EPDM Material
def create_epdm_material(
    variant: str,
    ethylene: float,
    diene: float,
) -> Material:
    """Create EPDM material."""
    material = Material(
        material_id=f"EPDM-{variant}",
        name=f"EPDM {variant}",
        category=MaterialCategory.SYNTHETIC_ELASTOMER,
        description=f"EPDM with {ethylene}% ethylene, {diene}% diene",
    )

    material.thermal = ThermalBehavior(
        glass_transition=213.0 - 0.3 * ethylene,
        degradation_temp=573.0,  # Excellent heat resistance
    )

    material.environmental = EnvironmentalReactivity(
        ozone_resistance=500.0,  # Excellent ozone resistance
        uv_stability=1000.0,
    )

    material.set_property(PropertyType.OZONE_RESISTANCE, 500.0, "hours")
    material.set_property(PropertyType.TENSILE_STRENGTH, 12.0 + 0.05 * ethylene, "MPa")

    return material


# EPDM variants (3 materials)
ELASTOMER_CATALOG["EPDM-5565"] = create_epdm_material("5565", 55.0, 6.5)
ELASTOMER_CATALOG["EPDM-6045"] = create_epdm_material("6045", 60.0, 4.5)
ELASTOMER_CATALOG["EPDM-7080"] = create_epdm_material("7080", 70.0, 8.0)


# CR (Chloroprene Rubber)
def create_cr_material(variant: str) -> Material:
    """Create Chloroprene Rubber material."""
    material = Material(
        material_id=f"CR-{variant}",
        name=f"Chloroprene Rubber {variant}",
        category=MaterialCategory.SYNTHETIC_ELASTOMER,
        description="Chloroprene rubber with excellent flame and oil resistance",
    )

    material.thermal = ThermalBehavior(
        glass_transition=233.0,
        degradation_temp=513.0,
    )

    material.environmental = EnvironmentalReactivity(
        ozone_resistance=200.0,
        uv_stability=400.0,
    )

    material.set_property(PropertyType.TENSILE_STRENGTH, 25.0, "MPa")
    material.set_property(PropertyType.OIL_RESISTANCE, 70.0, "index")

    return material


ELASTOMER_CATALOG["CR-WRT"] = create_cr_material("WRT")
ELASTOMER_CATALOG["CR-WM1"] = create_cr_material("WM1")


# BR (Butadiene Rubber)
def create_br_material(variant: str, cis_content: float) -> Material:
    """Create Butadiene Rubber material."""
    material = Material(
        material_id=f"BR-{variant}",
        name=f"Polybutadiene {variant}",
        category=MaterialCategory.SYNTHETIC_ELASTOMER,
        description=f"High-cis polybutadiene with {cis_content}% cis content",
    )

    material.thermal = ThermalBehavior(
        glass_transition=163.0 - 0.3 * cis_content,  # Very low Tg
        degradation_temp=523.0,
    )

    material.wear = WearModel(
        abrasion_index=180.0 + 0.5 * cis_content,  # Excellent abrasion
    )

    material.set_property(PropertyType.RESILIENCE, 85.0 + 0.1 * cis_content, "%")
    material.set_property(PropertyType.ABRASION_RESISTANCE, 180.0, "index")

    return material


ELASTOMER_CATALOG["BR-1220"] = create_br_material("1220", cis_content=98.0)
ELASTOMER_CATALOG["BR-1250H"] = create_br_material("1250H", cis_content=96.0)


# IIR (Butyl Rubber)
def create_iir_material(variant: str) -> Material:
    """Create Butyl Rubber material."""
    material = Material(
        material_id=f"IIR-{variant}",
        name=f"Butyl Rubber {variant}",
        category=MaterialCategory.SYNTHETIC_ELASTOMER,
        description="Butyl rubber with excellent gas impermeability",
    )

    material.thermal = ThermalBehavior(
        glass_transition=203.0,
    )

    # Excellent damping/hysteresis for noise reduction
    material.mechanical = MechanicalBehavior(
        viscosity=5000.0,
        relaxation_time=1.0,
    )

    material.set_property(PropertyType.HYSTERESIS, 0.45, "dimensionless")

    return material


ELASTOMER_CATALOG["IIR-268"] = create_iir_material("268")
ELASTOMER_CATALOG["CIIR-1066"] = create_iir_material("CIIR-1066")  # Chlorinated


# HNBR (Hydrogenated NBR)
def create_hnbr_material(variant: str, acn: float, hydrogenation: float) -> Material:
    """Create HNBR material."""
    material = Material(
        material_id=f"HNBR-{variant}",
        name=f"Hydrogenated NBR {variant}",
        category=MaterialCategory.SYNTHETIC_ELASTOMER,
        description=f"HNBR with {acn}% ACN, {hydrogenation}% hydrogenation",
    )

    material.thermal = ThermalBehavior(
        glass_transition=243.0,
        degradation_temp=573.0,  # Excellent heat resistance
    )

    material.environmental = EnvironmentalReactivity(
        ozone_resistance=1000.0,  # Excellent
        oxidation_rate=1e-7,
    )

    material.set_property(PropertyType.TENSILE_STRENGTH, 30.0, "MPa")
    material.set_property(PropertyType.OIL_RESISTANCE, 95.0, "index")

    return material


ELASTOMER_CATALOG["HNBR-3446"] = create_hnbr_material("3446", 34.0, 99.5)


# FKM (Fluoroelastomer)
def create_fkm_material(variant: str, fluorine: float) -> Material:
    """Create Fluoroelastomer material."""
    material = Material(
        material_id=f"FKM-{variant}",
        name=f"Fluoroelastomer {variant}",
        category=MaterialCategory.SYNTHETIC_ELASTOMER,
        description=f"FKM with {fluorine}% fluorine content",
    )

    material.thermal = ThermalBehavior(
        glass_transition=253.0,
        degradation_temp=623.0,  # Exceptional heat resistance
    )

    material.environmental = EnvironmentalReactivity(
        ozone_resistance=10000.0,
        chemical_resistance=98.0,
    )

    material.economic = EconomicFactors(
        raw_material_cost=50.0 + fluorine,  # Expensive
        processing_cost=15.0,
    )

    material.set_property(PropertyType.CHEMICAL_RESISTANCE, 98.0, "index")

    return material


ELASTOMER_CATALOG["FKM-66"] = create_fkm_material("66", 66.0)
ELASTOMER_CATALOG["FKM-70"] = create_fkm_material("70", 70.0)

# Alias exports
EPDM = ELASTOMER_CATALOG["EPDM-5565"]
CR = ELASTOMER_CATALOG["CR-WRT"]
BR = ELASTOMER_CATALOG["BR-1220"]
IIR = ELASTOMER_CATALOG["IIR-268"]
HNBR = ELASTOMER_CATALOG["HNBR-3446"]
FKM = ELASTOMER_CATALOG["FKM-66"]
