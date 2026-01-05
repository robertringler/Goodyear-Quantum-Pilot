"""Natural Rubber Materials Database.

Contains 15 natural rubber variants with complete specifications:
- SMR (Standard Malaysian Rubber)
- SVR (Standard Vietnamese Rubber)
- TSR (Technically Specified Rubber)
- Specialty natural rubbers
"""

from __future__ import annotations

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


def create_natural_rubber(
    variant: str,
    grade: str,
    dirt_content: float = 0.05,
    ash_content: float = 0.6,
    nitrogen_content: float = 0.6,
    volatile_matter: float = 0.8,
    plasticity: float = 30.0,
    pri: float = 60.0,  # Plasticity Retention Index
) -> Material:
    """Create a natural rubber material.

    Args:
        variant: Rubber variant (SMR, SVR, TSR, etc.)
        grade: Quality grade (CV, L, 5, 10, 20, etc.)
        dirt_content: Dirt content (% max)
        ash_content: Ash content (% max)
        nitrogen_content: Nitrogen content (% max)
        volatile_matter: Volatile matter (% max)
        plasticity: Initial Wallace plasticity (P0)
        pri: Plasticity Retention Index

    Returns:
        Configured natural rubber material
    """
    material = Material(
        material_id=f"NR-{variant}-{grade}",
        name=f"Natural Rubber {variant} {grade}",
        category=MaterialCategory.NATURAL_RUBBER,
        description=f"{variant} grade {grade} natural rubber (Hevea brasiliensis)",
    )

    # Molecular specification - cis-1,4-polyisoprene
    material.molecular = MolecularSpecification(
        formula="(C5H8)n",
        molecular_weight=68.12,
        smiles="CC(=C)C=C",
        monomer_units=5000,  # High MW natural rubber
        chain_length=200.0,  # Long chains
        crosslink_density=80.0,  # After vulcanization
        num_atoms=13,  # Per monomer
        num_electrons=40,  # Per monomer
        point_group="C2v",
    )

    # Quantum Hamiltonian
    num_orb = 10
    material.hamiltonian = HamiltonianParameters(
        one_body=np.random.randn(num_orb, num_orb) * 0.1 - 0.45 * np.eye(num_orb),
        two_body=np.random.randn(num_orb, num_orb, num_orb, num_orb) * 0.008,
        nuclear_repulsion=12.5,
        num_orbitals=num_orb,
        num_electrons=16,
        basis_set="cc-pVDZ",
        spin_multiplicity=1,
    )

    # Mechanical behavior - excellent for NR
    # Quality affects properties
    quality_factor = 1.0 + 0.01 * pri - 0.5 * dirt_content

    material.mechanical = MechanicalBehavior(
        model_type="mooney_rivlin",
        c10=0.35 * quality_factor,
        c01=0.08 * quality_factor,
        ogden_mu=[0.62, 0.0012, -0.01],
        ogden_alpha=[1.3, 5.0, -2.0],
        viscosity=800.0,
        relaxation_time=0.05,
        strain_rate_sensitivity=0.08,
    )

    # Thermal behavior
    material.thermal = ThermalBehavior(
        glass_transition=200.0,  # -73°C, very flexible
        melting_point=303.0,  # 30°C crystallization
        degradation_temp=473.0,  # Lower than synthetics
        thermal_conductivity=0.13,
        specific_heat=1880.0,
        thermal_expansion=2.2e-4,
        heat_buildup_rate=0.008,  # Lower than SBR
    )

    # Wear model - excellent for NR
    material.wear = WearModel(
        abrasion_index=120.0 * quality_factor,
        wear_rate=8e-9,
        fatigue_exponent=2.3,  # Excellent fatigue
        paris_c=5e-16,  # Good crack resistance
        paris_m=3.5,
        endurance_limit=60.0,  # Higher than SBR
        tearing_energy=40.0,  # Excellent tear
    )

    # Environmental reactivity - NR is less resistant
    material.environmental = EnvironmentalReactivity(
        ozone_resistance=30.0,  # Poor ozone resistance
        uv_stability=200.0,  # Moderate UV
        hydrolysis_rate=5e-7,
        oxidation_activation=70.0,
        oxidation_rate=2e-5,
    )

    # Economic factors
    # Price varies with grade
    base_cost = 1.80  # USD/kg
    if "CV" in grade or "L" in grade:
        base_cost = 2.20
    elif "5" in grade:
        base_cost = 1.90

    material.economic = EconomicFactors(
        raw_material_cost=base_cost,
        processing_cost=1.20,
        energy_intensity=35.0,  # Lower than synthetics
        manufacturability_score=90.0,
        sustainability_score=85.0,  # Renewable resource
        recyclability=20.0,
        carbon_footprint=1.5,  # Lower carbon footprint
    )

    # Properties
    material.set_property(
        PropertyType.TENSILE_STRENGTH,
        25.0 * quality_factor,
        "MPa",
        source="experimental",
        quantum_enhanced=False,
    )
    material.set_property(PropertyType.ELONGATION, 650.0, "%")
    material.set_property(PropertyType.HARDNESS, 45.0, "Shore A")
    material.set_property(PropertyType.RESILIENCE, 75.0, "%")  # Excellent
    material.set_property(PropertyType.HYSTERESIS, 0.08, "dimensionless")  # Low
    material.set_property(PropertyType.TEAR_STRENGTH, 80.0, "kN/m")
    material.set_property(PropertyType.ABRASION_RESISTANCE, 120.0, "index")
    material.set_property(PropertyType.GLASS_TRANSITION, 200.0, "K")
    material.set_property(PropertyType.FATIGUE_LIFE, 5e6, "cycles")

    # Store grade info
    material.description += f"\nDirt: {dirt_content}%, Ash: {ash_content}%, N: {nitrogen_content}%"
    material.description += f"\nP0: {plasticity}, PRI: {pri}"

    return material


# Natural Rubber Catalog - 15 materials

RUBBER_CATALOG: dict[str, Material] = {}

# SMR (Standard Malaysian Rubber) grades
_smr_configs = [
    ("SMR", "CV60", 0.03, 0.5, 0.6, 0.8, 60, 60),  # Constant Viscosity
    ("SMR", "L", 0.03, 0.5, 0.6, 0.8, 30, 60),  # Light colored
    ("SMR", "5", 0.05, 0.6, 0.6, 0.8, 30, 60),
    ("SMR", "10", 0.10, 0.75, 0.6, 0.8, 30, 50),
    ("SMR", "20", 0.20, 1.0, 0.6, 0.8, 30, 40),
]

for variant, grade, dirt, ash, n, vol, p0, pri in _smr_configs:
    key = f"{variant}-{grade}"
    RUBBER_CATALOG[key] = create_natural_rubber(variant, grade, dirt, ash, n, vol, p0, pri)

# SVR (Standard Vietnamese Rubber) grades
_svr_configs = [
    ("SVR", "CV60", 0.03, 0.5, 0.6, 0.8, 60, 60),
    ("SVR", "L", 0.03, 0.5, 0.6, 0.8, 30, 60),
    ("SVR", "3L", 0.03, 0.5, 0.6, 0.8, 35, 60),
    ("SVR", "5", 0.05, 0.6, 0.6, 0.8, 30, 60),
    ("SVR", "10", 0.10, 0.75, 0.6, 0.8, 30, 50),
]

for variant, grade, dirt, ash, n, vol, p0, pri in _svr_configs:
    key = f"{variant}-{grade}"
    RUBBER_CATALOG[key] = create_natural_rubber(variant, grade, dirt, ash, n, vol, p0, pri)

# TSR (Technically Specified Rubber) - generic designation
_tsr_configs = [
    ("TSR", "10", 0.10, 0.75, 0.6, 0.8, 30, 50),
    ("TSR", "20", 0.20, 1.0, 0.6, 0.8, 30, 40),
]

for variant, grade, dirt, ash, n, vol, p0, pri in _tsr_configs:
    key = f"{variant}-{grade}"
    RUBBER_CATALOG[key] = create_natural_rubber(variant, grade, dirt, ash, n, vol, p0, pri)


# Specialty Natural Rubbers
# Epoxidized Natural Rubber (ENR)
def create_enr_material(epoxidation: float) -> Material:
    """Create Epoxidized Natural Rubber."""
    material = create_natural_rubber("ENR", f"{int(epoxidation)}", 0.03, 0.5, 0.6, 0.8, 30, 50)
    material.material_id = f"ENR-{int(epoxidation)}"
    material.name = f"Epoxidized Natural Rubber {int(epoxidation)}%"
    material.description = f"Natural rubber with {epoxidation}% epoxidation"

    # ENR has improved oil resistance and lower Tg
    material.thermal.glass_transition = 200.0 + 1.0 * epoxidation

    # Better wet grip
    material.set_property(PropertyType.OIL_RESISTANCE, 30 + epoxidation, "index")

    return material


RUBBER_CATALOG["ENR-25"] = create_enr_material(25.0)
RUBBER_CATALOG["ENR-50"] = create_enr_material(50.0)


# Deproteinized Natural Rubber (DPNR)
def create_dpnr_material() -> Material:
    """Create Deproteinized Natural Rubber."""
    material = create_natural_rubber("DPNR", "HP", 0.02, 0.3, 0.1, 0.5, 30, 65)
    material.material_id = "DPNR"
    material.name = "Deproteinized Natural Rubber"
    material.description = "High purity NR with reduced protein allergens"

    # Lower nitrogen = reduced protein
    material.set_property(PropertyType.TENSILE_STRENGTH, 22.0, "MPa")  # Slightly lower

    return material


RUBBER_CATALOG["DPNR"] = create_dpnr_material()

# Alias exports
SMR = RUBBER_CATALOG["SMR-CV60"]
SVR = RUBBER_CATALOG["SVR-CV60"]
TSR = RUBBER_CATALOG["TSR-10"]


class NaturalRubber(Material):
    """Class representing natural rubber materials.

    Provides additional methods specific to natural rubber processing.
    """

    def __init__(self, variant: str = "SMR", grade: str = "CV60") -> None:
        """Initialize from catalog or create new."""
        key = f"{variant}-{grade}"
        if key in RUBBER_CATALOG:
            base = RUBBER_CATALOG[key]
            super().__init__(
                material_id=base.material_id,
                name=base.name,
                category=base.category,
                description=base.description,
            )
            self.molecular = base.molecular
            self.hamiltonian = base.hamiltonian
            self.mechanical = base.mechanical
            self.thermal = base.thermal
            self.wear = base.wear
            self.environmental = base.environmental
            self.economic = base.economic
            self.properties = base.properties.copy()
        else:
            base = create_natural_rubber(variant, grade)
            super().__init__(
                material_id=base.material_id,
                name=base.name,
                category=base.category,
            )

    def estimate_mastication_time(self, target_mooney: float) -> float:
        """Estimate mastication time to reach target Mooney viscosity.

        Args:
            target_mooney: Target Mooney viscosity (ML 1+4 at 100°C)

        Returns:
            Estimated mastication time (minutes)
        """
        # Simplified model: t = k * ln(ML0/MLt)
        initial_mooney = 80.0  # Typical for raw NR
        k = 2.5  # Rate constant

        if target_mooney >= initial_mooney:
            return 0.0

        return k * np.log(initial_mooney / target_mooney)

    def get_cure_characteristics(
        self,
        temperature: float = 433.0,  # 160°C
    ) -> dict[str, float]:
        """Get vulcanization cure characteristics.

        Args:
            temperature: Cure temperature (K)

        Returns:
            Dictionary with cure parameters
        """
        # Simplified cure kinetics
        # Activation energy for sulfur cure
        Ea = 80.0  # kJ/mol
        R = 8.314e-3  # kJ/mol·K

        # Reference cure at 150°C (423 K)
        t90_ref = 8.0  # minutes

        # Arrhenius temperature correction
        factor = np.exp(Ea / R * (1 / 423 - 1 / temperature))
        t90 = t90_ref * factor

        return {
            "ts2": t90 * 0.15,  # Scorch time
            "t50": t90 * 0.5,  # 50% cure
            "t90": t90,  # 90% cure
            "mh": 45.0,  # Maximum torque (dNm)
            "ml": 5.0,  # Minimum torque (dNm)
            "delta_torque": 40.0,
        }

    def predict_strain_crystallization(
        self,
        strain: float,
        temperature: float = 298.0,
    ) -> float:
        """Predict strain-induced crystallization.

        Natural rubber undergoes strain crystallization which
        provides self-reinforcement at high strains.

        Args:
            strain: Engineering strain
            temperature: Temperature (K)

        Returns:
            Crystallinity fraction (0-1)
        """
        if strain < 2.0:  # Below critical strain
            return 0.0

        # Temperature effect
        if temperature > 303:  # Above crystallization temperature
            temp_factor = max(0, 1 - (temperature - 303) / 30)
        else:
            temp_factor = 1.0

        # Strain crystallization onset ~200% strain
        crystallinity = min(0.3, 0.1 * (strain - 2.0)) * temp_factor

        return crystallinity
