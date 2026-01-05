"""Base material classes and property definitions.

Provides the foundational data structures for representing
tire materials with quantum-derived properties.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class PropertyType(Enum):
    """Types of material properties."""

    # Mechanical properties
    TENSILE_STRENGTH = auto()
    ELONGATION = auto()
    MODULUS = auto()
    HARDNESS = auto()
    TEAR_STRENGTH = auto()
    COMPRESSION_SET = auto()
    RESILIENCE = auto()
    HYSTERESIS = auto()

    # Thermal properties
    GLASS_TRANSITION = auto()
    THERMAL_CONDUCTIVITY = auto()
    HEAT_CAPACITY = auto()
    THERMAL_EXPANSION = auto()
    HEAT_BUILDUP = auto()

    # Chemical properties
    OZONE_RESISTANCE = auto()
    OIL_RESISTANCE = auto()
    CHEMICAL_RESISTANCE = auto()
    OXIDATION_RESISTANCE = auto()

    # Wear properties
    ABRASION_RESISTANCE = auto()
    CUT_RESISTANCE = auto()
    FATIGUE_LIFE = auto()
    CRACK_GROWTH = auto()

    # Quantum properties
    QUANTUM_COHERENCE = auto()
    TUNNELING_PROBABILITY = auto()
    ENTANGLEMENT_DENSITY = auto()
    CROSSLINK_STABILITY = auto()

    # Economic properties
    COST = auto()
    MANUFACTURABILITY = auto()
    SUSTAINABILITY = auto()

    # Self-healing properties
    HEALING_EFFICIENCY = auto()
    HEALING_TIME = auto()
    HEALING_CYCLES = auto()


class MaterialCategory(Enum):
    """Categories of tire materials."""

    SYNTHETIC_ELASTOMER = auto()
    NATURAL_RUBBER = auto()
    QUANTUM_ENGINEERED = auto()
    NANOARCHITECTURE = auto()
    SELF_HEALING = auto()
    ZERO_WEAR = auto()
    HIGH_ENTROPY = auto()
    COMPOSITE = auto()


@dataclass
class MaterialProperty:
    """A single material property with value and metadata.

    Attributes:
        property_type: Type of property
        value: Property value
        unit: Unit of measurement
        temperature: Temperature at which measured (K)
        uncertainty: Measurement uncertainty
        source: Data source (experimental, computed, etc.)
        quantum_enhanced: Whether quantum computation improved this value
    """

    property_type: PropertyType
    value: float
    unit: str
    temperature: float = 298.15  # Room temperature in K
    uncertainty: float = 0.0
    source: str = "computed"
    quantum_enhanced: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.property_type.name,
            "value": self.value,
            "unit": self.unit,
            "temperature": self.temperature,
            "uncertainty": self.uncertainty,
            "source": self.source,
            "quantum_enhanced": self.quantum_enhanced,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MaterialProperty:
        """Create from dictionary."""
        return cls(
            property_type=PropertyType[data["type"]],
            value=data["value"],
            unit=data["unit"],
            temperature=data.get("temperature", 298.15),
            uncertainty=data.get("uncertainty", 0.0),
            source=data.get("source", "computed"),
            quantum_enhanced=data.get("quantum_enhanced", False),
        )


@dataclass
class HamiltonianParameters:
    """Quantum Hamiltonian parameters for material simulation.

    Represents the molecular Hamiltonian:
    H = Σ_i h_i a†_i a_i + Σ_{ij} J_ij a†_i a_j + Σ_{ijkl} V_{ijkl} a†_i a†_j a_k a_l

    Attributes:
        one_body: One-body integrals (h_ij)
        two_body: Two-body integrals (V_ijkl)
        nuclear_repulsion: Nuclear repulsion energy
        num_orbitals: Number of molecular orbitals
        num_electrons: Number of electrons
        spin_multiplicity: Spin multiplicity (2S+1)
        basis_set: Basis set used for calculation
        frozen_core: Number of frozen core orbitals
    """

    one_body: np.ndarray = field(default_factory=lambda: np.zeros((4, 4)))
    two_body: np.ndarray = field(default_factory=lambda: np.zeros((4, 4, 4, 4)))
    nuclear_repulsion: float = 0.0
    num_orbitals: int = 4
    num_electrons: int = 4
    spin_multiplicity: int = 1
    basis_set: str = "STO-3G"
    frozen_core: int = 0

    @property
    def num_qubits_required(self) -> int:
        """Qubits needed for quantum simulation (Jordan-Wigner)."""
        return 2 * (self.num_orbitals - self.frozen_core)

    def to_qubit_hamiltonian(self) -> dict[str, complex]:
        """Convert to qubit Hamiltonian using Jordan-Wigner transform.

        Returns:
            Dictionary mapping Pauli strings to coefficients
        """
        # Simplified implementation
        hamiltonian = {}

        # Identity term (nuclear repulsion + one-body diagonal)
        identity_coeff = self.nuclear_repulsion
        for i in range(self.num_orbitals):
            identity_coeff += self.one_body[i, i]
        hamiltonian["I" * self.num_qubits_required] = identity_coeff

        # One-body terms
        for i in range(self.num_orbitals):
            for j in range(self.num_orbitals):
                if i != j and abs(self.one_body[i, j]) > 1e-10:
                    # Create Pauli string for a†_i a_j
                    pauli = self._create_hopping_pauli(i, j)
                    coeff = self.one_body[i, j] / 2
                    hamiltonian[pauli] = hamiltonian.get(pauli, 0) + coeff

        return hamiltonian

    def _create_hopping_pauli(self, i: int, j: int) -> str:
        """Create Pauli string for hopping term."""
        n = self.num_qubits_required
        pauli = ["I"] * n

        # Jordan-Wigner: a†_i a_j → (X_i - iY_i)(X_j + iY_j) Z_string / 4
        pauli[i] = "X"
        pauli[j] = "X"

        return "".join(pauli)


@dataclass
class MolecularSpecification:
    """Complete molecular specification for a material.

    Attributes:
        formula: Chemical formula
        molecular_weight: Molecular weight (g/mol)
        smiles: SMILES representation
        monomer_units: Number of monomer units
        chain_length: Average chain length (nm)
        crosslink_density: Crosslinks per unit volume (mol/m³)
        filler_content: Filler content by weight (%)
        vulcanization_system: Vulcanization chemistry type
    """

    formula: str = "C8H8"
    molecular_weight: float = 104.15
    smiles: str = "C=Cc1ccccc1"
    monomer_units: int = 1000
    chain_length: float = 50.0  # nm
    crosslink_density: float = 100.0  # mol/m³
    filler_content: float = 30.0  # %
    vulcanization_system: str = "sulfur"

    # Quantum-specific
    num_atoms: int = 16
    num_electrons: int = 58
    point_group: str = "C2v"

    def get_repeat_unit_atoms(self) -> int:
        """Get number of atoms in repeat unit."""
        # Parse from formula
        import re

        atoms = 0
        for match in re.finditer(r"([A-Z][a-z]?)(\d*)", self.formula):
            count = int(match.group(2)) if match.group(2) else 1
            atoms += count
        return atoms


@dataclass
class MechanicalBehavior:
    """Mechanical behavior model for the material.

    Includes constitutive model parameters for FEM simulation.

    Attributes:
        model_type: Constitutive model (Mooney-Rivlin, Ogden, etc.)
        c10: Mooney-Rivlin C10 coefficient (MPa)
        c01: Mooney-Rivlin C01 coefficient (MPa)
        ogden_mu: Ogden mu parameters
        ogden_alpha: Ogden alpha parameters
        viscosity: Dynamic viscosity (Pa·s)
        relaxation_time: Stress relaxation time constant (s)
    """

    model_type: str = "mooney_rivlin"
    c10: float = 0.5  # MPa
    c01: float = 0.1  # MPa
    ogden_mu: list[float] = field(default_factory=lambda: [0.69, 0.01, -0.0122])
    ogden_alpha: list[float] = field(default_factory=lambda: [1.3, 5.0, -2.0])
    viscosity: float = 1000.0  # Pa·s
    relaxation_time: float = 0.1  # s

    # Strain-rate dependence
    strain_rate_sensitivity: float = 0.1

    # Temperature dependence
    wlf_c1: float = 17.44  # WLF constant C1
    wlf_c2: float = 51.6  # WLF constant C2 (K)
    reference_temperature: float = 298.15  # K

    def get_shear_modulus(self) -> float:
        """Calculate initial shear modulus."""
        if self.model_type == "mooney_rivlin":
            return 2 * (self.c10 + self.c01)
        elif self.model_type == "ogden":
            G = 0
            for mu, alpha in zip(self.ogden_mu, self.ogden_alpha):
                G += mu * alpha / 2
            return G
        return 2 * self.c10

    def get_stress_strain_curve(
        self,
        max_strain: float = 2.0,
        num_points: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate stress-strain curve.

        Args:
            max_strain: Maximum engineering strain
            num_points: Number of data points

        Returns:
            Tuple of (strain, stress) arrays
        """
        strain = np.linspace(0, max_strain, num_points)
        stretch = 1 + strain

        if self.model_type == "mooney_rivlin":
            # σ = 2(λ - λ^-2)(C10 + C01/λ)
            stress = 2 * (stretch - 1 / stretch**2) * (self.c10 + self.c01 / stretch)
        elif self.model_type == "ogden":
            stress = np.zeros_like(strain)
            for mu, alpha in zip(self.ogden_mu, self.ogden_alpha):
                stress += mu * (stretch ** (alpha - 1) - stretch ** (-0.5 * alpha - 1))
        else:
            # Neo-Hookean fallback
            stress = 2 * self.c10 * (stretch - 1 / stretch**2)

        return strain, stress


@dataclass
class ThermalBehavior:
    """Thermal behavior model for the material.

    Attributes:
        glass_transition: Glass transition temperature (K)
        melting_point: Melting point (K)
        degradation_temp: Thermal degradation onset (K)
        thermal_conductivity: Thermal conductivity (W/m·K)
        specific_heat: Specific heat capacity (J/kg·K)
        thermal_expansion: Linear thermal expansion coefficient (1/K)
        heat_buildup_rate: Heat buildup rate under cyclic loading (K/cycle)
    """

    glass_transition: float = 213.0  # K (-60°C)
    melting_point: float = 453.0  # K (180°C)
    degradation_temp: float = 523.0  # K (250°C)
    thermal_conductivity: float = 0.15  # W/m·K
    specific_heat: float = 1900.0  # J/kg·K
    thermal_expansion: float = 2e-4  # 1/K
    heat_buildup_rate: float = 0.01  # K/cycle

    def get_shift_factor(self, temperature: float, reference: float = 298.15) -> float:
        """Calculate WLF time-temperature shift factor.

        Args:
            temperature: Current temperature (K)
            reference: Reference temperature (K)

        Returns:
            Shift factor log(aT)
        """
        c1, c2 = 17.44, 51.6
        dT = temperature - reference

        if abs(dT) < 1e-6:
            return 1.0

        log_aT = -c1 * dT / (c2 + dT)
        return 10**log_aT


@dataclass
class WearModel:
    """Wear and fatigue behavior model.

    Attributes:
        abrasion_index: DIN abrasion index (relative to standard)
        wear_rate: Wear rate coefficient (mm³/N·m)
        fatigue_exponent: Fatigue crack growth exponent
        paris_c: Paris law coefficient C
        paris_m: Paris law exponent m
        endurance_limit: Endurance limit strain energy density (kJ/m³)
        tearing_energy: Tearing energy (kJ/m²)
    """

    abrasion_index: float = 100.0  # % relative to standard
    wear_rate: float = 1e-8  # mm³/N·m
    fatigue_exponent: float = 2.5
    paris_c: float = 1e-15  # Crack growth coefficient
    paris_m: float = 4.0  # Crack growth exponent
    endurance_limit: float = 50.0  # kJ/m³
    tearing_energy: float = 30.0  # kJ/m²

    def predict_crack_growth_rate(self, tearing_energy: float) -> float:
        """Predict crack growth rate using Paris law.

        dc/dN = C * T^m

        Args:
            tearing_energy: Current tearing energy (kJ/m²)

        Returns:
            Crack growth rate (m/cycle)
        """
        return self.paris_c * (tearing_energy**self.paris_m)

    def predict_fatigue_life(
        self,
        strain_amplitude: float,
        initial_crack: float = 1e-5,
        critical_crack: float = 1e-3,
    ) -> int:
        """Predict fatigue life (number of cycles).

        Args:
            strain_amplitude: Strain amplitude
            initial_crack: Initial crack size (m)
            critical_crack: Critical crack size (m)

        Returns:
            Number of cycles to failure
        """
        # Simplified calculation
        T = self.tearing_energy * strain_amplitude**2
        da_dN = self.predict_crack_growth_rate(T)

        if da_dN > 0:
            cycles = int((critical_crack - initial_crack) / da_dN)
            return max(cycles, 1)
        return int(1e9)


@dataclass
class EnvironmentalReactivity:
    """Environmental aging and reactivity model.

    Attributes:
        ozone_resistance: Ozone cracking resistance (hours at 50 pphm)
        uv_stability: UV stability half-life (hours)
        hydrolysis_rate: Hydrolysis rate constant (1/hr)
        oxidation_activation: Oxidation activation energy (kJ/mol)
        oxidation_rate: Oxidation rate at reference temp (1/hr)
    """

    ozone_resistance: float = 100.0  # hours
    uv_stability: float = 500.0  # hours half-life
    hydrolysis_rate: float = 1e-6  # 1/hr
    oxidation_activation: float = 80.0  # kJ/mol
    oxidation_rate: float = 1e-5  # 1/hr at 25°C

    def predict_property_retention(
        self,
        property_type: str,
        exposure_hours: float,
        temperature: float = 298.15,
        ozone_pphm: float = 0.0,
        uv_intensity: float = 0.0,
    ) -> float:
        """Predict property retention after environmental exposure.

        Args:
            property_type: Property to track
            exposure_hours: Exposure time (hours)
            temperature: Temperature (K)
            ozone_pphm: Ozone concentration (parts per hundred million)
            uv_intensity: UV intensity (W/m²)

        Returns:
            Property retention fraction (0-1)
        """
        R = 8.314  # J/mol·K

        # Oxidation contribution (Arrhenius)
        k_ox = self.oxidation_rate * np.exp(
            -self.oxidation_activation * 1000 / R * (1 / temperature - 1 / 298.15)
        )
        ox_damage = 1 - np.exp(-k_ox * exposure_hours)

        # Ozone contribution
        oz_damage = 0
        if ozone_pphm > 0 and self.ozone_resistance > 0:
            oz_factor = ozone_pphm / 50.0  # Normalized to 50 pphm
            oz_damage = 1 - np.exp(-oz_factor * exposure_hours / self.ozone_resistance)

        # UV contribution
        uv_damage = 0
        if uv_intensity > 0 and self.uv_stability > 0:
            uv_damage = 1 - 2 ** (-exposure_hours / self.uv_stability)

        # Combined degradation
        total_damage = min(1.0, ox_damage + oz_damage + uv_damage)

        return max(0.0, 1.0 - total_damage)


@dataclass
class EconomicFactors:
    """Economic and manufacturing factors.

    Attributes:
        raw_material_cost: Cost per kg (USD)
        processing_cost: Processing cost per kg (USD)
        energy_intensity: Energy required per kg (MJ)
        manufacturability_score: Ease of manufacturing (0-100)
        sustainability_score: Environmental sustainability (0-100)
        recyclability: Recyclability percentage
        carbon_footprint: CO2 equivalent per kg
    """

    raw_material_cost: float = 2.50  # USD/kg
    processing_cost: float = 1.50  # USD/kg
    energy_intensity: float = 50.0  # MJ/kg
    manufacturability_score: float = 75.0  # 0-100
    sustainability_score: float = 50.0  # 0-100
    recyclability: float = 30.0  # %
    carbon_footprint: float = 3.5  # kg CO2/kg

    @property
    def total_cost(self) -> float:
        """Total cost per kg."""
        return self.raw_material_cost + self.processing_cost


@dataclass
class Material:
    """Complete material specification for tire simulation.

    This is the main material class that aggregates all properties,
    behaviors, and simulation parameters for a tire material.

    Attributes:
        material_id: Unique identifier
        name: Human-readable name
        category: Material category
        description: Detailed description
        molecular: Molecular specification
        hamiltonian: Quantum Hamiltonian parameters
        properties: Dictionary of material properties
        mechanical: Mechanical behavior model
        thermal: Thermal behavior model
        wear: Wear and fatigue model
        environmental: Environmental reactivity model
        economic: Economic factors
    """

    material_id: str
    name: str
    category: MaterialCategory
    description: str = ""

    # Specifications
    molecular: MolecularSpecification = field(default_factory=MolecularSpecification)
    hamiltonian: HamiltonianParameters = field(default_factory=HamiltonianParameters)

    # Properties
    properties: dict[PropertyType, MaterialProperty] = field(default_factory=dict)

    # Behavior models
    mechanical: MechanicalBehavior = field(default_factory=MechanicalBehavior)
    thermal: ThermalBehavior = field(default_factory=ThermalBehavior)
    wear: WearModel = field(default_factory=WearModel)
    environmental: EnvironmentalReactivity = field(default_factory=EnvironmentalReactivity)

    # Economic factors
    economic: EconomicFactors = field(default_factory=EconomicFactors)

    # Metadata
    version: str = "1.0.0"
    created_by: str = "quantum_simulation"
    validated: bool = False
    patents: list[str] = field(default_factory=list)

    def get_property(self, prop_type: PropertyType) -> float | None:
        """Get value of a specific property.

        Args:
            prop_type: Property type to retrieve

        Returns:
            Property value or None if not found
        """
        prop = self.properties.get(prop_type)
        return prop.value if prop else None

    def set_property(
        self,
        prop_type: PropertyType,
        value: float,
        unit: str,
        **kwargs: Any,
    ) -> None:
        """Set a material property.

        Args:
            prop_type: Property type
            value: Property value
            unit: Unit of measurement
            **kwargs: Additional property parameters
        """
        self.properties[prop_type] = MaterialProperty(
            property_type=prop_type,
            value=value,
            unit=unit,
            **kwargs,
        )

    def get_quantum_simulation_config(self) -> dict[str, Any]:
        """Get configuration for quantum simulation.

        Returns:
            Dictionary with quantum simulation parameters
        """
        return {
            "num_qubits": self.hamiltonian.num_qubits_required,
            "num_electrons": self.hamiltonian.num_electrons,
            "num_orbitals": self.hamiltonian.num_orbitals,
            "basis_set": self.hamiltonian.basis_set,
            "spin_multiplicity": self.hamiltonian.spin_multiplicity,
            "nuclear_repulsion": self.hamiltonian.nuclear_repulsion,
        }

    def predict_tire_performance(
        self,
        tire_type: str = "passenger",
    ) -> dict[str, float]:
        """Predict tire performance for a given application.

        Args:
            tire_type: Type of tire application

        Returns:
            Dictionary of performance scores (0-100)
        """
        scores = {}

        # Grip score based on glass transition and hysteresis
        Tg = self.thermal.glass_transition
        scores["wet_grip"] = min(100, max(0, 100 - 0.5 * (Tg - 213)))

        # Wear resistance
        wear_index = self.wear.abrasion_index
        scores["wear_resistance"] = min(100, wear_index)

        # Rolling resistance (inversely related to hysteresis)
        G = self.mechanical.get_shear_modulus()
        scores["rolling_resistance"] = min(100, 50 + 10 * G)

        # Temperature resistance
        scores["heat_resistance"] = min(100, 100 * (self.thermal.degradation_temp - 373) / 200)

        # Overall score
        weights = {
            "passenger": {
                "wet_grip": 0.3,
                "wear_resistance": 0.3,
                "rolling_resistance": 0.3,
                "heat_resistance": 0.1,
            },
            "performance": {
                "wet_grip": 0.4,
                "wear_resistance": 0.2,
                "rolling_resistance": 0.2,
                "heat_resistance": 0.2,
            },
            "racing": {
                "wet_grip": 0.5,
                "wear_resistance": 0.1,
                "rolling_resistance": 0.1,
                "heat_resistance": 0.3,
            },
            "off_road": {
                "wet_grip": 0.2,
                "wear_resistance": 0.4,
                "rolling_resistance": 0.2,
                "heat_resistance": 0.2,
            },
        }

        w = weights.get(tire_type, weights["passenger"])
        scores["overall"] = sum(scores[k] * w.get(k, 0) for k in scores if k != "overall")

        return scores

    def to_dict(self) -> dict[str, Any]:
        """Serialize material to dictionary.

        Returns:
            Complete material specification as dictionary
        """
        return {
            "material_id": self.material_id,
            "name": self.name,
            "category": self.category.name,
            "description": self.description,
            "version": self.version,
            "created_by": self.created_by,
            "validated": self.validated,
            "patents": self.patents,
            "properties": {k.name: v.to_dict() for k, v in self.properties.items()},
            # Add other serializable fields as needed
        }

    def __repr__(self) -> str:
        return f"Material(id={self.material_id}, name={self.name}, category={self.category.name})"


# Factory functions


def create_material(
    material_id: str,
    name: str,
    category: MaterialCategory,
    base_properties: dict[str, float] | None = None,
) -> Material:
    """Factory function to create a material with default properties.

    Args:
        material_id: Unique identifier
        name: Material name
        category: Material category
        base_properties: Optional base properties

    Returns:
        Configured Material instance
    """
    material = Material(
        material_id=material_id,
        name=name,
        category=category,
    )

    # Set default properties if provided
    if base_properties:
        if "tensile_strength" in base_properties:
            material.set_property(
                PropertyType.TENSILE_STRENGTH,
                base_properties["tensile_strength"],
                "MPa",
            )
        if "elongation" in base_properties:
            material.set_property(
                PropertyType.ELONGATION,
                base_properties["elongation"],
                "%",
            )

    return material
