"""Self-Healing Polymer Materials Database.

Contains 15 self-healing polymer systems including:
- Microcapsule-based healing
- Vascular network healing
- Intrinsic self-healing polymers
- Nanobot-driven repair systems
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

from goodyear_quantum_pilot.materials.base import (
    EconomicFactors,
    Material,
    MaterialCategory,
    MechanicalBehavior,
    MolecularSpecification,
    PropertyType,
    ThermalBehavior,
    WearModel,
)


class HealingMechanism(Enum):
    """Types of self-healing mechanisms."""

    MICROCAPSULE = auto()  # Embedded healing agent capsules
    VASCULAR = auto()  # Vascular network delivery
    INTRINSIC = auto()  # Inherent reversible bonds
    NANOBOT = auto()  # Nanobot-driven repair
    SHAPE_MEMORY = auto()  # Shape memory assisted
    PHOTOTRIGGERED = auto()  # Light-activated healing


@dataclass
class HealingParameters:
    """Parameters for self-healing behavior.

    Attributes:
        mechanism: Type of healing mechanism
        healing_efficiency: Maximum healing efficiency (0-1)
        healing_time: Time to achieve 90% healing (hours)
        healing_cycles: Maximum number of healing cycles
        trigger_threshold: Damage threshold to trigger healing
        temperature_range: Operating temperature range (K)
        autonomous: Whether healing is autonomous
    """

    mechanism: HealingMechanism = HealingMechanism.INTRINSIC
    healing_efficiency: float = 0.85  # 0-1
    healing_time: float = 24.0  # hours
    healing_cycles: int = 10
    trigger_threshold: float = 0.1  # Strain at trigger
    temperature_range: tuple[float, float] = (250.0, 373.0)  # K
    autonomous: bool = True

    def predict_healed_strength(
        self,
        original_strength: float,
        damage_fraction: float,
        cycle_number: int = 1,
    ) -> float:
        """Predict strength after healing.

        Args:
            original_strength: Original material strength
            damage_fraction: Fraction of material damaged
            cycle_number: Current healing cycle

        Returns:
            Healed strength
        """
        # Efficiency decreases with cycles
        cycle_factor = 0.95 ** (cycle_number - 1)
        effective_efficiency = self.healing_efficiency * cycle_factor

        # Healed portion
        healed_fraction = damage_fraction * effective_efficiency
        remaining_damage = damage_fraction - healed_fraction

        return original_strength * (1 - remaining_damage)


class SelfHealingPolymer(Material):
    """Self-healing polymer tire material.

    Materials that can autonomously repair damage to extend
    tire lifetime and improve safety.
    """

    healing: HealingParameters = field(default_factory=HealingParameters)

    def __init__(
        self,
        material_id: str,
        name: str,
        healing_params: HealingParameters | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize self-healing polymer."""
        super().__init__(
            material_id=material_id,
            name=name,
            category=MaterialCategory.SELF_HEALING,
            **kwargs,
        )
        self.healing = healing_params or HealingParameters()
        self.patents = ["Patent #96", "Patent #97"]

    def simulate_damage_healing_cycle(
        self,
        initial_strength: float,
        damage_events: list[float],
    ) -> list[float]:
        """Simulate multiple damage-healing cycles.

        Args:
            initial_strength: Initial material strength (MPa)
            damage_events: List of damage fractions for each event

        Returns:
            List of strengths after each healing event
        """
        strengths = [initial_strength]
        current_strength = initial_strength

        for cycle, damage in enumerate(damage_events, 1):
            if cycle > self.healing.healing_cycles:
                # Healing exhausted
                current_strength *= 1 - damage
            else:
                current_strength = self.healing.predict_healed_strength(
                    current_strength, damage, cycle
                )
            strengths.append(current_strength)

        return strengths

    def get_effective_lifetime_extension(self) -> float:
        """Calculate effective lifetime extension factor.

        Returns:
            Multiplicative lifetime extension (e.g., 2.0 = 2x lifetime)
        """
        # Based on healing efficiency and cycles
        avg_efficiency = self.healing.healing_efficiency * 0.95 ** (self.healing.healing_cycles / 2)

        # Each cycle extends life by efficiency fraction
        extension = 1 + sum(avg_efficiency * (0.95**i) for i in range(self.healing.healing_cycles))

        return extension


def create_microcapsule_healing_material(
    variant: str,
    capsule_size: float,
    capsule_loading: float,
    healing_agent: str,
) -> SelfHealingPolymer:
    """Create microcapsule-based self-healing material.

    Args:
        variant: Material variant ID
        capsule_size: Average capsule diameter (μm)
        capsule_loading: Capsule loading (vol%)
        healing_agent: Type of healing agent

    Returns:
        Microcapsule healing material
    """
    material = SelfHealingPolymer(
        material_id=f"MCH-{variant}",
        name=f"Microcapsule Healing {variant}",
        description=(
            f"Self-healing elastomer with embedded microcapsules. "
            f"Capsule size: {capsule_size:.0f} μm, Loading: {capsule_loading:.1f} vol%. "
            f"Healing agent: {healing_agent}."
        ),
        healing_params=HealingParameters(
            mechanism=HealingMechanism.MICROCAPSULE,
            healing_efficiency=0.80,
            healing_time=48.0,  # Slower for microcapsule
            healing_cycles=1,  # Single-use capsules
            autonomous=True,
        ),
    )

    material.molecular = MolecularSpecification(
        formula="SBR + capsules",
        crosslink_density=100.0,
    )

    # Mechanical - capsules slightly reduce properties
    capsule_factor = 1 - 0.02 * capsule_loading

    material.mechanical = MechanicalBehavior(
        c10=0.4 * capsule_factor,
        c01=0.1 * capsule_factor,
        viscosity=1200,
    )

    material.thermal = ThermalBehavior(
        glass_transition=218.0,
        degradation_temp=513.0,
    )

    material.wear = WearModel(
        abrasion_index=90.0 * capsule_factor,
        fatigue_exponent=2.8,  # Capsules can initiate cracks
    )

    material.economic = EconomicFactors(
        raw_material_cost=5.00 + 0.5 * capsule_loading,
        processing_cost=2.50,
        manufacturability_score=70.0,
    )

    # Properties
    material.set_property(PropertyType.TENSILE_STRENGTH, 18.0 * capsule_factor, "MPa")
    material.set_property(PropertyType.HEALING_EFFICIENCY, 0.80, "fraction")
    material.set_property(PropertyType.HEALING_TIME, 48.0, "hours")
    material.set_property(PropertyType.HEALING_CYCLES, 1, "cycles")

    material.patents = ["Patent #96"]

    return material


def create_vascular_healing_material(
    variant: str,
    channel_diameter: float,
    channel_density: float,
) -> SelfHealingPolymer:
    """Create vascular network self-healing material.

    Args:
        variant: Material variant ID
        channel_diameter: Vascular channel diameter (μm)
        channel_density: Channel density (channels/mm²)

    Returns:
        Vascular healing material
    """
    material = SelfHealingPolymer(
        material_id=f"VNH-{variant}",
        name=f"Vascular Network Healing {variant}",
        description=(
            f"Self-healing elastomer with vascular network. "
            f"Channel diameter: {channel_diameter:.0f} μm, "
            f"Density: {channel_density:.1f} channels/mm²."
        ),
        healing_params=HealingParameters(
            mechanism=HealingMechanism.VASCULAR,
            healing_efficiency=0.90,
            healing_time=24.0,
            healing_cycles=50,  # Continuous supply
            autonomous=True,
        ),
    )

    # Channels affect mechanical properties
    void_fraction = np.pi * (channel_diameter / 1000) ** 2 * channel_density / 4
    void_factor = 1 - void_fraction

    material.mechanical = MechanicalBehavior(
        c10=0.38 * void_factor,
        c01=0.09 * void_factor,
    )

    material.wear = WearModel(
        abrasion_index=85.0 * void_factor,
        fatigue_exponent=2.5,
    )

    material.economic = EconomicFactors(
        raw_material_cost=8.00,
        processing_cost=5.00,  # Complex manufacturing
        manufacturability_score=50.0,
    )

    material.set_property(PropertyType.TENSILE_STRENGTH, 16.0 * void_factor, "MPa")
    material.set_property(PropertyType.HEALING_EFFICIENCY, 0.90, "fraction")
    material.set_property(PropertyType.HEALING_TIME, 24.0, "hours")
    material.set_property(PropertyType.HEALING_CYCLES, 50, "cycles")

    material.patents = ["Patent #97"]

    return material


def create_intrinsic_healing_material(
    variant: str,
    bond_type: str,
    exchange_rate: float,
) -> SelfHealingPolymer:
    """Create intrinsic self-healing material.

    Args:
        variant: Material variant ID
        bond_type: Type of reversible bond
        exchange_rate: Bond exchange rate (1/s)

    Returns:
        Intrinsic healing material
    """
    # Healing time inversely related to exchange rate
    healing_time = 10.0 / exchange_rate  # hours

    material = SelfHealingPolymer(
        material_id=f"ISH-{variant}",
        name=f"Intrinsic Self-Healing {variant}",
        description=(
            f"Self-healing elastomer with reversible {bond_type} bonds. "
            f"Bond exchange rate: {exchange_rate:.2f} 1/s."
        ),
        healing_params=HealingParameters(
            mechanism=HealingMechanism.INTRINSIC,
            healing_efficiency=0.95,  # Higher for intrinsic
            healing_time=healing_time,
            healing_cycles=100,  # Many cycles possible
            autonomous=True,
        ),
    )

    material.molecular = MolecularSpecification(
        formula="Polymer-[Reversible-Bond]",
        crosslink_density=120.0,
        vulcanization_system="reversible_covalent",
    )

    material.mechanical = MechanicalBehavior(
        c10=0.42,
        c01=0.11,
        viscosity=1800,  # Higher viscosity from dynamic bonds
        relaxation_time=1.0 / exchange_rate,
    )

    material.thermal = ThermalBehavior(
        glass_transition=215.0,
        degradation_temp=493.0,  # Lower due to dynamic bonds
    )

    material.wear = WearModel(
        abrasion_index=110.0,
        fatigue_exponent=2.0,  # Excellent fatigue from healing
        endurance_limit=80.0,
    )

    material.economic = EconomicFactors(
        raw_material_cost=7.50,
        processing_cost=3.00,
        manufacturability_score=65.0,
    )

    material.set_property(PropertyType.TENSILE_STRENGTH, 20.0, "MPa")
    material.set_property(PropertyType.HEALING_EFFICIENCY, 0.95, "fraction")
    material.set_property(PropertyType.HEALING_TIME, healing_time, "hours")
    material.set_property(PropertyType.HEALING_CYCLES, 100, "cycles")

    material.patents = ["Patent #98"]

    return material


def create_nanobot_healing_material(
    variant: str,
    nanobot_density: float,
    repair_rate: float,
) -> SelfHealingPolymer:
    """Create nanobot-driven self-healing material.

    Args:
        variant: Material variant ID
        nanobot_density: Nanobot density (units/mm³)
        repair_rate: Repair rate per nanobot (nm³/s)

    Returns:
        Nanobot healing material
    """
    # Calculate healing time based on nanobot parameters
    # Assume 1 mm³ damage = 10^9 nm³
    damage_volume = 1e9  # nm³
    total_repair_rate = nanobot_density * repair_rate  # nm³/s
    healing_time = damage_volume / total_repair_rate / 3600  # hours

    material = SelfHealingPolymer(
        material_id=f"NBH-{variant}",
        name=f"Nanobot Healing {variant}",
        description=(
            f"Self-healing elastomer with embedded nanobots. "
            f"Nanobot density: {nanobot_density:.0f} units/mm³, "
            f"Repair rate: {repair_rate:.0f} nm³/s per bot."
        ),
        healing_params=HealingParameters(
            mechanism=HealingMechanism.NANOBOT,
            healing_efficiency=0.99,  # Near-perfect
            healing_time=healing_time,
            healing_cycles=1000,  # Reusable nanobots
            autonomous=True,
        ),
    )

    material.molecular = MolecularSpecification(
        formula="Elastomer + Nanobots",
        crosslink_density=130.0,
    )

    material.mechanical = MechanicalBehavior(
        c10=0.48,
        c01=0.12,
        viscosity=1100,
    )

    material.thermal = ThermalBehavior(
        glass_transition=212.0,
        degradation_temp=543.0,
    )

    # Excellent wear due to continuous repair
    material.wear = WearModel(
        abrasion_index=200.0,  # Effective wear near zero
        wear_rate=1e-10,
        fatigue_exponent=1.5,
        endurance_limit=150.0,
    )

    material.economic = EconomicFactors(
        raw_material_cost=50.00,  # Expensive nanobot tech
        processing_cost=20.00,
        manufacturability_score=30.0,  # Very difficult
    )

    material.set_property(PropertyType.TENSILE_STRENGTH, 25.0, "MPa")
    material.set_property(PropertyType.HEALING_EFFICIENCY, 0.99, "fraction")
    material.set_property(PropertyType.HEALING_TIME, healing_time, "hours")
    material.set_property(PropertyType.HEALING_CYCLES, 1000, "cycles")

    material.patents = ["Patent #99", "Patent #100"]

    return material


def create_shape_memory_healing_material(
    variant: str,
    activation_temp: float,
    recovery_ratio: float,
) -> SelfHealingPolymer:
    """Create shape memory assisted healing material.

    Args:
        variant: Material variant ID
        activation_temp: Shape recovery temperature (K)
        recovery_ratio: Shape recovery ratio (0-1)

    Returns:
        Shape memory healing material
    """
    material = SelfHealingPolymer(
        material_id=f"SMH-{variant}",
        name=f"Shape Memory Healing {variant}",
        description=(
            f"Self-healing elastomer with shape memory effect. "
            f"Activation temperature: {activation_temp - 273:.0f}°C, "
            f"Recovery ratio: {recovery_ratio:.0%}."
        ),
        healing_params=HealingParameters(
            mechanism=HealingMechanism.SHAPE_MEMORY,
            healing_efficiency=recovery_ratio * 0.9,
            healing_time=1.0,  # Fast once activated
            healing_cycles=20,
            trigger_threshold=0.05,
            temperature_range=(activation_temp - 30, activation_temp + 50),
            autonomous=False,  # Needs temperature trigger
        ),
    )

    material.thermal = ThermalBehavior(
        glass_transition=activation_temp - 30,
        degradation_temp=523.0,
    )

    material.mechanical = MechanicalBehavior(
        c10=0.50,
        c01=0.13,
    )

    material.set_property(PropertyType.TENSILE_STRENGTH, 22.0, "MPa")
    material.set_property(PropertyType.HEALING_EFFICIENCY, recovery_ratio * 0.9, "fraction")
    material.set_property(PropertyType.HEALING_TIME, 1.0, "hours")

    return material


# Build Self-Healing Catalog - 15 materials

HEALING_CATALOG: dict[str, SelfHealingPolymer] = {}

# Microcapsule Healing (3 materials)
_mch_configs = [
    ("1", 100, 5.0, "DCPD"),
    ("2", 50, 10.0, "DCPD"),
    ("3", 200, 3.0, "epoxy"),
]

for variant, size, loading, agent in _mch_configs:
    key = f"MCH-{variant}"
    HEALING_CATALOG[key] = create_microcapsule_healing_material(variant, size, loading, agent)

# Vascular Network Healing (3 materials)
_vnh_configs = [
    ("1", 100, 10),
    ("2", 50, 25),
    ("3", 200, 5),
]

for variant, diameter, density in _vnh_configs:
    key = f"VNH-{variant}"
    HEALING_CATALOG[key] = create_vascular_healing_material(variant, diameter, density)

# Intrinsic Self-Healing (4 materials)
_ish_configs = [
    ("1", "disulfide", 0.1),
    ("2", "Diels-Alder", 0.05),
    ("3", "hydrogen_bonding", 0.5),
    ("4", "metal-ligand", 0.2),
]

for variant, bond, rate in _ish_configs:
    key = f"ISH-{variant}"
    HEALING_CATALOG[key] = create_intrinsic_healing_material(variant, bond, rate)

# Nanobot Healing (3 materials)
_nbh_configs = [
    ("1", 1000, 100),
    ("2", 5000, 50),
    ("3", 10000, 200),
]

for variant, density, rate in _nbh_configs:
    key = f"NBH-{variant}"
    HEALING_CATALOG[key] = create_nanobot_healing_material(variant, density, rate)

# Shape Memory Healing (2 materials)
_smh_configs = [
    ("1", 333, 0.95),  # 60°C activation
    ("2", 353, 0.98),  # 80°C activation
]

for variant, temp, ratio in _smh_configs:
    key = f"SMH-{variant}"
    HEALING_CATALOG[key] = create_shape_memory_healing_material(variant, temp, ratio)

# Aliases
MicrocapsuleHealing = HEALING_CATALOG["MCH-2"]
VascularHealing = HEALING_CATALOG["VNH-2"]
IntrinsicHealing = HEALING_CATALOG["ISH-3"]
