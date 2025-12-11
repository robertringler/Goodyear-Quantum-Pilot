"""Environmental Degradation Simulation.

Simulates tire degradation from environmental exposure including:
- UV radiation damage
- Ozone cracking
- Temperature effects
- Humidity and moisture
- Chemical exposure
- Biological degradation

These simulations predict long-term tire aging and help
optimize antidegradant systems.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class DegradationMechanism(Enum):
    """Environmental degradation mechanisms."""
    
    UV_OXIDATION = auto()
    OZONE_ATTACK = auto()
    THERMAL_OXIDATION = auto()
    HYDROLYSIS = auto()
    FLEXING_FATIGUE = auto()
    CHEMICAL_ATTACK = auto()
    BIOLOGICAL = auto()


class ClimateZone(Enum):
    """Geographic climate zones."""
    
    TROPICAL = auto()
    SUBTROPICAL = auto()
    TEMPERATE = auto()
    CONTINENTAL = auto()
    POLAR = auto()
    ARID = auto()


@dataclass
class EnvironmentalConditions:
    """Environmental conditions for degradation.
    
    Attributes:
        temperature: Ambient temperature (K)
        humidity: Relative humidity (0-1)
        uv_intensity: UV irradiance (W/m²)
        ozone_concentration: Ozone level (ppb)
        strain_level: Applied strain (%)
        chemical_exposure: Chemical agents present
    """
    
    temperature: float = 298.0
    humidity: float = 0.5
    uv_intensity: float = 100.0
    ozone_concentration: float = 50.0
    strain_level: float = 0.0
    chemical_exposure: dict[str, float] = field(default_factory=dict)
    
    @classmethod
    def from_climate_zone(
        cls,
        zone: ClimateZone,
        season: str = "summer",
    ) -> "EnvironmentalConditions":
        """Create conditions from climate zone.
        
        Args:
            zone: Climate zone
            season: Season (summer, winter, spring, fall)
            
        Returns:
            Environmental conditions
        """
        # Base conditions by zone
        conditions = {
            ClimateZone.TROPICAL: {
                "temperature": 303,
                "humidity": 0.85,
                "uv_intensity": 150,
                "ozone_concentration": 40,
            },
            ClimateZone.SUBTROPICAL: {
                "temperature": 298,
                "humidity": 0.7,
                "uv_intensity": 130,
                "ozone_concentration": 50,
            },
            ClimateZone.TEMPERATE: {
                "temperature": 293,
                "humidity": 0.6,
                "uv_intensity": 100,
                "ozone_concentration": 60,
            },
            ClimateZone.CONTINENTAL: {
                "temperature": 288,
                "humidity": 0.5,
                "uv_intensity": 90,
                "ozone_concentration": 70,
            },
            ClimateZone.POLAR: {
                "temperature": 263,
                "humidity": 0.4,
                "uv_intensity": 30,
                "ozone_concentration": 30,
            },
            ClimateZone.ARID: {
                "temperature": 308,
                "humidity": 0.2,
                "uv_intensity": 180,
                "ozone_concentration": 80,
            },
        }
        
        base = conditions.get(zone, conditions[ClimateZone.TEMPERATE])
        
        # Season adjustments
        if season == "winter":
            base["temperature"] -= 15
            base["uv_intensity"] *= 0.5
        elif season == "spring" or season == "fall":
            base["temperature"] -= 5
            base["uv_intensity"] *= 0.8
        
        return cls(**base)


@dataclass
class MaterialState:
    """Rubber material state.
    
    Tracks property changes from degradation.
    
    Attributes:
        hardness: Shore A hardness
        tensile_strength: Tensile strength (MPa)
        elongation: Elongation at break (%)
        modulus: 100% modulus (MPa)
        tear_strength: Tear strength (kN/m)
        surface_condition: Surface quality (0-1)
        crosslink_density: Crosslink density (mol/m³)
    """
    
    hardness: float = 65.0
    tensile_strength: float = 20.0
    elongation: float = 500.0
    modulus: float = 2.0
    tear_strength: float = 50.0
    surface_condition: float = 1.0
    crosslink_density: float = 100.0
    
    def retention(self, original: "MaterialState") -> dict[str, float]:
        """Compute property retention percentages.
        
        Args:
            original: Original material state
            
        Returns:
            Property retention percentages
        """
        return {
            "hardness": (self.hardness / original.hardness) * 100,
            "tensile_strength": (self.tensile_strength / original.tensile_strength) * 100,
            "elongation": (self.elongation / original.elongation) * 100,
            "modulus": (self.modulus / original.modulus) * 100,
            "tear_strength": (self.tear_strength / original.tear_strength) * 100,
        }
    
    @property
    def is_serviceable(self) -> bool:
        """Check if material is still serviceable."""
        return (
            self.tensile_strength > 10.0
            and self.elongation > 200
            and self.surface_condition > 0.5
        )


class UVDegradation:
    """UV radiation degradation model.
    
    UV radiation causes:
    - Chain scission in polymer backbone
    - Surface chalking and cracking
    - Color fading
    - Loss of antioxidants
    
    The model uses Beer-Lambert law for depth profile
    and first-order kinetics for degradation.
    """
    
    # UV absorption coefficients
    ABSORPTION_COEFFICIENT = 100.0  # 1/m
    
    # Quantum yield for chain scission
    QUANTUM_YIELD = 1e-4
    
    def __init__(
        self,
        antioxidant_concentration: float = 1.0,  # phr
        carbon_black_loading: float = 30.0,  # phr
    ) -> None:
        """Initialize UV degradation model.
        
        Args:
            antioxidant_concentration: Antioxidant level (phr)
            carbon_black_loading: Carbon black loading (phr)
        """
        self.antioxidant = antioxidant_concentration
        self.carbon_black = carbon_black_loading
        
        # UV protection factors
        self.protection_factor = (
            1.0
            + 0.5 * self.antioxidant
            + 0.1 * self.carbon_black
        )
        
        logger.info(f"UV degradation model: protection factor = {self.protection_factor:.2f}")
    
    def compute_uv_penetration(
        self,
        depth: NDArray[np.float64],  # m
        uv_intensity: float,  # W/m²
    ) -> NDArray[np.float64]:
        """Compute UV intensity at depth.
        
        Beer-Lambert law:
        I(z) = I₀ * exp(-α * z)
        
        Args:
            depth: Depth array (m)
            uv_intensity: Surface UV intensity (W/m²)
            
        Returns:
            UV intensity at depth (W/m²)
        """
        alpha = self.ABSORPTION_COEFFICIENT * (1 + 0.05 * self.carbon_black)
        return uv_intensity * np.exp(-alpha * depth)
    
    def compute_degradation_rate(
        self,
        uv_intensity: float,  # W/m²
        temperature: float,  # K
    ) -> float:
        """Compute UV degradation rate.
        
        Args:
            uv_intensity: UV intensity (W/m²)
            temperature: Temperature (K)
            
        Returns:
            Degradation rate (1/s)
        """
        # Photon energy (average UV: 350 nm)
        h = 6.626e-34  # Planck's constant
        c = 3e8  # Speed of light
        lambda_uv = 350e-9  # 350 nm
        E_photon = h * c / lambda_uv
        
        # Photon flux
        photon_flux = uv_intensity / E_photon
        
        # Base degradation rate
        rate = self.QUANTUM_YIELD * photon_flux / self.protection_factor
        
        # Temperature acceleration (Arrhenius-like)
        E_a = 30000  # J/mol
        R = 8.314
        rate *= np.exp(-E_a / (R * temperature) + E_a / (R * 298))
        
        return rate
    
    def compute_property_loss(
        self,
        state: MaterialState,
        uv_intensity: float,
        temperature: float,
        duration: float,  # hours
    ) -> MaterialState:
        """Compute property changes from UV exposure.
        
        Args:
            state: Current material state
            uv_intensity: UV intensity (W/m²)
            temperature: Temperature (K)
            duration: Exposure duration (hours)
            
        Returns:
            Updated material state
        """
        rate = self.compute_degradation_rate(uv_intensity, temperature)
        time_s = duration * 3600
        
        # Degradation factor
        D = 1 - np.exp(-rate * time_s)
        
        # Property changes (surface-dominated)
        new_state = MaterialState(
            hardness=state.hardness + 5 * D,  # Hardening
            tensile_strength=state.tensile_strength * (1 - 0.3 * D),
            elongation=state.elongation * (1 - 0.4 * D),
            modulus=state.modulus * (1 + 0.2 * D),
            tear_strength=state.tear_strength * (1 - 0.35 * D),
            surface_condition=state.surface_condition * (1 - 0.5 * D),
            crosslink_density=state.crosslink_density * (1 - 0.2 * D),
        )
        
        return new_state


class OzoneDegradation:
    """Ozone cracking model.
    
    Ozone attacks double bonds in rubber, causing:
    - Surface cracks perpendicular to strain
    - Crack growth under static strain
    - Accelerated failure under dynamic conditions
    
    Critical for sidewall durability.
    """
    
    # Critical strain threshold (below this, no cracking)
    CRITICAL_STRAIN = 0.05  # 5%
    
    # Crack growth rate constant
    CRACK_RATE_CONSTANT = 1e-8  # m/(ppb·s) at critical strain
    
    def __init__(
        self,
        antiozonant_concentration: float = 2.0,  # phr
    ) -> None:
        """Initialize ozone degradation model.
        
        Args:
            antiozonant_concentration: Antiozonant level (phr)
        """
        self.antiozonant = antiozonant_concentration
        
        # Protection factor
        self.protection_factor = 1.0 + 0.8 * self.antiozonant
    
    def compute_crack_initiation_time(
        self,
        ozone_concentration: float,  # ppb
        strain: float,  # fraction
        temperature: float,  # K
    ) -> float:
        """Compute time to crack initiation.
        
        Args:
            ozone_concentration: Ozone level (ppb)
            strain: Applied strain (fraction)
            temperature: Temperature (K)
            
        Returns:
            Time to initiation (hours)
        """
        if strain < self.CRITICAL_STRAIN:
            return float("inf")
        
        # Base initiation time
        t_base = 1000 / (ozone_concentration * (strain - self.CRITICAL_STRAIN))
        
        # Protection
        t_protected = t_base * self.protection_factor
        
        # Temperature effect
        E_a = 25000  # J/mol
        R = 8.314
        t_protected *= np.exp(E_a / R * (1 / temperature - 1 / 298))
        
        return t_protected  # hours
    
    def compute_crack_growth_rate(
        self,
        ozone_concentration: float,  # ppb
        strain: float,  # fraction
        temperature: float,  # K
    ) -> float:
        """Compute ozone crack growth rate.
        
        Args:
            ozone_concentration: Ozone level (ppb)
            strain: Applied strain (fraction)
            temperature: Temperature (K)
            
        Returns:
            Crack growth rate (m/s)
        """
        if strain < self.CRITICAL_STRAIN:
            return 0.0
        
        # Strain factor
        strain_factor = (strain - self.CRITICAL_STRAIN) ** 0.5
        
        # Base rate
        rate = self.CRACK_RATE_CONSTANT * ozone_concentration * strain_factor
        
        # Protection
        rate /= self.protection_factor
        
        # Temperature activation
        E_a = 30000  # J/mol
        R = 8.314
        rate *= np.exp(-E_a / (R * temperature) + E_a / (R * 298))
        
        return rate
    
    def compute_crack_depth(
        self,
        ozone_concentration: float,
        strain: float,
        temperature: float,
        duration: float,  # hours
    ) -> float:
        """Compute ozone crack depth.
        
        Args:
            ozone_concentration: Ozone level (ppb)
            strain: Applied strain (fraction)
            temperature: Temperature (K)
            duration: Exposure duration (hours)
            
        Returns:
            Crack depth (mm)
        """
        t_init = self.compute_crack_initiation_time(ozone_concentration, strain, temperature)
        
        if duration < t_init:
            return 0.0
        
        growth_time = (duration - t_init) * 3600  # seconds
        rate = self.compute_crack_growth_rate(ozone_concentration, strain, temperature)
        
        depth = rate * growth_time * 1000  # mm
        
        return depth
    
    def assess_crack_severity(
        self,
        crack_depth: float,  # mm
        wall_thickness: float,  # mm
    ) -> str:
        """Assess crack severity.
        
        Args:
            crack_depth: Crack depth (mm)
            wall_thickness: Component thickness (mm)
            
        Returns:
            Severity rating
        """
        ratio = crack_depth / wall_thickness
        
        if ratio > 0.5:
            return "CRITICAL - Failure imminent"
        elif ratio > 0.25:
            return "SEVERE - Service life compromised"
        elif ratio > 0.1:
            return "MODERATE - Monitor closely"
        elif ratio > 0.01:
            return "MINOR - Normal aging"
        else:
            return "NONE - No visible cracking"


class ThermalOxidation:
    """Thermal oxidation model.
    
    Heat accelerates oxidation reactions:
    - Peroxide radical formation
    - Chain scission
    - Crosslink degradation
    - Hardening from additional crosslinking
    
    Uses Arrhenius kinetics for temperature dependence.
    """
    
    # Activation energies (J/mol)
    E_SCISSION = 80000
    E_CROSSLINK = 90000
    
    # Pre-exponential factors
    A_SCISSION = 1e10
    A_CROSSLINK = 5e9
    
    # Gas constant
    R = 8.314
    
    def __init__(
        self,
        antioxidant_concentration: float = 1.0,
    ) -> None:
        """Initialize thermal oxidation model.
        
        Args:
            antioxidant_concentration: Antioxidant level (phr)
        """
        self.antioxidant = antioxidant_concentration
        
        # Induction period (hours at 100°C)
        self.induction_period = 10 * (1 + 0.5 * self.antioxidant)
    
    def compute_oxidation_rate(
        self,
        temperature: float,  # K
        oxygen_partial_pressure: float = 21.0,  # kPa
    ) -> dict[str, float]:
        """Compute thermal oxidation rates.
        
        Args:
            temperature: Temperature (K)
            oxygen_partial_pressure: O2 pressure (kPa)
            
        Returns:
            Scission and crosslink rates (1/s)
        """
        # Oxygen effect
        O2_factor = oxygen_partial_pressure / 21.0
        
        # Scission rate
        k_scission = self.A_SCISSION * np.exp(-self.E_SCISSION / (self.R * temperature))
        k_scission *= O2_factor
        
        # Crosslink rate (competing reaction)
        k_crosslink = self.A_CROSSLINK * np.exp(-self.E_CROSSLINK / (self.R * temperature))
        
        return {
            "scission": k_scission,
            "crosslink": k_crosslink,
        }
    
    def compute_induction_time(
        self,
        temperature: float,
    ) -> float:
        """Compute oxidation induction time.
        
        Args:
            temperature: Temperature (K)
            
        Returns:
            Induction time (hours)
        """
        # Reference at 100°C (373 K)
        t_ref = self.induction_period
        T_ref = 373
        
        # Arrhenius shift
        E_a = 70000  # J/mol
        t_ind = t_ref * np.exp(E_a / self.R * (1 / temperature - 1 / T_ref))
        
        return t_ind
    
    def compute_property_changes(
        self,
        state: MaterialState,
        temperature_profile: NDArray[np.float64],
        duration: float,  # hours
    ) -> MaterialState:
        """Compute property changes from thermal aging.
        
        Args:
            state: Current material state
            temperature_profile: Temperature time series (K)
            duration: Total duration (hours)
            
        Returns:
            Updated material state
        """
        dt = duration * 3600 / len(temperature_profile)  # seconds
        
        # Integrate property changes
        delta_hardness = 0.0
        delta_tensile = 0.0
        delta_elongation = 0.0
        delta_crosslink = 0.0
        
        for T in temperature_profile:
            rates = self.compute_oxidation_rate(T)
            
            # Hardness increases from crosslinking
            delta_hardness += 0.001 * rates["crosslink"] * dt
            
            # Tensile and elongation decrease from scission
            delta_tensile -= 0.0001 * rates["scission"] * dt
            delta_elongation -= 0.0002 * rates["scission"] * dt
            
            # Net crosslink change
            delta_crosslink += (rates["crosslink"] - rates["scission"]) * dt
        
        new_state = MaterialState(
            hardness=state.hardness * (1 + delta_hardness),
            tensile_strength=state.tensile_strength * (1 + delta_tensile),
            elongation=state.elongation * (1 + delta_elongation),
            modulus=state.modulus * (1 + 0.5 * delta_hardness),
            tear_strength=state.tear_strength * (1 + 0.7 * delta_tensile),
            surface_condition=state.surface_condition,
            crosslink_density=state.crosslink_density * (1 + 0.001 * delta_crosslink),
        )
        
        return new_state
    
    def compute_equivalent_age(
        self,
        temperature_profile: NDArray[np.float64],
        reference_temperature: float = 298.0,  # K
    ) -> float:
        """Compute equivalent age at reference temperature.
        
        Arrhenius-based time-temperature superposition.
        
        Args:
            temperature_profile: Temperature time series (K)
            reference_temperature: Reference temperature (K)
            
        Returns:
            Equivalent age (hours)
        """
        E_a = 80000  # J/mol
        
        shift_factors = np.exp(E_a / self.R * (1 / reference_temperature - 1 / temperature_profile))
        
        # Assume uniform time spacing
        equivalent_age = np.sum(shift_factors) / len(temperature_profile)
        
        return equivalent_age


class HumidityDegradation:
    """Humidity effects on rubber properties.
    
    Moisture causes:
    - Hydrolysis of ester groups
    - Swelling
    - Plasticization
    - Steel belt corrosion
    - Reduced adhesion
    """
    
    # Diffusion coefficient (m²/s)
    D_WATER = 1e-11
    
    def __init__(
        self,
        rubber_type: str = "NR",
    ) -> None:
        """Initialize humidity model.
        
        Args:
            rubber_type: Rubber type (NR, SBR, EPDM, etc.)
        """
        self.rubber_type = rubber_type
        
        # Equilibrium moisture content at 100% RH (%)
        moisture_eq = {
            "NR": 1.5,
            "SBR": 1.0,
            "BR": 0.5,
            "EPDM": 0.3,
            "CR": 2.0,
        }
        self.M_eq = moisture_eq.get(rubber_type, 1.0)
    
    def compute_moisture_uptake(
        self,
        humidity: float,  # 0-1
        temperature: float,  # K
        thickness: float,  # m
        duration: float,  # hours
    ) -> float:
        """Compute moisture uptake.
        
        Fickian diffusion model.
        
        Args:
            humidity: Relative humidity (0-1)
            temperature: Temperature (K)
            thickness: Rubber thickness (m)
            duration: Exposure duration (hours)
            
        Returns:
            Moisture content (%)
        """
        t = duration * 3600
        L = thickness
        
        # Temperature-dependent diffusivity
        E_a = 40000  # J/mol
        D = self.D_WATER * np.exp(-E_a / (8.314 * 298) + E_a / (8.314 * temperature))
        
        # Characteristic time
        tau = L ** 2 / (np.pi ** 2 * D)
        
        # Moisture uptake
        M_eq = self.M_eq * humidity
        M = M_eq * (1 - np.exp(-t / tau))
        
        return M
    
    def compute_swelling(
        self,
        moisture_content: float,  # %
    ) -> float:
        """Compute volume swelling from moisture.
        
        Args:
            moisture_content: Moisture content (%)
            
        Returns:
            Volume swelling (%)
        """
        # Linear relation: ~0.5% swelling per 1% moisture
        return 0.5 * moisture_content
    
    def compute_property_effects(
        self,
        state: MaterialState,
        moisture_content: float,
    ) -> MaterialState:
        """Compute property changes from moisture.
        
        Args:
            state: Current material state
            moisture_content: Moisture content (%)
            
        Returns:
            Updated material state
        """
        # Plasticization reduces hardness and modulus
        softening_factor = 1 - 0.02 * moisture_content
        
        new_state = MaterialState(
            hardness=state.hardness * softening_factor,
            tensile_strength=state.tensile_strength * (1 - 0.05 * moisture_content),
            elongation=state.elongation * (1 + 0.02 * moisture_content),
            modulus=state.modulus * softening_factor,
            tear_strength=state.tear_strength * (1 - 0.03 * moisture_content),
            surface_condition=state.surface_condition,
            crosslink_density=state.crosslink_density,
        )
        
        return new_state


class ChemicalDegradation:
    """Chemical exposure effects.
    
    Various chemicals can damage rubber:
    - Fuels and oils (swelling, extraction)
    - Salt (corrosion, embrittlement)
    - Acids (hydrolysis)
    - Solvents (dissolution)
    """
    
    # Swelling factors (% per exposure hour)
    SWELLING_RATES = {
        "gasoline": 0.1,
        "diesel": 0.05,
        "motor_oil": 0.02,
        "brake_fluid": 0.15,
        "antifreeze": 0.03,
        "salt_solution": 0.01,
    }
    
    def __init__(
        self,
        rubber_type: str = "NR",
    ) -> None:
        """Initialize chemical degradation model.
        
        Args:
            rubber_type: Rubber type
        """
        self.rubber_type = rubber_type
        
        # Chemical resistance factor (higher = more resistant)
        resistance = {
            "NR": 0.5,
            "SBR": 0.6,
            "BR": 0.5,
            "NBR": 1.5,  # Nitrile rubber is oil resistant
            "EPDM": 1.2,
            "FKM": 2.0,  # Fluoroelastomer is very resistant
        }
        self.resistance = resistance.get(rubber_type, 1.0)
    
    def compute_swelling(
        self,
        chemical: str,
        exposure_hours: float,
        temperature: float = 298.0,
    ) -> float:
        """Compute swelling from chemical exposure.
        
        Args:
            chemical: Chemical type
            exposure_hours: Exposure duration (hours)
            temperature: Temperature (K)
            
        Returns:
            Volume swelling (%)
        """
        base_rate = self.SWELLING_RATES.get(chemical, 0.01)
        
        # Temperature acceleration
        T_factor = np.exp(0.05 * (temperature - 298))
        
        # Resistance factor
        rate = base_rate * T_factor / self.resistance
        
        # Saturation model
        swelling_max = 50.0  # Maximum swelling (%)
        tau = 100.0  # Time constant (hours)
        
        swelling = swelling_max * (1 - np.exp(-rate * exposure_hours / swelling_max))
        
        return swelling
    
    def compute_property_changes(
        self,
        state: MaterialState,
        swelling: float,
    ) -> MaterialState:
        """Compute property changes from chemical swelling.
        
        Args:
            state: Current material state
            swelling: Volume swelling (%)
            
        Returns:
            Updated material state
        """
        # Swelling reduces most properties
        factor = 1 / (1 + 0.01 * swelling)
        
        new_state = MaterialState(
            hardness=state.hardness * factor ** 0.5,
            tensile_strength=state.tensile_strength * factor,
            elongation=state.elongation * (1 + 0.5 * swelling / 100),
            modulus=state.modulus * factor ** 1.5,
            tear_strength=state.tear_strength * factor,
            surface_condition=state.surface_condition * factor ** 0.3,
            crosslink_density=state.crosslink_density * factor,
        )
        
        return new_state


class EnvironmentSimulator:
    """Complete environmental degradation simulator."""
    
    def __init__(
        self,
        antioxidant: float = 1.0,
        antiozonant: float = 2.0,
        carbon_black: float = 30.0,
        rubber_type: str = "NR",
    ) -> None:
        """Initialize environment simulator.
        
        Args:
            antioxidant: Antioxidant concentration (phr)
            antiozonant: Antiozonant concentration (phr)
            carbon_black: Carbon black loading (phr)
            rubber_type: Rubber type
        """
        self.uv = UVDegradation(antioxidant, carbon_black)
        self.ozone = OzoneDegradation(antiozonant)
        self.thermal = ThermalOxidation(antioxidant)
        self.humidity = HumidityDegradation(rubber_type)
        self.chemical = ChemicalDegradation(rubber_type)
        
        # Initial material state
        self.initial_state = MaterialState()
        self.current_state = MaterialState()
        
        logger.info("Initialized EnvironmentSimulator")
    
    def simulate_exposure(
        self,
        conditions: EnvironmentalConditions,
        duration: float,  # hours
    ) -> dict[str, Any]:
        """Simulate environmental exposure.
        
        Args:
            conditions: Environmental conditions
            duration: Exposure duration (hours)
            
        Returns:
            Simulation results
        """
        results = {}
        
        # UV degradation
        uv_state = self.uv.compute_property_loss(
            self.current_state,
            conditions.uv_intensity,
            conditions.temperature,
            duration,
        )
        
        # Ozone cracking
        ozone_crack = self.ozone.compute_crack_depth(
            conditions.ozone_concentration,
            conditions.strain_level / 100,
            conditions.temperature,
            duration,
        )
        
        # Thermal oxidation
        temp_profile = np.ones(int(duration * 60)) * conditions.temperature
        thermal_state = self.thermal.compute_property_changes(
            self.current_state,
            temp_profile,
            duration,
        )
        
        # Humidity effects
        moisture = self.humidity.compute_moisture_uptake(
            conditions.humidity,
            conditions.temperature,
            0.01,  # 10mm thickness
            duration,
        )
        
        # Chemical effects
        for chem, concentration in conditions.chemical_exposure.items():
            swelling = self.chemical.compute_swelling(
                chem,
                duration * concentration,
                conditions.temperature,
            )
            results[f"swelling_{chem}"] = swelling
        
        # Combine effects (use minimum properties)
        self.current_state = MaterialState(
            hardness=min(uv_state.hardness, thermal_state.hardness),
            tensile_strength=min(uv_state.tensile_strength, thermal_state.tensile_strength),
            elongation=min(uv_state.elongation, thermal_state.elongation),
            modulus=max(uv_state.modulus, thermal_state.modulus),  # Higher = worse
            tear_strength=min(uv_state.tear_strength, thermal_state.tear_strength),
            surface_condition=min(uv_state.surface_condition, thermal_state.surface_condition),
            crosslink_density=min(uv_state.crosslink_density, thermal_state.crosslink_density),
        )
        
        # Apply moisture softening
        if moisture > 0:
            self.current_state = self.humidity.compute_property_effects(
                self.current_state,
                moisture,
            )
        
        results.update({
            "duration_hours": duration,
            "ozone_crack_depth_mm": ozone_crack,
            "moisture_content_pct": moisture,
            "material_state": self.current_state,
            "property_retention": self.current_state.retention(self.initial_state),
            "is_serviceable": self.current_state.is_serviceable,
        })
        
        return results
    
    def simulate_field_aging(
        self,
        climate_zone: ClimateZone,
        years: float,
    ) -> dict[str, Any]:
        """Simulate multi-year field aging.
        
        Args:
            climate_zone: Geographic climate zone
            years: Aging duration (years)
            
        Returns:
            Aging results
        """
        results = []
        
        seasons = ["spring", "summer", "fall", "winter"]
        hours_per_season = 365.25 * 24 / 4
        
        for year in range(int(years)):
            for season in seasons:
                conditions = EnvironmentalConditions.from_climate_zone(
                    climate_zone, season
                )
                
                result = self.simulate_exposure(conditions, hours_per_season)
                result["year"] = year + 1
                result["season"] = season
                results.append(result)
                
                if not self.current_state.is_serviceable:
                    break
            
            if not self.current_state.is_serviceable:
                break
        
        return {
            "seasonal_results": results,
            "final_state": self.current_state,
            "years_serviceable": (
                sum(r["is_serviceable"] for r in results) * 0.25
            ),
            "final_retention": self.current_state.retention(self.initial_state),
        }
    
    def predict_service_life(
        self,
        climate_zone: ClimateZone,
        min_tensile_retention: float = 50.0,  # %
        min_elongation_retention: float = 40.0,  # %
    ) -> float:
        """Predict tire service life.
        
        Args:
            climate_zone: Operating climate zone
            min_tensile_retention: Minimum tensile retention (%)
            min_elongation_retention: Minimum elongation retention (%)
            
        Returns:
            Predicted service life (years)
        """
        # Reset state
        self.current_state = MaterialState()
        
        max_years = 20
        hours_per_year = 365.25 * 24
        
        for year in range(1, max_years + 1):
            # Simulate one year with average conditions
            conditions = EnvironmentalConditions.from_climate_zone(climate_zone)
            self.simulate_exposure(conditions, hours_per_year)
            
            retention = self.current_state.retention(self.initial_state)
            
            if (
                retention["tensile_strength"] < min_tensile_retention
                or retention["elongation"] < min_elongation_retention
            ):
                return year - 0.5
        
        return float(max_years)
