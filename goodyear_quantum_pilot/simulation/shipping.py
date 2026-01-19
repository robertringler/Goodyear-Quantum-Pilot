"""Shipping and Transport Simulation.

Simulates tire behavior during shipping and transport including:
- Vibration exposure and resonance effects
- Temperature cycling during transit
- Humidity effects on rubber properties
- Mechanical shock and impact
- Storage duration effects

These simulations predict shipping damage and aging to ensure
product quality at delivery.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class TransportMode(Enum):
    """Transportation modes."""

    TRUCK = auto()
    RAIL = auto()
    SHIP = auto()
    AIR = auto()
    WAREHOUSE = auto()


class PackagingType(Enum):
    """Tire packaging types."""

    STACKED = auto()
    PALLETIZED = auto()
    RACKED = auto()
    INDIVIDUAL = auto()


@dataclass
class TransportCondition:
    """Transport environmental conditions.

    Attributes:
        mode: Transport mode
        duration: Transport duration (hours)
        temperature_profile: Temperature time series (K)
        humidity_profile: Humidity time series (0-1)
        vibration_psd: Vibration power spectral density (g²/Hz)
        max_shock: Maximum shock level (g)
        stacking_load: Vertical stacking load (N)
    """

    mode: TransportMode
    duration: float
    temperature_profile: NDArray[np.float64] | None = None
    humidity_profile: NDArray[np.float64] | None = None
    vibration_psd: NDArray[np.float64] | None = None
    max_shock: float = 0.0
    stacking_load: float = 0.0

    def __post_init__(self) -> None:
        """Generate default profiles if not provided."""
        n_points = int(self.duration * 60)  # One point per minute

        if self.temperature_profile is None:
            # Default: ambient with daily cycling
            t = np.linspace(0, self.duration, n_points)
            self.temperature_profile = 298 + 5 * np.sin(2 * np.pi * t / 24)

        if self.humidity_profile is None:
            # Default: 50% RH
            self.humidity_profile = np.ones(n_points) * 0.5

        if self.vibration_psd is None:
            # Default: transport vibration spectrum
            self._generate_vibration_psd()

    def _generate_vibration_psd(self) -> None:
        """Generate typical transport vibration spectrum."""
        # Frequency range: 1-500 Hz
        freq = np.logspace(0, np.log10(500), 100)

        # Mode-specific vibration levels
        psd_levels = {
            TransportMode.TRUCK: 0.01,
            TransportMode.RAIL: 0.005,
            TransportMode.SHIP: 0.001,
            TransportMode.AIR: 0.02,
            TransportMode.WAREHOUSE: 0.0001,
        }

        base_level = psd_levels.get(self.mode, 0.01)

        # Random road/track profile (1/f spectrum)
        self.vibration_psd = base_level * (10 / freq) ** 0.5


@dataclass
class DegradationState:
    """Tire degradation state during transport.

    Attributes:
        oxidation_level: Surface oxidation level (0-1)
        ozone_damage: Ozone cracking level (0-1)
        fatigue_damage: Cumulative fatigue damage (0-1)
        compression_set: Permanent compression (%)
        moisture_content: Rubber moisture (%)
    """

    oxidation_level: float = 0.0
    ozone_damage: float = 0.0
    fatigue_damage: float = 0.0
    compression_set: float = 0.0
    moisture_content: float = 0.5

    @property
    def total_damage(self) -> float:
        """Total degradation factor."""
        return (
            self.oxidation_level
            + self.ozone_damage
            + self.fatigue_damage
            + self.compression_set / 100
        ) / 4

    @property
    def is_acceptable(self) -> bool:
        """Check if tire is within acceptable degradation."""
        return (
            self.oxidation_level < 0.1
            and self.ozone_damage < 0.05
            and self.fatigue_damage < 0.2
            and self.compression_set < 5.0
        )


class VibrationSimulator:
    """Tire vibration during transport.

    Simulates resonance effects and cumulative fatigue damage
    from transport vibrations.

    Key phenomena:
    - Tire natural frequencies (sidewall, belt, tread)
    - Resonance amplification
    - Cumulative fatigue damage (Miner's rule)
    """

    def __init__(
        self,
        tire_mass: float = 10.0,  # kg
        sidewall_stiffness: float = 200000.0,  # N/m
        damping_ratio: float = 0.05,
    ) -> None:
        """Initialize vibration simulator.

        Args:
            tire_mass: Tire mass (kg)
            sidewall_stiffness: Sidewall stiffness (N/m)
            damping_ratio: Damping ratio (dimensionless)
        """
        self.mass = tire_mass
        self.stiffness = sidewall_stiffness
        self.damping = damping_ratio

        # Natural frequency
        self.omega_n = np.sqrt(self.stiffness / self.mass)
        self.f_n = self.omega_n / (2 * np.pi)

        logger.info(f"VibrationSimulator: natural freq = {self.f_n:.1f} Hz")

    def compute_transmissibility(
        self,
        frequencies: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute vibration transmissibility.

        Transmissibility is the ratio of output to input
        vibration amplitude.

        Args:
            frequencies: Frequency array (Hz)

        Returns:
            Transmissibility array
        """
        r = frequencies / self.f_n
        zeta = self.damping

        numerator = 1 + (2 * zeta * r) ** 2
        denominator = (1 - r**2) ** 2 + (2 * zeta * r) ** 2

        return np.sqrt(numerator / denominator)

    def compute_fatigue_damage(
        self,
        vibration_psd: NDArray[np.float64],
        frequencies: NDArray[np.float64],
        duration: float,  # hours
    ) -> float:
        """Compute cumulative fatigue damage.

        Uses Miner's linear damage accumulation rule:
        D = Σ (n_i / N_f,i)

        Args:
            vibration_psd: Vibration PSD (g²/Hz)
            frequencies: Frequency array (Hz)
            duration: Exposure duration (hours)

        Returns:
            Fatigue damage fraction (0-1)
        """
        # Transmissibility
        T = self.compute_transmissibility(frequencies)

        # RMS acceleration at tire
        df = np.diff(frequencies, prepend=frequencies[0])
        g_rms = np.sqrt(np.sum(vibration_psd * T**2 * df))

        # Convert to stress (simplified)
        sigma = g_rms * 9.81 * self.mass * 0.1  # MPa equivalent

        # S-N curve parameters for rubber
        sigma_f = 10.0  # Fatigue strength coefficient (MPa)
        b = -0.1  # Fatigue exponent

        # Cycles to failure at this stress level
        if sigma > 0:
            N_f = (sigma / sigma_f) ** (1 / b)
        else:
            return 0.0

        # Number of cycles (assume dominant frequency)
        dominant_freq = frequencies[np.argmax(vibration_psd * T**2)]
        n_cycles = dominant_freq * duration * 3600

        # Miner's damage
        damage = n_cycles / N_f

        return min(1.0, damage)

    def compute_resonance_risk(
        self,
        vibration_psd: NDArray[np.float64],
        frequencies: NDArray[np.float64],
    ) -> dict[str, float]:
        """Compute resonance risk analysis.

        Args:
            vibration_psd: Vibration PSD (g²/Hz)
            frequencies: Frequency array (Hz)

        Returns:
            Resonance risk metrics
        """
        T = self.compute_transmissibility(frequencies)

        # Find resonance region
        resonance_mask = np.abs(frequencies - self.f_n) < 5  # ±5 Hz

        # Energy in resonance region
        if np.any(resonance_mask):
            resonance_energy = np.sum(vibration_psd[resonance_mask] * T[resonance_mask] ** 2)
            total_energy = np.sum(vibration_psd * T**2)
            resonance_fraction = resonance_energy / total_energy if total_energy > 0 else 0
        else:
            resonance_fraction = 0

        # Peak amplification
        peak_amplification = np.max(T)

        return {
            "natural_frequency_hz": self.f_n,
            "peak_amplification": peak_amplification,
            "resonance_energy_fraction": resonance_fraction,
            "risk_level": "HIGH" if resonance_fraction > 0.3 else "LOW",
        }


class TemperatureSimulator:
    """Tire temperature effects during transport.

    Models temperature-dependent degradation including:
    - Arrhenius-based oxidation
    - Thermal expansion/contraction
    - Heat-induced softening
    - Thermal cycling damage
    """

    # Activation energies (J/mol)
    E_OXIDATION = 50000  # Oxidation
    E_CROSSLINK = 80000  # Crosslink degradation

    # Gas constant
    R = 8.314  # J/(mol·K)

    def __init__(
        self,
        reference_temperature: float = 298.0,  # K
    ) -> None:
        """Initialize temperature simulator.

        Args:
            reference_temperature: Reference temperature (K)
        """
        self.T_ref = reference_temperature

        # Pre-exponential factors
        self.A_oxidation = 1e6  # 1/s
        self.A_crosslink = 1e8  # 1/s

    def compute_oxidation_rate(
        self,
        temperature: float,  # K
        humidity: float = 0.5,  # 0-1
    ) -> float:
        """Compute oxidation rate at given conditions.

        Arrhenius equation:
        k = A * exp(-E_a / (R * T))

        Args:
            temperature: Temperature (K)
            humidity: Relative humidity (0-1)

        Returns:
            Oxidation rate (1/s)
        """
        # Arrhenius rate
        k = self.A_oxidation * np.exp(-self.E_OXIDATION / (self.R * temperature))

        # Humidity correction (moisture accelerates oxidation)
        humidity_factor = 1 + humidity

        return k * humidity_factor

    def compute_oxidation(
        self,
        temperature_profile: NDArray[np.float64],
        humidity_profile: NDArray[np.float64],
        duration: float,  # hours
    ) -> float:
        """Compute cumulative oxidation.

        Args:
            temperature_profile: Temperature time series (K)
            humidity_profile: Humidity time series (0-1)
            duration: Total duration (hours)

        Returns:
            Oxidation level (0-1)
        """
        dt = duration * 3600 / len(temperature_profile)

        oxidation = 0.0
        for T, H in zip(temperature_profile, humidity_profile):
            rate = self.compute_oxidation_rate(T, H)
            oxidation += rate * dt

        return min(1.0, oxidation)

    def compute_thermal_cycling_damage(
        self,
        temperature_profile: NDArray[np.float64],
    ) -> float:
        """Compute damage from thermal cycling.

        Thermal cycling causes:
        - CTE mismatch between components
        - Interfacial stress buildup
        - Delamination initiation

        Args:
            temperature_profile: Temperature time series (K)

        Returns:
            Cycling damage factor (0-1)
        """
        # Count temperature cycles
        T_mean = np.mean(temperature_profile)
        crossings = np.diff(np.sign(temperature_profile - T_mean))
        n_cycles = np.sum(crossings != 0) / 2

        # Temperature amplitude
        T_max = np.max(temperature_profile)
        T_min = np.min(temperature_profile)
        delta_T = T_max - T_min

        # Coffin-Manson-like relationship for thermal fatigue
        N_f = 1000 * (30 / max(1, delta_T)) ** 2  # Cycles to failure

        damage = n_cycles / N_f

        return min(1.0, damage)

    def compute_heat_history(
        self,
        temperature_profile: NDArray[np.float64],
        duration: float,
    ) -> dict[str, float]:
        """Compute thermal history metrics.

        Args:
            temperature_profile: Temperature time series (K)
            duration: Duration (hours)

        Returns:
            Thermal history metrics
        """
        # Time above threshold
        T_threshold = 323  # 50°C
        time_above = np.sum(temperature_profile > T_threshold) / len(temperature_profile) * duration

        # Peak temperature
        T_peak = np.max(temperature_profile)

        # Average temperature
        T_avg = np.mean(temperature_profile)

        # Degree-hours above threshold
        degree_hours = (
            np.sum(np.maximum(0, temperature_profile - T_threshold))
            / len(temperature_profile)
            * duration
        )

        return {
            "peak_temperature_k": T_peak,
            "average_temperature_k": T_avg,
            "time_above_50c_hours": time_above,
            "degree_hours_above_50c": degree_hours,
        }


class HumiditySimulator:
    """Humidity effects on tire during transport.

    Models moisture effects including:
    - Moisture absorption into rubber
    - Hydrolysis reactions
    - Plasticization effects
    - Corrosion of steel belts
    """

    # Diffusion coefficients (m²/s)
    D_MOISTURE = 1e-11

    def __init__(
        self,
        rubber_thickness: float = 5.0,  # mm
    ) -> None:
        """Initialize humidity simulator.

        Args:
            rubber_thickness: Rubber thickness (mm)
        """
        self.thickness = rubber_thickness * 1e-3  # Convert to m

    def compute_moisture_absorption(
        self,
        humidity_profile: NDArray[np.float64],
        duration: float,  # hours
    ) -> float:
        """Compute moisture absorption.

        Fickian diffusion model:
        M/M∞ = 1 - (8/π²) * Σ exp(-(2n+1)²π²Dt/L²)

        Args:
            humidity_profile: Humidity time series (0-1)
            duration: Duration (hours)

        Returns:
            Moisture content (%)
        """
        t = duration * 3600  # Convert to seconds
        L = self.thickness
        D = self.D_MOISTURE

        # Simplified Fickian solution
        tau = L**2 / (np.pi**2 * D)

        if t > 0:
            M_ratio = 1 - np.exp(-t / tau)
        else:
            M_ratio = 0

        # Equilibrium moisture content depends on RH
        avg_humidity = np.mean(humidity_profile)
        M_equilibrium = 2.0 * avg_humidity  # 2% at 100% RH

        return M_ratio * M_equilibrium

    def compute_hydrolysis_damage(
        self,
        humidity_profile: NDArray[np.float64],
        temperature_profile: NDArray[np.float64],
        duration: float,
    ) -> float:
        """Compute hydrolysis damage.

        Water hydrolyzes ester linkages in some rubbers.

        Args:
            humidity_profile: Humidity time series
            temperature_profile: Temperature time series (K)
            duration: Duration (hours)

        Returns:
            Hydrolysis damage (0-1)
        """
        # Hydrolysis rate
        E_a = 60000  # J/mol
        R = 8.314
        A = 1e4  # Pre-exponential

        damage = 0.0
        dt = duration * 3600 / len(humidity_profile)

        for H, T in zip(humidity_profile, temperature_profile):
            if H > 0.5:  # Only significant at high humidity
                rate = A * H * np.exp(-E_a / (R * T))
                damage += rate * dt

        return min(1.0, damage)

    def compute_belt_corrosion_risk(
        self,
        humidity_profile: NDArray[np.float64],
        temperature_profile: NDArray[np.float64],
        duration: float,
    ) -> dict[str, Any]:
        """Assess steel belt corrosion risk.

        Args:
            humidity_profile: Humidity time series
            temperature_profile: Temperature time series (K)
            duration: Duration (hours)

        Returns:
            Corrosion risk assessment
        """
        # Time at corrosive conditions (high humidity + temperature)
        high_humidity = humidity_profile > 0.8
        high_temp = temperature_profile > 303  # 30°C

        corrosive_time = np.sum(high_humidity & high_temp) / len(humidity_profile) * duration

        # Dew point crossing (condensation events)
        T_dew = temperature_profile * (humidity_profile**0.125)  # Simplified
        dew_crossings = np.sum(np.diff(np.sign(temperature_profile - T_dew)) != 0)

        # Risk level
        if corrosive_time > 48 or dew_crossings > 20:
            risk = "HIGH"
        elif corrosive_time > 12 or dew_crossings > 5:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        return {
            "corrosive_hours": corrosive_time,
            "condensation_events": dew_crossings,
            "risk_level": risk,
        }


class ShockSimulator:
    """Mechanical shock during transport.

    Models impact events from:
    - Dropping during handling
    - Road impacts (trucks)
    - Coupling shocks (rail)
    - Wave impacts (ship)
    """

    # Drop height thresholds
    MAX_SAFE_DROP = 0.5  # m
    DAMAGE_THRESHOLD_G = 50.0  # g

    def __init__(
        self,
        tire_mass: float = 10.0,  # kg
        rim_protection: bool = True,
    ) -> None:
        """Initialize shock simulator.

        Args:
            tire_mass: Tire mass (kg)
            rim_protection: Whether rim protection is present
        """
        self.mass = tire_mass
        self.rim_protection = rim_protection

    def compute_drop_impact(
        self,
        drop_height: float,  # m
    ) -> dict[str, float]:
        """Compute impact from drop.

        Args:
            drop_height: Drop height (m)

        Returns:
            Impact assessment
        """
        # Impact velocity: v = sqrt(2*g*h)
        g = 9.81
        v_impact = np.sqrt(2 * g * drop_height)

        # Deceleration distance (tire deforms ~10% of diameter)
        d_stop = 0.05  # 50mm deformation

        # Peak acceleration: a = v²/(2*d)
        a_peak = v_impact**2 / (2 * d_stop) / g  # in g's

        # Force
        F_peak = self.mass * a_peak * g

        # Damage assessment
        if a_peak > self.DAMAGE_THRESHOLD_G:
            damage = "CRITICAL"
        elif a_peak > self.DAMAGE_THRESHOLD_G * 0.5:
            damage = "MODERATE"
        else:
            damage = "MINOR"

        return {
            "impact_velocity_ms": v_impact,
            "peak_acceleration_g": a_peak,
            "peak_force_n": F_peak,
            "damage_level": damage,
        }

    def compute_cumulative_shock_damage(
        self,
        shock_events: list[float],  # g levels
    ) -> float:
        """Compute cumulative damage from shock events.

        Args:
            shock_events: List of shock levels (g)

        Returns:
            Cumulative damage (0-1)
        """
        damage = 0.0

        for shock_g in shock_events:
            if shock_g > self.DAMAGE_THRESHOLD_G * 0.3:
                # Damage increases with (g/threshold)^3
                damage += (shock_g / self.DAMAGE_THRESHOLD_G) ** 3

        return min(1.0, damage)

    def generate_transport_shocks(
        self,
        mode: TransportMode,
        duration: float,  # hours
    ) -> list[float]:
        """Generate typical shock events for transport mode.

        Args:
            mode: Transport mode
            duration: Duration (hours)

        Returns:
            List of shock levels (g)
        """
        # Shock rate (events per hour)
        shock_rates = {
            TransportMode.TRUCK: 5.0,
            TransportMode.RAIL: 2.0,
            TransportMode.SHIP: 1.0,
            TransportMode.AIR: 0.5,
            TransportMode.WAREHOUSE: 0.1,
        }

        # Typical shock levels
        shock_levels = {
            TransportMode.TRUCK: (5.0, 2.0),  # mean, std
            TransportMode.RAIL: (10.0, 5.0),
            TransportMode.SHIP: (3.0, 1.0),
            TransportMode.AIR: (4.0, 2.0),
            TransportMode.WAREHOUSE: (8.0, 4.0),  # Drops during handling
        }

        rate = shock_rates.get(mode, 1.0)
        mean, std = shock_levels.get(mode, (5.0, 2.0))

        n_events = int(np.random.poisson(rate * duration))
        shocks = np.abs(np.random.normal(mean, std, n_events)).tolist()

        return shocks


class CompressionSimulator:
    """Compression effects from stacking/storage.

    Models permanent deformation (compression set) from
    sustained loading during transport and storage.
    """

    def __init__(
        self,
        tire_stiffness: float = 300000.0,  # N/m
        sidewall_thickness: float = 10.0,  # mm
    ) -> None:
        """Initialize compression simulator.

        Args:
            tire_stiffness: Tire vertical stiffness (N/m)
            sidewall_thickness: Sidewall thickness (mm)
        """
        self.stiffness = tire_stiffness
        self.thickness = sidewall_thickness * 1e-3

    def compute_deformation(
        self,
        stacking_load: float,  # N
    ) -> float:
        """Compute deformation under stacking load.

        Args:
            stacking_load: Vertical load (N)

        Returns:
            Deformation (mm)
        """
        return stacking_load / self.stiffness * 1000  # mm

    def compute_compression_set(
        self,
        stacking_load: float,
        temperature_profile: NDArray[np.float64],
        duration: float,  # hours
    ) -> float:
        """Compute compression set (permanent deformation).

        Compression set follows:
        CS = CS∞ * (1 - exp(-t/τ))

        where τ decreases with temperature (Arrhenius).

        Args:
            stacking_load: Stacking load (N)
            temperature_profile: Temperature time series (K)
            duration: Duration (hours)

        Returns:
            Compression set (%)
        """
        # Initial deformation
        delta = self.compute_deformation(stacking_load)
        strain = delta / (self.thickness * 1000) * 100  # %

        # Ultimate compression set (25-50% of applied strain)
        CS_max = 0.35 * strain

        # Time constant (hours)
        T_avg = np.mean(temperature_profile)
        E_a = 40000  # J/mol
        R = 8.314
        tau_ref = 100  # hours at 25°C

        tau = tau_ref * np.exp(E_a / R * (1 / T_avg - 1 / 298))

        # Compression set
        CS = CS_max * (1 - np.exp(-duration / tau))

        return CS

    def compute_stacking_limit(
        self,
        tire_weight: float,  # N
        max_strain: float = 0.1,  # 10%
    ) -> int:
        """Compute maximum safe stacking height.

        Args:
            tire_weight: Single tire weight (N)
            max_strain: Maximum allowable strain

        Returns:
            Maximum stacking height (number of tires)
        """
        max_load = max_strain * self.thickness * self.stiffness / 1e-3
        max_height = int(max_load / tire_weight)

        return max_height


class ShippingSimulator:
    """Complete shipping simulation combining all models."""

    def __init__(
        self,
        tire_mass: float = 10.0,
    ) -> None:
        """Initialize shipping simulator.

        Args:
            tire_mass: Tire mass (kg)
        """
        self.tire_mass = tire_mass

        self.vibration = VibrationSimulator(tire_mass)
        self.temperature = TemperatureSimulator()
        self.humidity = HumiditySimulator()
        self.shock = ShockSimulator(tire_mass)
        self.compression = CompressionSimulator()

        self.degradation = DegradationState()

        logger.info(f"Initialized ShippingSimulator: mass={tire_mass} kg")

    def simulate_transport(
        self,
        condition: TransportCondition,
    ) -> dict[str, Any]:
        """Simulate transport segment.

        Args:
            condition: Transport conditions

        Returns:
            Transport results and updated degradation
        """
        # Vibration damage
        freq = np.logspace(0, np.log10(500), 100)
        vib_damage = self.vibration.compute_fatigue_damage(
            condition.vibration_psd,
            freq,
            condition.duration,
        )

        # Oxidation
        oxidation = self.temperature.compute_oxidation(
            condition.temperature_profile,
            condition.humidity_profile,
            condition.duration,
        )

        # Thermal cycling
        thermal_damage = self.temperature.compute_thermal_cycling_damage(
            condition.temperature_profile
        )

        # Moisture
        moisture = self.humidity.compute_moisture_absorption(
            condition.humidity_profile,
            condition.duration,
        )

        # Shocks
        shocks = self.shock.generate_transport_shocks(
            condition.mode,
            condition.duration,
        )
        shock_damage = self.shock.compute_cumulative_shock_damage(shocks)

        # Compression set
        compression_set = self.compression.compute_compression_set(
            condition.stacking_load,
            condition.temperature_profile,
            condition.duration,
        )

        # Update degradation state
        self.degradation.oxidation_level += oxidation
        self.degradation.fatigue_damage += vib_damage + shock_damage
        self.degradation.compression_set += compression_set
        self.degradation.moisture_content = moisture

        return {
            "mode": condition.mode.name,
            "duration_hours": condition.duration,
            "vibration_damage": vib_damage,
            "oxidation": oxidation,
            "thermal_cycling_damage": thermal_damage,
            "moisture_content_pct": moisture,
            "shock_events": len(shocks),
            "shock_damage": shock_damage,
            "compression_set_pct": compression_set,
            "degradation_state": self.degradation,
            "is_acceptable": self.degradation.is_acceptable,
        }

    def simulate_supply_chain(
        self,
        segments: list[TransportCondition],
    ) -> dict[str, Any]:
        """Simulate complete supply chain.

        Args:
            segments: List of transport segments

        Returns:
            Supply chain results
        """
        results = []
        total_time = 0.0

        for segment in segments:
            result = self.simulate_transport(segment)
            results.append(result)
            total_time += segment.duration

        return {
            "segments": results,
            "total_duration_hours": total_time,
            "final_degradation": self.degradation,
            "product_acceptable": self.degradation.is_acceptable,
            "total_damage": self.degradation.total_damage,
        }

    def recommend_packaging(
        self,
        route: list[TransportCondition],
    ) -> dict[str, Any]:
        """Recommend packaging based on route.

        Args:
            route: Planned transport route

        Returns:
            Packaging recommendations
        """
        recommendations = []

        # Check for high vibration modes
        high_vib_modes = [TransportMode.TRUCK, TransportMode.AIR]
        if any(seg.mode in high_vib_modes for seg in route):
            recommendations.append("Use vibration-dampening packaging")

        # Check for long durations
        total_hours = sum(seg.duration for seg in route)
        if total_hours > 168:  # 1 week
            recommendations.append("Use moisture barrier packaging")
            recommendations.append("Add desiccant packets")

        # Check for temperature extremes
        for seg in route:
            if seg.temperature_profile is not None:
                if np.max(seg.temperature_profile) > 323:  # 50°C
                    recommendations.append("Use insulated packaging or climate control")
                    break

        # Check stacking loads
        max_height = self.compression.compute_stacking_limit(self.tire_mass * 9.81)
        recommendations.append(f"Maximum stacking height: {max_height} tires")

        # Packaging type recommendation
        if total_hours > 336:  # 2 weeks
            pkg_type = PackagingType.RACKED
        elif any(seg.mode == TransportMode.SHIP for seg in route):
            pkg_type = PackagingType.PALLETIZED
        else:
            pkg_type = PackagingType.STACKED

        return {
            "recommended_packaging": pkg_type.name,
            "max_stacking_height": max_height,
            "recommendations": recommendations,
        }
