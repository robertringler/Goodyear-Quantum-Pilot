"""Catastrophic Failure Simulation.

Simulates extreme tire failure modes including:
- Blowout prediction and dynamics
- High-speed failure
- Impact damage
- Structural failure
- Run-flat capability
- Failure mode analysis

These simulations support safety engineering and
failure prevention strategies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class FailureMode(Enum):
    """Tire failure modes."""
    
    BLOWOUT = auto()           # Rapid pressure loss
    TREAD_SEPARATION = auto()   # Belt separation
    SIDEWALL_RUPTURE = auto()   # Sidewall failure
    BEAD_FAILURE = auto()       # Bead unseating
    IMPACT_DAMAGE = auto()      # Road hazard impact
    OVERLOAD = auto()           # Excessive loading
    OVERSPEED = auto()          # Speed rating exceeded
    UNDERINFLATION = auto()     # Low pressure failure
    HEAT_BUILDUP = auto()       # Thermal degradation


class SeverityLevel(Enum):
    """Failure severity levels."""
    
    MINOR = auto()      # Repairable
    MODERATE = auto()   # Requires replacement
    SEVERE = auto()     # Immediate stop required
    CRITICAL = auto()   # Safety hazard


@dataclass
class FailureEvent:
    """Tire failure event.
    
    Attributes:
        mode: Failure mode
        severity: Severity level
        location: Failure location (degrees around circumference)
        time_to_failure: Time from initiation to failure (s)
        pressure_loss_rate: Pressure loss rate (kPa/s)
        energy_release: Energy released (kJ)
        debris_risk: Debris generation risk (0-1)
    """
    
    mode: FailureMode
    severity: SeverityLevel
    location: float = 0.0
    time_to_failure: float = 0.0
    pressure_loss_rate: float = 0.0
    energy_release: float = 0.0
    debris_risk: float = 0.0
    
    @property
    def is_catastrophic(self) -> bool:
        """Check if failure is catastrophic."""
        return self.severity in [SeverityLevel.SEVERE, SeverityLevel.CRITICAL]


@dataclass
class TireStructure:
    """Tire structural properties.
    
    Attributes:
        sidewall_thickness: Sidewall thickness (mm)
        tread_thickness: Tread thickness (mm)
        belt_layers: Number of belt layers
        carcass_plies: Number of carcass plies
        bead_strength: Bead seat strength (kN)
        burst_pressure: Burst pressure (kPa)
        max_load_rating: Maximum load rating (kg)
        max_speed_rating: Maximum speed rating (km/h)
    """
    
    sidewall_thickness: float = 8.0
    tread_thickness: float = 10.0
    belt_layers: int = 2
    carcass_plies: int = 2
    bead_strength: float = 50.0
    burst_pressure: float = 1200.0
    max_load_rating: float = 700.0
    max_speed_rating: float = 210.0


@dataclass
class DamageState:
    """Tire damage accumulation state.
    
    Attributes:
        fatigue_damage: Cumulative fatigue damage (0-1)
        thermal_damage: Thermal degradation (0-1)
        impact_damage: Impact damage history
        aging_factor: Aging degradation factor
        remaining_life: Estimated remaining life (%)
    """
    
    fatigue_damage: float = 0.0
    thermal_damage: float = 0.0
    impact_damage: list[dict[str, float]] = field(default_factory=list)
    aging_factor: float = 1.0
    remaining_life: float = 100.0
    
    @property
    def total_damage(self) -> float:
        """Total damage accumulation."""
        impact_sum = sum(d.get("severity", 0) for d in self.impact_damage)
        return min(1.0, self.fatigue_damage + self.thermal_damage + 0.1 * impact_sum)
    
    @property
    def failure_probability(self) -> float:
        """Probability of imminent failure."""
        # Weibull-like failure probability
        damage = self.total_damage
        if damage < 0.5:
            return 0.01 * damage ** 2
        else:
            return 0.1 + 0.9 * (damage - 0.5) ** 2


class BlowoutSimulator:
    """Tire blowout simulation.
    
    Models rapid pressure loss events including:
    - Initiation mechanisms
    - Propagation dynamics
    - Vehicle stability effects
    - Debris generation
    
    Critical for safety analysis and run-flat design.
    """
    
    def __init__(
        self,
        tire_volume: float = 30.0,  # liters
        operating_pressure: float = 220.0,  # kPa
    ) -> None:
        """Initialize blowout simulator.
        
        Args:
            tire_volume: Tire internal volume (liters)
            operating_pressure: Normal operating pressure (kPa)
        """
        self.volume = tire_volume * 1e-3  # Convert to m³
        self.pressure = operating_pressure * 1e3  # Convert to Pa
        
        # Air properties
        self.gamma = 1.4  # Specific heat ratio
        self.R_air = 287.0  # J/(kg·K)
        self.T = 300.0  # K
        
        # Calculate stored energy
        self.stored_energy = self._compute_stored_energy()
        
        logger.info(f"BlowoutSimulator: stored energy = {self.stored_energy:.1f} kJ")
    
    def _compute_stored_energy(self) -> float:
        """Compute stored pneumatic energy.
        
        E = P * V / (γ - 1) * [1 - (P_atm/P)^((γ-1)/γ)]
        
        Returns:
            Stored energy (kJ)
        """
        P_atm = 101325  # Pa
        P = self.pressure + P_atm  # Absolute pressure
        
        energy = (P * self.volume) / (self.gamma - 1) * (
            1 - (P_atm / P) ** ((self.gamma - 1) / self.gamma)
        )
        
        return energy / 1000  # Convert to kJ
    
    def simulate_blowout(
        self,
        hole_diameter: float,  # mm
        location: str = "sidewall",
    ) -> dict[str, Any]:
        """Simulate blowout event.
        
        Args:
            hole_diameter: Rupture hole diameter (mm)
            location: Failure location (tread, sidewall, shoulder)
            
        Returns:
            Blowout simulation results
        """
        A_hole = np.pi * (hole_diameter * 1e-3 / 2) ** 2  # m²
        
        # Mass flow rate (choked flow if pressure ratio high enough)
        P = self.pressure + 101325
        P_atm = 101325
        rho = P / (self.R_air * self.T)
        
        # Critical pressure ratio for choked flow
        P_crit = P * (2 / (self.gamma + 1)) ** (self.gamma / (self.gamma - 1))
        
        if P_atm < P_crit:
            # Choked flow
            m_dot = A_hole * P * np.sqrt(
                self.gamma / (self.R_air * self.T)
            ) * (2 / (self.gamma + 1)) ** ((self.gamma + 1) / (2 * (self.gamma - 1)))
        else:
            # Subsonic flow
            ratio = P_atm / P
            m_dot = A_hole * np.sqrt(2 * rho * P * (
                self.gamma / (self.gamma - 1)
            ) * (ratio ** (2 / self.gamma) - ratio ** ((self.gamma + 1) / self.gamma)))
        
        # Time to complete deflation
        initial_mass = rho * self.volume
        t_deflate = initial_mass / m_dot
        
        # Pressure loss rate
        pressure_rate = self.pressure / t_deflate / 1000  # kPa/s
        
        # Energy release rate
        energy_rate = self.stored_energy / t_deflate
        
        # Debris risk based on location and hole size
        debris_factors = {
            "tread": 0.8,
            "sidewall": 0.3,
            "shoulder": 0.6,
        }
        debris_risk = debris_factors.get(location, 0.5) * min(1.0, hole_diameter / 50)
        
        # Severity assessment
        if t_deflate < 0.5:
            severity = SeverityLevel.CRITICAL
        elif t_deflate < 2.0:
            severity = SeverityLevel.SEVERE
        elif t_deflate < 10.0:
            severity = SeverityLevel.MODERATE
        else:
            severity = SeverityLevel.MINOR
        
        event = FailureEvent(
            mode=FailureMode.BLOWOUT,
            severity=severity,
            location=0.0,
            time_to_failure=t_deflate,
            pressure_loss_rate=pressure_rate,
            energy_release=self.stored_energy,
            debris_risk=debris_risk,
        )
        
        return {
            "hole_diameter_mm": hole_diameter,
            "mass_flow_rate_kgs": m_dot,
            "deflation_time_s": t_deflate,
            "pressure_loss_rate_kpa_s": pressure_rate,
            "energy_release_kj": self.stored_energy,
            "energy_rate_kw": energy_rate,
            "debris_risk": debris_risk,
            "event": event,
        }
    
    def compute_vehicle_stability(
        self,
        deflation_time: float,
        vehicle_speed: float,  # km/h
        tire_position: str,  # FL, FR, RL, RR
    ) -> dict[str, Any]:
        """Compute vehicle stability after blowout.
        
        Args:
            deflation_time: Time to deflation (s)
            vehicle_speed: Vehicle speed (km/h)
            tire_position: Tire position on vehicle
            
        Returns:
            Stability assessment
        """
        speed_ms = vehicle_speed / 3.6
        
        # Deceleration from drag (deflated tire has high resistance)
        Cd_deflated = 0.3  # Drag coefficient
        
        # Distance traveled during deflation
        distance = speed_ms * deflation_time
        
        # Yaw moment from asymmetric drag
        yaw_coefficients = {
            "FL": -0.5,
            "FR": 0.5,
            "RL": -0.3,
            "RR": 0.3,
        }
        yaw_factor = yaw_coefficients.get(tire_position, 0)
        yaw_moment = yaw_factor * speed_ms ** 2
        
        # Stability margin (higher = more stable)
        stability_margin = 1.0 - abs(yaw_moment) / 1000
        
        # Correction time available
        reaction_time = 1.5  # seconds (typical driver)
        
        if stability_margin > 0.7 and deflation_time > reaction_time:
            controllability = "CONTROLLABLE"
        elif stability_margin > 0.4:
            controllability = "CHALLENGING"
        else:
            controllability = "CRITICAL"
        
        return {
            "distance_during_event_m": distance,
            "yaw_moment_factor": yaw_moment,
            "stability_margin": stability_margin,
            "controllability": controllability,
            "required_steering_correction_deg": abs(yaw_moment) * 0.1,
        }


class TreadSeparationSimulator:
    """Tread/belt separation simulation.
    
    Models belt separation failures:
    - Adhesion degradation
    - Thermal fatigue
    - Centrifugal loading
    - Progressive delamination
    """
    
    def __init__(
        self,
        structure: TireStructure,
    ) -> None:
        """Initialize tread separation simulator.
        
        Args:
            structure: Tire structural properties
        """
        self.structure = structure
        
        # Adhesion strength (N/mm)
        self.peel_strength = 15.0
        self.shear_strength = 50.0
    
    def compute_centrifugal_stress(
        self,
        speed: float,  # km/h
        outer_diameter: float,  # mm
    ) -> float:
        """Compute centrifugal stress on tread.
        
        σ = ρ * ω² * r
        
        Args:
            speed: Vehicle speed (km/h)
            outer_diameter: Tire outer diameter (mm)
            
        Returns:
            Centrifugal stress (MPa)
        """
        v = speed / 3.6  # m/s
        r = outer_diameter / 2000  # m
        omega = v / r  # rad/s
        
        # Tread density
        rho = 1100  # kg/m³
        
        # Stress at tread surface
        sigma = rho * omega ** 2 * r / 1e6  # MPa
        
        return sigma
    
    def compute_thermal_stress(
        self,
        tread_temp: float,  # K
        belt_temp: float,  # K
    ) -> float:
        """Compute thermal stress from temperature gradient.
        
        Args:
            tread_temp: Tread temperature (K)
            belt_temp: Belt temperature (K)
            
        Returns:
            Thermal stress (MPa)
        """
        delta_T = tread_temp - belt_temp
        
        # CTE difference
        alpha_rubber = 2e-4  # 1/K
        alpha_steel = 1.2e-5  # 1/K
        delta_alpha = alpha_rubber - alpha_steel
        
        # Modulus
        E = 10.0  # MPa (constrained modulus)
        
        sigma = E * delta_alpha * delta_T
        
        return abs(sigma)
    
    def compute_separation_risk(
        self,
        speed: float,
        tread_temp: float,
        belt_temp: float,
        outer_diameter: float,
        damage_state: DamageState,
    ) -> dict[str, Any]:
        """Compute tread separation risk.
        
        Args:
            speed: Vehicle speed (km/h)
            tread_temp: Tread temperature (K)
            belt_temp: Belt temperature (K)
            outer_diameter: Tire diameter (mm)
            damage_state: Current damage state
            
        Returns:
            Separation risk assessment
        """
        sigma_centrifugal = self.compute_centrifugal_stress(speed, outer_diameter)
        sigma_thermal = self.compute_thermal_stress(tread_temp, belt_temp)
        
        # Combined stress
        sigma_total = sigma_centrifugal + sigma_thermal
        
        # Degraded strength
        degradation = 1 - damage_state.total_damage
        current_strength = self.peel_strength * degradation
        
        # Stress ratio
        stress_ratio = sigma_total / current_strength
        
        # Risk level
        if stress_ratio > 0.9:
            risk = "CRITICAL"
            probability = 0.5 + 0.5 * (stress_ratio - 0.9) / 0.1
        elif stress_ratio > 0.7:
            risk = "HIGH"
            probability = 0.1 + 0.4 * (stress_ratio - 0.7) / 0.2
        elif stress_ratio > 0.5:
            risk = "MODERATE"
            probability = 0.01 + 0.09 * (stress_ratio - 0.5) / 0.2
        else:
            risk = "LOW"
            probability = 0.001 * stress_ratio / 0.5
        
        return {
            "centrifugal_stress_mpa": sigma_centrifugal,
            "thermal_stress_mpa": sigma_thermal,
            "total_stress_mpa": sigma_total,
            "current_strength_n_mm": current_strength,
            "stress_ratio": stress_ratio,
            "risk_level": risk,
            "failure_probability": probability,
        }
    
    def simulate_separation_event(
        self,
        initiation_location: float,  # degrees
        speed: float,  # km/h
    ) -> dict[str, Any]:
        """Simulate tread separation event.
        
        Args:
            initiation_location: Initiation point (degrees)
            speed: Vehicle speed at failure (km/h)
            
        Returns:
            Separation event results
        """
        # Propagation rate (degrees per revolution)
        prop_rate = 5.0  # degrees/rev
        
        # Revolutions per second
        outer_diameter = 650  # mm typical
        circumference = np.pi * outer_diameter / 1000  # m
        rps = (speed / 3.6) / circumference
        
        # Time to full separation
        t_separation = 360 / (prop_rate * rps)
        
        # Energy released (kinetic energy of tread strip)
        tread_mass = 2.0  # kg (approximate tread strip mass)
        v = speed / 3.6
        kinetic_energy = 0.5 * tread_mass * v ** 2 / 1000  # kJ
        
        # Debris trajectory
        launch_angle = 30  # degrees
        launch_velocity = v + (outer_diameter / 2000) * 2 * np.pi * rps
        
        # Range
        g = 9.81
        range_m = (launch_velocity ** 2 * np.sin(2 * np.radians(launch_angle))) / g
        
        event = FailureEvent(
            mode=FailureMode.TREAD_SEPARATION,
            severity=SeverityLevel.SEVERE,
            location=initiation_location,
            time_to_failure=t_separation,
            pressure_loss_rate=0.0,  # No immediate pressure loss
            energy_release=kinetic_energy,
            debris_risk=0.95,
        )
        
        return {
            "separation_time_s": t_separation,
            "debris_kinetic_energy_kj": kinetic_energy,
            "debris_launch_velocity_ms": launch_velocity,
            "debris_range_m": range_m,
            "event": event,
        }


class ImpactDamageSimulator:
    """Road hazard impact simulation.
    
    Models tire damage from:
    - Potholes
    - Curb strikes
    - Road debris
    - Sharp objects
    """
    
    # Impact severity thresholds
    MINOR_ENERGY = 10  # J
    MODERATE_ENERGY = 50  # J
    SEVERE_ENERGY = 200  # J
    
    def __init__(
        self,
        structure: TireStructure,
    ) -> None:
        """Initialize impact simulator.
        
        Args:
            structure: Tire structural properties
        """
        self.structure = structure
        
        # Impact resistance (J/mm²)
        self.impact_strength = 0.5
    
    def compute_impact_energy(
        self,
        vehicle_speed: float,  # km/h
        obstacle_height: float,  # mm
        vehicle_mass: float = 1500.0,  # kg
    ) -> float:
        """Compute impact energy.
        
        Args:
            vehicle_speed: Vehicle speed (km/h)
            obstacle_height: Obstacle height (mm)
            vehicle_mass: Vehicle mass (kg)
            
        Returns:
            Impact energy (J)
        """
        v = vehicle_speed / 3.6  # m/s
        h = obstacle_height / 1000  # m
        
        # Vertical velocity component from obstacle
        # Simplified: assume 45° impact angle
        v_vertical = v * np.sin(np.arctan(h / 0.1))
        
        # Effective mass (single wheel)
        m_effective = vehicle_mass / 4 * 0.3  # Unsprung mass fraction
        
        energy = 0.5 * m_effective * v_vertical ** 2
        
        return energy
    
    def compute_penetration_depth(
        self,
        impact_energy: float,
        contact_area: float,  # mm²
    ) -> float:
        """Compute impact penetration depth.
        
        Args:
            impact_energy: Impact energy (J)
            contact_area: Contact area (mm²)
            
        Returns:
            Penetration depth (mm)
        """
        # Energy absorbed per unit penetration
        work_per_mm = self.impact_strength * contact_area
        
        depth = impact_energy / work_per_mm
        
        return depth
    
    def assess_impact_damage(
        self,
        vehicle_speed: float,
        obstacle_height: float,
        obstacle_type: str,
    ) -> dict[str, Any]:
        """Assess damage from road hazard impact.
        
        Args:
            vehicle_speed: Vehicle speed (km/h)
            obstacle_height: Obstacle height (mm)
            obstacle_type: Type (pothole, curb, debris, puncture)
            
        Returns:
            Damage assessment
        """
        energy = self.compute_impact_energy(vehicle_speed, obstacle_height)
        
        # Contact area by type
        contact_areas = {
            "pothole": 5000,  # mm² (distributed)
            "curb": 1000,  # mm² (edge impact)
            "debris": 500,  # mm² (localized)
            "puncture": 50,  # mm² (point)
        }
        area = contact_areas.get(obstacle_type, 1000)
        
        penetration = self.compute_penetration_depth(energy, area)
        
        # Damage type
        if obstacle_type == "puncture":
            damage_type = "PENETRATION"
            if penetration > self.structure.tread_thickness:
                severity = SeverityLevel.SEVERE
            else:
                severity = SeverityLevel.MINOR
        elif penetration > self.structure.sidewall_thickness * 0.5:
            damage_type = "STRUCTURAL"
            severity = SeverityLevel.SEVERE
        elif penetration > self.structure.sidewall_thickness * 0.2:
            damage_type = "BRUISE"
            severity = SeverityLevel.MODERATE
        else:
            damage_type = "SUPERFICIAL"
            severity = SeverityLevel.MINOR
        
        # Latent damage (may cause delayed failure)
        latent_damage = (penetration / self.structure.sidewall_thickness) ** 2
        
        return {
            "impact_energy_j": energy,
            "penetration_depth_mm": penetration,
            "damage_type": damage_type,
            "severity": severity,
            "latent_damage_factor": latent_damage,
            "immediate_failure": penetration > self.structure.sidewall_thickness,
        }
    
    def simulate_puncture(
        self,
        object_diameter: float,  # mm
        object_sharpness: float = 1.0,  # 0-1 (dull to sharp)
        vehicle_speed: float = 60.0,  # km/h
    ) -> dict[str, Any]:
        """Simulate puncture event.
        
        Args:
            object_diameter: Penetrating object diameter (mm)
            object_sharpness: Sharpness factor
            vehicle_speed: Vehicle speed (km/h)
            
        Returns:
            Puncture simulation results
        """
        # Puncture force required
        shear_strength = 20.0  # MPa
        A_shear = np.pi * object_diameter * self.structure.tread_thickness
        F_puncture = shear_strength * A_shear * (2 - object_sharpness)
        
        # Energy to puncture
        E_puncture = F_puncture * self.structure.tread_thickness / 1000
        
        # Check if puncture occurs
        impact_energy = 0.5 * 100 * (vehicle_speed / 3.6) ** 2 * 0.001  # Simplified
        
        if impact_energy > E_puncture:
            punctured = True
            hole_size = object_diameter
            
            # Pressure loss (slow leak vs blowout)
            if hole_size < 5:
                leak_rate = 0.1  # kPa/s (slow)
                failure_mode = "SLOW_LEAK"
            else:
                leak_rate = 10 * (hole_size / 5) ** 2  # kPa/s
                failure_mode = "RAPID_DEFLATION"
        else:
            punctured = False
            hole_size = 0
            leak_rate = 0
            failure_mode = "NO_PUNCTURE"
        
        return {
            "object_diameter_mm": object_diameter,
            "puncture_force_n": F_puncture,
            "puncture_energy_j": E_puncture,
            "impact_energy_j": impact_energy,
            "punctured": punctured,
            "hole_size_mm": hole_size,
            "leak_rate_kpa_s": leak_rate,
            "failure_mode": failure_mode,
        }


class RunFlatSimulator:
    """Run-flat tire capability simulation.
    
    Models extended mobility after pressure loss:
    - Self-supporting sidewall
    - Auxiliary support ring
    - Sealant systems
    """
    
    def __init__(
        self,
        run_flat_type: str = "self_supporting",
        max_run_flat_distance: float = 80.0,  # km
        max_run_flat_speed: float = 80.0,  # km/h
    ) -> None:
        """Initialize run-flat simulator.
        
        Args:
            run_flat_type: Type (self_supporting, auxiliary_ring, sealant)
            max_run_flat_distance: Maximum run-flat distance (km)
            max_run_flat_speed: Maximum run-flat speed (km/h)
        """
        self.type = run_flat_type
        self.max_distance = max_run_flat_distance
        self.max_speed = max_run_flat_speed
    
    def compute_run_flat_performance(
        self,
        speed: float,  # km/h
        load: float,  # kg
        ambient_temp: float = 298.0,  # K
    ) -> dict[str, Any]:
        """Compute run-flat performance.
        
        Args:
            speed: Vehicle speed (km/h)
            load: Tire load (kg)
            ambient_temp: Ambient temperature (K)
            
        Returns:
            Run-flat performance metrics
        """
        # Speed factor (higher speed = shorter range)
        speed_factor = 1 - 0.5 * min(1.0, (speed / self.max_speed) ** 2)
        
        # Load factor
        load_factor = 1 - 0.3 * min(1.0, (load / 800) ** 2)
        
        # Temperature factor (heat buildup limits range)
        temp_factor = 1 - 0.2 * max(0, (ambient_temp - 298) / 20)
        
        # Effective range
        range_km = self.max_distance * speed_factor * load_factor * temp_factor
        
        # Heat generation (run-flat generates more heat)
        heat_multiplier = 3.0 if self.type == "self_supporting" else 1.5
        
        # Sidewall temperature estimate
        time_to_limit = range_km / max(1, speed)  # hours
        delta_T = heat_multiplier * speed * time_to_limit * 0.5
        sidewall_temp = ambient_temp + delta_T
        
        # Handling degradation
        if self.type == "self_supporting":
            handling_factor = 0.6
        elif self.type == "auxiliary_ring":
            handling_factor = 0.8
        else:
            handling_factor = 0.7
        
        return {
            "effective_range_km": range_km,
            "max_safe_speed_kmh": min(speed, self.max_speed),
            "estimated_sidewall_temp_k": sidewall_temp,
            "handling_degradation": 1 - handling_factor,
            "time_at_speed_hours": time_to_limit,
        }
    
    def simulate_extended_mobility(
        self,
        profile: list[tuple[float, float]],  # (speed km/h, distance km)
    ) -> dict[str, Any]:
        """Simulate extended mobility drive.
        
        Args:
            profile: List of (speed, distance) segments
            
        Returns:
            Extended mobility results
        """
        total_distance = 0.0
        total_time = 0.0
        max_temp = 298.0
        
        for speed, distance in profile:
            perf = self.compute_run_flat_performance(speed, 600)
            
            if total_distance + distance > self.max_distance:
                remaining = self.max_distance - total_distance
                total_distance = self.max_distance
                break
            
            total_distance += distance
            total_time += distance / max(1, speed)
            max_temp = max(max_temp, perf["estimated_sidewall_temp_k"])
        
        success = total_distance >= sum(d for _, d in profile)
        
        return {
            "total_distance_km": total_distance,
            "total_time_hours": total_time,
            "max_temperature_k": max_temp,
            "completed_successfully": success,
            "remaining_capacity_km": max(0, self.max_distance - total_distance),
        }


class CatastrophicSimulator:
    """Complete catastrophic failure simulator."""
    
    def __init__(
        self,
        structure: TireStructure | None = None,
        tire_volume: float = 30.0,
        operating_pressure: float = 220.0,
    ) -> None:
        """Initialize catastrophic simulator.
        
        Args:
            structure: Tire structural properties
            tire_volume: Tire volume (liters)
            operating_pressure: Operating pressure (kPa)
        """
        self.structure = structure or TireStructure()
        
        self.blowout = BlowoutSimulator(tire_volume, operating_pressure)
        self.separation = TreadSeparationSimulator(self.structure)
        self.impact = ImpactDamageSimulator(self.structure)
        self.runflat = RunFlatSimulator()
        
        self.damage_state = DamageState()
        
        logger.info("Initialized CatastrophicSimulator")
    
    def assess_failure_risk(
        self,
        speed: float,
        load: float,
        temperature: float,
        inflation_pressure: float,
    ) -> dict[str, Any]:
        """Comprehensive failure risk assessment.
        
        Args:
            speed: Vehicle speed (km/h)
            load: Tire load (kg)
            temperature: Operating temperature (K)
            inflation_pressure: Inflation pressure (kPa)
            
        Returns:
            Comprehensive risk assessment
        """
        risks = {}
        
        # Speed rating check
        speed_margin = 1 - speed / self.structure.max_speed_rating
        if speed_margin < 0:
            risks["overspeed"] = {
                "risk": "CRITICAL",
                "margin": speed_margin,
                "probability": 0.5,
            }
        elif speed_margin < 0.1:
            risks["overspeed"] = {
                "risk": "HIGH",
                "margin": speed_margin,
                "probability": 0.1,
            }
        
        # Load rating check
        load_margin = 1 - load / self.structure.max_load_rating
        if load_margin < 0:
            risks["overload"] = {
                "risk": "CRITICAL",
                "margin": load_margin,
                "probability": 0.3,
            }
        elif load_margin < 0.1:
            risks["overload"] = {
                "risk": "HIGH",
                "margin": load_margin,
                "probability": 0.05,
            }
        
        # Underinflation check
        pressure_margin = (inflation_pressure - 100) / 100  # Minimum ~100 kPa
        if pressure_margin < 0.3:
            risks["underinflation"] = {
                "risk": "HIGH",
                "margin": pressure_margin,
                "probability": 0.2,
            }
        
        # Temperature check
        if temperature > 393:  # 120°C
            risks["overheating"] = {
                "risk": "CRITICAL",
                "temperature_k": temperature,
                "probability": 0.4,
            }
        elif temperature > 373:  # 100°C
            risks["overheating"] = {
                "risk": "HIGH",
                "temperature_k": temperature,
                "probability": 0.1,
            }
        
        # Tread separation risk
        sep_risk = self.separation.compute_separation_risk(
            speed, temperature, temperature - 20, 650, self.damage_state
        )
        if sep_risk["risk_level"] in ["HIGH", "CRITICAL"]:
            risks["tread_separation"] = sep_risk
        
        # Overall risk
        if any(r.get("risk") == "CRITICAL" for r in risks.values()):
            overall = "CRITICAL"
        elif any(r.get("risk") == "HIGH" or r.get("risk_level") == "HIGH" for r in risks.values()):
            overall = "HIGH"
        elif risks:
            overall = "MODERATE"
        else:
            overall = "LOW"
        
        return {
            "individual_risks": risks,
            "overall_risk": overall,
            "damage_state": self.damage_state,
            "cumulative_failure_probability": self.damage_state.failure_probability,
        }
    
    def simulate_failure_scenario(
        self,
        failure_mode: FailureMode,
        vehicle_speed: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Simulate specific failure scenario.
        
        Args:
            failure_mode: Failure mode to simulate
            vehicle_speed: Vehicle speed (km/h)
            **kwargs: Mode-specific parameters
            
        Returns:
            Failure scenario results
        """
        if failure_mode == FailureMode.BLOWOUT:
            hole_diameter = kwargs.get("hole_diameter", 25)
            location = kwargs.get("location", "sidewall")
            result = self.blowout.simulate_blowout(hole_diameter, location)
            
            # Add vehicle stability
            tire_position = kwargs.get("tire_position", "FL")
            stability = self.blowout.compute_vehicle_stability(
                result["deflation_time_s"],
                vehicle_speed,
                tire_position,
            )
            result["vehicle_stability"] = stability
            
        elif failure_mode == FailureMode.TREAD_SEPARATION:
            location = kwargs.get("initiation_location", 0)
            result = self.separation.simulate_separation_event(location, vehicle_speed)
            
        elif failure_mode == FailureMode.IMPACT_DAMAGE:
            obstacle_height = kwargs.get("obstacle_height", 50)
            obstacle_type = kwargs.get("obstacle_type", "pothole")
            result = self.impact.assess_impact_damage(
                vehicle_speed, obstacle_height, obstacle_type
            )
            
        else:
            result = {"error": f"Unsupported failure mode: {failure_mode}"}
        
        return result
    
    def generate_failure_report(
        self,
        event: FailureEvent,
    ) -> str:
        """Generate failure analysis report.
        
        Args:
            event: Failure event to analyze
            
        Returns:
            Formatted report string
        """
        report = f"""
TIRE FAILURE ANALYSIS REPORT
============================

Failure Mode: {event.mode.name}
Severity: {event.severity.name}
Location: {event.location:.1f}°

Time Characteristics:
  - Time to Complete Failure: {event.time_to_failure:.2f} s
  - Pressure Loss Rate: {event.pressure_loss_rate:.1f} kPa/s

Energy Release:
  - Total Energy: {event.energy_release:.1f} kJ

Safety Assessment:
  - Catastrophic: {"YES" if event.is_catastrophic else "NO"}
  - Debris Risk: {event.debris_risk * 100:.0f}%

Recommendations:
"""
        
        if event.severity == SeverityLevel.CRITICAL:
            report += "  - IMMEDIATE vehicle stop required\n"
            report += "  - Do not attempt to drive\n"
            report += "  - Professional recovery recommended\n"
        elif event.severity == SeverityLevel.SEVERE:
            report += "  - Safely reduce speed\n"
            report += "  - Activate hazard lights\n"
            report += "  - Stop at earliest safe location\n"
        elif event.severity == SeverityLevel.MODERATE:
            report += "  - Reduce speed to 50 km/h maximum\n"
            report += "  - Drive to nearest service location\n"
        else:
            report += "  - Monitor tire pressure\n"
            report += "  - Inspect at convenience\n"
        
        return report
