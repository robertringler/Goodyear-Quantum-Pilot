"""On-Vehicle Tire Simulation.

Simulates tire behavior during actual vehicle operation including:
- Rolling dynamics and handling
- Thermal management
- Wear progression
- Traction and braking
- Noise and vibration

These simulations predict real-world tire performance and
lifetime based on usage patterns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from .core import LoadCase, OperatingCondition, TireGeometry

logger = logging.getLogger(__name__)


class RoadSurface(Enum):
    """Road surface types."""
    
    DRY_ASPHALT = auto()
    WET_ASPHALT = auto()
    DRY_CONCRETE = auto()
    WET_CONCRETE = auto()
    GRAVEL = auto()
    SNOW = auto()
    ICE = auto()
    SAND = auto()


class DrivingManeuver(Enum):
    """Driving maneuvers."""
    
    STRAIGHT = auto()
    CORNERING = auto()
    BRAKING = auto()
    ACCELERATION = auto()
    COMBINED = auto()
    EMERGENCY = auto()


@dataclass
class TireForces:
    """Tire force and moment outputs.
    
    Uses SAE tire axis system:
    - X: Forward (positive = propulsion)
    - Y: Lateral (positive = left)
    - Z: Vertical (positive = down)
    
    Attributes:
        Fx: Longitudinal force (N)
        Fy: Lateral force (N)
        Fz: Vertical load (N)
        Mx: Overturning moment (N·m)
        My: Rolling resistance moment (N·m)
        Mz: Self-aligning moment (N·m)
    """
    
    Fx: float = 0.0
    Fy: float = 0.0
    Fz: float = 0.0
    Mx: float = 0.0
    My: float = 0.0
    Mz: float = 0.0
    
    @property
    def combined_force(self) -> float:
        """Combined horizontal force magnitude."""
        return np.sqrt(self.Fx ** 2 + self.Fy ** 2)
    
    @property
    def friction_coefficient(self) -> float:
        """Effective friction coefficient."""
        if self.Fz > 0:
            return self.combined_force / self.Fz
        return 0.0


@dataclass
class WearState:
    """Tire wear state.
    
    Tracks wear depth across the tread surface.
    
    Attributes:
        depth: Wear depth array (mm) - [circumferential, lateral]
        initial_tread: Initial tread depth (mm)
        accumulated_distance: Distance traveled (km)
        wear_rate: Current wear rate (mm/1000km)
    """
    
    depth: NDArray[np.float64]
    initial_tread: float = 10.0
    accumulated_distance: float = 0.0
    wear_rate: float = 0.01
    
    @property
    def remaining_tread(self) -> float:
        """Minimum remaining tread depth (mm)."""
        return self.initial_tread - np.max(self.depth)
    
    @property
    def wear_percentage(self) -> float:
        """Percentage of tread worn."""
        return np.max(self.depth) / self.initial_tread * 100
    
    @property
    def is_worn_out(self) -> bool:
        """Check if tire needs replacement (< 1.6mm)."""
        return self.remaining_tread < 1.6


@dataclass
class ThermalState:
    """Tire thermal state.
    
    Tracks temperature distribution through tire structure.
    
    Attributes:
        surface_temp: Tread surface temperature (K)
        bulk_temp: Bulk rubber temperature (K)
        belt_temp: Belt temperature (K)
        sidewall_temp: Sidewall temperature (K)
        ambient_temp: Ambient temperature (K)
    """
    
    surface_temp: float = 300.0
    bulk_temp: float = 300.0
    belt_temp: float = 300.0
    sidewall_temp: float = 300.0
    ambient_temp: float = 298.0
    
    @property
    def max_temp(self) -> float:
        """Maximum temperature in tire."""
        return max(
            self.surface_temp,
            self.bulk_temp,
            self.belt_temp,
            self.sidewall_temp,
        )
    
    @property
    def is_overheating(self) -> bool:
        """Check if tire is overheating (>120°C)."""
        return self.max_temp > 393


class RollingDynamics:
    """Tire rolling dynamics and handling simulation.
    
    Implements the Magic Formula (Pacejka) tire model for
    accurate force prediction under combined slip conditions.
    
    The Magic Formula:
    Y = D * sin(C * arctan(B*X - E*(B*X - arctan(B*X))))
    
    where:
    - Y: Force or moment
    - X: Slip angle or slip ratio
    - B, C, D, E: Model coefficients
    
    Example:
        >>> dynamics = RollingDynamics(
        ...     geometry=TireGeometry(),
        ...     lateral_stiffness=50000,
        ...     longitudinal_stiffness=100000,
        ... )
        >>> 
        >>> load = LoadCase(
        ...     vertical_load=5000,
        ...     slip_angle=5.0,
        ...     slip_ratio=0.1,
        ... )
        >>> 
        >>> forces = dynamics.compute_forces(load)
        >>> print(f"Lateral force: {forces.Fy:.0f} N")
    """
    
    def __init__(
        self,
        geometry: TireGeometry,
        lateral_stiffness: float = 50000.0,  # N/rad
        longitudinal_stiffness: float = 100000.0,  # N
        peak_friction: float = 1.0,
    ) -> None:
        """Initialize rolling dynamics model.
        
        Args:
            geometry: Tire geometry
            lateral_stiffness: Cornering stiffness (N/rad)
            longitudinal_stiffness: Longitudinal stiffness (N)
            peak_friction: Peak friction coefficient
        """
        self.geometry = geometry
        self.C_alpha = lateral_stiffness
        self.C_kappa = longitudinal_stiffness
        self.mu_peak = peak_friction
        
        # Magic Formula coefficients
        self._setup_magic_formula()
        
        logger.info(
            f"Initialized RollingDynamics: C_alpha={lateral_stiffness:.0f} N/rad"
        )
    
    def _setup_magic_formula(self) -> None:
        """Initialize Magic Formula coefficients."""
        # Lateral force coefficients
        self.By = 10.0   # Stiffness factor
        self.Cy = 1.3    # Shape factor
        self.Dy = self.mu_peak  # Peak factor (normalized by Fz)
        self.Ey = -0.5   # Curvature factor
        
        # Longitudinal force coefficients
        self.Bx = 12.0
        self.Cx = 1.65
        self.Dx = self.mu_peak
        self.Ex = -0.3
        
        # Self-aligning moment coefficients
        self.Bz = 8.0
        self.Cz = 2.5
        self.Dz = 0.03  # Pneumatic trail (m)
        self.Ez = -0.2
    
    def compute_forces(
        self,
        load: LoadCase,
        surface: RoadSurface = RoadSurface.DRY_ASPHALT,
    ) -> TireForces:
        """Compute tire forces and moments.
        
        Args:
            load: Loading condition
            surface: Road surface type
            
        Returns:
            TireForces with all force/moment components
        """
        Fz = load.vertical_load
        alpha = np.radians(load.slip_angle)
        kappa = load.slip_ratio
        
        # Surface friction modifier
        mu_surface = self._get_surface_friction(surface)
        
        # Pure slip forces
        Fy_pure = self._lateral_force(alpha, Fz, mu_surface)
        Fx_pure = self._longitudinal_force(kappa, Fz, mu_surface)
        
        # Combined slip (friction ellipse)
        sigma_x = kappa / (1 + abs(kappa))
        sigma_y = np.tan(alpha) / (1 + abs(kappa))
        sigma = np.sqrt(sigma_x ** 2 + sigma_y ** 2)
        
        if sigma > 0:
            Fx = Fx_pure * sigma_x / sigma
            Fy = Fy_pure * sigma_y / sigma
        else:
            Fx = 0.0
            Fy = 0.0
        
        # Self-aligning moment
        Mz = self._self_aligning_moment(alpha, Fz, mu_surface)
        
        # Rolling resistance moment
        Cr = 0.01  # Rolling resistance coefficient
        My = -Cr * Fz * self.geometry.rolling_radius * 1e-3
        
        # Overturning moment (from lateral force and camber)
        h_cg = 0.05  # Height of contact patch centroid (m)
        Mx = Fy * h_cg
        
        return TireForces(
            Fx=Fx,
            Fy=Fy,
            Fz=Fz,
            Mx=Mx,
            My=My,
            Mz=Mz,
        )
    
    def _lateral_force(
        self,
        alpha: float,
        Fz: float,
        mu: float,
    ) -> float:
        """Compute pure lateral force (Magic Formula).
        
        Args:
            alpha: Slip angle (rad)
            Fz: Vertical load (N)
            mu: Friction coefficient
            
        Returns:
            Lateral force (N)
        """
        D = self.Dy * mu * Fz
        C = self.Cy
        B = self.C_alpha / (C * D)
        E = self.Ey
        
        phi = B * alpha - E * (B * alpha - np.arctan(B * alpha))
        Fy = D * np.sin(C * np.arctan(phi))
        
        return Fy
    
    def _longitudinal_force(
        self,
        kappa: float,
        Fz: float,
        mu: float,
    ) -> float:
        """Compute pure longitudinal force (Magic Formula).
        
        Args:
            kappa: Slip ratio
            Fz: Vertical load (N)
            mu: Friction coefficient
            
        Returns:
            Longitudinal force (N)
        """
        D = self.Dx * mu * Fz
        C = self.Cx
        B = self.C_kappa / (C * D)
        E = self.Ex
        
        phi = B * kappa - E * (B * kappa - np.arctan(B * kappa))
        Fx = D * np.sin(C * np.arctan(phi))
        
        return Fx
    
    def _self_aligning_moment(
        self,
        alpha: float,
        Fz: float,
        mu: float,
    ) -> float:
        """Compute self-aligning moment.
        
        Args:
            alpha: Slip angle (rad)
            Fz: Vertical load (N)
            mu: Friction coefficient
            
        Returns:
            Self-aligning moment (N·m)
        """
        D = self.Dz * Fz
        C = self.Cz
        B = self.Bz
        E = self.Ez
        
        phi = B * alpha - E * (B * alpha - np.arctan(B * alpha))
        Mz = -D * np.sin(C * np.arctan(phi))
        
        return Mz
    
    def _get_surface_friction(self, surface: RoadSurface) -> float:
        """Get friction coefficient for road surface."""
        friction_map = {
            RoadSurface.DRY_ASPHALT: 1.0,
            RoadSurface.WET_ASPHALT: 0.7,
            RoadSurface.DRY_CONCRETE: 0.9,
            RoadSurface.WET_CONCRETE: 0.6,
            RoadSurface.GRAVEL: 0.6,
            RoadSurface.SNOW: 0.3,
            RoadSurface.ICE: 0.1,
            RoadSurface.SAND: 0.5,
        }
        return friction_map.get(surface, 1.0)
    
    def compute_relaxation_length(self) -> float:
        """Compute tire relaxation length.
        
        The relaxation length determines how quickly forces
        build up in response to slip.
        
        Returns:
            Relaxation length (m)
        """
        # Empirical: σ ≈ a / (3 * C_alpha / C_Fz)
        a = 0.1  # Half contact length (m)
        C_Fz = 100000  # Vertical stiffness (N/m)
        
        return a / (3 * self.C_alpha / C_Fz)


class ThermalModel:
    """Tire thermal model.
    
    Simulates heat generation and dissipation in the tire
    during operation.
    
    Heat sources:
    - Hysteresis in rubber deformation
    - Sliding friction in contact patch
    - Carcass flexing
    
    Heat transfer:
    - Conduction through tire structure
    - Convection to ambient air
    - Convection to road surface
    - Radiation to surroundings
    """
    
    def __init__(
        self,
        geometry: TireGeometry,
        thermal_conductivity: float = 0.3,  # W/(m·K)
        specific_heat: float = 2000.0,  # J/(kg·K)
        density: float = 1100.0,  # kg/m³
    ) -> None:
        """Initialize thermal model.
        
        Args:
            geometry: Tire geometry
            thermal_conductivity: Thermal conductivity
            specific_heat: Specific heat capacity
            density: Material density
        """
        self.geometry = geometry
        self.k = thermal_conductivity
        self.cp = specific_heat
        self.rho = density
        
        # Thermal diffusivity
        self.alpha = self.k / (self.rho * self.cp)
        
        # Convection coefficients
        self.h_air = 25.0  # W/(m²·K), forced convection to air
        self.h_road = 500.0  # W/(m²·K), conduction to road
        
        logger.info("Initialized ThermalModel")
    
    def compute_heat_generation(
        self,
        forces: TireForces,
        load: LoadCase,
    ) -> dict[str, float]:
        """Compute heat generation rates.
        
        Args:
            forces: Current tire forces
            load: Loading condition
            
        Returns:
            Dictionary of heat sources (W)
        """
        # Speed
        v = load.vehicle_speed
        omega = load.angular_velocity
        r = self.geometry.rolling_radius * 1e-3  # m
        
        # Rolling resistance heat
        Q_rolling = abs(forces.My) * omega
        
        # Sliding friction heat
        slip_velocity = v * abs(load.slip_ratio) + r * omega * np.tan(np.radians(load.slip_angle))
        Q_sliding = forces.combined_force * abs(slip_velocity)
        
        # Hysteresis heat (proportional to deformation energy)
        tan_delta = 0.1  # Loss tangent
        strain_energy = 0.5 * load.vertical_load * 0.01  # Simplified
        Q_hysteresis = tan_delta * strain_energy * omega
        
        return {
            "rolling": Q_rolling,
            "sliding": Q_sliding,
            "hysteresis": Q_hysteresis,
            "total": Q_rolling + Q_sliding + Q_hysteresis,
        }
    
    def compute_heat_dissipation(
        self,
        thermal_state: ThermalState,
        load: LoadCase,
    ) -> dict[str, float]:
        """Compute heat dissipation rates.
        
        Args:
            thermal_state: Current thermal state
            load: Loading condition
            
        Returns:
            Dictionary of heat sinks (W)
        """
        # Surface areas (approximate)
        A_tread = (
            np.pi * self.geometry.overall_diameter * 1e-3
            * self.geometry.tread_width * 1e-3
        )
        A_sidewall = 2 * np.pi * (
            (self.geometry.overall_diameter * 1e-3) ** 2
            - (self.geometry.rim_diameter * 25.4e-3) ** 2
        ) / 4
        
        # Contact patch area
        A_contact = 0.01 * load.vertical_load / 200000  # m²
        
        # Convection to air (tread and sidewalls)
        # Enhanced by vehicle speed
        v = load.vehicle_speed
        h_eff = self.h_air * (1 + 0.1 * v)
        
        Q_air_tread = h_eff * A_tread * (thermal_state.surface_temp - load.ambient_temperature)
        Q_air_sidewall = h_eff * A_sidewall * (thermal_state.sidewall_temp - load.ambient_temperature)
        
        # Conduction to road
        Q_road = self.h_road * A_contact * (thermal_state.surface_temp - load.road_temperature)
        
        # Radiation (Stefan-Boltzmann)
        sigma = 5.67e-8
        epsilon = 0.9
        Q_radiation = epsilon * sigma * A_tread * (
            thermal_state.surface_temp ** 4 - load.ambient_temperature ** 4
        )
        
        return {
            "air_convection": Q_air_tread + Q_air_sidewall,
            "road_conduction": Q_road,
            "radiation": Q_radiation,
            "total": Q_air_tread + Q_air_sidewall + Q_road + Q_radiation,
        }
    
    def update_temperatures(
        self,
        thermal_state: ThermalState,
        heat_gen: dict[str, float],
        heat_diss: dict[str, float],
        dt: float,
    ) -> ThermalState:
        """Update temperatures for one time step.
        
        Args:
            thermal_state: Current thermal state
            heat_gen: Heat generation rates
            heat_diss: Heat dissipation rates
            dt: Time step (s)
            
        Returns:
            Updated thermal state
        """
        # Approximate tire mass
        mass = 10.0  # kg
        
        # Net heat rate
        Q_net = heat_gen["total"] - heat_diss["total"]
        
        # Temperature change
        dT = Q_net * dt / (mass * self.cp)
        
        # Update all temperatures (simplified uniform heating)
        return ThermalState(
            surface_temp=thermal_state.surface_temp + dT,
            bulk_temp=thermal_state.bulk_temp + dT * 0.8,
            belt_temp=thermal_state.belt_temp + dT * 0.6,
            sidewall_temp=thermal_state.sidewall_temp + dT * 0.7,
            ambient_temp=thermal_state.ambient_temp,
        )


class WearSimulator:
    """Tire wear simulation.
    
    Models tread wear using physics-based models including:
    - Abrasive wear (road surface interaction)
    - Adhesive wear (rubber transfer)
    - Fatigue wear (cyclic deformation)
    - Chemical wear (oxidation, ozone)
    
    Wear patterns predicted:
    - Shoulder wear (underinflation, overloading)
    - Center wear (overinflation)
    - Cupping (suspension issues)
    - Feathering (alignment issues)
    - Flat spots (hard braking)
    """
    
    # Wear coefficients for different mechanisms (mm³/(N·mm))
    WEAR_COEFFICIENTS = {
        "abrasive": 1e-9,
        "adhesive": 5e-10,
        "fatigue": 2e-10,
        "chemical": 1e-10,
    }
    
    def __init__(
        self,
        geometry: TireGeometry,
        hardness: float = 65.0,  # Shore A
        abrasion_resistance: float = 100.0,  # DIN abrasion index
    ) -> None:
        """Initialize wear simulator.
        
        Args:
            geometry: Tire geometry
            hardness: Rubber hardness (Shore A)
            abrasion_resistance: Abrasion resistance index
        """
        self.geometry = geometry
        self.hardness = hardness
        self.abrasion_resistance = abrasion_resistance
        
        # Initialize wear state
        n_circumferential = 180
        n_lateral = 30
        self.wear_state = WearState(
            depth=np.zeros((n_circumferential, n_lateral)),
            initial_tread=geometry.tread_depth,
        )
        
        logger.info(
            f"Initialized WearSimulator: hardness={hardness} Shore A"
        )
    
    def compute_wear_rate(
        self,
        forces: TireForces,
        load: LoadCase,
        surface: RoadSurface = RoadSurface.DRY_ASPHALT,
    ) -> float:
        """Compute instantaneous wear rate.
        
        Args:
            forces: Current tire forces
            load: Loading condition
            surface: Road surface type
            
        Returns:
            Wear rate (mm/km)
        """
        # Contact pressure
        A_contact = 0.01 * forces.Fz / 200000  # m²
        pressure = forces.Fz / (A_contact * 1e6)  # MPa
        
        # Sliding distance per revolution
        circumference = np.pi * self.geometry.overall_diameter * 1e-3  # m
        slip_distance = circumference * (
            abs(load.slip_ratio) + abs(np.tan(np.radians(load.slip_angle)))
        )
        
        # Surface roughness factor
        roughness_factors = {
            RoadSurface.DRY_ASPHALT: 1.0,
            RoadSurface.WET_ASPHALT: 0.8,
            RoadSurface.DRY_CONCRETE: 1.2,
            RoadSurface.GRAVEL: 2.0,
        }
        roughness = roughness_factors.get(surface, 1.0)
        
        # Archard wear equation: V = K * F * s / H
        K = self.WEAR_COEFFICIENTS["abrasive"] * (100 / self.abrasion_resistance)
        H = self.hardness * 0.1  # Convert Shore A to MPa-like hardness
        
        wear_volume = K * forces.Fz * slip_distance * roughness / H
        
        # Convert to depth (assuming uniform contact area)
        wear_depth = wear_volume / A_contact
        
        # Wear per km
        revolutions_per_km = 1000 / circumference
        wear_per_km = wear_depth * revolutions_per_km
        
        return wear_per_km
    
    def compute_wear_distribution(
        self,
        forces: TireForces,
        load: LoadCase,
    ) -> NDArray[np.float64]:
        """Compute wear distribution across tread.
        
        Args:
            forces: Current tire forces
            load: Loading condition
            
        Returns:
            Wear distribution array (mm) - [lateral profile]
        """
        n_lateral = self.wear_state.depth.shape[1]
        
        # Lateral position (-1 to 1)
        y = np.linspace(-1, 1, n_lateral)
        
        # Base pressure distribution (parabolic)
        pressure_base = 1 - 0.5 * y ** 2
        
        # Camber effect (shifts pressure)
        camber = np.radians(load.camber_angle)
        pressure_base *= 1 + 0.5 * camber * y
        
        # Cornering effect (loads outside edge)
        slip_angle = np.radians(load.slip_angle)
        if slip_angle != 0:
            sign = np.sign(slip_angle)
            pressure_base *= 1 + 0.3 * sign * y
        
        # Normalize
        pressure_base /= np.mean(pressure_base)
        
        # Wear proportional to pressure
        base_wear = self.compute_wear_rate(forces, load)
        wear_distribution = base_wear * pressure_base
        
        return wear_distribution
    
    def update_wear(
        self,
        forces: TireForces,
        load: LoadCase,
        distance_km: float,
    ) -> WearState:
        """Update wear state after driving distance.
        
        Args:
            forces: Tire forces during driving
            load: Loading condition
            distance_km: Distance driven (km)
            
        Returns:
            Updated wear state
        """
        # Get wear distribution
        wear_rate = self.compute_wear_distribution(forces, load)
        
        # Apply wear (uniform around circumference for steady driving)
        n_circ = self.wear_state.depth.shape[0]
        for i in range(n_circ):
            self.wear_state.depth[i, :] += wear_rate * distance_km
        
        self.wear_state.accumulated_distance += distance_km
        self.wear_state.wear_rate = np.mean(wear_rate)
        
        return self.wear_state
    
    def predict_remaining_life(
        self,
        average_forces: TireForces,
        average_load: LoadCase,
    ) -> float:
        """Predict remaining tire life.
        
        Args:
            average_forces: Average operating forces
            average_load: Average operating conditions
            
        Returns:
            Remaining life (km)
        """
        current_wear = np.max(self.wear_state.depth)
        remaining_tread = self.geometry.tread_depth - current_wear - 1.6  # 1.6mm minimum
        
        if remaining_tread <= 0:
            return 0.0
        
        wear_rate = self.compute_wear_rate(average_forces, average_load)
        
        if wear_rate > 0:
            return remaining_tread / wear_rate
        return float("inf")
    
    def get_wear_pattern(self) -> str:
        """Analyze wear pattern and return diagnosis."""
        depth = self.wear_state.depth
        lateral_profile = np.mean(depth, axis=0)
        
        center = lateral_profile[len(lateral_profile) // 2]
        edges = (lateral_profile[0] + lateral_profile[-1]) / 2
        left = lateral_profile[:len(lateral_profile) // 4].mean()
        right = lateral_profile[-len(lateral_profile) // 4:].mean()
        
        if center > edges * 1.3:
            return "CENTER_WEAR - Overinflation suspected"
        elif edges > center * 1.3:
            return "SHOULDER_WEAR - Underinflation or overloading"
        elif abs(left - right) > 0.2 * (left + right):
            return "UNEVEN_WEAR - Alignment issue suspected"
        else:
            return "NORMAL_WEAR - Uniform wear pattern"


class TractionModel:
    """Tire traction and grip model.
    
    Predicts tire grip under various conditions:
    - Wet braking
    - Dry cornering
    - Snow/ice traction
    - Hydroplaning threshold
    """
    
    def __init__(
        self,
        geometry: TireGeometry,
        tread_pattern: str = "directional",
        sipe_density: float = 10.0,  # sipes per cm
    ) -> None:
        """Initialize traction model.
        
        Args:
            geometry: Tire geometry
            tread_pattern: Pattern type (directional, symmetric, asymmetric)
            sipe_density: Sipe density for wet/snow grip
        """
        self.geometry = geometry
        self.tread_pattern = tread_pattern
        self.sipe_density = sipe_density
    
    def compute_hydroplaning_speed(
        self,
        inflation_pressure: float,  # kPa
        water_depth: float = 5.0,  # mm
    ) -> float:
        """Compute hydroplaning onset speed.
        
        Uses NASA hydroplaning equation:
        V = 10.35 * sqrt(P)
        
        Args:
            inflation_pressure: Tire pressure (kPa)
            water_depth: Water depth on road (mm)
            
        Returns:
            Hydroplaning speed (km/h)
        """
        P_psi = inflation_pressure / 6.895  # Convert to psi
        
        # Base hydroplaning speed (NASA equation)
        V_base = 10.35 * np.sqrt(P_psi) * 1.609  # Convert to km/h
        
        # Corrections for tread depth and pattern
        tread_remaining = self.geometry.tread_depth  # Assume new tire
        tread_factor = min(1.0, tread_remaining / 8.0)  # 8mm reference
        
        # Water depth effect
        water_factor = 1.0 - 0.05 * (water_depth - 2.0)
        
        # Tread pattern effect
        pattern_factors = {
            "directional": 1.1,
            "asymmetric": 1.05,
            "symmetric": 1.0,
        }
        pattern_factor = pattern_factors.get(self.tread_pattern, 1.0)
        
        V_hydro = V_base * tread_factor * max(0.5, water_factor) * pattern_factor
        
        return V_hydro
    
    def compute_wet_grip(
        self,
        speed: float,  # km/h
        water_depth: float = 2.0,  # mm
    ) -> float:
        """Compute wet grip coefficient.
        
        Args:
            speed: Vehicle speed (km/h)
            water_depth: Water depth (mm)
            
        Returns:
            Wet grip coefficient (fraction of dry)
        """
        V_hydro = self.compute_hydroplaning_speed(220, water_depth)
        
        # Grip reduction with speed
        if speed >= V_hydro:
            return 0.1  # Near-complete loss
        
        # Linear interpolation
        grip = 1.0 - 0.7 * (speed / V_hydro) ** 2
        
        # Sipe effect (helps channel water)
        sipe_bonus = 0.1 * min(1.0, self.sipe_density / 15.0)
        
        return min(0.95, grip + sipe_bonus)
    
    def compute_snow_grip(self) -> float:
        """Compute snow grip capability."""
        # Based on sipe density and tread depth
        base_grip = 0.3
        
        sipe_bonus = 0.2 * min(1.0, self.sipe_density / 20.0)
        tread_bonus = 0.1 * min(1.0, self.geometry.tread_depth / 10.0)
        
        return min(0.7, base_grip + sipe_bonus + tread_bonus)
    
    def compute_ice_grip(self) -> float:
        """Compute ice grip capability."""
        # Very limited grip on ice without studs
        base_grip = 0.1
        
        sipe_bonus = 0.05 * min(1.0, self.sipe_density / 25.0)
        
        return min(0.25, base_grip + sipe_bonus)


class OnVehicleSimulator:
    """Complete on-vehicle simulation combining all models."""
    
    def __init__(
        self,
        geometry: TireGeometry,
    ) -> None:
        """Initialize on-vehicle simulator.
        
        Args:
            geometry: Tire geometry
        """
        self.geometry = geometry
        
        self.dynamics = RollingDynamics(geometry)
        self.thermal = ThermalModel(geometry)
        self.wear = WearSimulator(geometry)
        self.traction = TractionModel(geometry)
        
        # Current state
        self.thermal_state = ThermalState()
        
        logger.info(f"Initialized OnVehicleSimulator for {geometry.tire_code}")
    
    def step(
        self,
        load: LoadCase,
        dt: float = 0.1,  # seconds
    ) -> dict[str, Any]:
        """Advance simulation by one time step.
        
        Args:
            load: Current loading condition
            dt: Time step (seconds)
            
        Returns:
            Dictionary of current state
        """
        # Compute forces
        forces = self.dynamics.compute_forces(load)
        
        # Update thermal state
        heat_gen = self.thermal.compute_heat_generation(forces, load)
        heat_diss = self.thermal.compute_heat_dissipation(self.thermal_state, load)
        self.thermal_state = self.thermal.update_temperatures(
            self.thermal_state, heat_gen, heat_diss, dt
        )
        
        # Update wear (convert time to distance)
        distance_km = load.vehicle_speed * dt / 1000
        self.wear.update_wear(forces, load, distance_km)
        
        return {
            "forces": forces,
            "thermal_state": self.thermal_state,
            "wear_state": self.wear.wear_state,
            "heat_generation": heat_gen["total"],
        }
    
    def simulate_drive_cycle(
        self,
        conditions: list[OperatingCondition],
    ) -> dict[str, Any]:
        """Simulate a complete drive cycle.
        
        Args:
            conditions: List of operating conditions
            
        Returns:
            Summary of drive cycle results
        """
        results = {
            "total_distance": 0.0,
            "max_temperature": 0.0,
            "total_wear": 0.0,
            "time_series": [],
        }
        
        time = 0.0
        
        for condition in conditions:
            n_steps = int(condition.duration * 3600 / 0.1)  # 0.1s steps
            
            for _ in range(n_steps):
                state = self.step(condition.load_case, dt=0.1)
                time += 0.1
                
                results["max_temperature"] = max(
                    results["max_temperature"],
                    self.thermal_state.max_temp,
                )
            
            results["total_distance"] += (
                condition.load_case.vehicle_speed * condition.duration
            )
        
        results["total_wear"] = np.max(self.wear.wear_state.depth)
        
        return results
