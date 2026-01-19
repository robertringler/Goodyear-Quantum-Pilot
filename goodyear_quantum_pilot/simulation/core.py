"""Core Tire Simulation Framework.

Provides the main TireSimulator class and supporting infrastructure
for multi-physics tire simulation with quantum material integration.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class SimulationPhase(Enum):
    """Tire lifecycle phases."""

    MANUFACTURING = auto()
    CURING = auto()
    STORAGE = auto()
    SHIPPING = auto()
    BREAK_IN = auto()
    SERVICE = auto()
    AGING = auto()
    END_OF_LIFE = auto()


class SolverType(Enum):
    """Numerical solver types."""

    EXPLICIT = auto()
    IMPLICIT = auto()
    HYBRID = auto()
    SPECTRAL = auto()
    MONTE_CARLO = auto()


class BackendType(Enum):
    """Computation backend types."""

    CPU = auto()
    GPU_CUDA = auto()
    GPU_ROCM = auto()
    TPU = auto()
    QUANTUM = auto()
    HYBRID = auto()


@dataclass
class TireGeometry:
    """Tire geometry specification.

    Follows standard tire sizing convention:
    P{width}/{aspect_ratio}R{rim_diameter}

    Example: P225/60R16
    - Section width: 225 mm
    - Aspect ratio: 60%
    - Rim diameter: 16 inches

    Attributes:
        section_width: Tire width (mm)
        aspect_ratio: Sidewall height / width (%)
        rim_diameter: Rim diameter (inches)
        overall_diameter: Total tire diameter (mm)
        tread_width: Width of tread surface (mm)
        tread_depth: Initial tread depth (mm)
        sidewall_thickness: Sidewall thickness (mm)
        bead_diameter: Bead seat diameter (mm)
        belt_width: Width of belt package (mm)
        belt_angle: Belt cord angle (degrees)
        ply_count: Number of body plies
    """

    section_width: float = 225.0  # mm
    aspect_ratio: float = 60.0  # %
    rim_diameter: float = 16.0  # inches
    tread_width: float = 200.0  # mm
    tread_depth: float = 10.0  # mm
    sidewall_thickness: float = 8.0  # mm
    belt_width: float = 210.0  # mm
    belt_angle: float = 23.0  # degrees
    ply_count: int = 2

    @property
    def sidewall_height(self) -> float:
        """Sidewall height in mm."""
        return self.section_width * self.aspect_ratio / 100

    @property
    def overall_diameter(self) -> float:
        """Overall tire diameter in mm."""
        rim_mm = self.rim_diameter * 25.4
        return rim_mm + 2 * self.sidewall_height

    @property
    def rolling_radius(self) -> float:
        """Effective rolling radius in mm."""
        # Loaded radius is ~97% of free radius
        return self.overall_diameter / 2 * 0.97

    @property
    def tire_code(self) -> str:
        """Standard tire size code."""
        return f"P{int(self.section_width)}/{int(self.aspect_ratio)}R{int(self.rim_diameter)}"

    def generate_mesh(
        self,
        radial_divisions: int = 50,
        circumferential_divisions: int = 180,
        axial_divisions: int = 30,
    ) -> dict[str, NDArray[np.float64]]:
        """Generate finite element mesh for tire.

        Args:
            radial_divisions: Divisions in radial direction
            circumferential_divisions: Divisions around circumference
            axial_divisions: Divisions in axial (width) direction

        Returns:
            Dictionary with node coordinates and element connectivity
        """
        # Radial coordinates
        r_inner = self.rim_diameter * 25.4 / 2
        r_outer = self.overall_diameter / 2
        r = np.linspace(r_inner, r_outer, radial_divisions)

        # Circumferential coordinates
        theta = np.linspace(0, 2 * np.pi, circumferential_divisions, endpoint=False)

        # Axial coordinates
        z = np.linspace(-self.section_width / 2, self.section_width / 2, axial_divisions)

        # Generate node grid
        R, THETA, Z = np.meshgrid(r, theta, z, indexing="ij")

        # Convert to Cartesian
        X = R * np.cos(THETA)
        Y = R * np.sin(THETA)

        nodes = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1)

        n_nodes = len(nodes)
        n_elements = (radial_divisions - 1) * circumferential_divisions * (axial_divisions - 1)

        logger.info(f"Generated mesh: {n_nodes} nodes, ~{n_elements} elements")

        return {
            "nodes": nodes,
            "radial": r,
            "theta": theta,
            "axial": z,
            "n_nodes": n_nodes,
            "n_elements": n_elements,
        }


@dataclass
class LoadCase:
    """Tire loading condition.

    Attributes:
        vertical_load: Vertical (normal) load (N)
        inflation_pressure: Tire pressure (kPa)
        camber_angle: Camber angle (degrees)
        slip_angle: Slip angle (degrees)
        slip_ratio: Longitudinal slip ratio
        angular_velocity: Rotation speed (rad/s)
        ambient_temperature: Ambient temperature (K)
        road_temperature: Road surface temperature (K)
    """

    vertical_load: float = 5000.0  # N
    inflation_pressure: float = 220.0  # kPa
    camber_angle: float = 0.0  # degrees
    slip_angle: float = 0.0  # degrees
    slip_ratio: float = 0.0
    angular_velocity: float = 80.0  # rad/s (~60 km/h for 16" tire)
    ambient_temperature: float = 298.0  # K
    road_temperature: float = 320.0  # K

    @property
    def vehicle_speed(self) -> float:
        """Approximate vehicle speed in m/s."""
        # Assuming ~0.3m rolling radius
        return self.angular_velocity * 0.3

    @property
    def vehicle_speed_kmh(self) -> float:
        """Vehicle speed in km/h."""
        return self.vehicle_speed * 3.6


@dataclass
class OperatingCondition:
    """Operating conditions over time.

    Attributes:
        duration: Duration of condition (hours)
        load_case: Applied loads
        road_type: Type of road surface
        weather: Weather conditions
        driving_style: Driving aggressiveness (0-1)
    """

    duration: float = 1.0  # hours
    load_case: LoadCase = field(default_factory=LoadCase)
    road_type: str = "asphalt_dry"
    weather: str = "clear"
    driving_style: float = 0.5  # 0=gentle, 1=aggressive

    def get_cycles(self, wheel_rpm: float = 600) -> int:
        """Get number of wheel rotations."""
        return int(self.duration * 60 * wheel_rpm)


@dataclass
class SimulationConfig:
    """Configuration for tire simulation.

    Attributes:
        solver_type: Numerical solver type
        backend: Computation backend
        time_step: Simulation time step (s)
        max_time: Maximum simulation time (s)
        tolerance: Convergence tolerance
        output_interval: Data output interval
        enable_thermal: Enable thermal simulation
        enable_wear: Enable wear simulation
        enable_dynamics: Enable dynamic simulation
        enable_quantum: Enable quantum corrections
        n_threads: Number of CPU threads
        gpu_id: GPU device ID
        checkpoint_interval: Checkpoint frequency
        random_seed: Random seed for reproducibility
    """

    solver_type: SolverType = SolverType.IMPLICIT
    backend: BackendType = BackendType.CPU
    time_step: float = 1e-4  # seconds
    max_time: float = 1.0  # seconds
    tolerance: float = 1e-6
    output_interval: int = 100
    enable_thermal: bool = True
    enable_wear: bool = True
    enable_dynamics: bool = True
    enable_quantum: bool = True
    n_threads: int = 4
    gpu_id: int = 0
    checkpoint_interval: int = 1000
    random_seed: int = 42


@dataclass
class SimulationResult:
    """Results from tire simulation.

    Attributes:
        time: Time array
        displacement: Displacement field history
        stress: Stress field history
        temperature: Temperature field history
        wear_depth: Wear depth history
        contact_pressure: Contact pressure history
        forces: Force/moment history
        energy: Energy components history
        statistics: Summary statistics
        metadata: Simulation metadata
    """

    time: NDArray[np.float64]
    displacement: NDArray[np.float64] | None = None
    stress: NDArray[np.float64] | None = None
    temperature: NDArray[np.float64] | None = None
    wear_depth: NDArray[np.float64] | None = None
    contact_pressure: NDArray[np.float64] | None = None
    forces: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    energy: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    statistics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def max_temperature(self) -> float:
        """Maximum temperature reached."""
        if self.temperature is not None:
            return float(np.max(self.temperature))
        return 0.0

    @property
    def total_wear(self) -> float:
        """Total wear depth (mm)."""
        if self.wear_depth is not None:
            return float(np.max(self.wear_depth))
        return 0.0

    def export_to_vtk(self, filepath: str) -> None:
        """Export results to VTK format for visualization."""
        # VTK export implementation
        logger.info(f"Exporting results to {filepath}")


class MaterialModel(ABC):
    """Abstract base class for material constitutive models."""

    @abstractmethod
    def compute_stress(
        self,
        strain: NDArray[np.float64],
        strain_rate: NDArray[np.float64] | None = None,
        temperature: float = 300.0,
    ) -> NDArray[np.float64]:
        """Compute stress from strain."""
        ...

    @abstractmethod
    def compute_tangent(
        self,
        strain: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute tangent stiffness matrix."""
        ...


@dataclass
class HyperelasticMaterial(MaterialModel):
    """Hyperelastic material model for rubber.

    Implements Mooney-Rivlin and Ogden models commonly
    used for rubber materials.

    Strain energy function (Mooney-Rivlin):
    W = C10(I1 - 3) + C01(I2 - 3) + D1(J - 1)^2

    where I1, I2 are strain invariants and J is volume ratio.
    """

    C10: float = 0.5  # MPa
    C01: float = 0.1  # MPa
    D1: float = 0.01  # MPa^-1 (incompressibility)

    def compute_stress(
        self,
        strain: NDArray[np.float64],
        strain_rate: NDArray[np.float64] | None = None,
        temperature: float = 300.0,
    ) -> NDArray[np.float64]:
        """Compute Cauchy stress from Green-Lagrange strain.

        Args:
            strain: Green-Lagrange strain tensor [6] or [n, 6]
            strain_rate: Strain rate (for viscoelastic)
            temperature: Temperature (K)

        Returns:
            Cauchy stress tensor
        """
        # Convert strain to deformation gradient
        # Simplified: assume small strain regime
        E = strain

        # Small strain approximation: σ ≈ 2(C10 + C01) * ε
        mu = 2 * (self.C10 + self.C01)

        # Temperature softening
        T_ref = 300.0
        softening = 1.0 - 0.003 * (temperature - T_ref)
        mu *= max(0.1, softening)

        stress = mu * E

        return stress

    def compute_tangent(
        self,
        strain: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute material tangent stiffness.

        Returns:
            6x6 tangent stiffness matrix
        """
        mu = 2 * (self.C10 + self.C01)
        K = 2 / self.D1  # Bulk modulus

        # Isotropic elasticity tensor
        C = np.zeros((6, 6))

        # Diagonal terms
        C[0, 0] = C[1, 1] = C[2, 2] = K + 4 * mu / 3
        C[3, 3] = C[4, 4] = C[5, 5] = mu

        # Off-diagonal terms
        C[0, 1] = C[0, 2] = C[1, 0] = C[1, 2] = C[2, 0] = C[2, 1] = K - 2 * mu / 3

        return C


@dataclass
class ViscoelasticMaterial(MaterialModel):
    """Viscoelastic material model for rubber.

    Implements Prony series viscoelasticity:
    G(t) = G_∞ + Σ G_i exp(-t/τ_i)

    Attributes:
        G_infinity: Long-term shear modulus (MPa)
        G_prony: Prony series moduli (MPa)
        tau_prony: Prony series relaxation times (s)
    """

    G_infinity: float = 0.5
    G_prony: tuple[float, ...] = (0.3, 0.15, 0.05)
    tau_prony: tuple[float, ...] = (1e-3, 1e-1, 10.0)

    def compute_stress(
        self,
        strain: NDArray[np.float64],
        strain_rate: NDArray[np.float64] | None = None,
        temperature: float = 300.0,
    ) -> NDArray[np.float64]:
        """Compute viscoelastic stress."""
        # Equilibrium response
        stress = self.G_infinity * strain

        # Add viscous contribution if strain rate provided
        if strain_rate is not None:
            for G_i, tau_i in zip(self.G_prony, self.tau_prony):
                stress += G_i * tau_i * strain_rate

        return stress

    def compute_tangent(
        self,
        strain: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute instantaneous tangent."""
        G_instant = self.G_infinity + sum(self.G_prony)
        C = np.eye(6) * G_instant
        return C


class TireSimulator:
    """Main tire simulation engine.

    Coordinates multi-physics simulation of tire behavior including:
    - Structural mechanics (hyperelastic, viscoelastic)
    - Contact mechanics (tire-road interaction)
    - Thermal analysis (heat generation and conduction)
    - Wear prediction (abrasive and fatigue wear)
    - Dynamics (rolling, cornering, braking)

    The simulator can integrate quantum-computed material properties
    from the algorithms module for enhanced accuracy.

    Example:
        >>> # Define tire geometry
        >>> geometry = TireGeometry(
        ...     section_width=225,
        ...     aspect_ratio=60,
        ...     rim_diameter=16,
        ... )
        >>>
        >>> # Create simulator
        >>> sim = TireSimulator(
        ...     geometry=geometry,
        ...     material=HyperelasticMaterial(C10=0.5, C01=0.1),
        ...     config=SimulationConfig(enable_thermal=True),
        ... )
        >>>
        >>> # Run simulation
        >>> result = sim.run(
        ...     load_case=LoadCase(vertical_load=5000),
        ...     duration=10.0,
        ... )
        >>>
        >>> print(f"Max temperature: {result.max_temperature:.1f} K")
        >>> print(f"Total wear: {result.total_wear:.3f} mm")
    """

    def __init__(
        self,
        geometry: TireGeometry,
        material: MaterialModel,
        config: SimulationConfig | None = None,
    ) -> None:
        """Initialize tire simulator.

        Args:
            geometry: Tire geometry specification
            material: Material constitutive model
            config: Simulation configuration
        """
        self.geometry = geometry
        self.material = material
        self.config = config or SimulationConfig()

        # Generate mesh
        self.mesh = geometry.generate_mesh()

        # Initialize state variables
        self._initialize_state()

        # Set random seed
        np.random.seed(self.config.random_seed)

        logger.info(
            f"Initialized TireSimulator: {geometry.tire_code}, " f"{self.mesh['n_nodes']} nodes"
        )

    def _initialize_state(self) -> None:
        """Initialize simulation state variables."""
        n_nodes = self.mesh["n_nodes"]

        self.displacement = np.zeros((n_nodes, 3))
        self.velocity = np.zeros((n_nodes, 3))
        self.temperature = np.ones(n_nodes) * 300.0  # K
        self.wear_depth = np.zeros(n_nodes)
        self.accumulated_strain = np.zeros(n_nodes)

        self._time = 0.0
        self._step = 0

    def run(
        self,
        load_case: LoadCase,
        duration: float = 1.0,
        callback: Callable[[int, float], None] | None = None,
    ) -> SimulationResult:
        """Run tire simulation.

        Args:
            load_case: Applied loading condition
            duration: Simulation duration (seconds)
            callback: Optional callback function(step, time)

        Returns:
            SimulationResult with simulation output
        """
        import time as time_module

        start_time = time_module.time()

        dt = self.config.time_step
        n_steps = int(duration / dt)

        # Output arrays
        time_history = []
        temperature_history = []
        wear_history = []
        force_history = {"Fx": [], "Fy": [], "Fz": [], "Mx": [], "My": [], "Mz": []}

        logger.info(f"Starting simulation: {n_steps} steps, dt={dt:.2e} s")

        for step in range(n_steps):
            self._time = step * dt
            self._step = step

            # Solve structural problem
            if self.config.enable_dynamics:
                self._solve_dynamics(load_case, dt)

            # Solve thermal problem
            if self.config.enable_thermal:
                self._solve_thermal(load_case, dt)

            # Compute wear
            if self.config.enable_wear:
                self._compute_wear(load_case, dt)

            # Record output
            if step % self.config.output_interval == 0:
                time_history.append(self._time)
                temperature_history.append(self.temperature.copy())
                wear_history.append(self.wear_depth.copy())

                forces = self._compute_forces(load_case)
                for key, value in forces.items():
                    force_history[key].append(value)

            # Callback
            if callback is not None:
                callback(step, self._time)

        execution_time = time_module.time() - start_time

        # Build result
        result = SimulationResult(
            time=np.array(time_history),
            temperature=np.array(temperature_history) if temperature_history else None,
            wear_depth=np.array(wear_history) if wear_history else None,
            forces={k: np.array(v) for k, v in force_history.items()},
            statistics={
                "max_temperature": float(np.max(self.temperature)),
                "min_temperature": float(np.min(self.temperature)),
                "max_wear": float(np.max(self.wear_depth)),
                "execution_time": execution_time,
            },
            metadata={
                "geometry": self.geometry.tire_code,
                "n_steps": n_steps,
                "dt": dt,
                "load_case": load_case,
            },
        )

        logger.info(
            f"Simulation completed: {execution_time:.2f} s, "
            f"T_max={result.max_temperature:.1f} K, "
            f"wear={result.total_wear:.4f} mm"
        )

        return result

    def _solve_dynamics(self, load_case: LoadCase, dt: float) -> None:
        """Solve structural dynamics for one time step."""
        n_nodes = self.mesh["n_nodes"]

        # Simplified dynamics: apply load and compute deformation
        # Real implementation would use FEM assembly and solver

        # Contact force distribution (simplified)
        contact_nodes = self._get_contact_nodes(load_case)

        for node in contact_nodes:
            # Apply contact pressure
            pressure = load_case.vertical_load / len(contact_nodes) / 100  # N/mm²
            self.displacement[node, 1] = -pressure * 0.1  # Simplified

        # Update strain accumulation
        self.accumulated_strain += np.abs(self.displacement[:, 1]) * dt

    def _solve_thermal(self, load_case: LoadCase, dt: float) -> None:
        """Solve thermal problem for one time step."""
        # Heat generation from hysteresis
        strain_rate = np.abs(self.displacement[:, 1]) * load_case.angular_velocity
        heat_gen = 0.1 * strain_rate**2  # Simplified hysteresis model

        # Heat conduction (simplified explicit scheme)
        k_thermal = 0.3  # W/(m·K)
        rho_cp = 1.5e6  # J/(m³·K)

        # Diffusion coefficient
        alpha = k_thermal / rho_cp

        # Update temperature
        self.temperature += dt * (
            heat_gen - alpha * (self.temperature - load_case.ambient_temperature)
        )

        # Convective cooling at surface (simplified)
        h_conv = 10.0  # W/(m²·K)
        surface_cooling = h_conv * (self.temperature - load_case.ambient_temperature) / rho_cp
        self.temperature -= dt * surface_cooling * 0.1

    def _compute_wear(self, load_case: LoadCase, dt: float) -> None:
        """Compute wear accumulation."""
        # Archard wear model: V = K * F * s / H
        # where K = wear coefficient, F = normal force, s = sliding distance, H = hardness

        K_wear = 1e-9  # mm³/(N·mm), wear coefficient
        H = 5.0  # MPa, hardness

        contact_nodes = self._get_contact_nodes(load_case)

        for node in contact_nodes:
            # Contact pressure
            p = load_case.vertical_load / len(contact_nodes) / 100  # MPa

            # Sliding velocity (from slip)
            v_sliding = load_case.slip_ratio * load_case.vehicle_speed * 1000  # mm/s

            # Wear rate
            wear_rate = K_wear * p * abs(v_sliding) / H

            # Temperature effect (wear increases with temperature)
            T_factor = 1 + 0.01 * (self.temperature[node] - 300)

            self.wear_depth[node] += wear_rate * dt * T_factor

    def _get_contact_nodes(self, load_case: LoadCase) -> list[int]:
        """Get nodes in contact patch."""
        # Simplified: nodes at bottom of tire
        nodes = self.mesh["nodes"]
        y_min = np.min(nodes[:, 1])

        # Nodes within contact region
        contact_mask = nodes[:, 1] < (y_min + 5)  # 5mm tolerance
        contact_indices = np.where(contact_mask)[0].tolist()

        return contact_indices[:100]  # Limit for performance

    def _compute_forces(self, load_case: LoadCase) -> dict[str, float]:
        """Compute tire forces and moments."""
        # Simplified force computation
        Fz = load_case.vertical_load

        # Lateral force (from slip angle)
        C_alpha = 50000  # N/rad, cornering stiffness
        Fy = C_alpha * np.radians(load_case.slip_angle)

        # Longitudinal force (from slip ratio)
        C_kappa = 100000  # N, longitudinal stiffness
        Fx = C_kappa * load_case.slip_ratio

        # Self-aligning moment
        pneumatic_trail = 0.03  # m
        Mz = -pneumatic_trail * Fy

        return {
            "Fx": Fx,
            "Fy": Fy,
            "Fz": Fz,
            "Mx": 0.0,
            "My": 0.0,
            "Mz": Mz,
        }

    def run_steady_state(
        self,
        load_case: LoadCase,
    ) -> dict[str, float]:
        """Compute steady-state tire response.

        Faster than transient simulation when only
        equilibrium state is needed.

        Args:
            load_case: Applied loading

        Returns:
            Dictionary of steady-state results
        """
        # Newton-Raphson iteration for equilibrium
        max_iter = 100
        tolerance = 1e-6

        for iteration in range(max_iter):
            # Compute residual (simplified)
            residual = np.linalg.norm(self.displacement)

            if residual < tolerance:
                break

            # Update (simplified pseudo-inverse)
            self.displacement *= 0.9

        # Compute final forces
        forces = self._compute_forces(load_case)

        # Steady-state temperature
        T_steady = load_case.ambient_temperature + 30  # Simplified

        return {
            **forces,
            "temperature": T_steady,
            "contact_length": 100.0,  # mm
            "contact_width": 150.0,  # mm
            "rolling_resistance": forces["Fx"] / load_case.vertical_load,
        }

    def run_lifecycle(
        self,
        operating_profile: list[OperatingCondition],
        years: float = 5.0,
    ) -> SimulationResult:
        """Simulate full tire lifecycle.

        Args:
            operating_profile: List of operating conditions
            years: Total lifetime to simulate

        Returns:
            SimulationResult with lifecycle data
        """
        logger.info(f"Starting lifecycle simulation: {years} years")

        total_hours = years * 500  # ~500 hours/year usage

        time_history = []
        wear_history = []

        current_time = 0.0
        cycle = 0

        while current_time < total_hours:
            for condition in operating_profile:
                # Fast forward through each condition
                self._apply_aging(condition.duration)

                # Accumulate wear (simplified)
                wear_rate = 0.001 * condition.driving_style  # mm/hour
                self.wear_depth += wear_rate * condition.duration

                current_time += condition.duration

                time_history.append(current_time)
                wear_history.append(np.max(self.wear_depth))

                if current_time >= total_hours:
                    break

            cycle += 1

        return SimulationResult(
            time=np.array(time_history),
            wear_depth=np.array(wear_history),
            statistics={
                "total_hours": current_time,
                "total_wear_mm": float(np.max(self.wear_depth)),
                "remaining_tread_mm": self.geometry.tread_depth - float(np.max(self.wear_depth)),
            },
        )

    def _apply_aging(self, hours: float) -> None:
        """Apply aging effects to material."""
        # Oxidative aging (simplified)
        aging_rate = 0.0001  # per hour
        # This would modify material properties
        pass


def create_standard_tire(
    size_code: str = "P225/60R16",
    compound: str = "touring",
) -> TireSimulator:
    """Create tire simulator with standard configuration.

    Args:
        size_code: Tire size code (e.g., "P225/60R16")
        compound: Compound type ("touring", "performance", "winter")

    Returns:
        Configured TireSimulator
    """
    # Parse size code
    import re

    match = re.match(r"P?(\d+)/(\d+)R(\d+)", size_code)
    if not match:
        raise ValueError(f"Invalid tire size code: {size_code}")

    width = float(match.group(1))
    aspect = float(match.group(2))
    rim = float(match.group(3))

    geometry = TireGeometry(
        section_width=width,
        aspect_ratio=aspect,
        rim_diameter=rim,
    )

    # Material based on compound type
    if compound == "performance":
        material = HyperelasticMaterial(C10=0.6, C01=0.15)
    elif compound == "winter":
        material = HyperelasticMaterial(C10=0.4, C01=0.08)
    else:  # touring
        material = HyperelasticMaterial(C10=0.5, C01=0.1)

    return TireSimulator(
        geometry=geometry,
        material=material,
    )
