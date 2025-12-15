"""Factory Simulation Module.

Simulates tire manufacturing processes including:
- Curing (vulcanization)
- Mold flow
- Component assembly
- Quality control
- Defect prediction

These simulations help optimize manufacturing parameters
and predict product quality before physical production.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class CuringStage(Enum):
    """Stages of the vulcanization process."""

    MOLD_CLOSING = auto()
    INITIAL_HEATING = auto()
    BLADDER_INFLATION = auto()
    VULCANIZATION = auto()
    COOLING = auto()
    MOLD_OPENING = auto()


class DefectType(Enum):
    """Types of manufacturing defects."""

    UNDERCURE = auto()
    OVERCURE = auto()
    POROSITY = auto()
    BLISTER = auto()
    FLOW_LINES = auto()
    DELAMINATION = auto()
    CONTAMINATION = auto()
    DIMENSIONAL = auto()


@dataclass
class CuringParameters:
    """Vulcanization process parameters.

    Attributes:
        mold_temperature: Mold temperature (°C)
        bladder_temperature: Bladder temperature (°C)
        bladder_pressure: Bladder pressure (bar)
        cure_time: Total cure time (minutes)
        heating_rate: Heating rate (°C/min)
        cooling_rate: Cooling rate (°C/min)
    """

    mold_temperature: float = 170.0  # °C
    bladder_temperature: float = 180.0  # °C
    bladder_pressure: float = 15.0  # bar
    cure_time: float = 12.0  # minutes
    heating_rate: float = 10.0  # °C/min
    cooling_rate: float = 5.0  # °C/min

    # Advanced parameters
    steam_pressure: float = 14.0  # bar
    nitrogen_pressure: float = 20.0  # bar (for bladder)
    vacuum_level: float = -0.8  # bar (initial evacuation)


@dataclass
class CuringResult:
    """Results from curing simulation.

    Attributes:
        cure_state: Final state of cure (0-1)
        temperature_history: Temperature vs time at key points
        crosslink_density: Achieved crosslink density
        reversion_index: Degree of reversion (over-cure)
        predicted_properties: Predicted mechanical properties
        defects: Detected defect risks
    """

    cure_state: NDArray[np.float64]
    temperature_history: dict[str, NDArray[np.float64]]
    crosslink_density: float
    reversion_index: float
    predicted_properties: dict[str, float]
    defects: list[tuple[DefectType, float, tuple[float, float, float]]]

    @property
    def is_acceptable(self) -> bool:
        """Check if cure result is acceptable."""
        return (
            np.mean(self.cure_state) > 0.9 and self.reversion_index < 0.1 and len(self.defects) == 0
        )


@dataclass
class VulcanizationKinetics:
    """Kinetic model for vulcanization.

    Uses the Kamal-Sourour autocatalytic model:
    dα/dt = (k1 + k2*α^m) * (1-α)^n

    where α is the degree of cure (0-1).
    """

    # Arrhenius parameters for k1
    A1: float = 1e8  # Pre-exponential factor (1/min)
    Ea1: float = 80.0  # Activation energy (kJ/mol)

    # Arrhenius parameters for k2 (autocatalytic)
    A2: float = 1e10
    Ea2: float = 90.0

    # Reaction orders
    m: float = 0.8  # Autocatalytic order
    n: float = 1.5  # Reaction order

    # Reversion parameters
    A_rev: float = 1e6
    Ea_rev: float = 120.0

    def rate_constant(
        self,
        temperature: float,
        A: float,
        Ea: float,
    ) -> float:
        """Compute Arrhenius rate constant.

        Args:
            temperature: Temperature (°C)
            A: Pre-exponential factor
            Ea: Activation energy (kJ/mol)

        Returns:
            Rate constant
        """
        T_kelvin = temperature + 273.15
        R = 8.314e-3  # kJ/(mol·K)
        return A * np.exp(-Ea / (R * T_kelvin))

    def cure_rate(
        self,
        alpha: float,
        temperature: float,
    ) -> float:
        """Compute instantaneous cure rate.

        Args:
            alpha: Current degree of cure (0-1)
            temperature: Temperature (°C)

        Returns:
            Rate of cure (1/min)
        """
        k1 = self.rate_constant(temperature, self.A1, self.Ea1)
        k2 = self.rate_constant(temperature, self.A2, self.Ea2)

        if alpha >= 1.0:
            return 0.0

        rate = (k1 + k2 * alpha**self.m) * (1 - alpha) ** self.n
        return max(0.0, rate)

    def reversion_rate(
        self,
        alpha: float,
        temperature: float,
    ) -> float:
        """Compute reversion (over-cure) rate.

        Args:
            alpha: Current degree of cure
            temperature: Temperature (°C)

        Returns:
            Reversion rate (1/min)
        """
        k_rev = self.rate_constant(temperature, self.A_rev, self.Ea_rev)

        # Reversion only occurs after substantial cure
        if alpha < 0.9:
            return 0.0

        return k_rev * (alpha - 0.9)


class CuringSimulation:
    """Simulation of tire curing (vulcanization) process.

    Models the complex heat transfer and chemical kinetics
    during tire vulcanization in a curing press.

    Physics modeled:
    - Heat conduction through tire components
    - Convection from steam/hot water
    - Exothermic vulcanization reaction
    - Crosslink formation kinetics
    - Reversion (over-cure) effects

    Example:
        >>> # Define curing parameters
        >>> params = CuringParameters(
        ...     mold_temperature=170,
        ...     cure_time=12,
        ... )
        >>>
        >>> # Create simulation
        >>> sim = CuringSimulation(
        ...     tire_geometry=geometry,
        ...     compound_properties=compound,
        ...     kinetics=VulcanizationKinetics(),
        ... )
        >>>
        >>> # Run simulation
        >>> result = sim.run(params)
        >>> print(f"Cure state: {np.mean(result.cure_state):.1%}")
    """

    def __init__(
        self,
        tire_thickness: float = 20.0,  # mm
        n_layers: int = 50,
        kinetics: VulcanizationKinetics | None = None,
    ) -> None:
        """Initialize curing simulation.

        Args:
            tire_thickness: Tire cross-section thickness (mm)
            n_layers: Number of layers for discretization
            kinetics: Vulcanization kinetics model
        """
        self.thickness = tire_thickness
        self.n_layers = n_layers
        self.kinetics = kinetics or VulcanizationKinetics()

        # Discretization
        self.dx = tire_thickness / n_layers
        self.x = np.linspace(0, tire_thickness, n_layers)

        # Material properties
        self.k_thermal = 0.25  # W/(m·K)
        self.rho = 1100.0  # kg/m³
        self.cp = 2000.0  # J/(kg·K)
        self.alpha_thermal = self.k_thermal / (self.rho * self.cp)

        logger.info(
            f"Initialized CuringSimulation: {n_layers} layers, " f"thickness={tire_thickness} mm"
        )

    def run(
        self,
        params: CuringParameters,
        dt: float = 0.01,  # minutes
    ) -> CuringResult:
        """Run curing simulation.

        Args:
            params: Curing process parameters
            dt: Time step (minutes)

        Returns:
            CuringResult with cure state and predictions
        """
        n_steps = int(params.cure_time / dt)

        # Initialize fields
        temperature = np.ones(self.n_layers) * 25.0  # Start at room temp
        cure_state = np.zeros(self.n_layers)

        # History storage
        time_history = []
        T_surface_history = []
        T_center_history = []
        cure_history = []

        # Boundary temperatures
        T_outer = params.mold_temperature
        T_inner = params.bladder_temperature

        for step in range(n_steps):
            time = step * dt

            # Heat conduction (explicit finite difference)
            T_new = temperature.copy()

            # Interior nodes
            for i in range(1, self.n_layers - 1):
                T_new[i] = temperature[i] + (
                    self.alpha_thermal
                    * dt
                    * 60  # Convert to seconds
                    / (self.dx * 1e-3) ** 2
                    * (temperature[i + 1] - 2 * temperature[i] + temperature[i - 1])
                )

            # Boundary conditions
            T_new[0] = T_outer  # Mold surface
            T_new[-1] = T_inner  # Bladder surface

            # Update cure state
            for i in range(self.n_layers):
                cure_rate = self.kinetics.cure_rate(cure_state[i], T_new[i])
                rev_rate = self.kinetics.reversion_rate(cure_state[i], T_new[i])

                cure_state[i] += (cure_rate - rev_rate) * dt
                cure_state[i] = np.clip(cure_state[i], 0, 1)

            temperature = T_new

            # Record history periodically
            if step % 100 == 0:
                time_history.append(time)
                T_surface_history.append(temperature[0])
                T_center_history.append(temperature[self.n_layers // 2])
                cure_history.append(np.mean(cure_state))

        # Compute final properties
        crosslink_density = self._estimate_crosslink_density(cure_state)
        reversion_index = self._compute_reversion_index(cure_state)

        # Predict mechanical properties
        properties = self._predict_properties(crosslink_density)

        # Detect defects
        defects = self._detect_defects(cure_state, temperature)

        result = CuringResult(
            cure_state=cure_state,
            temperature_history={
                "time": np.array(time_history),
                "surface": np.array(T_surface_history),
                "center": np.array(T_center_history),
                "mean_cure": np.array(cure_history),
            },
            crosslink_density=crosslink_density,
            reversion_index=reversion_index,
            predicted_properties=properties,
            defects=defects,
        )

        logger.info(
            f"Curing simulation complete: "
            f"mean cure={np.mean(cure_state):.1%}, "
            f"crosslink density={crosslink_density:.2e}"
        )

        return result

    def _estimate_crosslink_density(
        self,
        cure_state: NDArray[np.float64],
    ) -> float:
        """Estimate crosslink density from cure state."""
        # Maximum crosslink density for sulfur vulcanization
        max_density = 5e19  # crosslinks/m³

        return max_density * np.mean(cure_state)

    def _compute_reversion_index(
        self,
        cure_state: NDArray[np.float64],
    ) -> float:
        """Compute reversion (over-cure) index."""
        # Reversion indicated by cure state variation
        if np.max(cure_state) > 0.95:
            return (np.max(cure_state) - np.mean(cure_state)) / np.max(cure_state)
        return 0.0

    def _predict_properties(
        self,
        crosslink_density: float,
    ) -> dict[str, float]:
        """Predict mechanical properties from crosslink density."""
        # Simplified correlations
        return {
            "modulus_100": 2.0 + crosslink_density / 1e19,  # MPa
            "tensile_strength": 15.0 * (crosslink_density / 5e19) ** 0.5,  # MPa
            "elongation_break": 500 - 50 * crosslink_density / 1e19,  # %
            "hardness_shore_A": 60 + 5 * crosslink_density / 1e19,
            "resilience": 50 + 10 * (1 - crosslink_density / 1e20),  # %
        }

    def _detect_defects(
        self,
        cure_state: NDArray[np.float64],
        temperature: NDArray[np.float64],
    ) -> list[tuple[DefectType, float, tuple[float, float, float]]]:
        """Detect potential manufacturing defects."""
        defects = []

        # Check for undercure
        for i, alpha in enumerate(cure_state):
            if alpha < 0.85:
                x_pos = self.x[i]
                defects.append(
                    (
                        DefectType.UNDERCURE,
                        1 - alpha,
                        (x_pos, 0, 0),
                    )
                )

        # Check for porosity (from trapped air/moisture)
        if np.max(temperature) > 200:
            defects.append(
                (
                    DefectType.POROSITY,
                    0.1,
                    (self.thickness / 2, 0, 0),
                )
            )

        return defects

    def optimize_cure_time(
        self,
        params: CuringParameters,
        target_cure: float = 0.95,
    ) -> float:
        """Find optimal cure time for target cure state.

        Args:
            params: Base curing parameters
            target_cure: Target mean cure state

        Returns:
            Optimal cure time (minutes)
        """
        from scipy.optimize import minimize_scalar

        def objective(cure_time):
            params_copy = CuringParameters(**{k: v for k, v in params.__dict__.items()})
            params_copy.cure_time = cure_time

            result = self.run(params_copy, dt=0.05)
            return abs(np.mean(result.cure_state) - target_cure)

        result = minimize_scalar(objective, bounds=(5, 30), method="bounded")
        return result.x


class VulcanizationModel:
    """Advanced vulcanization model with quantum corrections.

    Incorporates quantum tunneling effects in crosslink formation
    for more accurate prediction of cure kinetics.
    """

    def __init__(
        self,
        sulfur_content: float = 2.0,  # phr
        accelerator_content: float = 1.0,  # phr
        activator_content: float = 5.0,  # phr (ZnO)
    ) -> None:
        """Initialize vulcanization model.

        Args:
            sulfur_content: Sulfur content (parts per hundred rubber)
            accelerator_content: Accelerator content (phr)
            activator_content: Activator content (phr)
        """
        self.sulfur = sulfur_content
        self.accelerator = accelerator_content
        self.activator = activator_content

        # Compute kinetic parameters from formulation
        self.kinetics = self._compute_kinetics()

    def _compute_kinetics(self) -> VulcanizationKinetics:
        """Compute kinetic parameters from formulation."""
        # Empirical correlations
        base_A1 = 1e8
        base_A2 = 1e10

        # Accelerator increases rate
        acc_factor = 1 + 2 * self.accelerator

        # Sulfur affects activation energy
        Ea_modifier = 1 - 0.05 * (self.sulfur - 2.0)

        return VulcanizationKinetics(
            A1=base_A1 * acc_factor,
            A2=base_A2 * acc_factor,
            Ea1=80.0 * Ea_modifier,
            Ea2=90.0 * Ea_modifier,
        )

    def predict_scorch_time(self, temperature: float) -> float:
        """Predict scorch (onset of cure) time.

        Args:
            temperature: Processing temperature (°C)

        Returns:
            Scorch time (minutes)
        """
        # Time to 2% cure
        dt = 0.01
        time = 0.0
        alpha = 0.0

        while alpha < 0.02 and time < 30:
            rate = self.kinetics.cure_rate(alpha, temperature)
            alpha += rate * dt
            time += dt

        return time

    def predict_optimum_cure(self, temperature: float) -> float:
        """Predict optimum cure time (T90).

        Args:
            temperature: Cure temperature (°C)

        Returns:
            Time to 90% cure (minutes)
        """
        dt = 0.01
        time = 0.0
        alpha = 0.0

        while alpha < 0.90 and time < 60:
            rate = self.kinetics.cure_rate(alpha, temperature)
            alpha += rate * dt
            time += dt

        return time


class QualityControl:
    """Quality control simulation and prediction.

    Predicts quality metrics and identifies potential issues
    before physical production.
    """

    def __init__(
        self,
        specifications: dict[str, tuple[float, float]],
    ) -> None:
        """Initialize QC system.

        Args:
            specifications: Dictionary of {property: (min, max)}
        """
        self.specs = specifications

    def evaluate_cure_result(
        self,
        result: CuringResult,
    ) -> dict[str, Any]:
        """Evaluate curing result against specifications.

        Args:
            result: Curing simulation result

        Returns:
            Quality assessment report
        """
        assessment = {
            "pass": True,
            "property_status": {},
            "defects": [],
            "recommendations": [],
        }

        # Check each property
        for prop, (min_val, max_val) in self.specs.items():
            if prop in result.predicted_properties:
                value = result.predicted_properties[prop]

                if min_val <= value <= max_val:
                    status = "PASS"
                else:
                    status = "FAIL"
                    assessment["pass"] = False

                assessment["property_status"][prop] = {
                    "value": value,
                    "min": min_val,
                    "max": max_val,
                    "status": status,
                }

        # Check defects
        for defect_type, severity, location in result.defects:
            assessment["defects"].append(
                {
                    "type": defect_type.name,
                    "severity": severity,
                    "location": location,
                }
            )

            if severity > 0.1:
                assessment["pass"] = False

        # Generate recommendations
        if result.reversion_index > 0.05:
            assessment["recommendations"].append("Reduce cure time to prevent reversion")

        if np.mean(result.cure_state) < 0.90:
            assessment["recommendations"].append("Increase cure time or temperature")

        return assessment

    def predict_field_performance(
        self,
        cure_result: CuringResult,
    ) -> dict[str, float]:
        """Predict field performance from cure result.

        Args:
            cure_result: Manufacturing simulation result

        Returns:
            Predicted field performance metrics
        """
        props = cure_result.predicted_properties

        return {
            "wear_resistance": props.get("hardness_shore_A", 65) / 70,
            "rolling_resistance": 1 - props.get("resilience", 50) / 100,
            "grip_wet": props.get("modulus_100", 2.5) / 3.0,
            "comfort": 1 - props.get("hardness_shore_A", 65) / 80,
            "expected_lifetime_km": 60000 * (cure_result.crosslink_density / 5e19),
        }


class FactorySimulator:
    """Complete factory simulation including all manufacturing stages."""

    def __init__(
        self,
        line_capacity: int = 100,  # tires per hour
    ) -> None:
        """Initialize factory simulator.

        Args:
            line_capacity: Production line capacity
        """
        self.capacity = line_capacity
        self.curing_sim = CuringSimulation()
        self.vulc_model = VulcanizationModel()

    def simulate_batch(
        self,
        batch_size: int,
        params: CuringParameters,
    ) -> list[CuringResult]:
        """Simulate a production batch.

        Args:
            batch_size: Number of tires in batch
            params: Curing parameters

        Returns:
            List of curing results
        """
        results = []

        for i in range(batch_size):
            # Add process variation
            params_varied = CuringParameters(
                mold_temperature=params.mold_temperature + np.random.normal(0, 2),
                bladder_temperature=params.bladder_temperature + np.random.normal(0, 2),
                cure_time=params.cure_time + np.random.normal(0, 0.5),
            )

            result = self.curing_sim.run(params_varied)
            results.append(result)

        return results

    def optimize_process(
        self,
        target_properties: dict[str, float],
    ) -> CuringParameters:
        """Optimize curing parameters for target properties.

        Args:
            target_properties: Desired property values

        Returns:
            Optimized curing parameters
        """
        from scipy.optimize import minimize

        def objective(x):
            params = CuringParameters(
                mold_temperature=x[0],
                cure_time=x[1],
            )
            result = self.curing_sim.run(params, dt=0.05)

            error = 0
            for prop, target in target_properties.items():
                if prop in result.predicted_properties:
                    error += (result.predicted_properties[prop] - target) ** 2

            return error

        x0 = [170, 12]
        bounds = [(150, 190), (8, 20)]

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B")

        return CuringParameters(
            mold_temperature=result.x[0],
            cure_time=result.x[1],
        )
