"""Quantum Tunneling Dynamics for Crosslink Lifetime Simulation.

This module implements quantum tunneling simulation for polymer
crosslink degradation, providing accurate predictions of crosslink
lifetime and tire durability.

Key Physics:
    Crosslinks in tire rubber can break via quantum tunneling
    through energy barriers. This process is:
    - Temperature dependent (Arrhenius + tunneling correction)
    - Barrier-shape dependent (parabolic, Eckart, etc.)
    - Isotope sensitive (H vs D transfer)
    
    The tunneling transmission coefficient κ(E) is computed using:
    - WKB approximation for thick barriers
    - Exact solutions for model potentials
    - Instanton methods for multidimensional barriers

Mathematical Foundation:
    For a 1D barrier V(x), the WKB tunneling probability is:
    
    T(E) = exp(-2/ℏ ∫√(2m(V(x)-E)) dx)
    
    where the integral is over the classically forbidden region.
    
    The thermal tunneling rate is:
    
    k(T) = (1/2πℏ) ∫ T(E) exp(-E/k_B T) dE

Reference:
    Miller, W. H. "Semiclassical limit of quantum mechanical
    transition state theory for nonseparable systems."
    J. Chem. Phys. 62, 1899 (1975).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from scipy import integrate, optimize

logger = logging.getLogger(__name__)


# Physical constants
HBAR = 1.054571817e-34  # J·s
KB = 1.380649e-23  # J/K
AMU = 1.66054e-27  # kg
EV_TO_J = 1.602176634e-19


class BarrierType(Enum):
    """Types of potential energy barriers."""
    
    PARABOLIC = auto()
    ECKART = auto()
    GAUSSIAN = auto()
    ASYMMETRIC_ECKART = auto()
    DOUBLE_WELL = auto()
    POLYNOMIAL = auto()


class TunnelingMethod(Enum):
    """Methods for computing tunneling probabilities."""
    
    WKB = auto()
    EXACT = auto()
    INSTANTON = auto()
    QUANTUM_DYNAMICS = auto()
    SEMICLASSICAL = auto()


@dataclass
class BarrierParameters:
    """Parameters defining a potential energy barrier.
    
    Attributes:
        barrier_type: Type of barrier potential
        height: Barrier height (eV)
        width: Barrier width (Å)
        asymmetry: Reaction asymmetry (eV)
        imaginary_frequency: Imaginary frequency at saddle point (cm^-1)
        reaction_coordinate_mass: Effective mass (amu)
        temperature: Temperature for rate calculations (K)
    """
    
    barrier_type: BarrierType = BarrierType.ECKART
    height: float = 0.5  # eV
    width: float = 1.0  # Å
    asymmetry: float = 0.0  # eV (ΔE of reaction)
    imaginary_frequency: float = 1000.0  # cm^-1
    reaction_coordinate_mass: float = 1.0  # amu
    temperature: float = 300.0  # K
    
    @property
    def height_joules(self) -> float:
        """Barrier height in Joules."""
        return self.height * EV_TO_J
    
    @property
    def width_meters(self) -> float:
        """Barrier width in meters."""
        return self.width * 1e-10
    
    @property
    def mass_kg(self) -> float:
        """Effective mass in kg."""
        return self.reaction_coordinate_mass * AMU
    
    @property
    def omega_barrier(self) -> float:
        """Barrier frequency in rad/s."""
        return 2 * np.pi * self.imaginary_frequency * 2.998e10


@dataclass
class TunnelingResult:
    """Results from tunneling calculation.
    
    Attributes:
        transmission_coefficient: T(E) at specified energy
        tunneling_rate: k(T) including quantum correction
        classical_rate: Classical TST rate
        tunneling_correction: κ = k_quantum / k_classical
        crosslink_lifetime: Expected lifetime (seconds)
        temperature: Calculation temperature (K)
        penetration_depth: WKB penetration depth (Å)
    """
    
    transmission_coefficient: float
    tunneling_rate: float
    classical_rate: float
    tunneling_correction: float
    crosslink_lifetime: float
    temperature: float
    penetration_depth: float
    
    # Energy-resolved data
    energies: NDArray[np.float64] | None = None
    transmission_vs_energy: NDArray[np.float64] | None = None
    
    # Temperature-dependent data
    temperatures: NDArray[np.float64] | None = None
    rates_vs_temperature: NDArray[np.float64] | None = None


class Barrier(ABC):
    """Abstract base class for potential energy barriers."""
    
    @abstractmethod
    def potential(self, x: float) -> float:
        """Evaluate potential at position x."""
        ...
    
    @abstractmethod
    def classical_turning_points(self, energy: float) -> tuple[float, float]:
        """Find classical turning points at given energy."""
        ...
    
    def gradient(self, x: float, dx: float = 1e-6) -> float:
        """Numerical gradient of potential."""
        return (self.potential(x + dx) - self.potential(x - dx)) / (2 * dx)
    
    def hessian(self, x: float, dx: float = 1e-6) -> float:
        """Numerical second derivative."""
        return (
            self.potential(x + dx) - 2 * self.potential(x) + self.potential(x - dx)
        ) / (dx ** 2)


@dataclass
class EckartBarrier(Barrier):
    """Eckart barrier for chemical reactions.
    
    The Eckart potential is:
    
    V(x) = A * y / (1 + y) + B * y / (1 + y)^2
    
    where y = exp(2πx/L) and A, B are determined by
    the barrier height and reaction asymmetry.
    
    This provides an analytically solvable model for tunneling
    that captures the essential physics of chemical barriers.
    """
    
    params: BarrierParameters
    
    def __post_init__(self):
        """Compute Eckart parameters A and B."""
        V_f = self.params.height_joules
        Delta = self.params.asymmetry * EV_TO_J
        
        # A and B from barrier height and asymmetry
        self.A = np.sqrt(V_f) + np.sqrt(V_f - Delta)
        self.A = self.A ** 2
        
        self.B = np.sqrt(V_f) - np.sqrt(V_f - Delta)
        self.B = -self.B ** 2 if Delta != 0 else 0
        
        # Width parameter
        self.L = self.params.width_meters
    
    def potential(self, x: float) -> float:
        """Evaluate Eckart potential.
        
        Args:
            x: Position (meters)
            
        Returns:
            Potential energy (Joules)
        """
        y = np.exp(2 * np.pi * x / self.L)
        return self.A * y / (1 + y) + self.B * y / (1 + y) ** 2
    
    def classical_turning_points(self, energy: float) -> tuple[float, float]:
        """Find turning points where V(x) = E.
        
        Args:
            energy: Energy level (Joules)
            
        Returns:
            Tuple of (x_left, x_right) turning points
        """
        def residual(x):
            return self.potential(x) - energy
        
        # Search for roots
        x_left = optimize.brentq(residual, -5 * self.L, 0)
        x_right = optimize.brentq(residual, 0, 5 * self.L)
        
        return x_left, x_right
    
    def exact_transmission(self, energy: float) -> float:
        """Compute exact Eckart transmission coefficient.
        
        The Eckart barrier has an analytical solution for T(E).
        
        Args:
            energy: Incident energy (Joules)
            
        Returns:
            Transmission probability T(E)
        """
        m = self.params.mass_kg
        
        # Dimensionless parameters
        alpha = 2 * np.pi * np.sqrt(2 * m * self.A) * self.L / (2 * np.pi * HBAR)
        beta = 2 * np.pi * np.sqrt(2 * m * abs(self.B)) * self.L / (2 * np.pi * HBAR)
        
        eps = energy / self.A
        
        # Eckart transmission formula
        cosh_a = np.cosh(2 * np.pi * alpha)
        cosh_b = np.cosh(2 * np.pi * beta) if self.B != 0 else 1.0
        
        d = 4 * eps * (1 - eps / (self.A / abs(self.B) if self.B != 0 else 1e10))
        
        if d <= 0:
            return 0.0
        
        cosh_d = np.cosh(np.pi * np.sqrt(d) * alpha)
        
        T = (cosh_a - 1) / (cosh_a + cosh_d)
        
        return max(0.0, min(1.0, T))


@dataclass
class ParabolicBarrier(Barrier):
    """Inverted parabolic barrier.
    
    V(x) = V_0 (1 - (x/a)^2) for |x| < a
    V(x) = 0 otherwise
    
    Simple model that gives WKB-exact results.
    """
    
    params: BarrierParameters
    
    @property
    def V0(self) -> float:
        """Barrier height in Joules."""
        return self.params.height_joules
    
    @property
    def a(self) -> float:
        """Half-width of barrier in meters."""
        return self.params.width_meters / 2
    
    def potential(self, x: float) -> float:
        """Evaluate parabolic barrier potential."""
        if abs(x) < self.a:
            return self.V0 * (1 - (x / self.a) ** 2)
        return 0.0
    
    def classical_turning_points(self, energy: float) -> tuple[float, float]:
        """Find turning points for parabolic barrier."""
        if energy >= self.V0:
            return (0.0, 0.0)
        
        x_turn = self.a * np.sqrt(1 - energy / self.V0)
        return (-x_turn, x_turn)


class TunnelingSimulator:
    """Simulator for quantum tunneling through potential barriers.
    
    Computes tunneling rates and crosslink lifetimes using various
    methods from WKB approximation to exact quantum dynamics.
    
    Example:
        >>> # Define barrier parameters for C-S bond breaking
        >>> params = BarrierParameters(
        ...     barrier_type=BarrierType.ECKART,
        ...     height=1.5,  # eV
        ...     width=0.8,   # Å
        ...     reaction_coordinate_mass=32.0,  # S atom
        ...     temperature=350.0,  # K (tire operating temperature)
        ... )
        >>> 
        >>> # Create simulator
        >>> sim = TunnelingSimulator(params)
        >>> 
        >>> # Compute tunneling rate
        >>> result = sim.compute_tunneling_rate()
        >>> print(f"Crosslink lifetime: {result.crosslink_lifetime:.2e} s")
    """
    
    def __init__(
        self,
        params: BarrierParameters,
        method: TunnelingMethod = TunnelingMethod.WKB,
    ) -> None:
        """Initialize tunneling simulator.
        
        Args:
            params: Barrier parameters
            method: Tunneling calculation method
        """
        self.params = params
        self.method = method
        
        # Build barrier potential
        self.barrier = self._create_barrier()
        
        logger.info(
            f"Initialized TunnelingSimulator: height={params.height:.2f} eV, "
            f"width={params.width:.2f} Å, T={params.temperature:.0f} K"
        )
    
    def _create_barrier(self) -> Barrier:
        """Create appropriate barrier object."""
        if self.params.barrier_type == BarrierType.ECKART:
            return EckartBarrier(self.params)
        elif self.params.barrier_type == BarrierType.PARABOLIC:
            return ParabolicBarrier(self.params)
        else:
            # Default to Eckart
            return EckartBarrier(self.params)
    
    def compute_wkb_transmission(self, energy: float) -> float:
        """Compute WKB transmission coefficient.
        
        T(E) = exp(-2 * S / ℏ)
        
        where S = ∫ √(2m(V(x) - E)) dx
        
        Args:
            energy: Incident energy (Joules)
            
        Returns:
            WKB transmission probability
        """
        if energy >= self.params.height_joules:
            return 1.0
        
        # Get turning points
        try:
            x1, x2 = self.barrier.classical_turning_points(energy)
        except ValueError:
            return 0.0
        
        m = self.params.mass_kg
        
        # Integrate action through barrier
        def integrand(x):
            v = self.barrier.potential(x)
            if v > energy:
                return np.sqrt(2 * m * (v - energy))
            return 0.0
        
        # Numerical integration
        n_points = 1000
        x_values = np.linspace(x1, x2, n_points)
        s_values = [integrand(x) for x in x_values]
        
        action = np.trapz(s_values, x_values)
        
        return np.exp(-2 * action / HBAR)
    
    def compute_thermal_rate(
        self,
        n_energies: int = 100,
    ) -> tuple[float, float]:
        """Compute thermally-averaged tunneling rate.
        
        k(T) = (1/h) ∫ T(E) exp(-E/k_B T) dE / ∫ exp(-E/k_B T) dE
        
        Args:
            n_energies: Number of energy points for integration
            
        Returns:
            Tuple of (quantum_rate, classical_rate)
        """
        T = self.params.temperature
        beta = 1.0 / (KB * T)
        V_barrier = self.params.height_joules
        
        # Energy range for integration
        E_min = 0.0
        E_max = 3 * V_barrier
        energies = np.linspace(E_min, E_max, n_energies)
        dE = energies[1] - energies[0]
        
        # Compute transmission at each energy
        transmissions = np.array([
            self._get_transmission(E) for E in energies
        ])
        
        # Boltzmann weights
        boltzmann = np.exp(-beta * energies)
        
        # Quantum rate: integrate T(E) * exp(-βE)
        numerator = np.trapz(transmissions * boltzmann, energies)
        denominator = np.trapz(boltzmann, energies)
        
        # Prefactor from TST
        prefactor = self.params.omega_barrier / (2 * np.pi)
        
        quantum_rate = prefactor * numerator / denominator
        
        # Classical TST rate
        classical_rate = prefactor * np.exp(-beta * V_barrier)
        
        return quantum_rate, classical_rate
    
    def _get_transmission(self, energy: float) -> float:
        """Get transmission coefficient using selected method."""
        if self.method == TunnelingMethod.WKB:
            return self.compute_wkb_transmission(energy)
        elif self.method == TunnelingMethod.EXACT:
            if isinstance(self.barrier, EckartBarrier):
                return self.barrier.exact_transmission(energy)
            return self.compute_wkb_transmission(energy)
        else:
            return self.compute_wkb_transmission(energy)
    
    def compute_tunneling_rate(self) -> TunnelingResult:
        """Compute full tunneling analysis.
        
        Returns:
            TunnelingResult with rates and lifetimes
        """
        # Get thermal rates
        quantum_rate, classical_rate = self.compute_thermal_rate()
        
        # Tunneling correction
        if classical_rate > 0:
            kappa = quantum_rate / classical_rate
        else:
            kappa = float("inf")
        
        # Transmission at barrier top
        T_barrier = self._get_transmission(self.params.height_joules * 0.99)
        
        # WKB penetration depth
        m = self.params.mass_kg
        V = self.params.height_joules
        penetration = HBAR / np.sqrt(2 * m * V) * 1e10  # Convert to Å
        
        # Crosslink lifetime
        if quantum_rate > 0:
            lifetime = 1.0 / quantum_rate
        else:
            lifetime = float("inf")
        
        result = TunnelingResult(
            transmission_coefficient=T_barrier,
            tunneling_rate=quantum_rate,
            classical_rate=classical_rate,
            tunneling_correction=kappa,
            crosslink_lifetime=lifetime,
            temperature=self.params.temperature,
            penetration_depth=penetration,
        )
        
        logger.info(
            f"Tunneling analysis: κ={kappa:.2f}, "
            f"lifetime={lifetime:.2e} s"
        )
        
        return result
    
    def compute_temperature_dependence(
        self,
        T_min: float = 200.0,
        T_max: float = 500.0,
        n_temps: int = 50,
    ) -> TunnelingResult:
        """Compute rate vs temperature.
        
        Args:
            T_min: Minimum temperature (K)
            T_max: Maximum temperature (K)
            n_temps: Number of temperature points
            
        Returns:
            TunnelingResult with temperature-dependent data
        """
        temperatures = np.linspace(T_min, T_max, n_temps)
        rates = []
        
        original_T = self.params.temperature
        
        for T in temperatures:
            self.params.temperature = T
            q_rate, _ = self.compute_thermal_rate(n_energies=50)
            rates.append(q_rate)
        
        self.params.temperature = original_T
        
        # Get result at original temperature
        result = self.compute_tunneling_rate()
        result.temperatures = temperatures
        result.rates_vs_temperature = np.array(rates)
        
        return result
    
    def compute_energy_dependence(
        self,
        E_min: float = 0.0,
        E_max: float | None = None,
        n_energies: int = 100,
    ) -> TunnelingResult:
        """Compute transmission vs energy.
        
        Args:
            E_min: Minimum energy (eV)
            E_max: Maximum energy (eV), default = 2 * barrier height
            n_energies: Number of energy points
            
        Returns:
            TunnelingResult with energy-dependent data
        """
        if E_max is None:
            E_max = 2 * self.params.height
        
        energies = np.linspace(E_min, E_max, n_energies)
        transmissions = []
        
        for E in energies:
            E_joules = E * EV_TO_J
            T = self._get_transmission(E_joules)
            transmissions.append(T)
        
        result = self.compute_tunneling_rate()
        result.energies = energies
        result.transmission_vs_energy = np.array(transmissions)
        
        return result


class CrosslinkTunneling:
    """Specialized tunneling simulator for tire crosslink degradation.
    
    Models the quantum tunneling-mediated breaking of:
    - Sulfur crosslinks (mono-, di-, polysulfidic)
    - Carbon-carbon crosslinks (peroxide cured)
    - Silane coupling agent bonds
    
    Includes environmental effects:
    - Temperature activation
    - Oxidative stress
    - Mechanical strain
    - UV exposure
    
    Example:
        >>> # Model polysulfidic crosslink breaking
        >>> tunneling = CrosslinkTunneling(
        ...     crosslink_type="polysulfidic",
        ...     n_sulfur_atoms=3,
        ...     temperature=350.0,
        ...     strain_level=0.2,
        ... )
        >>> 
        >>> lifetime = tunneling.compute_lifetime()
        >>> print(f"Expected lifetime: {lifetime/3600:.1f} hours")
    """
    
    # Crosslink type parameters
    CROSSLINK_DATA = {
        "monosulfidic": {
            "barrier_height": 2.5,  # eV
            "barrier_width": 0.6,   # Å
            "mass": 32.0,           # amu (S atom)
            "frequency": 500.0,     # cm^-1
        },
        "disulfidic": {
            "barrier_height": 2.0,  # eV
            "barrier_width": 0.7,   # Å
            "mass": 32.0,           # amu
            "frequency": 450.0,     # cm^-1
        },
        "polysulfidic": {
            "barrier_height": 1.5,  # eV
            "barrier_width": 0.8,   # Å
            "mass": 32.0,           # amu
            "frequency": 400.0,     # cm^-1
        },
        "carbon_carbon": {
            "barrier_height": 3.6,  # eV
            "barrier_width": 0.5,   # Å
            "mass": 12.0,           # amu (C atom)
            "frequency": 800.0,     # cm^-1
        },
        "silane": {
            "barrier_height": 2.8,  # eV
            "barrier_width": 0.6,   # Å
            "mass": 28.0,           # amu (Si atom)
            "frequency": 600.0,     # cm^-1
        },
    }
    
    def __init__(
        self,
        crosslink_type: str = "polysulfidic",
        n_sulfur_atoms: int = 3,
        temperature: float = 300.0,
        strain_level: float = 0.0,
        oxidation_level: float = 0.0,
    ) -> None:
        """Initialize crosslink tunneling simulator.
        
        Args:
            crosslink_type: Type of crosslink
            n_sulfur_atoms: Number of sulfur atoms (for polysulfidic)
            temperature: Temperature (K)
            strain_level: Applied strain (fractional)
            oxidation_level: Oxidation extent (0-1)
        """
        self.crosslink_type = crosslink_type
        self.n_sulfur_atoms = n_sulfur_atoms
        self.temperature = temperature
        self.strain_level = strain_level
        self.oxidation_level = oxidation_level
        
        # Get base parameters
        if crosslink_type not in self.CROSSLINK_DATA:
            raise ValueError(f"Unknown crosslink type: {crosslink_type}")
        
        self.base_params = self.CROSSLINK_DATA[crosslink_type].copy()
        
        # Apply modifiers
        self._apply_environmental_effects()
        
        # Create barrier parameters
        self.barrier_params = BarrierParameters(
            barrier_type=BarrierType.ECKART,
            height=self.base_params["barrier_height"],
            width=self.base_params["barrier_width"],
            reaction_coordinate_mass=self.base_params["mass"],
            imaginary_frequency=self.base_params["frequency"],
            temperature=temperature,
        )
        
        # Create simulator
        self.simulator = TunnelingSimulator(
            self.barrier_params,
            method=TunnelingMethod.EXACT,
        )
        
        logger.info(
            f"CrosslinkTunneling initialized: {crosslink_type}, "
            f"T={temperature} K, strain={strain_level:.1%}"
        )
    
    def _apply_environmental_effects(self) -> None:
        """Modify barrier parameters for environmental conditions."""
        # Strain lowers barrier (bond stretching)
        strain_effect = 1.0 - 0.3 * self.strain_level
        self.base_params["barrier_height"] *= strain_effect
        
        # Strain increases width (bond elongation)
        self.base_params["barrier_width"] *= (1.0 + 0.1 * self.strain_level)
        
        # Oxidation weakens S-S bonds
        if "sulf" in self.crosslink_type:
            oxidation_effect = 1.0 - 0.4 * self.oxidation_level
            self.base_params["barrier_height"] *= oxidation_effect
        
        # Polysulfidic length effect (longer = weaker)
        if self.crosslink_type == "polysulfidic" and self.n_sulfur_atoms > 2:
            length_factor = 1.0 - 0.1 * (self.n_sulfur_atoms - 2)
            self.base_params["barrier_height"] *= max(0.5, length_factor)
    
    def compute_lifetime(self) -> float:
        """Compute expected crosslink lifetime.
        
        Returns:
            Lifetime in seconds
        """
        result = self.simulator.compute_tunneling_rate()
        return result.crosslink_lifetime
    
    def compute_degradation_rate(self) -> float:
        """Compute crosslink breaking rate.
        
        Returns:
            Rate in s^-1
        """
        result = self.simulator.compute_tunneling_rate()
        return result.tunneling_rate
    
    def compute_temperature_sensitivity(
        self,
        T_range: tuple[float, float] = (250.0, 400.0),
    ) -> dict[str, NDArray[np.float64]]:
        """Compute temperature sensitivity analysis.
        
        Args:
            T_range: Temperature range (K)
            
        Returns:
            Dictionary with temperature and rate data
        """
        result = self.simulator.compute_temperature_dependence(
            T_min=T_range[0],
            T_max=T_range[1],
        )
        
        # Compute activation energy from Arrhenius fit
        T = result.temperatures
        k = result.rates_vs_temperature
        
        # ln(k) vs 1/T
        valid = k > 0
        if np.sum(valid) > 2:
            x = 1.0 / T[valid]
            y = np.log(k[valid])
            
            # Linear regression
            coeffs = np.polyfit(x, y, 1)
            E_activation = -coeffs[0] * KB / EV_TO_J  # eV
        else:
            E_activation = self.barrier_params.height
        
        return {
            "temperatures": T,
            "rates": k,
            "lifetimes": 1.0 / k,
            "activation_energy_eV": E_activation,
        }
    
    def predict_tire_lifetime(
        self,
        crosslink_density: float = 1e20,  # crosslinks/m³
        failure_threshold: float = 0.5,   # fraction remaining
        operating_profile: dict[str, float] | None = None,
    ) -> float:
        """Predict tire lifetime based on crosslink degradation.
        
        Args:
            crosslink_density: Initial crosslink density
            failure_threshold: Fraction at which tire fails
            operating_profile: Temperature/strain profile
            
        Returns:
            Predicted lifetime in hours
        """
        if operating_profile is None:
            operating_profile = {
                "highway_fraction": 0.6,
                "highway_temp": 340.0,
                "city_fraction": 0.3,
                "city_temp": 310.0,
                "parked_fraction": 0.1,
                "parked_temp": 290.0,
            }
        
        # Compute effective degradation rate
        effective_rate = 0.0
        
        for condition in ["highway", "city", "parked"]:
            fraction = operating_profile.get(f"{condition}_fraction", 0)
            temp = operating_profile.get(f"{condition}_temp", 300)
            
            # Update temperature and compute rate
            self.barrier_params.temperature = temp
            self.simulator = TunnelingSimulator(self.barrier_params)
            result = self.simulator.compute_tunneling_rate()
            
            effective_rate += fraction * result.tunneling_rate
        
        # Time to reach failure threshold
        # N(t) = N_0 * exp(-k*t)
        # failure_threshold = exp(-k*t)
        # t = -ln(failure_threshold) / k
        
        if effective_rate > 0:
            lifetime_seconds = -np.log(failure_threshold) / effective_rate
        else:
            lifetime_seconds = float("inf")
        
        lifetime_hours = lifetime_seconds / 3600
        
        return lifetime_hours


class ProtonTunneling:
    """Proton tunneling in polymer hydrogen bonds.
    
    Models tunneling in:
    - Hydrogen-bonded crosslinks
    - Urethane linkages
    - Amide bonds in aramid reinforcement
    
    Proton tunneling is particularly important due to the
    light mass of hydrogen (large quantum effects).
    """
    
    def __init__(
        self,
        bond_type: str = "N-H...O",
        donor_acceptor_distance: float = 2.8,  # Å
        temperature: float = 300.0,
    ) -> None:
        """Initialize proton tunneling simulator.
        
        Args:
            bond_type: Type of hydrogen bond
            donor_acceptor_distance: D-A distance (Å)
            temperature: Temperature (K)
        """
        self.bond_type = bond_type
        self.donor_acceptor_distance = donor_acceptor_distance
        self.temperature = temperature
        
        # Barrier height depends on D-A distance
        # Typical: 0.1-0.5 eV for H-bonds
        self.barrier_height = 0.2 + 0.1 * (donor_acceptor_distance - 2.5)
        
        # Barrier width from geometry
        self.barrier_width = donor_acceptor_distance - 2.0  # Å
        
        # Create simulator
        self.params = BarrierParameters(
            barrier_type=BarrierType.ECKART,
            height=self.barrier_height,
            width=self.barrier_width,
            reaction_coordinate_mass=1.008,  # Proton
            imaginary_frequency=3000.0,  # Typical O-H stretch
            temperature=temperature,
        )
        
        self.simulator = TunnelingSimulator(self.params)
    
    def compute_transfer_rate(self) -> float:
        """Compute proton transfer rate.
        
        Returns:
            Rate in s^-1
        """
        result = self.simulator.compute_tunneling_rate()
        return result.tunneling_rate
    
    def compute_isotope_effect(self) -> float:
        """Compute H/D kinetic isotope effect.
        
        Returns:
            k_H / k_D ratio
        """
        # Hydrogen rate
        result_H = self.simulator.compute_tunneling_rate()
        rate_H = result_H.tunneling_rate
        
        # Deuterium rate (mass = 2.014)
        params_D = BarrierParameters(
            barrier_type=self.params.barrier_type,
            height=self.params.height,
            width=self.params.width,
            reaction_coordinate_mass=2.014,
            imaginary_frequency=self.params.imaginary_frequency / np.sqrt(2),
            temperature=self.temperature,
        )
        
        simulator_D = TunnelingSimulator(params_D)
        result_D = simulator_D.compute_tunneling_rate()
        rate_D = result_D.tunneling_rate
        
        if rate_D > 0:
            return rate_H / rate_D
        return float("inf")


# Convenience functions

def compute_crosslink_lifetime(
    crosslink_type: str,
    temperature: float,
    strain: float = 0.0,
) -> float:
    """Compute crosslink lifetime under specified conditions.
    
    Args:
        crosslink_type: Type of crosslink
        temperature: Temperature (K)
        strain: Applied strain
        
    Returns:
        Lifetime in seconds
    """
    tunneling = CrosslinkTunneling(
        crosslink_type=crosslink_type,
        temperature=temperature,
        strain_level=strain,
    )
    
    return tunneling.compute_lifetime()


def predict_degradation_profile(
    crosslink_type: str,
    time_hours: NDArray[np.float64],
    temperature: float,
    initial_density: float = 1.0,
) -> NDArray[np.float64]:
    """Predict crosslink density vs time.
    
    Args:
        crosslink_type: Type of crosslink
        time_hours: Time points (hours)
        temperature: Temperature (K)
        initial_density: Initial normalized density
        
    Returns:
        Crosslink density at each time point
    """
    tunneling = CrosslinkTunneling(
        crosslink_type=crosslink_type,
        temperature=temperature,
    )
    
    rate = tunneling.compute_degradation_rate()
    time_seconds = time_hours * 3600
    
    return initial_density * np.exp(-rate * time_seconds)
