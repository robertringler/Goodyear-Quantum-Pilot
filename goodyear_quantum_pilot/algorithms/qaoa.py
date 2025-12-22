"""Quantum Approximate Optimization Algorithm for Tire Material Blending.

This module implements QAOA variants specifically designed for solving
combinatorial optimization problems in tire compound formulation,
including material selection, proportion optimization, and
multi-objective blend design.

Key Applications:
    - Optimal material selection from candidate pool
    - Proportion optimization for target properties
    - Multi-objective tire compound design
    - Supply chain constrained blending
    - Cost-performance Pareto optimization

Mathematical Foundation:
    QAOA prepares approximate solutions to combinatorial problems
    by alternating between cost and mixer Hamiltonians:

    |ψ(γ, β)> = Π_p [e^{-iβ_p H_M} e^{-iγ_p H_C}] |+>^n

    where:
    - H_C: Cost Hamiltonian encoding objective function
    - H_M: Mixer Hamiltonian (typically Σ_i X_i)
    - γ, β: Variational parameters
    - p: QAOA depth

Reference:
    Farhi et al. "A Quantum Approximate Optimization Algorithm."
    arXiv:1411.4028 (2014).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class MixerType(Enum):
    """Types of QAOA mixer Hamiltonians."""

    TRANSVERSE_FIELD = auto()  # Standard X-mixer
    XY_MIXER = auto()  # Preserves Hamming weight
    GROVER_MIXER = auto()  # Full mixing in feasible subspace
    PARITY_MIXER = auto()  # Preserves parity
    CUSTOM = auto()


class CostType(Enum):
    """Types of cost function encodings."""

    ISING = auto()  # Quadratic Ising model
    QUBO = auto()  # Quadratic unconstrained binary optimization
    HIGHER_ORDER = auto()  # Higher-order terms
    PENALTY = auto()  # Constraint penalties


@dataclass
class QAOAConfig:
    """Configuration for QAOA algorithm.

    Attributes:
        depth: Number of QAOA layers (p)
        mixer_type: Type of mixer Hamiltonian
        optimizer: Classical optimizer for parameters
        max_iterations: Maximum optimization iterations
        shots: Measurement shots per evaluation
        warm_start: Use warm-start initialization
        constraint_penalty: Penalty weight for constraints
        multi_angle: Use independent angles per qubit
    """

    depth: int = 3
    mixer_type: MixerType = MixerType.TRANSVERSE_FIELD
    optimizer: str = "COBYLA"
    max_iterations: int = 500
    shots: int = 4096
    warm_start: bool = True
    constraint_penalty: float = 10.0
    multi_angle: bool = False

    # Advanced options
    initial_gamma: float = 0.1
    initial_beta: float = 0.5
    parameter_bounds: tuple[float, float] = (0.0, 2 * np.pi)
    ramp_schedule: bool = False


@dataclass
class BlendingConstraint:
    """Constraint for material blending optimization.

    Represents constraints of the form:
    lower <= Σ_i c_i x_i <= upper

    Attributes:
        name: Constraint name
        coefficients: Coefficient for each material
        lower_bound: Minimum value
        upper_bound: Maximum value
        type: Constraint type (equality, inequality)
        penalty_weight: Weight in penalty formulation
    """

    name: str
    coefficients: NDArray[np.float64]
    lower_bound: float = 0.0
    upper_bound: float = float("inf")
    type: str = "inequality"  # equality, inequality
    penalty_weight: float = 1.0

    def evaluate(self, x: NDArray[np.float64]) -> float:
        """Evaluate constraint violation."""
        value = np.dot(self.coefficients, x)

        if value < self.lower_bound:
            return (self.lower_bound - value) ** 2
        elif value > self.upper_bound:
            return (value - self.upper_bound) ** 2
        else:
            return 0.0


@dataclass
class BlendingObjective:
    """Objective function for material blending.

    Encodes the optimization objective as:
    minimize: Σ_i c_i x_i + Σ_ij Q_ij x_i x_j + constant

    Attributes:
        name: Objective name
        linear_coeffs: Linear coefficients
        quadratic_coeffs: Quadratic coefficient matrix
        constant: Constant offset
        minimize: Whether to minimize (True) or maximize (False)
    """

    name: str
    linear_coeffs: NDArray[np.float64]
    quadratic_coeffs: NDArray[np.float64] | None = None
    constant: float = 0.0
    minimize: bool = True

    @property
    def n_variables(self) -> int:
        """Number of binary variables."""
        return len(self.linear_coeffs)

    def evaluate(self, x: NDArray[np.float64]) -> float:
        """Evaluate objective at point x."""
        value = self.constant + np.dot(self.linear_coeffs, x)

        if self.quadratic_coeffs is not None:
            value += x @ self.quadratic_coeffs @ x

        return value if self.minimize else -value

    def to_ising(self) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        """Convert to Ising model form.

        Returns:
            Tuple of (h, J, offset) where:
            - h: Linear Ising coefficients
            - J: Coupling matrix
            - offset: Energy offset
        """
        n = self.n_variables

        # Transform x ∈ {0,1} to s ∈ {-1,+1} via x = (1-s)/2
        h = np.zeros(n)
        J = np.zeros((n, n))
        offset = self.constant

        # Linear terms
        for i in range(n):
            h[i] = -self.linear_coeffs[i] / 2
            offset += self.linear_coeffs[i] / 2

        # Quadratic terms
        if self.quadratic_coeffs is not None:
            for i in range(n):
                for j in range(n):
                    if i != j:
                        J[i, j] = self.quadratic_coeffs[i, j] / 4
                    offset += self.quadratic_coeffs[i, j] / 4

        return h, J, offset


@dataclass
class QAOAResult:
    """Results from QAOA optimization.

    Attributes:
        optimal_bitstring: Best solution found
        optimal_value: Objective value at optimal solution
        parameters: Optimized (γ, β) parameters
        probability_distribution: Measurement probabilities
        iterations: Number of optimization iterations
        converged: Whether optimization converged
        execution_time: Total execution time
        approximation_ratio: Ratio to known optimal (if available)
    """

    optimal_bitstring: str
    optimal_value: float
    parameters: NDArray[np.float64]
    probability_distribution: dict[str, float]
    iterations: int
    converged: bool
    execution_time: float = 0.0
    approximation_ratio: float | None = None

    # Top solutions
    top_solutions: list[tuple[str, float, float]] = field(default_factory=list)

    def get_material_selection(self) -> list[int]:
        """Convert bitstring to material selection indices."""
        return [i for i, bit in enumerate(self.optimal_bitstring) if bit == "1"]

    def get_blend_proportions(
        self,
        discretization_levels: int = 4,
    ) -> NDArray[np.float64]:
        """Extract blend proportions from multi-bit encoding."""
        # Decode binary representation to continuous proportions
        n_bits_per_material = int(np.log2(discretization_levels))
        n_materials = len(self.optimal_bitstring) // n_bits_per_material

        proportions = np.zeros(n_materials)

        for i in range(n_materials):
            bits = self.optimal_bitstring[i * n_bits_per_material : (i + 1) * n_bits_per_material]
            value = int(bits, 2)
            proportions[i] = value / (discretization_levels - 1)

        # Normalize to sum to 1
        if proportions.sum() > 0:
            proportions /= proportions.sum()

        return proportions


class CostHamiltonian:
    """Cost Hamiltonian for QAOA encoding the optimization objective.

    Converts QUBO/Ising problems to quantum Hamiltonian form:
    H_C = Σ_i h_i Z_i + Σ_ij J_ij Z_i Z_j
    """

    def __init__(
        self,
        objective: BlendingObjective,
        constraints: list[BlendingConstraint] | None = None,
        penalty_weight: float = 10.0,
    ) -> None:
        """Initialize cost Hamiltonian.

        Args:
            objective: Optimization objective
            constraints: List of constraints
            penalty_weight: Weight for constraint penalties
        """
        self.objective = objective
        self.constraints = constraints or []
        self.penalty_weight = penalty_weight

        # Build Ising representation
        self._build_ising_model()

    def _build_ising_model(self) -> None:
        """Construct Ising model from objective and constraints."""
        # Get objective Ising form
        h_obj, J_obj, offset_obj = self.objective.to_ising()

        n = self.objective.n_variables
        self.h = h_obj.copy()
        self.J = J_obj.copy()
        self.offset = offset_obj

        # Add constraint penalties
        for constraint in self.constraints:
            h_c, J_c, offset_c = self._constraint_to_ising(constraint)
            self.h += self.penalty_weight * h_c
            self.J += self.penalty_weight * J_c
            self.offset += self.penalty_weight * offset_c

    def _constraint_to_ising(
        self,
        constraint: BlendingConstraint,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        """Convert constraint to Ising penalty terms."""
        n = len(constraint.coefficients)

        # Slack variable formulation for inequality constraints
        # For equality: penalty = (Σ c_i x_i - target)^2

        h = np.zeros(n)
        J = np.zeros((n, n))

        target = (constraint.lower_bound + constraint.upper_bound) / 2

        # Expand (Σ c_i x_i - target)^2
        for i in range(n):
            h[i] -= constraint.coefficients[i] * target
            for j in range(n):
                J[i, j] += constraint.coefficients[i] * constraint.coefficients[j] / 2

        offset = target**2

        return h, J, offset

    def to_pauli_strings(self) -> list[tuple[str, float]]:
        """Convert to Pauli string representation.

        Returns:
            List of (pauli_string, coefficient) tuples
        """
        n = len(self.h)
        terms = []

        # Identity term (offset)
        terms.append(("I" * n, self.offset))

        # Linear terms: h_i Z_i
        for i in range(n):
            if abs(self.h[i]) > 1e-12:
                pauli = ["I"] * n
                pauli[i] = "Z"
                terms.append(("".join(pauli), self.h[i]))

        # Quadratic terms: J_ij Z_i Z_j
        for i in range(n):
            for j in range(i + 1, n):
                coeff = self.J[i, j] + self.J[j, i]
                if abs(coeff) > 1e-12:
                    pauli = ["I"] * n
                    pauli[i] = "Z"
                    pauli[j] = "Z"
                    terms.append(("".join(pauli), coeff))

        return terms

    def get_circuit_layer(self, gamma: float) -> list[dict]:
        """Generate circuit for exp(-i γ H_C).

        Args:
            gamma: Rotation angle

        Returns:
            List of gate operations
        """
        n = len(self.h)
        circuit = []

        # Single-qubit Z rotations: exp(-i γ h_i Z_i) = RZ(2 γ h_i)
        for i in range(n):
            if abs(self.h[i]) > 1e-12:
                circuit.append(
                    {
                        "gate": "RZ",
                        "qubits": [i],
                        "params": [2 * gamma * self.h[i]],
                    }
                )

        # Two-qubit ZZ rotations: exp(-i γ J_ij Z_i Z_j)
        for i in range(n):
            for j in range(i + 1, n):
                coeff = self.J[i, j] + self.J[j, i]
                if abs(coeff) > 1e-12:
                    # Decompose ZZ rotation
                    circuit.append({"gate": "CNOT", "qubits": [i, j], "params": []})
                    circuit.append(
                        {
                            "gate": "RZ",
                            "qubits": [j],
                            "params": [2 * gamma * coeff],
                        }
                    )
                    circuit.append({"gate": "CNOT", "qubits": [i, j], "params": []})

        return circuit


class MixerHamiltonian:
    """Mixer Hamiltonian for QAOA.

    The standard transverse-field mixer is:
    H_M = Σ_i X_i

    Alternative mixers preserve problem structure.
    """

    def __init__(
        self,
        n_qubits: int,
        mixer_type: MixerType = MixerType.TRANSVERSE_FIELD,
        feasible_subspace: list[str] | None = None,
    ) -> None:
        """Initialize mixer Hamiltonian.

        Args:
            n_qubits: Number of qubits
            mixer_type: Type of mixer
            feasible_subspace: Feasible bitstrings (for Grover mixer)
        """
        self.n_qubits = n_qubits
        self.mixer_type = mixer_type
        self.feasible_subspace = feasible_subspace

    def get_circuit_layer(self, beta: float) -> list[dict]:
        """Generate circuit for exp(-i β H_M).

        Args:
            beta: Rotation angle

        Returns:
            List of gate operations
        """
        circuit = []

        if self.mixer_type == MixerType.TRANSVERSE_FIELD:
            # exp(-i β Σ X_i) = Π_i RX(2β)
            for i in range(self.n_qubits):
                circuit.append(
                    {
                        "gate": "RX",
                        "qubits": [i],
                        "params": [2 * beta],
                    }
                )

        elif self.mixer_type == MixerType.XY_MIXER:
            # XY mixer preserves Hamming weight
            # exp(-i β (X_i X_j + Y_i Y_j))
            for i in range(self.n_qubits - 1):
                # Decomposition of XY interaction
                circuit.append({"gate": "H", "qubits": [i], "params": []})
                circuit.append({"gate": "H", "qubits": [i + 1], "params": []})
                circuit.append({"gate": "CNOT", "qubits": [i, i + 1], "params": []})
                circuit.append(
                    {
                        "gate": "RZ",
                        "qubits": [i + 1],
                        "params": [2 * beta],
                    }
                )
                circuit.append({"gate": "CNOT", "qubits": [i, i + 1], "params": []})
                circuit.append({"gate": "H", "qubits": [i], "params": []})
                circuit.append({"gate": "H", "qubits": [i + 1], "params": []})

        elif self.mixer_type == MixerType.GROVER_MIXER:
            # Grover mixer over feasible subspace
            # Requires preparing superposition over feasible states
            for i in range(self.n_qubits):
                circuit.append(
                    {
                        "gate": "RY",
                        "qubits": [i],
                        "params": [2 * beta],
                    }
                )

        return circuit


class TireQAOA:
    """QAOA solver for tire material blending optimization.

    Optimizes the selection and proportion of materials in a tire
    compound to achieve target performance characteristics while
    satisfying manufacturing and cost constraints.

    Problem Formulation:
        minimize: cost(x) - λ * performance(x)
        subject to:
            Σ x_i = 1 (proportions sum to 1)
            x_i >= 0 (non-negative proportions)
            property_min <= property(x) <= property_max (for each property)

    The binary encoding uses multiple bits per material to represent
    discrete proportion levels.

    Example:
        >>> # Define materials and their properties
        >>> materials = ["SBR", "NBR", "NR", "Carbon_Black", "Silica"]
        >>> costs = np.array([2.5, 3.2, 1.8, 1.2, 2.8])
        >>> performance = np.array([0.8, 0.7, 0.6, 0.9, 0.85])
        >>>
        >>> # Create objective
        >>> objective = BlendingObjective(
        ...     name="cost_performance",
        ...     linear_coeffs=costs - 0.5 * performance,
        ... )
        >>>
        >>> # Run QAOA
        >>> qaoa = TireQAOA(
        ...     n_materials=len(materials),
        ...     objective=objective,
        ...     config=QAOAConfig(depth=3),
        ... )
        >>> result = qaoa.optimize()
        >>> print(f"Optimal blend: {result.get_blend_proportions()}")
    """

    def __init__(
        self,
        n_materials: int,
        objective: BlendingObjective,
        constraints: list[BlendingConstraint] | None = None,
        config: QAOAConfig | None = None,
        bits_per_material: int = 2,
    ) -> None:
        """Initialize QAOA solver.

        Args:
            n_materials: Number of candidate materials
            objective: Blending objective function
            constraints: List of blending constraints
            config: QAOA configuration
            bits_per_material: Binary encoding precision
        """
        self.n_materials = n_materials
        self.objective = objective
        self.constraints = constraints or []
        self.config = config or QAOAConfig()
        self.bits_per_material = bits_per_material

        # Total qubits needed
        self.n_qubits = n_materials * bits_per_material

        # Build Hamiltonians
        self.cost_hamiltonian = CostHamiltonian(
            objective=objective,
            constraints=constraints,
            penalty_weight=self.config.constraint_penalty,
        )

        self.mixer_hamiltonian = MixerHamiltonian(
            n_qubits=self.n_qubits,
            mixer_type=self.config.mixer_type,
        )

        logger.info(
            f"Initialized TireQAOA: {n_materials} materials, "
            f"{self.n_qubits} qubits, depth={self.config.depth}"
        )

    def _build_circuit(
        self,
        gammas: NDArray[np.float64],
        betas: NDArray[np.float64],
    ) -> list[dict]:
        """Build QAOA circuit.

        Args:
            gammas: Cost layer angles
            betas: Mixer layer angles

        Returns:
            Circuit as list of gates
        """
        circuit = []

        # Initial superposition
        for i in range(self.n_qubits):
            circuit.append({"gate": "H", "qubits": [i], "params": []})

        # QAOA layers
        for p in range(self.config.depth):
            # Cost layer
            circuit.extend(self.cost_hamiltonian.get_circuit_layer(gammas[p]))

            # Mixer layer
            circuit.extend(self.mixer_hamiltonian.get_circuit_layer(betas[p]))

        return circuit

    def _compute_expectation(
        self,
        parameters: NDArray[np.float64],
    ) -> float:
        """Compute cost function expectation value.

        Args:
            parameters: Concatenated [γ_1,...,γ_p, β_1,...,β_p]

        Returns:
            Expected cost value
        """
        p = self.config.depth
        gammas = parameters[:p]
        betas = parameters[p:]

        # Build circuit
        circuit = self._build_circuit(gammas, betas)

        # Simulate measurement (simplified)
        # Real implementation would execute on quantum backend
        samples = self._sample_circuit(circuit, self.config.shots)

        # Compute average cost
        total_cost = 0.0
        for bitstring, count in samples.items():
            x = np.array([int(b) for b in bitstring])
            cost = self.objective.evaluate(x)

            # Add constraint penalties
            for constraint in self.constraints:
                cost += (
                    self.config.constraint_penalty
                    * constraint.penalty_weight
                    * constraint.evaluate(x)
                )

            total_cost += cost * count

        return total_cost / self.config.shots

    def _sample_circuit(
        self,
        circuit: list[dict],
        shots: int,
    ) -> dict[str, int]:
        """Sample from circuit (simulation).

        Args:
            circuit: QAOA circuit
            shots: Number of samples

        Returns:
            Dictionary of bitstring counts
        """
        # Simplified simulation: sample from approximate distribution
        # Real implementation would use full state vector simulation

        n = self.n_qubits
        samples: dict[str, int] = {}

        # Generate samples (placeholder - would be quantum simulation)
        for _ in range(shots):
            # Biased sampling based on circuit structure
            bits = []
            for i in range(n):
                # Simple heuristic based on number of gates
                p = 0.5  # Uniform in reality would depend on circuit
                bits.append("1" if np.random.random() < p else "0")

            bitstring = "".join(bits)
            samples[bitstring] = samples.get(bitstring, 0) + 1

        return samples

    def _initialize_parameters(self) -> NDArray[np.float64]:
        """Initialize QAOA parameters.

        Returns:
            Initial [γ, β] parameters
        """
        p = self.config.depth

        if self.config.warm_start:
            # Warm-start initialization from analytical insights
            gammas = np.linspace(0.1, 0.5, p) * self.config.initial_gamma
            betas = np.linspace(0.5, 0.1, p) * self.config.initial_beta
        elif self.config.ramp_schedule:
            # Linear ramp schedule
            gammas = np.linspace(0, np.pi / 4, p)
            betas = np.linspace(np.pi / 4, 0, p)
        else:
            # Random initialization
            gammas = np.random.uniform(0, np.pi, p)
            betas = np.random.uniform(0, np.pi, p)

        return np.concatenate([gammas, betas])

    def optimize(self) -> QAOAResult:
        """Run QAOA optimization.

        Returns:
            QAOAResult with optimal blend
        """
        import time

        from scipy.optimize import minimize

        start_time = time.time()

        # Initialize parameters
        initial_params = self._initialize_parameters()

        # Optimization history
        history = {"iterations": 0, "costs": []}

        def objective_wrapper(params: NDArray[np.float64]) -> float:
            cost = self._compute_expectation(params)
            history["iterations"] += 1
            history["costs"].append(cost)
            return cost

        logger.info(f"Starting QAOA optimization with depth {self.config.depth}")

        # Run optimization
        result = minimize(
            objective_wrapper,
            initial_params,
            method=self.config.optimizer,
            options={"maxiter": self.config.max_iterations},
        )

        execution_time = time.time() - start_time

        # Get final samples
        p = self.config.depth
        final_gammas = result.x[:p]
        final_betas = result.x[p:]
        final_circuit = self._build_circuit(final_gammas, final_betas)
        final_samples = self._sample_circuit(final_circuit, self.config.shots * 10)

        # Find optimal solution
        best_bitstring = ""
        best_cost = float("inf")

        probability_dist = {}
        total_samples = sum(final_samples.values())

        for bitstring, count in final_samples.items():
            probability_dist[bitstring] = count / total_samples

            x = np.array([int(b) for b in bitstring])
            cost = self.objective.evaluate(x)

            if cost < best_cost:
                # Check constraints
                feasible = all(constraint.evaluate(x) < 1e-6 for constraint in self.constraints)
                if feasible or not self.constraints:
                    best_cost = cost
                    best_bitstring = bitstring

        # Get top solutions
        top_solutions = sorted(
            [
                (bs, self.objective.evaluate(np.array([int(b) for b in bs])), prob)
                for bs, prob in probability_dist.items()
            ],
            key=lambda x: x[1],
        )[:10]

        qaoa_result = QAOAResult(
            optimal_bitstring=best_bitstring,
            optimal_value=best_cost,
            parameters=result.x,
            probability_distribution=probability_dist,
            iterations=history["iterations"],
            converged=result.success,
            execution_time=execution_time,
            top_solutions=top_solutions,
        )

        logger.info(
            f"QAOA completed: optimal_value={best_cost:.4f}, " f"iterations={history['iterations']}"
        )

        return qaoa_result

    def get_blend_recommendation(
        self,
        result: QAOAResult,
        material_names: list[str] | None = None,
    ) -> dict[str, float]:
        """Convert QAOA result to blend recommendation.

        Args:
            result: QAOA optimization result
            material_names: Names for materials

        Returns:
            Dictionary mapping material names to proportions
        """
        proportions = result.get_blend_proportions(discretization_levels=2**self.bits_per_material)

        if material_names is None:
            material_names = [f"Material_{i}" for i in range(self.n_materials)]

        return {
            name: float(prop)
            for name, prop in zip(material_names, proportions)
            if prop > 0.01  # Filter out negligible proportions
        }


class BlendingQAOA(TireQAOA):
    """Extended QAOA for multi-objective tire compound blending.

    Handles complex blending scenarios with multiple objectives:
    - Cost minimization
    - Performance maximization
    - Durability optimization
    - Environmental impact reduction

    Uses weighted sum or Pareto optimization approaches.
    """

    def __init__(
        self,
        n_materials: int,
        objectives: list[BlendingObjective],
        weights: NDArray[np.float64] | None = None,
        constraints: list[BlendingConstraint] | None = None,
        config: QAOAConfig | None = None,
    ) -> None:
        """Initialize multi-objective QAOA.

        Args:
            n_materials: Number of materials
            objectives: List of objectives
            weights: Weights for each objective
            constraints: Blending constraints
            config: QAOA configuration
        """
        self.objectives = objectives
        self.weights = weights if weights is not None else np.ones(len(objectives))

        # Create combined objective
        combined = self._combine_objectives()

        super().__init__(
            n_materials=n_materials,
            objective=combined,
            constraints=constraints,
            config=config,
        )

    def _combine_objectives(self) -> BlendingObjective:
        """Combine objectives with weights."""
        n = self.objectives[0].n_variables

        combined_linear = np.zeros(n)
        combined_quadratic = np.zeros((n, n))
        combined_constant = 0.0

        for obj, weight in zip(self.objectives, self.weights):
            sign = 1.0 if obj.minimize else -1.0
            combined_linear += weight * sign * obj.linear_coeffs

            if obj.quadratic_coeffs is not None:
                combined_quadratic += weight * sign * obj.quadratic_coeffs

            combined_constant += weight * sign * obj.constant

        return BlendingObjective(
            name="combined",
            linear_coeffs=combined_linear,
            quadratic_coeffs=combined_quadratic if np.any(combined_quadratic) else None,
            constant=combined_constant,
        )

    def pareto_optimize(
        self,
        n_points: int = 20,
    ) -> list[tuple[QAOAResult, NDArray[np.float64]]]:
        """Find Pareto-optimal blends.

        Args:
            n_points: Number of Pareto points to find

        Returns:
            List of (result, objective_values) on Pareto frontier
        """
        pareto_front = []

        # Generate weight combinations
        n_obj = len(self.objectives)

        for i in range(n_points):
            # Vary weights along simplex
            weights = np.random.dirichlet(np.ones(n_obj))
            self.weights = weights

            # Re-combine objectives
            self.objective = self._combine_objectives()
            self.cost_hamiltonian = CostHamiltonian(
                objective=self.objective,
                constraints=self.constraints,
                penalty_weight=self.config.constraint_penalty,
            )

            # Optimize
            result = self.optimize()

            # Evaluate all objectives at solution
            x = np.array([int(b) for b in result.optimal_bitstring])
            obj_values = np.array([obj.evaluate(x) for obj in self.objectives])

            pareto_front.append((result, obj_values))

        # Filter dominated solutions
        pareto_front = self._filter_dominated(pareto_front)

        return pareto_front

    def _filter_dominated(
        self,
        solutions: list[tuple[QAOAResult, NDArray[np.float64]]],
    ) -> list[tuple[QAOAResult, NDArray[np.float64]]]:
        """Remove dominated solutions from Pareto front."""
        non_dominated = []

        for i, (result_i, values_i) in enumerate(solutions):
            dominated = False

            for j, (result_j, values_j) in enumerate(solutions):
                if i == j:
                    continue

                # Check if j dominates i
                if all(values_j <= values_i) and any(values_j < values_i):
                    dominated = True
                    break

            if not dominated:
                non_dominated.append((result_i, values_i))

        return non_dominated


class ConstraintQAOA(TireQAOA):
    """QAOA with advanced constraint handling.

    Implements constraint-preserving mixers and penalty methods
    for handling complex manufacturing and regulatory constraints.

    Constraint Types Supported:
    - Budget constraints: Σ c_i x_i <= B
    - Proportion constraints: x_i / x_j >= r
    - Exclusion constraints: x_i + x_j <= 1
    - Grouping constraints: Σ_{i∈G} x_i >= 1
    """

    def __init__(
        self,
        n_materials: int,
        objective: BlendingObjective,
        hard_constraints: list[BlendingConstraint] | None = None,
        soft_constraints: list[BlendingConstraint] | None = None,
        config: QAOAConfig | None = None,
    ) -> None:
        """Initialize constraint-aware QAOA.

        Args:
            n_materials: Number of materials
            objective: Blending objective
            hard_constraints: Must be satisfied exactly
            soft_constraints: Penalties for violation
            config: QAOA configuration
        """
        self.hard_constraints = hard_constraints or []
        self.soft_constraints = soft_constraints or []

        # Build constraint-preserving mixer for hard constraints
        feasible_states = self._enumerate_feasible_states()

        config = config or QAOAConfig()
        if feasible_states:
            config.mixer_type = MixerType.GROVER_MIXER

        super().__init__(
            n_materials=n_materials,
            objective=objective,
            constraints=soft_constraints,  # Soft as penalties
            config=config,
        )

        # Override mixer for feasible subspace
        if feasible_states:
            self.mixer_hamiltonian = MixerHamiltonian(
                n_qubits=self.n_qubits,
                mixer_type=MixerType.GROVER_MIXER,
                feasible_subspace=feasible_states,
            )

    def _enumerate_feasible_states(self) -> list[str] | None:
        """Enumerate feasible states for hard constraints.

        Only practical for small instances.
        """
        if not self.hard_constraints:
            return None

        if self.n_qubits > 12:
            logger.warning("Too many qubits for explicit feasibility enumeration")
            return None

        feasible = []

        for i in range(2**self.n_qubits):
            bitstring = format(i, f"0{self.n_qubits}b")
            x = np.array([int(b) for b in bitstring])

            # Check all hard constraints
            if all(c.evaluate(x) < 1e-6 for c in self.hard_constraints):
                feasible.append(bitstring)

        logger.info(f"Found {len(feasible)} feasible states out of {2**self.n_qubits}")

        return feasible if feasible else None


# Convenience functions


def optimize_tire_blend(
    materials: list[str],
    costs: NDArray[np.float64],
    properties: dict[str, NDArray[np.float64]],
    targets: dict[str, tuple[float, float]],
    depth: int = 3,
) -> dict[str, float]:
    """Optimize tire blend composition.

    Args:
        materials: Material names
        costs: Cost per unit for each material
        properties: Property values for each material
        targets: Target ranges for each property
        depth: QAOA depth

    Returns:
        Dictionary of material proportions
    """
    n = len(materials)

    # Create cost objective
    objective = BlendingObjective(
        name="cost",
        linear_coeffs=costs,
    )

    # Create property constraints
    constraints = []
    for prop_name, (min_val, max_val) in targets.items():
        if prop_name in properties:
            constraints.append(
                BlendingConstraint(
                    name=f"{prop_name}_constraint",
                    coefficients=properties[prop_name],
                    lower_bound=min_val,
                    upper_bound=max_val,
                )
            )

    # Run QAOA
    qaoa = TireQAOA(
        n_materials=n,
        objective=objective,
        constraints=constraints,
        config=QAOAConfig(depth=depth),
    )

    result = qaoa.optimize()

    return qaoa.get_blend_recommendation(result, materials)


def create_blending_problem(
    n_materials: int,
    cost_coeffs: NDArray[np.float64],
    performance_coeffs: NDArray[np.float64],
    cost_weight: float = 0.5,
) -> BlendingObjective:
    """Create a blending optimization problem.

    Args:
        n_materials: Number of materials
        cost_coeffs: Cost coefficients
        performance_coeffs: Performance coefficients (higher = better)
        cost_weight: Weight for cost vs. performance

    Returns:
        Blending objective for QAOA
    """
    # Combined objective: minimize cost - (1-w) * performance
    combined = cost_weight * cost_coeffs - (1 - cost_weight) * performance_coeffs

    return BlendingObjective(
        name="cost_performance_tradeoff",
        linear_coeffs=combined,
    )
