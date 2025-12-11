"""Quantum state management for materials simulation.

Provides quantum state representations including:
- State vectors for pure states
- Density matrices for mixed states
- Tensor network representations for large systems
- State tomography utilities
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QuantumState:
    """Abstract quantum state representation.
    
    Base class for different quantum state representations
    used in materials simulation.
    
    Attributes:
        num_qubits: Number of qubits in the system
        label: Optional state label
    """
    
    num_qubits: int
    label: str = "state"
    
    @property
    def dimension(self) -> int:
        """Hilbert space dimension."""
        return 2 ** self.num_qubits
    
    def to_statevector(self) -> "StateVector":
        """Convert to state vector representation."""
        raise NotImplementedError
    
    def to_density_matrix(self) -> "DensityMatrix":
        """Convert to density matrix representation."""
        raise NotImplementedError
    
    def expectation_value(self, observable: np.ndarray) -> complex:
        """Compute expectation value of an observable.
        
        Args:
            observable: Hermitian operator as matrix
            
        Returns:
            Expectation value ⟨ψ|O|ψ⟩
        """
        raise NotImplementedError
    
    def probability(self, bitstring: str) -> float:
        """Get measurement probability for a bitstring.
        
        Args:
            bitstring: Computational basis state (e.g., "0101")
            
        Returns:
            Probability of measuring this state
        """
        raise NotImplementedError
    
    def entropy(self) -> float:
        """Compute von Neumann entropy.
        
        Returns:
            Entropy S = -Tr(ρ log ρ)
        """
        raise NotImplementedError
    
    def fidelity(self, other: "QuantumState") -> float:
        """Compute fidelity with another state.
        
        Args:
            other: State to compare with
            
        Returns:
            Fidelity F(ρ, σ) = (Tr√(√ρ σ √ρ))²
        """
        raise NotImplementedError


@dataclass
class StateVector(QuantumState):
    """Pure quantum state as state vector.
    
    Represents a pure state |ψ⟩ as a complex vector in
    the 2^n dimensional Hilbert space.
    
    Attributes:
        amplitudes: Complex amplitudes of the state vector
    """
    
    amplitudes: np.ndarray = field(default_factory=lambda: np.array([1.0 + 0j]))
    
    def __post_init__(self) -> None:
        """Validate and normalize state vector."""
        self.amplitudes = np.asarray(self.amplitudes, dtype=np.complex128)
        
        # Infer num_qubits if not set
        if self.amplitudes.size > 1:
            inferred_qubits = int(np.log2(self.amplitudes.size))
            if 2 ** inferred_qubits != self.amplitudes.size:
                raise ValueError("State vector size must be power of 2")
            self.num_qubits = inferred_qubits
        
        # Normalize
        norm = np.linalg.norm(self.amplitudes)
        if norm > 0:
            self.amplitudes /= norm
    
    @classmethod
    def zero_state(cls, num_qubits: int) -> "StateVector":
        """Create |0...0⟩ computational basis state.
        
        Args:
            num_qubits: Number of qubits
            
        Returns:
            Ground state vector
        """
        dim = 2 ** num_qubits
        amplitudes = np.zeros(dim, dtype=np.complex128)
        amplitudes[0] = 1.0
        return cls(num_qubits=num_qubits, amplitudes=amplitudes)
    
    @classmethod
    def uniform_superposition(cls, num_qubits: int) -> "StateVector":
        """Create uniform superposition state.
        
        Args:
            num_qubits: Number of qubits
            
        Returns:
            State |+...+⟩ = H⊗n |0...0⟩
        """
        dim = 2 ** num_qubits
        amplitudes = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
        return cls(num_qubits=num_qubits, amplitudes=amplitudes)
    
    @classmethod
    def from_bitstring(cls, bitstring: str) -> "StateVector":
        """Create state from computational basis bitstring.
        
        Args:
            bitstring: Binary string (e.g., "0101")
            
        Returns:
            Computational basis state
        """
        num_qubits = len(bitstring)
        dim = 2 ** num_qubits
        idx = int(bitstring, 2)
        
        amplitudes = np.zeros(dim, dtype=np.complex128)
        amplitudes[idx] = 1.0
        
        return cls(num_qubits=num_qubits, amplitudes=amplitudes)
    
    @classmethod
    def random(cls, num_qubits: int, seed: int | None = None) -> "StateVector":
        """Create random normalized state.
        
        Args:
            num_qubits: Number of qubits
            seed: Random seed for reproducibility
            
        Returns:
            Random normalized state vector
        """
        if seed is not None:
            np.random.seed(seed)
        
        dim = 2 ** num_qubits
        real = np.random.randn(dim)
        imag = np.random.randn(dim)
        amplitudes = real + 1j * imag
        
        return cls(num_qubits=num_qubits, amplitudes=amplitudes)
    
    def to_statevector(self) -> "StateVector":
        """Return self (already a state vector)."""
        return self
    
    def to_density_matrix(self) -> "DensityMatrix":
        """Convert to density matrix |ψ⟩⟨ψ|."""
        rho = np.outer(self.amplitudes, np.conj(self.amplitudes))
        return DensityMatrix(num_qubits=self.num_qubits, matrix=rho)
    
    def expectation_value(self, observable: np.ndarray) -> complex:
        """Compute ⟨ψ|O|ψ⟩."""
        return np.conj(self.amplitudes) @ observable @ self.amplitudes
    
    def probability(self, bitstring: str) -> float:
        """Get measurement probability for bitstring."""
        if len(bitstring) != self.num_qubits:
            raise ValueError(f"Bitstring length must be {self.num_qubits}")
        
        idx = int(bitstring, 2)
        return float(np.abs(self.amplitudes[idx]) ** 2)
    
    def probabilities(self) -> np.ndarray:
        """Get all measurement probabilities."""
        return np.abs(self.amplitudes) ** 2
    
    def sample(self, shots: int = 1000, seed: int | None = None) -> dict[str, int]:
        """Sample measurements from the state.
        
        Args:
            shots: Number of measurements
            seed: Random seed
            
        Returns:
            Dictionary of bitstring counts
        """
        if seed is not None:
            np.random.seed(seed)
        
        probs = self.probabilities()
        indices = np.random.choice(self.dimension, size=shots, p=probs)
        
        counts = {}
        for idx in indices:
            bitstring = format(idx, f'0{self.num_qubits}b')
            counts[bitstring] = counts.get(bitstring, 0) + 1
        
        return counts
    
    def entropy(self) -> float:
        """Von Neumann entropy (0 for pure state)."""
        return 0.0
    
    def fidelity(self, other: QuantumState) -> float:
        """Compute fidelity with another state."""
        other_sv = other.to_statevector()
        overlap = np.abs(np.conj(self.amplitudes) @ other_sv.amplitudes) ** 2
        return float(overlap)
    
    def apply_operator(self, operator: np.ndarray) -> "StateVector":
        """Apply a unitary operator to the state.
        
        Args:
            operator: Unitary matrix
            
        Returns:
            New state vector
        """
        new_amplitudes = operator @ self.amplitudes
        return StateVector(num_qubits=self.num_qubits, amplitudes=new_amplitudes)
    
    def partial_trace(self, qubits_to_keep: list[int]) -> "DensityMatrix":
        """Compute partial trace over specified qubits.
        
        Args:
            qubits_to_keep: Indices of qubits to keep
            
        Returns:
            Reduced density matrix
        """
        # Convert to density matrix and trace out
        rho = self.to_density_matrix()
        return rho.partial_trace(qubits_to_keep)
    
    def __repr__(self) -> str:
        return f"StateVector(num_qubits={self.num_qubits}, norm={np.linalg.norm(self.amplitudes):.4f})"


@dataclass
class DensityMatrix(QuantumState):
    """Mixed quantum state as density matrix.
    
    Represents a general (possibly mixed) state as a density matrix ρ.
    
    Attributes:
        matrix: Density matrix (Hermitian, positive semi-definite, trace 1)
    """
    
    matrix: np.ndarray = field(default_factory=lambda: np.array([[1.0 + 0j]]))
    
    def __post_init__(self) -> None:
        """Validate density matrix properties."""
        self.matrix = np.asarray(self.matrix, dtype=np.complex128)
        
        # Infer num_qubits
        dim = self.matrix.shape[0]
        if self.matrix.shape != (dim, dim):
            raise ValueError("Density matrix must be square")
        
        inferred_qubits = int(np.log2(dim))
        if 2 ** inferred_qubits != dim:
            raise ValueError("Matrix dimension must be power of 2")
        self.num_qubits = inferred_qubits
        
        # Normalize trace
        trace = np.trace(self.matrix)
        if np.abs(trace) > 0:
            self.matrix /= trace
    
    @classmethod
    def from_statevector(cls, sv: StateVector) -> "DensityMatrix":
        """Create density matrix from pure state.
        
        Args:
            sv: Pure state vector
            
        Returns:
            Density matrix |ψ⟩⟨ψ|
        """
        return sv.to_density_matrix()
    
    @classmethod
    def maximally_mixed(cls, num_qubits: int) -> "DensityMatrix":
        """Create maximally mixed state.
        
        Args:
            num_qubits: Number of qubits
            
        Returns:
            Density matrix I/2^n
        """
        dim = 2 ** num_qubits
        matrix = np.eye(dim, dtype=np.complex128) / dim
        return cls(num_qubits=num_qubits, matrix=matrix)
    
    @classmethod
    def thermal(cls, hamiltonian: np.ndarray, temperature: float) -> "DensityMatrix":
        """Create thermal equilibrium state.
        
        Args:
            hamiltonian: System Hamiltonian
            temperature: Temperature in energy units (kT)
            
        Returns:
            Thermal density matrix ρ = exp(-H/kT) / Z
        """
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        
        beta = 1.0 / temperature
        rho = np.linalg.matrix_power(
            np.diag(np.exp(-beta * np.linalg.eigvalsh(hamiltonian))),
            1
        )
        
        # Transform to original basis
        eigvecs = np.linalg.eigh(hamiltonian)[1]
        rho = eigvecs @ rho @ eigvecs.conj().T
        
        dim = hamiltonian.shape[0]
        num_qubits = int(np.log2(dim))
        
        return cls(num_qubits=num_qubits, matrix=rho)
    
    def to_statevector(self) -> StateVector:
        """Convert to state vector (only for pure states)."""
        # Check if pure
        if not self.is_pure(tolerance=1e-10):
            raise ValueError("Cannot convert mixed state to state vector")
        
        # Extract state vector from rank-1 density matrix
        eigenvalues, eigenvectors = np.linalg.eigh(self.matrix)
        idx = np.argmax(eigenvalues)
        amplitudes = eigenvectors[:, idx]
        
        return StateVector(num_qubits=self.num_qubits, amplitudes=amplitudes)
    
    def to_density_matrix(self) -> "DensityMatrix":
        """Return self."""
        return self
    
    def is_pure(self, tolerance: float = 1e-10) -> bool:
        """Check if state is pure.
        
        Args:
            tolerance: Numerical tolerance
            
        Returns:
            True if Tr(ρ²) ≈ 1
        """
        purity = np.real(np.trace(self.matrix @ self.matrix))
        return np.abs(purity - 1.0) < tolerance
    
    def purity(self) -> float:
        """Compute purity Tr(ρ²)."""
        return float(np.real(np.trace(self.matrix @ self.matrix)))
    
    def expectation_value(self, observable: np.ndarray) -> complex:
        """Compute Tr(ρO)."""
        return np.trace(self.matrix @ observable)
    
    def probability(self, bitstring: str) -> float:
        """Get measurement probability for bitstring."""
        if len(bitstring) != self.num_qubits:
            raise ValueError(f"Bitstring length must be {self.num_qubits}")
        
        idx = int(bitstring, 2)
        return float(np.real(self.matrix[idx, idx]))
    
    def probabilities(self) -> np.ndarray:
        """Get all measurement probabilities."""
        return np.real(np.diag(self.matrix))
    
    def entropy(self) -> float:
        """Compute von Neumann entropy S = -Tr(ρ log ρ)."""
        eigenvalues = np.linalg.eigvalsh(self.matrix)
        eigenvalues = eigenvalues[eigenvalues > 1e-15]  # Avoid log(0)
        return float(-np.sum(eigenvalues * np.log2(eigenvalues)))
    
    def fidelity(self, other: QuantumState) -> float:
        """Compute fidelity F(ρ, σ)."""
        sigma = other.to_density_matrix()
        
        # F(ρ, σ) = (Tr√(√ρ σ √ρ))²
        sqrt_rho = self._matrix_sqrt(self.matrix)
        product = sqrt_rho @ sigma.matrix @ sqrt_rho
        sqrt_product = self._matrix_sqrt(product)
        
        return float(np.real(np.trace(sqrt_product)) ** 2)
    
    def _matrix_sqrt(self, matrix: np.ndarray) -> np.ndarray:
        """Compute matrix square root."""
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        sqrt_eigenvalues = np.sqrt(np.maximum(eigenvalues, 0))
        return eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.conj().T
    
    def partial_trace(self, qubits_to_keep: list[int]) -> "DensityMatrix":
        """Compute partial trace.
        
        Args:
            qubits_to_keep: Indices of subsystem qubits to keep
            
        Returns:
            Reduced density matrix
        """
        qubits_to_trace = [q for q in range(self.num_qubits) if q not in qubits_to_keep]
        
        if not qubits_to_trace:
            return self
        
        # Reshape and trace
        # This is a simplified implementation
        num_keep = len(qubits_to_keep)
        dim_keep = 2 ** num_keep
        
        reduced = np.zeros((dim_keep, dim_keep), dtype=np.complex128)
        
        for i in range(dim_keep):
            for j in range(dim_keep):
                # Sum over traced-out dimensions
                for k in range(2 ** len(qubits_to_trace)):
                    # Build full indices
                    idx_i = self._build_index(i, k, qubits_to_keep, qubits_to_trace)
                    idx_j = self._build_index(j, k, qubits_to_keep, qubits_to_trace)
                    reduced[i, j] += self.matrix[idx_i, idx_j]
        
        return DensityMatrix(num_qubits=num_keep, matrix=reduced)
    
    def _build_index(
        self,
        keep_idx: int,
        trace_idx: int,
        qubits_to_keep: list[int],
        qubits_to_trace: list[int],
    ) -> int:
        """Build full index from partial indices."""
        full_bits = ['0'] * self.num_qubits
        
        # Set kept qubit bits
        keep_bits = format(keep_idx, f'0{len(qubits_to_keep)}b')
        for i, q in enumerate(qubits_to_keep):
            full_bits[q] = keep_bits[i]
        
        # Set traced qubit bits
        trace_bits = format(trace_idx, f'0{len(qubits_to_trace)}b')
        for i, q in enumerate(qubits_to_trace):
            full_bits[q] = trace_bits[i]
        
        return int(''.join(full_bits), 2)
    
    def evolve(self, unitary: np.ndarray) -> "DensityMatrix":
        """Evolve state under unitary evolution.
        
        Args:
            unitary: Unitary operator
            
        Returns:
            Evolved density matrix UρU†
        """
        new_matrix = unitary @ self.matrix @ unitary.conj().T
        return DensityMatrix(num_qubits=self.num_qubits, matrix=new_matrix)
    
    def __repr__(self) -> str:
        return f"DensityMatrix(num_qubits={self.num_qubits}, purity={self.purity():.4f})"


@dataclass
class MixedState:
    """Statistical mixture of quantum states.
    
    Represents an ensemble {(p_i, |ψ_i⟩)} where p_i are probabilities.
    
    Attributes:
        states: List of (probability, state) tuples
    """
    
    states: list[tuple[float, StateVector]] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Validate probabilities sum to 1."""
        if self.states:
            total_prob = sum(p for p, _ in self.states)
            if np.abs(total_prob - 1.0) > 1e-10:
                # Normalize
                self.states = [(p / total_prob, s) for p, s in self.states]
    
    def add_state(self, probability: float, state: StateVector) -> None:
        """Add a state to the ensemble.
        
        Args:
            probability: Probability weight
            state: Quantum state
        """
        self.states.append((probability, state))
        # Renormalize
        total = sum(p for p, _ in self.states)
        self.states = [(p / total, s) for p, s in self.states]
    
    def to_density_matrix(self) -> DensityMatrix:
        """Convert ensemble to density matrix.
        
        Returns:
            ρ = Σ_i p_i |ψ_i⟩⟨ψ_i|
        """
        if not self.states:
            raise ValueError("Empty ensemble")
        
        num_qubits = self.states[0][1].num_qubits
        dim = 2 ** num_qubits
        
        matrix = np.zeros((dim, dim), dtype=np.complex128)
        for prob, state in self.states:
            rho_i = np.outer(state.amplitudes, np.conj(state.amplitudes))
            matrix += prob * rho_i
        
        return DensityMatrix(num_qubits=num_qubits, matrix=matrix)
    
    @property
    def num_states(self) -> int:
        """Number of states in ensemble."""
        return len(self.states)


class StateTomography:
    """Quantum state tomography utilities.
    
    Provides methods for reconstructing quantum states from
    measurement data.
    """
    
    def __init__(self, num_qubits: int) -> None:
        """Initialize tomography.
        
        Args:
            num_qubits: Number of qubits in the system
        """
        self.num_qubits = num_qubits
        self.dimension = 2 ** num_qubits
    
    def generate_measurement_bases(self) -> list[list[str]]:
        """Generate measurement bases for full tomography.
        
        Returns:
            List of measurement basis specifications
        """
        # For each qubit: X, Y, Z measurements
        bases = ["X", "Y", "Z"]
        
        from itertools import product
        return [list(combo) for combo in product(bases, repeat=self.num_qubits)]
    
    def reconstruct_state(
        self,
        measurement_results: dict[str, dict[str, int]],
    ) -> DensityMatrix:
        """Reconstruct density matrix from measurement results.
        
        Args:
            measurement_results: Dict mapping basis to measurement counts
            
        Returns:
            Reconstructed density matrix
        """
        # Linear inversion tomography (simplified)
        dim = self.dimension
        rho = np.zeros((dim, dim), dtype=np.complex128)
        
        # Process each measurement basis
        for basis, counts in measurement_results.items():
            total = sum(counts.values())
            for bitstring, count in counts.items():
                prob = count / total
                idx = int(bitstring, 2)
                rho[idx, idx] += prob
        
        # Normalize and ensure valid density matrix
        rho /= len(measurement_results)
        
        # Make Hermitian
        rho = (rho + rho.conj().T) / 2
        
        # Make positive semi-definite
        eigenvalues, eigenvectors = np.linalg.eigh(rho)
        eigenvalues = np.maximum(eigenvalues, 0)
        rho = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.conj().T
        
        # Normalize trace
        rho /= np.trace(rho)
        
        return DensityMatrix(num_qubits=self.num_qubits, matrix=rho)
    
    def compute_fidelity_from_measurements(
        self,
        target: QuantumState,
        measurement_results: dict[str, dict[str, int]],
    ) -> float:
        """Estimate fidelity with target from measurements.
        
        Args:
            target: Target quantum state
            measurement_results: Measurement outcome counts
            
        Returns:
            Estimated fidelity
        """
        reconstructed = self.reconstruct_state(measurement_results)
        return reconstructed.fidelity(target)
