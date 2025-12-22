"""Tensor network simulation for large-scale quantum systems.

Provides efficient simulation of quantum systems using tensor networks:
- Matrix Product States (MPS)
- Projected Entangled Pair States (PEPS)
- Tree Tensor Networks (TTN)
- Multiscale Entanglement Renormalization Ansatz (MERA)
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class TensorNetworkType(Enum):
    """Supported tensor network architectures."""

    MPS = auto()  # Matrix Product State
    PEPS = auto()  # Projected Entangled Pair States
    TTN = auto()  # Tree Tensor Network
    MERA = auto()  # Multiscale Entanglement Renormalization Ansatz


@dataclass
class TensorNode:
    """A node (tensor) in a tensor network.

    Attributes:
        data: Tensor data as numpy array
        indices: Names of tensor indices
        physical_dims: Physical dimension indices
        bond_dims: Bond/virtual dimension indices
    """

    data: np.ndarray
    indices: list[str] = field(default_factory=list)
    physical_dims: list[str] = field(default_factory=list)
    bond_dims: list[str] = field(default_factory=list)

    @property
    def rank(self) -> int:
        """Tensor rank (number of indices)."""
        return len(self.data.shape)

    @property
    def shape(self) -> tuple[int, ...]:
        """Tensor shape."""
        return self.data.shape

    @property
    def total_elements(self) -> int:
        """Total number of elements in tensor."""
        return int(np.prod(self.data.shape))

    def contract(self, other: TensorNode, indices: list[tuple[str, str]]) -> TensorNode:
        """Contract this tensor with another over specified indices.

        Args:
            other: Other tensor to contract with
            indices: List of (self_index, other_index) pairs to contract

        Returns:
            Contracted tensor
        """
        # Build contraction axes
        self_axes = []
        other_axes = []

        for self_idx, other_idx in indices:
            if self_idx in self.indices:
                self_axes.append(self.indices.index(self_idx))
            if other_idx in other.indices:
                other_axes.append(other.indices.index(other_idx))

        # Contract using tensordot
        result_data = np.tensordot(self.data, other.data, axes=(self_axes, other_axes))

        # Compute remaining indices
        contracted_self = [indices[i][0] for i in range(len(indices))]
        contracted_other = [indices[i][1] for i in range(len(indices))]

        remaining_self = [idx for idx in self.indices if idx not in contracted_self]
        remaining_other = [idx for idx in other.indices if idx not in contracted_other]

        return TensorNode(
            data=result_data,
            indices=remaining_self + remaining_other,
        )

    def reshape(self, new_shape: tuple[int, ...]) -> TensorNode:
        """Reshape tensor.

        Args:
            new_shape: New shape

        Returns:
            Reshaped tensor
        """
        return TensorNode(
            data=self.data.reshape(new_shape),
            indices=self.indices[: len(new_shape)],
        )

    def conjugate(self) -> TensorNode:
        """Complex conjugate of tensor.

        Returns:
            Conjugated tensor
        """
        return TensorNode(
            data=np.conj(self.data),
            indices=self.indices.copy(),
            physical_dims=self.physical_dims.copy(),
            bond_dims=self.bond_dims.copy(),
        )

    def norm(self) -> float:
        """Frobenius norm of tensor."""
        return float(np.linalg.norm(self.data))

    def normalize(self) -> TensorNode:
        """Return normalized tensor."""
        n = self.norm()
        if n > 0:
            return TensorNode(
                data=self.data / n,
                indices=self.indices.copy(),
                physical_dims=self.physical_dims.copy(),
                bond_dims=self.bond_dims.copy(),
            )
        return self


class TensorContraction:
    """Efficient tensor network contraction engine.

    Implements various contraction strategies for optimal performance.
    """

    def __init__(
        self,
        use_gpu: bool = False,
        optimization: str = "auto",
    ) -> None:
        """Initialize contraction engine.

        Args:
            use_gpu: Use GPU acceleration if available
            optimization: Contraction optimization strategy
        """
        self.use_gpu = use_gpu
        self.optimization = optimization

        if use_gpu:
            try:
                import cupy as cp

                self._xp = cp
                logger.info("Using CuPy for GPU-accelerated contraction")
            except ImportError:
                self._xp = np
                logger.warning("CuPy not available, falling back to NumPy")
        else:
            self._xp = np

    def contract_pair(
        self,
        tensor1: TensorNode,
        tensor2: TensorNode,
        shared_indices: list[tuple[str, str]],
    ) -> TensorNode:
        """Contract two tensors.

        Args:
            tensor1: First tensor
            tensor2: Second tensor
            shared_indices: Pairs of indices to contract

        Returns:
            Contracted tensor
        """
        return tensor1.contract(tensor2, shared_indices)

    def contract_network(
        self,
        tensors: list[TensorNode],
        contraction_order: list[tuple[int, int]] | None = None,
    ) -> TensorNode:
        """Contract an entire tensor network.

        Args:
            tensors: List of tensors in the network
            contraction_order: Order of pairwise contractions

        Returns:
            Final contracted tensor
        """
        if not tensors:
            raise ValueError("Empty tensor network")

        if len(tensors) == 1:
            return tensors[0]

        # Use provided order or find optimal
        if contraction_order is None:
            contraction_order = self._find_optimal_order(tensors)

        # Perform contractions
        working_tensors = list(tensors)

        for i, j in contraction_order:
            if i >= len(working_tensors) or j >= len(working_tensors):
                continue

            t1, t2 = working_tensors[i], working_tensors[j]

            # Find shared indices
            shared = []
            for idx1 in t1.indices:
                for idx2 in t2.indices:
                    if idx1 == idx2:
                        shared.append((idx1, idx2))

            if shared:
                contracted = self.contract_pair(t1, t2, shared)

                # Update tensor list
                working_tensors[min(i, j)] = contracted
                working_tensors.pop(max(i, j))

        return working_tensors[0]

    def _find_optimal_order(
        self,
        tensors: list[TensorNode],
    ) -> list[tuple[int, int]]:
        """Find optimal contraction order using greedy algorithm.

        Args:
            tensors: List of tensors

        Returns:
            Ordered list of contraction pairs
        """
        n = len(tensors)
        order = []

        # Greedy: contract smallest tensors first
        remaining = list(range(n))

        while len(remaining) > 1:
            # Find pair with minimum contraction cost
            best_cost = float("inf")
            best_pair = (0, 1)

            for i in range(len(remaining)):
                for j in range(i + 1, len(remaining)):
                    cost = (
                        tensors[remaining[i]].total_elements + tensors[remaining[j]].total_elements
                    )
                    if cost < best_cost:
                        best_cost = cost
                        best_pair = (remaining[i], remaining[j])

            order.append(best_pair)
            remaining.remove(best_pair[1])

        return order


class TensorNetwork:
    """General tensor network representation.

    Provides unified interface for different tensor network types.
    """

    def __init__(
        self,
        network_type: TensorNetworkType = TensorNetworkType.MPS,
    ) -> None:
        """Initialize tensor network.

        Args:
            network_type: Type of tensor network architecture
        """
        self.network_type = network_type
        self.tensors: list[TensorNode] = []
        self.edges: list[tuple[int, str, int, str]] = []  # (tensor1, idx1, tensor2, idx2)
        self._contraction_engine = TensorContraction()

    def add_tensor(
        self,
        data: np.ndarray,
        indices: list[str] | None = None,
    ) -> int:
        """Add a tensor to the network.

        Args:
            data: Tensor data
            indices: Index names

        Returns:
            Index of added tensor
        """
        if indices is None:
            indices = [f"i{i}" for i in range(len(data.shape))]

        node = TensorNode(data=data, indices=indices)
        self.tensors.append(node)
        return len(self.tensors) - 1

    def connect(
        self,
        tensor1: int,
        index1: str,
        tensor2: int,
        index2: str,
    ) -> None:
        """Connect two tensors by identifying indices.

        Args:
            tensor1: Index of first tensor
            index1: Index name on first tensor
            tensor2: Index of second tensor
            index2: Index name on second tensor
        """
        self.edges.append((tensor1, index1, tensor2, index2))

    def contract(self) -> TensorNode:
        """Contract the entire network.

        Returns:
            Result of contraction
        """
        return self._contraction_engine.contract_network(self.tensors)

    def to_statevector(self, num_qubits: int) -> np.ndarray:
        """Contract network and return as state vector.

        Args:
            num_qubits: Number of qubits

        Returns:
            State vector as numpy array
        """
        result = self.contract()
        dim = 2**num_qubits
        return result.data.reshape(dim)

    @property
    def num_tensors(self) -> int:
        """Number of tensors in network."""
        return len(self.tensors)

    @property
    def total_parameters(self) -> int:
        """Total number of parameters in network."""
        return sum(t.total_elements for t in self.tensors)


@dataclass
class MPSState(TensorNetwork):
    """Matrix Product State representation.

    Efficient representation of quantum states with limited entanglement.

    Attributes:
        num_sites: Number of lattice sites (qubits)
        bond_dimension: Maximum bond dimension
        physical_dimension: Local Hilbert space dimension (default 2 for qubits)
    """

    num_sites: int = 1
    bond_dimension: int = 32
    physical_dimension: int = 2

    def __post_init__(self) -> None:
        """Initialize MPS structure."""
        super().__init__(TensorNetworkType.MPS)
        self._initialize_random()

    def _initialize_random(self) -> None:
        """Initialize with random tensors."""
        for i in range(self.num_sites):
            if i == 0:
                # Left boundary: (physical, right_bond)
                shape = (self.physical_dimension, min(self.bond_dimension, self.physical_dimension))
            elif i == self.num_sites - 1:
                # Right boundary: (left_bond, physical)
                left_dim = min(self.bond_dimension, self.physical_dimension**i)
                shape = (left_dim, self.physical_dimension)
            else:
                # Bulk: (left_bond, physical, right_bond)
                left_dim = min(self.bond_dimension, self.physical_dimension**i)
                right_dim = min(
                    self.bond_dimension, self.physical_dimension ** (self.num_sites - i - 1)
                )
                shape = (left_dim, self.physical_dimension, right_dim)

            data = np.random.randn(*shape) + 1j * np.random.randn(*shape)
            data /= np.linalg.norm(data)

            indices = [f"l{i}", f"p{i}", f"r{i}"][: len(shape)]
            self.add_tensor(data, indices)

    @classmethod
    def from_statevector(
        cls,
        statevector: np.ndarray,
        bond_dimension: int = 32,
    ) -> MPSState:
        """Create MPS from state vector using SVD.

        Args:
            statevector: Full state vector
            bond_dimension: Maximum bond dimension

        Returns:
            MPS representation
        """
        dim = len(statevector)
        num_sites = int(np.log2(dim))

        mps = cls(
            num_sites=num_sites,
            bond_dimension=bond_dimension,
        )

        # SVD decomposition
        remaining = statevector.reshape(2, -1)
        mps.tensors = []

        for i in range(num_sites - 1):
            u, s, vh = np.linalg.svd(remaining, full_matrices=False)

            # Truncate to bond dimension
            chi = min(len(s), bond_dimension)
            u = u[:, :chi]
            s = s[:chi]
            vh = vh[:chi, :]

            mps.add_tensor(u, [f"l{i}", f"r{i}"])

            remaining = np.diag(s) @ vh
            if i < num_sites - 2:
                remaining = remaining.reshape(chi * 2, -1)

        mps.add_tensor(remaining, [f"l{num_sites-1}", f"p{num_sites-1}"])

        return mps

    def get_overlap(self, other: MPSState) -> complex:
        """Compute overlap ⟨ψ|φ⟩.

        Args:
            other: Other MPS state

        Returns:
            Complex overlap
        """
        if self.num_sites != other.num_sites:
            raise ValueError("MPS must have same number of sites")

        # Contract from left to right
        result = np.array([[1.0 + 0j]])

        for i in range(self.num_sites):
            # Contract with conjugated self and other
            self_tensor = np.conj(self.tensors[i].data)
            other_tensor = other.tensors[i].data

            # Simplified contraction
            result = np.tensordot(result, self_tensor, axes=1)
            result = np.tensordot(result, other_tensor, axes=1)

        return complex(result.flat[0])

    def normalize(self) -> MPSState:
        """Normalize the MPS.

        Returns:
            Normalized MPS
        """
        norm = np.sqrt(np.abs(self.get_overlap(self)))

        if norm > 0:
            # Normalize first tensor
            self.tensors[0].data /= norm

        return self

    def expectation_value(self, operator: np.ndarray, sites: list[int]) -> complex:
        """Compute expectation value of local operator.

        Args:
            operator: Local operator matrix
            sites: Sites the operator acts on

        Returns:
            Expectation value
        """
        # Simplified implementation for single-site operators
        if len(sites) != 1:
            raise NotImplementedError("Only single-site operators supported")

        site = sites[0]
        tensor = self.tensors[site]

        # ⟨ψ|O|ψ⟩ at site
        contracted = np.tensordot(tensor.data, operator, axes=1)
        contracted = np.tensordot(np.conj(tensor.data), contracted, axes=([0, 1], [0, 1]))

        return complex(contracted)

    def entanglement_entropy(self, site: int) -> float:
        """Compute entanglement entropy at bipartition.

        Args:
            site: Site index for bipartition (left of site | right of site)

        Returns:
            Von Neumann entropy
        """
        # Contract left part up to site
        if site == 0 or site >= self.num_sites:
            return 0.0

        # SVD at the bond
        bond_tensor = self.tensors[site].data

        if len(bond_tensor.shape) == 2:
            u, s, vh = np.linalg.svd(bond_tensor)
        else:
            # Reshape for SVD
            shape = bond_tensor.shape
            reshaped = bond_tensor.reshape(shape[0], -1)
            u, s, vh = np.linalg.svd(reshaped)

        # Normalize singular values
        s = s[s > 1e-15]
        s = s / np.linalg.norm(s)

        # Von Neumann entropy
        entropy = -np.sum(s**2 * np.log2(s**2 + 1e-15))

        return float(entropy)


@dataclass
class PEPSState(TensorNetwork):
    """Projected Entangled Pair States for 2D systems.

    Efficient representation of 2D quantum states.

    Attributes:
        rows: Number of rows in lattice
        cols: Number of columns in lattice
        bond_dimension: Virtual bond dimension
        physical_dimension: Local Hilbert space dimension
    """

    rows: int = 4
    cols: int = 4
    bond_dimension: int = 4
    physical_dimension: int = 2

    def __post_init__(self) -> None:
        """Initialize PEPS structure."""
        super().__init__(TensorNetworkType.PEPS)
        self._initialize_random()

    def _initialize_random(self) -> None:
        """Initialize with random tensors."""
        for i in range(self.rows):
            for j in range(self.cols):
                # Determine bond dimensions for each direction
                dims = [self.physical_dimension]  # Physical

                if i > 0:  # Up bond
                    dims.append(self.bond_dimension)
                if j > 0:  # Left bond
                    dims.append(self.bond_dimension)
                if i < self.rows - 1:  # Down bond
                    dims.append(self.bond_dimension)
                if j < self.cols - 1:  # Right bond
                    dims.append(self.bond_dimension)

                data = np.random.randn(*dims) + 1j * np.random.randn(*dims)
                data /= np.linalg.norm(data)

                self.add_tensor(data, [f"t_{i}_{j}_{k}" for k in range(len(dims))])

    @property
    def num_sites(self) -> int:
        """Total number of lattice sites."""
        return self.rows * self.cols


@dataclass
class TTNState(TensorNetwork):
    """Tree Tensor Network state.

    Hierarchical tensor network for multi-scale quantum systems.

    Attributes:
        num_leaves: Number of physical sites (leaves)
        bond_dimension: Virtual bond dimension
        branching_factor: Number of children per node
    """

    num_leaves: int = 8
    bond_dimension: int = 16
    branching_factor: int = 2

    def __post_init__(self) -> None:
        """Initialize TTN structure."""
        super().__init__(TensorNetworkType.TTN)
        self._initialize_tree()

    def _initialize_tree(self) -> None:
        """Initialize tree structure with random tensors."""
        # Build from leaves to root
        current_level = self.num_leaves
        level = 0

        while current_level > 1:
            nodes_at_level = current_level // self.branching_factor

            for i in range(nodes_at_level):
                if level == 0:
                    # Leaf nodes: physical index + parent bond
                    dims = [2, self.bond_dimension]
                elif nodes_at_level == 1:
                    # Root node: only child bonds
                    dims = [self.bond_dimension] * self.branching_factor
                else:
                    # Internal nodes: child bonds + parent bond
                    dims = [self.bond_dimension] * (self.branching_factor + 1)

                data = np.random.randn(*dims) + 1j * np.random.randn(*dims)
                data /= np.linalg.norm(data)

                self.add_tensor(data, [f"ttn_{level}_{i}_{k}" for k in range(len(dims))])

            current_level = nodes_at_level
            level += 1


# Utility functions


def create_mps_product_state(
    bitstring: str,
    bond_dimension: int = 1,
) -> MPSState:
    """Create MPS for a computational basis state.

    Args:
        bitstring: Binary string (e.g., "0101")
        bond_dimension: Bond dimension (1 for product state)

    Returns:
        MPS representing |bitstring⟩
    """
    num_sites = len(bitstring)
    mps = MPSState(num_sites=num_sites, bond_dimension=bond_dimension)
    mps.tensors = []

    for i, bit in enumerate(bitstring):
        if bit == "0":
            data = np.array([1.0 + 0j, 0.0 + 0j])
        else:
            data = np.array([0.0 + 0j, 1.0 + 0j])

        if i == 0:
            data = data.reshape(2, 1)
        elif i == num_sites - 1:
            data = data.reshape(1, 2)
        else:
            data = data.reshape(1, 2, 1)

        mps.add_tensor(data, [f"t{i}"])

    return mps


def create_mps_ghz_state(num_sites: int, bond_dimension: int = 2) -> MPSState:
    """Create MPS for GHZ state.

    Args:
        num_sites: Number of qubits
        bond_dimension: Bond dimension (2 for exact GHZ)

    Returns:
        MPS representing (|00...0⟩ + |11...1⟩)/√2
    """
    mps = MPSState(num_sites=num_sites, bond_dimension=bond_dimension)
    mps.tensors = []

    # First tensor
    A0 = np.zeros((2, 2), dtype=complex)
    A0[0, 0] = 1.0 / np.sqrt(2)
    A0[1, 1] = 1.0 / np.sqrt(2)
    mps.add_tensor(A0, ["p0", "r0"])

    # Middle tensors
    for i in range(1, num_sites - 1):
        A = np.zeros((2, 2, 2), dtype=complex)
        A[0, 0, 0] = 1.0
        A[1, 1, 1] = 1.0
        mps.add_tensor(A, [f"l{i}", f"p{i}", f"r{i}"])

    # Last tensor
    AN = np.zeros((2, 2), dtype=complex)
    AN[0, 0] = 1.0
    AN[1, 1] = 1.0
    mps.add_tensor(AN, [f"l{num_sites-1}", f"p{num_sites-1}"])

    return mps
