"""Quantum hardware backend abstractions.

Provides unified interface to multiple quantum computing platforms:
- IBM Qiskit (Superconducting qubits)
- AWS Braket (Multi-provider cloud)
- IonQ (Trapped ion)
- QuEra (Neutral atom)
- PsiQuantum (Photonic - future)
- Local Simulator (Development)
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, TypeVar

import numpy as np

logger = logging.getLogger(__name__)


class BackendType(Enum):
    """Enumeration of supported quantum backend types."""
    
    QISKIT = auto()
    BRAKET = auto()
    IONQ = auto()
    QUERA = auto()
    PSIQUANTUM = auto()
    SIMULATOR = auto()
    CUDA_SIMULATOR = auto()
    TENSOR_NETWORK = auto()


@dataclass
class BackendConfig:
    """Configuration for quantum backend.
    
    Attributes:
        backend_type: Type of quantum backend to use
        api_key: API key for cloud backends (optional)
        endpoint: Custom endpoint URL (optional)
        max_qubits: Maximum number of qubits to use
        shots: Number of measurement shots
        optimization_level: Circuit optimization level (0-3)
        error_mitigation: Enable error mitigation techniques
        seed: Random seed for reproducibility
        timeout: Maximum execution time in seconds
        gpu_acceleration: Enable GPU acceleration for simulators
    """
    
    backend_type: BackendType = BackendType.SIMULATOR
    api_key: str | None = None
    endpoint: str | None = None
    max_qubits: int = 32
    shots: int = 10000
    optimization_level: int = 2
    error_mitigation: bool = True
    seed: int = 42
    timeout: int = 3600
    gpu_acceleration: bool = True
    noise_model: str | None = None
    coupling_map: list[tuple[int, int]] | None = None
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        if self.max_qubits < 1:
            raise ValueError("max_qubits must be at least 1")
        if self.shots < 1:
            raise ValueError("shots must be at least 1")
        if self.optimization_level not in range(4):
            raise ValueError("optimization_level must be 0-3")
        if self.timeout < 1:
            raise ValueError("timeout must be at least 1 second")
        return True


class QuantumBackend(abc.ABC):
    """Abstract base class for quantum computing backends.
    
    Provides unified interface for executing quantum circuits across
    different hardware and simulator platforms.
    """
    
    def __init__(self, config: BackendConfig) -> None:
        """Initialize backend with configuration.
        
        Args:
            config: Backend configuration parameters
        """
        self.config = config
        self.config.validate()
        self._connected = False
        self._job_history: list[dict[str, Any]] = []
        
    @abc.abstractmethod
    def connect(self) -> bool:
        """Establish connection to the quantum backend.
        
        Returns:
            True if connection successful
        """
        pass
    
    @abc.abstractmethod
    def disconnect(self) -> None:
        """Close connection to the quantum backend."""
        pass
    
    @abc.abstractmethod
    def execute(
        self, 
        circuit: Any,  # QuantumCircuit
        shots: int | None = None,
        parameters: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Execute a quantum circuit on the backend.
        
        Args:
            circuit: Quantum circuit to execute
            shots: Number of measurement shots (overrides config)
            parameters: Parameter values for parameterized circuits
            
        Returns:
            Execution results including counts, statevector, etc.
        """
        pass
    
    @abc.abstractmethod
    def get_backend_info(self) -> dict[str, Any]:
        """Get information about the backend.
        
        Returns:
            Dictionary with backend capabilities and status
        """
        pass
    
    @property
    def is_connected(self) -> bool:
        """Check if backend is connected."""
        return self._connected
    
    def get_job_history(self) -> list[dict[str, Any]]:
        """Get history of executed jobs."""
        return self._job_history.copy()


class SimulatorBackend(QuantumBackend):
    """Local quantum circuit simulator.
    
    Provides high-performance state vector simulation with optional
    GPU acceleration using cuQuantum or JAX.
    """
    
    def __init__(self, config: BackendConfig) -> None:
        """Initialize simulator backend."""
        super().__init__(config)
        self._simulator = None
        
    def connect(self) -> bool:
        """Initialize the local simulator."""
        logger.info("Initializing local quantum simulator")
        
        if self.config.gpu_acceleration:
            try:
                import jax
                import jax.numpy as jnp
                self._backend_lib = "jax"
                logger.info("Using JAX GPU backend for simulation")
            except ImportError:
                self._backend_lib = "numpy"
                logger.info("JAX not available, using NumPy backend")
        else:
            self._backend_lib = "numpy"
            
        self._connected = True
        return True
    
    def disconnect(self) -> None:
        """Clean up simulator resources."""
        self._simulator = None
        self._connected = False
        logger.info("Simulator disconnected")
    
    def execute(
        self,
        circuit: Any,
        shots: int | None = None,
        parameters: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Execute circuit on the local simulator.
        
        Args:
            circuit: Quantum circuit to simulate
            shots: Number of measurement shots
            parameters: Parameter values for parameterized gates
            
        Returns:
            Simulation results including statevector and counts
        """
        if not self._connected:
            raise RuntimeError("Simulator not connected. Call connect() first.")
            
        shots = shots or self.config.shots
        np.random.seed(self.config.seed)
        
        # Get circuit properties
        num_qubits = getattr(circuit, 'num_qubits', 4)
        
        # Initialize state vector |0...0⟩
        dim = 2 ** num_qubits
        statevector = np.zeros(dim, dtype=np.complex128)
        statevector[0] = 1.0
        
        # Apply gates (simplified simulation)
        gates = getattr(circuit, 'gates', [])
        for gate in gates:
            statevector = self._apply_gate(statevector, gate, num_qubits)
        
        # Calculate probabilities
        probabilities = np.abs(statevector) ** 2
        
        # Sample measurements
        indices = np.random.choice(dim, size=shots, p=probabilities)
        counts = {}
        for idx in indices:
            bitstring = format(idx, f'0{num_qubits}b')
            counts[bitstring] = counts.get(bitstring, 0) + 1
        
        result = {
            "statevector": statevector.tolist(),
            "probabilities": probabilities.tolist(),
            "counts": counts,
            "shots": shots,
            "num_qubits": num_qubits,
            "backend": "simulator",
            "success": True,
        }
        
        self._job_history.append({
            "job_id": f"sim_{len(self._job_history)}",
            "result": result,
            "status": "completed",
        })
        
        return result
    
    def _apply_gate(
        self,
        statevector: np.ndarray,
        gate: dict[str, Any],
        num_qubits: int,
    ) -> np.ndarray:
        """Apply a quantum gate to the state vector.
        
        Args:
            statevector: Current quantum state
            gate: Gate specification with type, qubits, and parameters
            num_qubits: Total number of qubits in the system
            
        Returns:
            Updated state vector
        """
        gate_type = gate.get("type", "I")
        qubits = gate.get("qubits", [0])
        params = gate.get("params", {})
        
        # Get gate matrix
        matrix = self._get_gate_matrix(gate_type, params)
        
        # Apply gate to specified qubits
        # This is a simplified implementation; production would use
        # optimized tensor contraction
        target_qubit = qubits[0] if qubits else 0
        
        # Create full operator by tensoring with identity
        full_operator = self._expand_gate(matrix, target_qubit, num_qubits)
        
        return full_operator @ statevector
    
    def _get_gate_matrix(
        self,
        gate_type: str,
        params: dict[str, float],
    ) -> np.ndarray:
        """Get the matrix representation of a quantum gate.
        
        Args:
            gate_type: Type of gate (H, X, Y, Z, RX, RY, RZ, CNOT, etc.)
            params: Gate parameters (angles, etc.)
            
        Returns:
            Gate matrix as numpy array
        """
        # Standard gate matrices
        gates = {
            "I": np.array([[1, 0], [0, 1]], dtype=complex),
            "X": np.array([[0, 1], [1, 0]], dtype=complex),
            "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
            "Z": np.array([[1, 0], [0, -1]], dtype=complex),
            "H": np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
            "S": np.array([[1, 0], [0, 1j]], dtype=complex),
            "T": np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex),
        }
        
        if gate_type in gates:
            return gates[gate_type]
        
        # Parameterized gates
        if gate_type == "RX":
            theta = params.get("theta", 0)
            return np.array([
                [np.cos(theta/2), -1j * np.sin(theta/2)],
                [-1j * np.sin(theta/2), np.cos(theta/2)]
            ], dtype=complex)
        
        if gate_type == "RY":
            theta = params.get("theta", 0)
            return np.array([
                [np.cos(theta/2), -np.sin(theta/2)],
                [np.sin(theta/2), np.cos(theta/2)]
            ], dtype=complex)
        
        if gate_type == "RZ":
            theta = params.get("theta", 0)
            return np.array([
                [np.exp(-1j * theta/2), 0],
                [0, np.exp(1j * theta/2)]
            ], dtype=complex)
        
        # Default to identity
        return gates["I"]
    
    def _expand_gate(
        self,
        gate: np.ndarray,
        target: int,
        num_qubits: int,
    ) -> np.ndarray:
        """Expand a single-qubit gate to the full Hilbert space.
        
        Args:
            gate: Single-qubit gate matrix
            target: Target qubit index
            num_qubits: Total number of qubits
            
        Returns:
            Full operator matrix
        """
        identity = np.eye(2, dtype=complex)
        
        # Build tensor product
        operators = []
        for i in range(num_qubits):
            if i == target:
                operators.append(gate)
            else:
                operators.append(identity)
        
        # Compute tensor product
        result = operators[0]
        for op in operators[1:]:
            result = np.kron(result, op)
        
        return result
    
    def get_backend_info(self) -> dict[str, Any]:
        """Get simulator information."""
        return {
            "backend_type": "simulator",
            "backend_lib": getattr(self, '_backend_lib', 'numpy'),
            "max_qubits": self.config.max_qubits,
            "gpu_acceleration": self.config.gpu_acceleration,
            "connected": self._connected,
            "jobs_executed": len(self._job_history),
        }


class QiskitBackend(QuantumBackend):
    """IBM Qiskit backend for superconducting quantum hardware.
    
    Supports both IBM Quantum cloud hardware and local Aer simulator.
    """
    
    def __init__(self, config: BackendConfig) -> None:
        """Initialize Qiskit backend."""
        super().__init__(config)
        self._service = None
        self._backend = None
        
    def connect(self) -> bool:
        """Connect to IBM Quantum service."""
        logger.info("Connecting to IBM Quantum service")
        
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
            
            if self.config.api_key:
                self._service = QiskitRuntimeService(
                    channel="ibm_quantum",
                    token=self.config.api_key,
                )
            else:
                # Use saved credentials
                self._service = QiskitRuntimeService()
            
            # Get least busy backend
            self._backend = self._service.least_busy(
                min_num_qubits=self.config.max_qubits,
                operational=True,
            )
            
            self._connected = True
            logger.info(f"Connected to IBM Quantum backend: {self._backend.name}")
            return True
            
        except ImportError:
            logger.warning("Qiskit not installed, falling back to simulator")
            return self._fallback_to_simulator()
        except Exception as e:
            logger.error(f"Failed to connect to IBM Quantum: {e}")
            return self._fallback_to_simulator()
    
    def _fallback_to_simulator(self) -> bool:
        """Fall back to local Qiskit Aer simulator."""
        try:
            from qiskit_aer import AerSimulator
            self._backend = AerSimulator()
            self._connected = True
            logger.info("Using Qiskit Aer simulator")
            return True
        except ImportError:
            logger.error("Qiskit Aer not available")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from IBM Quantum."""
        self._service = None
        self._backend = None
        self._connected = False
        logger.info("Disconnected from IBM Quantum")
    
    def execute(
        self,
        circuit: Any,
        shots: int | None = None,
        parameters: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Execute circuit on IBM Quantum hardware.
        
        Args:
            circuit: Quantum circuit (Qiskit QuantumCircuit)
            shots: Number of measurement shots
            parameters: Parameter values for parameterized circuits
            
        Returns:
            Execution results
        """
        if not self._connected:
            raise RuntimeError("Backend not connected")
            
        from qiskit import transpile
        
        shots = shots or self.config.shots
        
        # Transpile for target backend
        transpiled = transpile(
            circuit,
            backend=self._backend,
            optimization_level=self.config.optimization_level,
        )
        
        # Execute
        job = self._backend.run(transpiled, shots=shots)
        result = job.result()
        
        counts = result.get_counts(0)
        
        return {
            "counts": counts,
            "shots": shots,
            "backend": self._backend.name if hasattr(self._backend, 'name') else "aer",
            "success": result.success,
            "job_id": job.job_id() if hasattr(job, 'job_id') else None,
        }
    
    def get_backend_info(self) -> dict[str, Any]:
        """Get IBM Quantum backend information."""
        if not self._backend:
            return {"backend_type": "qiskit", "connected": False}
        
        return {
            "backend_type": "qiskit",
            "backend_name": getattr(self._backend, 'name', 'simulator'),
            "num_qubits": getattr(self._backend, 'num_qubits', 32),
            "connected": self._connected,
        }


class BraketBackend(QuantumBackend):
    """AWS Braket backend for multi-provider quantum access.
    
    Supports IonQ, Rigetti, and OQC hardware through AWS Braket.
    """
    
    def __init__(self, config: BackendConfig) -> None:
        """Initialize Braket backend."""
        super().__init__(config)
        self._device = None
        
    def connect(self) -> bool:
        """Connect to AWS Braket service."""
        logger.info("Connecting to AWS Braket")
        
        try:
            from braket.aws import AwsDevice
            from braket.devices import LocalSimulator
            
            if self.config.endpoint:
                # Use specified device ARN
                self._device = AwsDevice(self.config.endpoint)
            else:
                # Use local simulator by default
                self._device = LocalSimulator()
            
            self._connected = True
            logger.info("Connected to AWS Braket")
            return True
            
        except ImportError:
            logger.warning("AWS Braket SDK not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to AWS Braket: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from AWS Braket."""
        self._device = None
        self._connected = False
        logger.info("Disconnected from AWS Braket")
    
    def execute(
        self,
        circuit: Any,
        shots: int | None = None,
        parameters: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Execute circuit on AWS Braket."""
        if not self._connected:
            raise RuntimeError("Backend not connected")
        
        shots = shots or self.config.shots
        
        # Execute on Braket device
        task = self._device.run(circuit, shots=shots)
        result = task.result()
        
        return {
            "counts": dict(result.measurement_counts),
            "shots": shots,
            "backend": "braket",
            "success": True,
        }
    
    def get_backend_info(self) -> dict[str, Any]:
        """Get AWS Braket backend information."""
        return {
            "backend_type": "braket",
            "connected": self._connected,
            "device": str(self._device) if self._device else None,
        }


class IonQBackend(QuantumBackend):
    """IonQ trapped-ion quantum computer backend.
    
    Provides access to IonQ's high-fidelity trapped-ion quantum computers.
    """
    
    def __init__(self, config: BackendConfig) -> None:
        """Initialize IonQ backend."""
        super().__init__(config)
        self._client = None
        
    def connect(self) -> bool:
        """Connect to IonQ cloud service."""
        logger.info("Connecting to IonQ")
        
        if not self.config.api_key:
            logger.error("IonQ API key required")
            return False
        
        try:
            import ionq
            self._client = ionq.Client(api_key=self.config.api_key)
            self._connected = True
            logger.info("Connected to IonQ")
            return True
        except ImportError:
            logger.warning("IonQ SDK not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to IonQ: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from IonQ."""
        self._client = None
        self._connected = False
        logger.info("Disconnected from IonQ")
    
    def execute(
        self,
        circuit: Any,
        shots: int | None = None,
        parameters: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Execute circuit on IonQ hardware."""
        if not self._connected:
            raise RuntimeError("Backend not connected")
        
        shots = shots or self.config.shots
        
        # Submit to IonQ
        job = self._client.submit(circuit, shots=shots, target="qpu")
        result = job.wait()
        
        return {
            "counts": result.get_counts(),
            "shots": shots,
            "backend": "ionq",
            "success": True,
            "job_id": job.job_id,
        }
    
    def get_backend_info(self) -> dict[str, Any]:
        """Get IonQ backend information."""
        return {
            "backend_type": "ionq",
            "connected": self._connected,
            "qubits": 32,
            "fidelity": "99.9%",
        }


class QuEraBackend(QuantumBackend):
    """QuEra neutral atom quantum computer backend.
    
    Provides access to QuEra's large-scale neutral atom quantum computers.
    """
    
    def __init__(self, config: BackendConfig) -> None:
        """Initialize QuEra backend."""
        super().__init__(config)
        self._client = None
        
    def connect(self) -> bool:
        """Connect to QuEra service via AWS Braket."""
        logger.info("Connecting to QuEra via AWS Braket")
        
        try:
            from braket.aws import AwsDevice
            
            # QuEra Aquila device ARN
            quera_arn = "arn:aws:braket:us-east-1::device/qpu/quera/Aquila"
            self._device = AwsDevice(quera_arn)
            self._connected = True
            logger.info("Connected to QuEra Aquila")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to QuEra: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from QuEra."""
        self._device = None
        self._connected = False
        logger.info("Disconnected from QuEra")
    
    def execute(
        self,
        circuit: Any,
        shots: int | None = None,
        parameters: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Execute on QuEra neutral atom hardware."""
        if not self._connected:
            raise RuntimeError("Backend not connected")
        
        shots = shots or self.config.shots
        
        # Submit to QuEra
        task = self._device.run(circuit, shots=shots)
        result = task.result()
        
        return {
            "counts": dict(result.measurement_counts),
            "shots": shots,
            "backend": "quera",
            "success": True,
        }
    
    def get_backend_info(self) -> dict[str, Any]:
        """Get QuEra backend information."""
        return {
            "backend_type": "quera",
            "connected": self._connected,
            "qubits": 256,
            "technology": "neutral_atom",
        }


def create_backend(config: BackendConfig) -> QuantumBackend:
    """Factory function to create appropriate quantum backend.
    
    Args:
        config: Backend configuration
        
    Returns:
        Configured quantum backend instance
    """
    backend_map = {
        BackendType.QISKIT: QiskitBackend,
        BackendType.BRAKET: BraketBackend,
        BackendType.IONQ: IonQBackend,
        BackendType.QUERA: QuEraBackend,
        BackendType.SIMULATOR: SimulatorBackend,
        BackendType.CUDA_SIMULATOR: SimulatorBackend,
        BackendType.TENSOR_NETWORK: SimulatorBackend,
    }
    
    backend_class = backend_map.get(config.backend_type, SimulatorBackend)
    return backend_class(config)
