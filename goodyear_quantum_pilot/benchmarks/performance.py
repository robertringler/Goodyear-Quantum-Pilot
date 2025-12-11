"""Performance Benchmarking Module.

Provides comprehensive performance benchmarking for:
- Quantum algorithm execution time
- Classical baseline comparisons
- Hardware acceleration metrics
- Scalability analysis
- Cost efficiency evaluation
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class ComputeBackend(Enum):
    """Available compute backends."""
    
    CPU = auto()
    GPU_NVIDIA = auto()
    GPU_AMD = auto()
    QUANTUM_SIMULATOR = auto()
    QUANTUM_IONQ = auto()
    QUANTUM_IBMQ = auto()
    QUANTUM_RIGETTI = auto()
    QUANTUM_QUERA = auto()


@dataclass
class BenchmarkResult:
    """Performance benchmark result.
    
    Attributes:
        name: Benchmark name
        backend: Compute backend used
        execution_time: Total execution time (s)
        memory_usage: Peak memory usage (MB)
        flops: Floating point operations performed
        energy_consumption: Energy consumed (J)
        accuracy: Result accuracy vs reference
        cost: Compute cost ($)
    """
    
    name: str
    backend: ComputeBackend
    execution_time: float
    memory_usage: float = 0.0
    flops: float = 0.0
    energy_consumption: float = 0.0
    accuracy: float = 1.0
    cost: float = 0.0
    
    @property
    def throughput(self) -> float:
        """Compute throughput (GFLOPS)."""
        if self.execution_time > 0:
            return self.flops / self.execution_time / 1e9
        return 0.0
    
    @property
    def efficiency(self) -> float:
        """Compute efficiency (GFLOPS/W)."""
        if self.energy_consumption > 0:
            return self.throughput / (self.energy_consumption / self.execution_time)
        return 0.0


@dataclass
class ScalingResult:
    """Scaling analysis result.
    
    Attributes:
        problem_sizes: Problem sizes tested
        execution_times: Execution times for each size
        fitted_exponent: Fitted scaling exponent
        fitted_coefficient: Fitted scaling coefficient
        r_squared: Fit quality (R²)
    """
    
    problem_sizes: NDArray[np.float64]
    execution_times: NDArray[np.float64]
    fitted_exponent: float = 0.0
    fitted_coefficient: float = 0.0
    r_squared: float = 0.0
    
    @property
    def scaling_class(self) -> str:
        """Determine complexity class."""
        exp = self.fitted_exponent
        if exp < 1.5:
            return "O(n)"
        elif exp < 2.5:
            return "O(n²)"
        elif exp < 3.5:
            return "O(n³)"
        else:
            return f"O(n^{exp:.1f})"


class PerformanceBenchmark:
    """General performance benchmarking framework.
    
    Provides timing, profiling, and resource monitoring
    for arbitrary computational tasks.
    
    Example:
        >>> bench = PerformanceBenchmark()
        >>> 
        >>> @bench.benchmark("my_function")
        ... def my_function(n):
        ...     return sum(range(n))
        >>> 
        >>> result = my_function(1000000)
        >>> print(bench.get_results())
    """
    
    def __init__(
        self,
        warmup_iterations: int = 3,
        timing_iterations: int = 10,
    ) -> None:
        """Initialize benchmarking framework.
        
        Args:
            warmup_iterations: Warmup iterations before timing
            timing_iterations: Number of timing iterations
        """
        self.warmup = warmup_iterations
        self.iterations = timing_iterations
        self.results: list[BenchmarkResult] = []
        
        logger.info(f"PerformanceBenchmark: {timing_iterations} iterations")
    
    def benchmark(
        self,
        name: str,
        backend: ComputeBackend = ComputeBackend.CPU,
    ) -> Callable:
        """Decorator to benchmark a function.
        
        Args:
            name: Benchmark name
            backend: Compute backend
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                # Warmup
                for _ in range(self.warmup):
                    func(*args, **kwargs)
                
                # Timing
                times = []
                for _ in range(self.iterations):
                    start = time.perf_counter()
                    result = func(*args, **kwargs)
                    end = time.perf_counter()
                    times.append(end - start)
                
                # Record result
                self.results.append(BenchmarkResult(
                    name=name,
                    backend=backend,
                    execution_time=np.mean(times),
                ))
                
                return result
            
            return wrapper
        return decorator
    
    def time_function(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: dict | None = None,
        name: str = "unnamed",
        backend: ComputeBackend = ComputeBackend.CPU,
    ) -> BenchmarkResult:
        """Time a function execution.
        
        Args:
            func: Function to time
            args: Function arguments
            kwargs: Function keyword arguments
            name: Benchmark name
            backend: Compute backend
            
        Returns:
            Benchmark result
        """
        kwargs = kwargs or {}
        
        # Warmup
        for _ in range(self.warmup):
            func(*args, **kwargs)
        
        # Timing
        times = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            func(*args, **kwargs)
            end = time.perf_counter()
            times.append(end - start)
        
        result = BenchmarkResult(
            name=name,
            backend=backend,
            execution_time=np.mean(times),
        )
        self.results.append(result)
        
        return result
    
    def get_results(self) -> list[BenchmarkResult]:
        """Get all benchmark results."""
        return self.results
    
    def get_summary(self) -> dict[str, Any]:
        """Get summary of all benchmarks."""
        if not self.results:
            return {}
        
        return {
            "total_benchmarks": len(self.results),
            "fastest": min(self.results, key=lambda r: r.execution_time).name,
            "slowest": max(self.results, key=lambda r: r.execution_time).name,
            "average_time": np.mean([r.execution_time for r in self.results]),
        }


class QuantumBenchmark:
    """Quantum algorithm benchmarking.
    
    Benchmarks quantum algorithms across:
    - Different qubit counts
    - Various circuit depths
    - Multiple backends
    - Noise levels
    
    Example:
        >>> qbench = QuantumBenchmark()
        >>> 
        >>> results = qbench.benchmark_vqe(
        ...     molecule_sizes=[2, 4, 6, 8],
        ...     backends=["simulator", "ionq"],
        ... )
    """
    
    # Quantum hardware costs ($/shot)
    HARDWARE_COSTS = {
        ComputeBackend.QUANTUM_SIMULATOR: 0.0,
        ComputeBackend.QUANTUM_IONQ: 0.01,
        ComputeBackend.QUANTUM_IBMQ: 0.008,
        ComputeBackend.QUANTUM_RIGETTI: 0.005,
        ComputeBackend.QUANTUM_QUERA: 0.015,
    }
    
    def __init__(
        self,
        shots: int = 1000,
        optimization_iterations: int = 100,
    ) -> None:
        """Initialize quantum benchmarking.
        
        Args:
            shots: Measurement shots per circuit
            optimization_iterations: VQE/QAOA iterations
        """
        self.shots = shots
        self.opt_iterations = optimization_iterations
        self.results: list[BenchmarkResult] = []
    
    def benchmark_vqe(
        self,
        qubit_counts: list[int],
        circuit_depth: int = 4,
        backends: list[ComputeBackend] | None = None,
    ) -> list[BenchmarkResult]:
        """Benchmark VQE algorithm.
        
        Args:
            qubit_counts: List of qubit counts to test
            circuit_depth: Ansatz circuit depth
            backends: Quantum backends to use
            
        Returns:
            Benchmark results for each configuration
        """
        backends = backends or [ComputeBackend.QUANTUM_SIMULATOR]
        results = []
        
        for n_qubits in qubit_counts:
            for backend in backends:
                # Simulate VQE execution time
                # Real implementation would call actual quantum hardware
                time_per_circuit = self._estimate_circuit_time(
                    n_qubits, circuit_depth, backend
                )
                
                total_time = (
                    time_per_circuit
                    * self.opt_iterations
                    * self.shots
                )
                
                # Memory scales exponentially for simulators
                if backend == ComputeBackend.QUANTUM_SIMULATOR:
                    memory = 2 ** n_qubits * 16 / 1e6  # MB
                else:
                    memory = n_qubits * 0.1  # Minimal for real hardware
                
                # Cost calculation
                cost = self.HARDWARE_COSTS.get(backend, 0.0) * self.shots * self.opt_iterations
                
                result = BenchmarkResult(
                    name=f"VQE_{n_qubits}q_{backend.name}",
                    backend=backend,
                    execution_time=total_time,
                    memory_usage=memory,
                    cost=cost,
                )
                results.append(result)
        
        self.results.extend(results)
        return results
    
    def benchmark_qaoa(
        self,
        problem_sizes: list[int],
        p_values: list[int] = [1, 2, 4],
        backends: list[ComputeBackend] | None = None,
    ) -> list[BenchmarkResult]:
        """Benchmark QAOA algorithm.
        
        Args:
            problem_sizes: Problem sizes (number of variables)
            p_values: QAOA p-values (circuit depth)
            backends: Quantum backends to use
            
        Returns:
            Benchmark results
        """
        backends = backends or [ComputeBackend.QUANTUM_SIMULATOR]
        results = []
        
        for size in problem_sizes:
            for p in p_values:
                for backend in backends:
                    time_per_circuit = self._estimate_circuit_time(
                        size, p * 2, backend
                    )
                    
                    total_time = (
                        time_per_circuit
                        * 2 * p  # 2p parameters to optimize
                        * self.opt_iterations
                        * self.shots
                    )
                    
                    result = BenchmarkResult(
                        name=f"QAOA_{size}v_p{p}_{backend.name}",
                        backend=backend,
                        execution_time=total_time,
                        cost=self.HARDWARE_COSTS.get(backend, 0.0) * self.shots * self.opt_iterations,
                    )
                    results.append(result)
        
        self.results.extend(results)
        return results
    
    def benchmark_monte_carlo(
        self,
        sample_counts: list[int],
        system_size: int = 100,
    ) -> list[BenchmarkResult]:
        """Benchmark Quantum Monte Carlo.
        
        Args:
            sample_counts: Number of samples to test
            system_size: System size (atoms/sites)
            
        Returns:
            Benchmark results
        """
        results = []
        
        for n_samples in sample_counts:
            # QMC scales roughly as O(N³) per sample
            time_per_sample = 1e-6 * system_size ** 3
            total_time = time_per_sample * n_samples
            
            result = BenchmarkResult(
                name=f"QMC_{n_samples}samples_{system_size}sites",
                backend=ComputeBackend.GPU_NVIDIA,  # QMC typically on GPU
                execution_time=total_time,
                flops=system_size ** 3 * n_samples * 100,  # Approximate
            )
            results.append(result)
        
        self.results.extend(results)
        return results
    
    def _estimate_circuit_time(
        self,
        n_qubits: int,
        depth: int,
        backend: ComputeBackend,
    ) -> float:
        """Estimate circuit execution time.
        
        Args:
            n_qubits: Number of qubits
            depth: Circuit depth
            backend: Compute backend
            
        Returns:
            Estimated time (s)
        """
        # Base gate times (s)
        gate_times = {
            ComputeBackend.QUANTUM_SIMULATOR: 1e-9,  # Nanoseconds on simulator
            ComputeBackend.QUANTUM_IONQ: 100e-6,  # 100 μs per gate
            ComputeBackend.QUANTUM_IBMQ: 50e-6,  # 50 μs per gate
            ComputeBackend.QUANTUM_RIGETTI: 30e-6,  # 30 μs per gate
            ComputeBackend.QUANTUM_QUERA: 1e-6,  # 1 μs (Rydberg)
        }
        
        gate_time = gate_times.get(backend, 1e-6)
        
        # Simulator scales exponentially
        if backend == ComputeBackend.QUANTUM_SIMULATOR:
            return gate_time * depth * 2 ** n_qubits
        
        return gate_time * depth * n_qubits
    
    def compute_quantum_advantage(
        self,
        quantum_result: BenchmarkResult,
        classical_result: BenchmarkResult,
    ) -> dict[str, float]:
        """Compute quantum advantage metrics.
        
        Args:
            quantum_result: Quantum benchmark result
            classical_result: Classical benchmark result
            
        Returns:
            Advantage metrics
        """
        speedup = classical_result.execution_time / quantum_result.execution_time
        cost_ratio = classical_result.cost / quantum_result.cost if quantum_result.cost > 0 else float("inf")
        
        return {
            "speedup": speedup,
            "time_saved_s": classical_result.execution_time - quantum_result.execution_time,
            "cost_ratio": cost_ratio,
            "quantum_advantage": speedup > 1,
        }


class ClassicalBenchmark:
    """Classical algorithm benchmarking.
    
    Provides baselines for quantum comparison:
    - DFT/HF calculations
    - Classical optimization
    - Monte Carlo methods
    - Machine learning inference
    """
    
    def __init__(self) -> None:
        """Initialize classical benchmarking."""
        self.results: list[BenchmarkResult] = []
    
    def benchmark_dft(
        self,
        system_sizes: list[int],
        basis: str = "6-31G*",
        backend: ComputeBackend = ComputeBackend.CPU,
    ) -> list[BenchmarkResult]:
        """Benchmark DFT calculations.
        
        Args:
            system_sizes: Number of atoms/electrons
            basis: Basis set
            backend: Compute backend
            
        Returns:
            Benchmark results
        """
        results = []
        
        for size in system_sizes:
            # DFT scales as O(N³) with N = basis functions
            n_basis = size * 10  # Approximate basis functions
            time = 1e-8 * n_basis ** 3
            
            if backend == ComputeBackend.GPU_NVIDIA:
                time *= 0.1  # 10x GPU speedup
            
            result = BenchmarkResult(
                name=f"DFT_{basis}_{size}atoms",
                backend=backend,
                execution_time=time,
                flops=n_basis ** 3 * 100,
            )
            results.append(result)
        
        self.results.extend(results)
        return results
    
    def benchmark_md(
        self,
        atom_counts: list[int],
        timesteps: int = 10000,
        backend: ComputeBackend = ComputeBackend.GPU_NVIDIA,
    ) -> list[BenchmarkResult]:
        """Benchmark molecular dynamics.
        
        Args:
            atom_counts: Number of atoms
            timesteps: Simulation timesteps
            backend: Compute backend
            
        Returns:
            Benchmark results
        """
        results = []
        
        for n_atoms in atom_counts:
            # MD scales as O(N log N) with neighbor lists
            time_per_step = 1e-9 * n_atoms * np.log(n_atoms)
            
            if backend == ComputeBackend.GPU_NVIDIA:
                time_per_step *= 0.01  # GPU highly parallel
            
            total_time = time_per_step * timesteps
            
            result = BenchmarkResult(
                name=f"MD_{n_atoms}atoms_{timesteps}steps",
                backend=backend,
                execution_time=total_time,
            )
            results.append(result)
        
        self.results.extend(results)
        return results
    
    def benchmark_fea(
        self,
        element_counts: list[int],
        degrees_of_freedom: int = 3,
        backend: ComputeBackend = ComputeBackend.CPU,
    ) -> list[BenchmarkResult]:
        """Benchmark finite element analysis.
        
        Args:
            element_counts: Number of elements
            degrees_of_freedom: DOFs per node
            backend: Compute backend
            
        Returns:
            Benchmark results
        """
        results = []
        
        for n_elements in element_counts:
            # FEA solving scales as O(N^1.5) to O(N^2) with sparse solvers
            n_dofs = n_elements * degrees_of_freedom
            time = 1e-7 * n_dofs ** 1.5
            
            result = BenchmarkResult(
                name=f"FEA_{n_elements}elements",
                backend=backend,
                execution_time=time,
            )
            results.append(result)
        
        self.results.extend(results)
        return results


class SpeedupAnalyzer:
    """Analyze speedup and performance gains.
    
    Computes:
    - Speedup factors
    - Parallel efficiency
    - Amdahl's law limits
    - Strong/weak scaling
    """
    
    @staticmethod
    def compute_speedup(
        baseline_time: float,
        optimized_time: float,
    ) -> float:
        """Compute speedup factor.
        
        Args:
            baseline_time: Baseline execution time
            optimized_time: Optimized execution time
            
        Returns:
            Speedup factor
        """
        if optimized_time > 0:
            return baseline_time / optimized_time
        return float("inf")
    
    @staticmethod
    def compute_parallel_efficiency(
        speedup: float,
        num_processors: int,
    ) -> float:
        """Compute parallel efficiency.
        
        Efficiency = Speedup / P
        
        Args:
            speedup: Achieved speedup
            num_processors: Number of processors
            
        Returns:
            Efficiency (0-1)
        """
        return speedup / num_processors
    
    @staticmethod
    def amdahl_limit(
        parallel_fraction: float,
        num_processors: int,
    ) -> float:
        """Compute Amdahl's law speedup limit.
        
        S = 1 / ((1-P) + P/N)
        
        Args:
            parallel_fraction: Fraction of parallel code
            num_processors: Number of processors
            
        Returns:
            Maximum theoretical speedup
        """
        serial_fraction = 1 - parallel_fraction
        return 1 / (serial_fraction + parallel_fraction / num_processors)
    
    @staticmethod
    def gustafson_limit(
        parallel_fraction: float,
        num_processors: int,
    ) -> float:
        """Compute Gustafson's law speedup.
        
        S = N - (1-P) * (N-1)
        
        Args:
            parallel_fraction: Fraction of parallel code
            num_processors: Number of processors
            
        Returns:
            Scaled speedup
        """
        serial_fraction = 1 - parallel_fraction
        return num_processors - serial_fraction * (num_processors - 1)
    
    def analyze_scaling(
        self,
        times_by_processors: dict[int, float],
    ) -> dict[str, Any]:
        """Analyze parallel scaling.
        
        Args:
            times_by_processors: Execution time by processor count
            
        Returns:
            Scaling analysis
        """
        processors = sorted(times_by_processors.keys())
        times = [times_by_processors[p] for p in processors]
        
        baseline = times[0]
        speedups = [baseline / t for t in times]
        efficiencies = [s / p for s, p in zip(speedups, processors)]
        
        return {
            "processors": processors,
            "speedups": speedups,
            "efficiencies": efficiencies,
            "max_speedup": max(speedups),
            "average_efficiency": np.mean(efficiencies),
        }


class ScalingAnalyzer:
    """Analyze computational scaling behavior.
    
    Fits power law and exponential models to
    determine algorithmic complexity.
    """
    
    def analyze_complexity(
        self,
        sizes: list[int | float],
        times: list[float],
    ) -> ScalingResult:
        """Analyze computational complexity.
        
        Fits T = a * N^b to timing data.
        
        Args:
            sizes: Problem sizes
            times: Execution times
            
        Returns:
            Scaling analysis result
        """
        sizes_arr = np.array(sizes, dtype=np.float64)
        times_arr = np.array(times, dtype=np.float64)
        
        # Log-log regression for power law
        log_sizes = np.log(sizes_arr)
        log_times = np.log(times_arr)
        
        # Linear fit in log space
        coeffs = np.polyfit(log_sizes, log_times, 1)
        exponent = coeffs[0]
        coefficient = np.exp(coeffs[1])
        
        # R² calculation
        fitted = coefficient * sizes_arr ** exponent
        ss_res = np.sum((times_arr - fitted) ** 2)
        ss_tot = np.sum((times_arr - np.mean(times_arr)) ** 2)
        r_squared = 1 - ss_res / ss_tot
        
        return ScalingResult(
            problem_sizes=sizes_arr,
            execution_times=times_arr,
            fitted_exponent=exponent,
            fitted_coefficient=coefficient,
            r_squared=r_squared,
        )
    
    def extrapolate(
        self,
        result: ScalingResult,
        target_sizes: list[int | float],
    ) -> NDArray[np.float64]:
        """Extrapolate timing to larger sizes.
        
        Args:
            result: Scaling analysis result
            target_sizes: Sizes to extrapolate to
            
        Returns:
            Predicted times
        """
        target_arr = np.array(target_sizes, dtype=np.float64)
        return result.fitted_coefficient * target_arr ** result.fitted_exponent
    
    def compare_algorithms(
        self,
        algorithm_results: dict[str, ScalingResult],
    ) -> dict[str, Any]:
        """Compare scaling of multiple algorithms.
        
        Args:
            algorithm_results: Scaling results by algorithm name
            
        Returns:
            Comparison analysis
        """
        comparison = {
            "algorithms": list(algorithm_results.keys()),
            "scaling_exponents": {
                name: r.fitted_exponent
                for name, r in algorithm_results.items()
            },
            "r_squared": {
                name: r.r_squared
                for name, r in algorithm_results.items()
            },
            "complexity_classes": {
                name: r.scaling_class
                for name, r in algorithm_results.items()
            },
        }
        
        # Find best scaling algorithm
        best = min(
            algorithm_results.items(),
            key=lambda x: x[1].fitted_exponent,
        )
        comparison["best_scaling"] = best[0]
        
        return comparison
