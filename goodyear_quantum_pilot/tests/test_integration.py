"""
Integration Test Suite for Goodyear Quantum Pilot Platform.

This module provides comprehensive integration tests verifying:
- Materials database integrity and accessibility
- Quantum algorithm implementations
- Simulation engine correctness
- Benchmarking system functionality
- End-to-end workflow validation

Run with: pytest tests/test_integration.py -v
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# Test Configuration
# =============================================================================

@dataclass
class TestConfig:
    """Test configuration settings."""
    
    materials_min_count: int = 90
    simulation_timeout: float = 30.0
    benchmark_iterations: int = 5
    convergence_threshold: float = 1e-6
    accuracy_tolerance: float = 0.05


CONFIG = TestConfig()


# =============================================================================
# Materials Database Tests
# =============================================================================

class TestMaterialsDatabase:
    """Test suite for materials database functionality."""
    
    def test_database_loads_successfully(self):
        """Verify materials database loads without errors."""
        from goodyear_quantum_pilot.materials.database import MaterialsDatabase
        
        db = MaterialsDatabase()
        assert db is not None
    
    def test_minimum_materials_count(self):
        """Verify database contains required minimum materials."""
        from goodyear_quantum_pilot.materials.database import MaterialsDatabase
        
        all_materials = MaterialsDatabase.get_all_materials()
        assert len(all_materials) >= CONFIG.materials_min_count
    
    def test_all_categories_present(self):
        """Verify all required material categories are present."""
        from goodyear_quantum_pilot.materials.database import (
            MaterialsDatabase,
            MaterialCategory,
        )
        
        required_categories = [
            MaterialCategory.ELASTOMERS,
            MaterialCategory.REINFORCING_FILLERS,
            MaterialCategory.PROCESSING_AIDS,
            MaterialCategory.QUANTUM_ENGINEERED,
            MaterialCategory.SELF_HEALING,
            MaterialCategory.NANOARCHITECTURES,
        ]
        
        for category in required_categories:
            materials = MaterialsDatabase.get_by_category(category)
            assert len(materials) > 0, f"No materials in category {category}"
    
    def test_material_properties_complete(self):
        """Verify all materials have required properties."""
        from goodyear_quantum_pilot.materials.database import MaterialsDatabase
        
        all_materials = MaterialsDatabase.get_all_materials()
        
        required_attrs = ["id", "name", "category", "mechanical", "thermal"]
        
        for material in all_materials[:10]:  # Check first 10
            for attr in required_attrs:
                assert hasattr(material, attr), (
                    f"Material {material.id} missing attribute {attr}"
                )
    
    def test_material_lookup_by_id(self):
        """Verify material lookup by ID works correctly."""
        from goodyear_quantum_pilot.materials.database import MaterialsDatabase
        
        material = MaterialsDatabase.get_by_id("NR001")
        assert material is not None
        assert material.id == "NR001"
    
    def test_material_lookup_by_name(self):
        """Verify material lookup by name works correctly."""
        from goodyear_quantum_pilot.materials.database import MaterialsDatabase
        
        material = MaterialsDatabase.get_by_name("natural_rubber_smr_l")
        assert material is not None
        assert "natural" in material.name.lower()
    
    def test_material_search(self):
        """Verify material search functionality."""
        from goodyear_quantum_pilot.materials.database import MaterialsDatabase
        
        results = MaterialsDatabase.search("silica")
        assert len(results) > 0
        for material in results:
            assert "silica" in material.name.lower() or \
                   "silica" in str(material.description).lower()


# =============================================================================
# Quantum Algorithm Tests
# =============================================================================

class TestVQESolver:
    """Test suite for VQE solver implementation."""
    
    def test_vqe_initialization(self):
        """Verify VQE solver initializes correctly."""
        from goodyear_quantum_pilot.algorithms.vqe import VQESolver
        
        solver = VQESolver(backend="simulator")
        assert solver is not None
        assert solver.backend == "simulator"
    
    def test_vqe_ansatz_construction(self):
        """Verify VQE constructs valid ansatz circuits."""
        from goodyear_quantum_pilot.algorithms.vqe import VQESolver
        
        solver = VQESolver(backend="simulator", ansatz="UCCSD")
        circuit = solver.build_ansatz(n_qubits=4, n_electrons=2)
        
        assert circuit is not None
        assert circuit.n_qubits == 4
    
    def test_vqe_energy_calculation(self):
        """Verify VQE computes reasonable energy values."""
        from goodyear_quantum_pilot.algorithms.vqe import VQESolver
        
        solver = VQESolver(backend="simulator")
        
        # Simple test Hamiltonian (H2-like)
        hamiltonian = solver.create_test_hamiltonian()
        result = solver.compute_ground_state(hamiltonian, max_iterations=10)
        
        assert result.converged or result.iterations == 10
        assert result.energy < 0  # Ground state should be negative
    
    def test_vqe_parameter_optimization(self):
        """Verify VQE optimizer reduces energy."""
        from goodyear_quantum_pilot.algorithms.vqe import VQESolver
        
        solver = VQESolver(backend="simulator", optimizer="COBYLA")
        hamiltonian = solver.create_test_hamiltonian()
        
        # Get initial energy
        initial_energy = solver.evaluate_energy(
            hamiltonian, 
            solver.initial_parameters
        )
        
        # Run optimization
        result = solver.compute_ground_state(hamiltonian, max_iterations=20)
        
        # Final energy should be lower
        assert result.energy <= initial_energy


class TestQAOAOptimizer:
    """Test suite for QAOA optimizer implementation."""
    
    def test_qaoa_initialization(self):
        """Verify QAOA optimizer initializes correctly."""
        from goodyear_quantum_pilot.algorithms.qaoa import QAOAOptimizer
        
        optimizer = QAOAOptimizer(backend="simulator", p_layers=2)
        assert optimizer is not None
        assert optimizer.p_layers == 2
    
    def test_qubo_encoding(self):
        """Verify QUBO problem encoding is correct."""
        from goodyear_quantum_pilot.algorithms.qaoa import QAOAOptimizer
        
        optimizer = QAOAOptimizer(backend="simulator")
        
        # Simple QUBO: minimize x1 + x2 - 2*x1*x2
        Q = {(0, 0): 1, (1, 1): 1, (0, 1): -2}
        
        cost_hamiltonian = optimizer.encode_qubo(Q)
        assert cost_hamiltonian is not None
    
    def test_qaoa_optimization(self):
        """Verify QAOA finds reasonable solutions."""
        from goodyear_quantum_pilot.algorithms.qaoa import QAOAOptimizer
        
        optimizer = QAOAOptimizer(backend="simulator", p_layers=2)
        
        # Simple MAX-CUT on triangle
        edges = [(0, 1), (1, 2), (0, 2)]
        result = optimizer.solve_max_cut(edges, max_iterations=10)
        
        assert result.solution is not None
        assert len(result.solution) == 3


class TestQuantumMonteCarlo:
    """Test suite for Quantum Monte Carlo implementation."""
    
    def test_qmc_initialization(self):
        """Verify QMC engine initializes correctly."""
        from goodyear_quantum_pilot.algorithms.monte_carlo import QuantumMonteCarlo
        
        qmc = QuantumMonteCarlo(method="VMC", n_walkers=100)
        assert qmc is not None
        assert qmc.n_walkers == 100
    
    def test_vmc_energy_estimation(self):
        """Verify VMC computes energy with reasonable variance."""
        from goodyear_quantum_pilot.algorithms.monte_carlo import QuantumMonteCarlo
        
        qmc = QuantumMonteCarlo(method="VMC", n_walkers=100, n_steps=100)
        
        trial_wf = qmc.create_test_wavefunction()
        result = qmc.compute_energy(trial_wf)
        
        assert result.energy is not None
        assert result.error > 0  # Should have statistical error
        assert result.error < abs(result.energy) * 0.1  # Error < 10% of energy
    
    def test_configuration_sampling(self):
        """Verify configuration sampling produces valid samples."""
        from goodyear_quantum_pilot.algorithms.monte_carlo import QuantumMonteCarlo
        
        qmc = QuantumMonteCarlo(method="VMC", n_walkers=50)
        trial_wf = qmc.create_test_wavefunction()
        
        configs = qmc.sample_configurations(100)
        assert len(configs) == 100
        
        for config in configs:
            assert hasattr(config, "local_energy")


class TestTunnelingCalculator:
    """Test suite for quantum tunneling calculations."""
    
    def test_tunneling_initialization(self):
        """Verify tunneling calculator initializes correctly."""
        from goodyear_quantum_pilot.algorithms.tunneling import TunnelingCalculator
        
        calc = TunnelingCalculator()
        assert calc is not None
    
    def test_wkb_tunneling_rate(self):
        """Verify WKB tunneling rate calculation."""
        from goodyear_quantum_pilot.algorithms.tunneling import (
            TunnelingCalculator,
            PotentialBarrier,
        )
        
        calc = TunnelingCalculator(method="WKB")
        
        # Test barrier: 1 eV height, 2 Å width, hydrogen mass
        barrier = PotentialBarrier(height=1.0, width=2.0, mass=1.0)
        result = calc.calculate_tunneling_rate(barrier)
        
        assert result.rate > 0
        assert result.transmission > 0
        assert result.transmission <= 1.0


class TestQuantumStress:
    """Test suite for quantum stress tensor calculations."""
    
    def test_stress_initialization(self):
        """Verify stress calculator initializes correctly."""
        from goodyear_quantum_pilot.algorithms.stress import QuantumStressTensor
        
        calc = QuantumStressTensor()
        assert calc is not None
    
    def test_stress_tensor_symmetry(self):
        """Verify computed stress tensor is symmetric."""
        from goodyear_quantum_pilot.algorithms.stress import QuantumStressTensor
        
        calc = QuantumStressTensor()
        test_system = calc.create_test_system()
        
        stress = calc.compute_stress(test_system)
        
        # Stress tensor should be symmetric
        for i in range(3):
            for j in range(3):
                assert abs(stress.tensor[i, j] - stress.tensor[j, i]) < 1e-10


# =============================================================================
# Simulation Engine Tests
# =============================================================================

class TestTireSimulator:
    """Test suite for main tire simulator."""
    
    def test_simulator_initialization(self):
        """Verify tire simulator initializes correctly."""
        from goodyear_quantum_pilot.simulation.core import TireSimulator
        from goodyear_quantum_pilot.materials.database import MaterialsDatabase
        
        rubber = MaterialsDatabase.get_by_id("NR001")
        simulator = TireSimulator(materials=[rubber])
        
        assert simulator is not None
    
    def test_simulation_config_validation(self):
        """Verify simulation config validation works."""
        from goodyear_quantum_pilot.simulation.core import (
            TireSimulator,
            SimulationConfig,
            SimulationType,
        )
        from goodyear_quantum_pilot.materials.database import MaterialsDatabase
        
        rubber = MaterialsDatabase.get_by_id("NR001")
        simulator = TireSimulator(materials=[rubber])
        
        config = SimulationConfig(
            simulation_type=SimulationType.FACTORY,
            duration=100.0,
            time_step=0.1,
        )
        
        assert simulator.validate_config(config)
    
    def test_factory_simulation_runs(self):
        """Verify factory simulation executes successfully."""
        from goodyear_quantum_pilot.simulation.factory import FactorySimulator
        from goodyear_quantum_pilot.materials.database import MaterialsDatabase
        
        rubber = MaterialsDatabase.get_by_id("NR001")
        simulator = FactorySimulator(materials=[rubber])
        
        result = simulator.simulate_curing(
            temperature=160.0,
            time=100.0,
            pressure=15.0
        )
        
        assert result is not None
        assert 0 <= result.cure_degree <= 1.0


class TestOnVehicleSimulation:
    """Test suite for on-vehicle simulation."""
    
    def test_on_vehicle_initialization(self):
        """Verify on-vehicle simulator initializes correctly."""
        from goodyear_quantum_pilot.simulation.on_vehicle import OnVehicleSimulator
        from goodyear_quantum_pilot.materials.database import MaterialsDatabase
        
        rubber = MaterialsDatabase.get_by_id("NR001")
        simulator = OnVehicleSimulator(tire_material=rubber)
        
        assert simulator is not None
    
    def test_rolling_dynamics(self):
        """Verify rolling dynamics simulation runs."""
        from goodyear_quantum_pilot.simulation.on_vehicle import OnVehicleSimulator
        from goodyear_quantum_pilot.materials.database import MaterialsDatabase
        
        rubber = MaterialsDatabase.get_by_id("NR001")
        simulator = OnVehicleSimulator(tire_material=rubber)
        
        result = simulator.simulate_rolling(
            speed=100.0,  # km/h
            load=5000.0,  # N
            duration=10.0  # seconds
        )
        
        assert result.temperature > 0
        assert result.rolling_resistance > 0


class TestCatastrophicSimulation:
    """Test suite for catastrophic failure simulation."""
    
    def test_blowout_model(self):
        """Verify blowout model produces realistic results."""
        from goodyear_quantum_pilot.simulation.catastrophic import CatastrophicSimulator
        from goodyear_quantum_pilot.materials.database import MaterialsDatabase
        
        rubber = MaterialsDatabase.get_by_id("NR001")
        simulator = CatastrophicSimulator(tire_material=rubber)
        
        result = simulator.simulate_blowout(
            speed=120.0,
            ambient_temp=35.0,
            defect_size=5.0
        )
        
        assert result.time_to_failure > 0
        assert result.failure_probability > 0
        assert result.failure_probability <= 1.0
    
    def test_failure_probability_monotonic(self):
        """Verify failure probability increases with defect size."""
        from goodyear_quantum_pilot.simulation.catastrophic import CatastrophicSimulator
        from goodyear_quantum_pilot.materials.database import MaterialsDatabase
        
        rubber = MaterialsDatabase.get_by_id("NR001")
        simulator = CatastrophicSimulator(tire_material=rubber)
        
        defect_sizes = [1.0, 5.0, 10.0]
        probabilities = []
        
        for size in defect_sizes:
            result = simulator.predict_failure_probability(
                defect_size=size,
                operating_conditions={"speed": 100, "temp": 30}
            )
            probabilities.append(result)
        
        # Probability should increase with defect size
        assert probabilities[0] < probabilities[1] < probabilities[2]


# =============================================================================
# Benchmarking Tests
# =============================================================================

class TestPerformanceBenchmark:
    """Test suite for performance benchmarking."""
    
    def test_benchmark_initialization(self):
        """Verify benchmark system initializes correctly."""
        from goodyear_quantum_pilot.benchmarks.performance import PerformanceBenchmark
        
        benchmark = PerformanceBenchmark()
        assert benchmark is not None
    
    def test_timing_measurement(self):
        """Verify timing measurements are accurate."""
        from goodyear_quantum_pilot.benchmarks.performance import PerformanceBenchmark
        import time
        
        benchmark = PerformanceBenchmark()
        
        def slow_function():
            time.sleep(0.1)
            return True
        
        result = benchmark.time_function(slow_function, iterations=3)
        
        assert result.mean_time >= 0.1
        assert result.mean_time < 0.2


class TestAccuracyValidator:
    """Test suite for accuracy validation."""
    
    def test_validator_initialization(self):
        """Verify accuracy validator initializes correctly."""
        from goodyear_quantum_pilot.benchmarks.accuracy import AccuracyValidator
        
        validator = AccuracyValidator()
        assert validator is not None
    
    def test_r_squared_calculation(self):
        """Verify R² calculation is correct."""
        from goodyear_quantum_pilot.benchmarks.accuracy import AccuracyValidator
        import numpy as np
        
        validator = AccuracyValidator()
        
        # Perfect fit
        experimental = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        simulated = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        result = validator.validate(simulated, experimental)
        assert abs(result.r_squared - 1.0) < 1e-10
    
    def test_rmse_calculation(self):
        """Verify RMSE calculation is correct."""
        from goodyear_quantum_pilot.benchmarks.accuracy import AccuracyValidator
        import numpy as np
        
        validator = AccuracyValidator()
        
        experimental = np.array([1.0, 2.0, 3.0])
        simulated = np.array([1.1, 2.1, 3.1])  # 0.1 error each
        
        result = validator.validate(simulated, experimental)
        assert abs(result.rmse - 0.1) < 1e-10


# =============================================================================
# End-to-End Workflow Tests
# =============================================================================

class TestEndToEndWorkflows:
    """End-to-end workflow tests."""
    
    def test_material_to_simulation_workflow(self):
        """Test complete workflow from material selection to simulation."""
        from goodyear_quantum_pilot.materials.database import MaterialsDatabase
        from goodyear_quantum_pilot.simulation.factory import FactorySimulator
        
        # 1. Select material
        rubber = MaterialsDatabase.get_by_name("natural_rubber_smr_l")
        assert rubber is not None
        
        # 2. Configure simulation
        simulator = FactorySimulator(materials=[rubber])
        
        # 3. Run simulation
        result = simulator.simulate_mixing(
            mixing_time=300,
            rotor_speed=60,
            fill_factor=0.7
        )
        
        # 4. Verify results
        assert result.mooney_viscosity > 0
        assert result.dispersion_index >= 0
        assert result.dispersion_index <= 1.0
    
    def test_quantum_enhanced_property_workflow(self):
        """Test workflow using quantum-computed properties."""
        from goodyear_quantum_pilot.materials.database import MaterialsDatabase
        from goodyear_quantum_pilot.algorithms.vqe import VQESolver
        
        # 1. Load material with quantum properties
        qe_materials = MaterialsDatabase.get_by_category("quantum_engineered")
        assert len(qe_materials) > 0
        
        material = qe_materials[0]
        
        # 2. Verify quantum properties are available
        assert hasattr(material, "quantum")
        assert material.quantum is not None
        
        # 3. Run enhanced calculation
        solver = VQESolver(backend="simulator")
        # Use cached properties or recompute as needed
        assert material.quantum.ground_state_energy is not None
    
    def test_full_tire_lifecycle_workflow(self):
        """Test complete tire lifecycle simulation workflow."""
        from goodyear_quantum_pilot.materials.database import MaterialsDatabase
        from goodyear_quantum_pilot.simulation import (
            FactorySimulator,
            ShippingSimulator,
            OnVehicleSimulator,
        )
        
        # 1. Select formulation
        rubber = MaterialsDatabase.get_by_id("NR001")
        silica = MaterialsDatabase.get_by_id("RF001")
        
        # 2. Factory simulation
        factory = FactorySimulator(materials=[rubber, silica])
        factory_result = factory.simulate_complete_process()
        
        assert factory_result.quality_score > 0.8
        
        # 3. Shipping simulation
        shipping = ShippingSimulator(tire_state=factory_result.tire_state)
        shipping_result = shipping.simulate_transport(
            duration_days=30,
            storage_temp=25.0
        )
        
        assert shipping_result.degradation < 0.05  # Less than 5% degradation
        
        # 4. Vehicle simulation
        vehicle = OnVehicleSimulator(tire_state=shipping_result.tire_state)
        vehicle_result = vehicle.simulate_drive_cycle(
            cycle="highway",
            distance=10000  # km
        )
        
        assert vehicle_result.remaining_tread > 0
        assert vehicle_result.safety_score > 0.9


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def materials_database():
    """Provide materials database instance."""
    from goodyear_quantum_pilot.materials.database import MaterialsDatabase
    return MaterialsDatabase()


@pytest.fixture
def sample_rubber():
    """Provide sample rubber material for tests."""
    from goodyear_quantum_pilot.materials.database import MaterialsDatabase
    return MaterialsDatabase.get_by_id("NR001")


@pytest.fixture
def vqe_solver():
    """Provide configured VQE solver for tests."""
    from goodyear_quantum_pilot.algorithms.vqe import VQESolver
    return VQESolver(backend="simulator", shots=1024)


@pytest.fixture
def tire_simulator(sample_rubber):
    """Provide configured tire simulator for tests."""
    from goodyear_quantum_pilot.simulation.core import TireSimulator
    return TireSimulator(materials=[sample_rubber])


# =============================================================================
# Main Test Runner
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
