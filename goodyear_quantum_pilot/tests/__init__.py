"""
Goodyear Quantum Pilot - Test Suite.

This package contains comprehensive tests for the quantum-enhanced
tire materials simulation platform.

Test Modules:
- test_integration.py: End-to-end integration tests
- test_materials.py: Materials database unit tests
- test_algorithms.py: Quantum algorithm unit tests
- test_simulation.py: Simulation engine unit tests
- test_benchmarks.py: Benchmarking system unit tests

Running Tests:
    pytest tests/ -v                    # Run all tests
    pytest tests/ -v -k "materials"     # Run materials tests only
    pytest tests/ -v --cov=.            # Run with coverage
"""

from __future__ import annotations

__all__ = [
    "test_integration",
]
