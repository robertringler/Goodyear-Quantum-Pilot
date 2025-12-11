"""Comparison Analysis Module.

Provides comparative analysis tools for:
- Backend performance comparison (CPU, GPU, Quantum)
- Algorithm comparison (VQE, QAOA, Classical)
- Material property comparison
- Cost-benefit analysis
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .performance import BenchmarkResult, ComputeBackend

logger = logging.getLogger(__name__)


class ComparisonMetric(Enum):
    """Metrics for comparison."""
    
    EXECUTION_TIME = auto()
    MEMORY_USAGE = auto()
    ACCURACY = auto()
    COST = auto()
    ENERGY = auto()
    THROUGHPUT = auto()


@dataclass
class ComparisonResult:
    """Result of comparison analysis.
    
    Attributes:
        metric: Metric compared
        items: Items being compared
        values: Values for each item
        winner: Best performing item
        ratios: Ratios vs winner
    """
    
    metric: ComparisonMetric
    items: list[str]
    values: list[float]
    winner: str = ""
    ratios: list[float] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Compute winner and ratios."""
        if self.values:
            best_idx = np.argmin(self.values)
            if self.metric in [ComparisonMetric.ACCURACY, ComparisonMetric.THROUGHPUT]:
                best_idx = np.argmax(self.values)
            
            self.winner = self.items[best_idx]
            best_val = self.values[best_idx]
            
            if best_val > 0:
                self.ratios = [v / best_val for v in self.values]
            else:
                self.ratios = [1.0] * len(self.values)


class BackendComparison:
    """Compare performance across compute backends.
    
    Provides systematic comparison of:
    - CPU vs GPU
    - GPU vs Quantum
    - Different quantum hardware
    - Cost efficiency
    
    Example:
        >>> comparison = BackendComparison()
        >>> 
        >>> # Add benchmark results
        >>> comparison.add_result("VQE", cpu_result)
        >>> comparison.add_result("VQE", gpu_result)
        >>> comparison.add_result("VQE", quantum_result)
        >>> 
        >>> # Get comparison
        >>> result = comparison.compare("VQE", ComparisonMetric.EXECUTION_TIME)
    """
    
    def __init__(self) -> None:
        """Initialize backend comparison."""
        self.results: dict[str, list[BenchmarkResult]] = {}
    
    def add_result(
        self,
        task_name: str,
        result: BenchmarkResult,
    ) -> None:
        """Add benchmark result.
        
        Args:
            task_name: Name of the task
            result: Benchmark result
        """
        if task_name not in self.results:
            self.results[task_name] = []
        self.results[task_name].append(result)
    
    def compare(
        self,
        task_name: str,
        metric: ComparisonMetric,
    ) -> ComparisonResult:
        """Compare backends for a task.
        
        Args:
            task_name: Task to compare
            metric: Metric to compare on
            
        Returns:
            Comparison result
        """
        if task_name not in self.results:
            return ComparisonResult(
                metric=metric,
                items=[],
                values=[],
            )
        
        results = self.results[task_name]
        
        items = [r.backend.name for r in results]
        
        metric_extractors = {
            ComparisonMetric.EXECUTION_TIME: lambda r: r.execution_time,
            ComparisonMetric.MEMORY_USAGE: lambda r: r.memory_usage,
            ComparisonMetric.ACCURACY: lambda r: r.accuracy,
            ComparisonMetric.COST: lambda r: r.cost,
            ComparisonMetric.ENERGY: lambda r: r.energy_consumption,
            ComparisonMetric.THROUGHPUT: lambda r: r.throughput,
        }
        
        extractor = metric_extractors.get(metric, lambda r: r.execution_time)
        values = [extractor(r) for r in results]
        
        return ComparisonResult(
            metric=metric,
            items=items,
            values=values,
        )
    
    def speedup_matrix(
        self,
        task_name: str,
    ) -> dict[str, dict[str, float]]:
        """Compute speedup matrix between backends.
        
        Args:
            task_name: Task to analyze
            
        Returns:
            Speedup matrix {backend1: {backend2: speedup}}
        """
        if task_name not in self.results:
            return {}
        
        results = self.results[task_name]
        backends = {r.backend.name: r.execution_time for r in results}
        
        matrix = {}
        for b1, t1 in backends.items():
            matrix[b1] = {}
            for b2, t2 in backends.items():
                matrix[b1][b2] = t2 / t1 if t1 > 0 else float("inf")
        
        return matrix
    
    def quantum_advantage_analysis(
        self,
        task_name: str,
    ) -> dict[str, Any]:
        """Analyze quantum advantage for a task.
        
        Args:
            task_name: Task to analyze
            
        Returns:
            Quantum advantage analysis
        """
        if task_name not in self.results:
            return {"error": "No results for task"}
        
        results = self.results[task_name]
        
        # Separate classical and quantum results
        classical = [r for r in results if r.backend in [
            ComputeBackend.CPU, ComputeBackend.GPU_NVIDIA, ComputeBackend.GPU_AMD
        ]]
        quantum = [r for r in results if r.backend not in [
            ComputeBackend.CPU, ComputeBackend.GPU_NVIDIA, ComputeBackend.GPU_AMD
        ]]
        
        if not classical or not quantum:
            return {"quantum_advantage": False, "reason": "Missing classical or quantum results"}
        
        # Best classical time
        best_classical = min(classical, key=lambda r: r.execution_time)
        best_quantum = min(quantum, key=lambda r: r.execution_time)
        
        speedup = best_classical.execution_time / best_quantum.execution_time
        cost_ratio = best_quantum.cost / best_classical.cost if best_classical.cost > 0 else float("inf")
        
        return {
            "task": task_name,
            "best_classical_backend": best_classical.backend.name,
            "best_classical_time": best_classical.execution_time,
            "best_quantum_backend": best_quantum.backend.name,
            "best_quantum_time": best_quantum.execution_time,
            "speedup": speedup,
            "quantum_advantage": speedup > 1,
            "cost_ratio": cost_ratio,
            "cost_efficient": cost_ratio < 1,
        }
    
    def generate_report(self) -> str:
        """Generate comparison report.
        
        Returns:
            Formatted report string
        """
        report = """
# Backend Comparison Report

## Summary

| Task | Best Backend | Time (s) | Speedup vs CPU |
|------|--------------|----------|----------------|
"""
        
        for task_name in self.results:
            time_comparison = self.compare(task_name, ComparisonMetric.EXECUTION_TIME)
            
            if not time_comparison.items:
                continue
            
            best_idx = time_comparison.values.index(min(time_comparison.values))
            best_backend = time_comparison.items[best_idx]
            best_time = time_comparison.values[best_idx]
            
            # Calculate speedup vs CPU
            cpu_time = None
            for i, item in enumerate(time_comparison.items):
                if "CPU" in item:
                    cpu_time = time_comparison.values[i]
                    break
            
            speedup = cpu_time / best_time if cpu_time and best_time > 0 else 1.0
            
            report += f"| {task_name} | {best_backend} | {best_time:.4f} | {speedup:.2f}x |\n"
        
        report += "\n## Quantum Advantage Analysis\n\n"
        
        for task_name in self.results:
            qa = self.quantum_advantage_analysis(task_name)
            if "error" not in qa:
                status = "✓ ACHIEVED" if qa["quantum_advantage"] else "✗ NOT ACHIEVED"
                report += f"### {task_name}\n"
                report += f"- Quantum Advantage: {status}\n"
                report += f"- Speedup: {qa['speedup']:.2f}x\n"
                report += f"- Best Quantum: {qa['best_quantum_backend']}\n\n"
        
        return report


class AlgorithmComparison:
    """Compare different algorithms.
    
    Provides comparison of:
    - VQE vs QAOA vs classical
    - Different ansatz designs
    - Optimization strategies
    - Convergence behavior
    """
    
    def __init__(self) -> None:
        """Initialize algorithm comparison."""
        self.results: dict[str, dict[str, Any]] = {}
    
    def add_result(
        self,
        problem: str,
        algorithm: str,
        result: dict[str, Any],
    ) -> None:
        """Add algorithm result.
        
        Args:
            problem: Problem name
            algorithm: Algorithm name
            result: Result data
        """
        if problem not in self.results:
            self.results[problem] = {}
        self.results[problem][algorithm] = result
    
    def compare_accuracy(
        self,
        problem: str,
        reference_value: float,
    ) -> dict[str, Any]:
        """Compare algorithm accuracy.
        
        Args:
            problem: Problem name
            reference_value: Reference (exact) value
            
        Returns:
            Accuracy comparison
        """
        if problem not in self.results:
            return {"error": "No results for problem"}
        
        comparison = {}
        for algorithm, result in self.results[problem].items():
            computed = result.get("value", 0)
            error = abs(computed - reference_value)
            error_pct = error / abs(reference_value) * 100 if reference_value != 0 else 0
            
            comparison[algorithm] = {
                "value": computed,
                "error": error,
                "error_pct": error_pct,
            }
        
        # Rank by accuracy
        ranked = sorted(comparison.items(), key=lambda x: x[1]["error"])
        
        return {
            "problem": problem,
            "reference": reference_value,
            "algorithms": comparison,
            "ranking": [r[0] for r in ranked],
            "best": ranked[0][0] if ranked else None,
        }
    
    def compare_convergence(
        self,
        problem: str,
    ) -> dict[str, Any]:
        """Compare convergence behavior.
        
        Args:
            problem: Problem name
            
        Returns:
            Convergence comparison
        """
        if problem not in self.results:
            return {"error": "No results for problem"}
        
        comparison = {}
        for algorithm, result in self.results[problem].items():
            history = result.get("convergence_history", [])
            
            if history:
                comparison[algorithm] = {
                    "iterations": len(history),
                    "final_value": history[-1],
                    "improvement": history[0] - history[-1] if len(history) > 1 else 0,
                    "converged": result.get("converged", False),
                }
        
        return {
            "problem": problem,
            "algorithms": comparison,
            "fastest_convergence": min(
                comparison.items(),
                key=lambda x: x[1]["iterations"]
            )[0] if comparison else None,
        }
    
    def compare_resources(
        self,
        problem: str,
    ) -> dict[str, Any]:
        """Compare resource requirements.
        
        Args:
            problem: Problem name
            
        Returns:
            Resource comparison
        """
        if problem not in self.results:
            return {"error": "No results for problem"}
        
        comparison = {}
        for algorithm, result in self.results[problem].items():
            comparison[algorithm] = {
                "time": result.get("execution_time", 0),
                "memory": result.get("memory_mb", 0),
                "qubits": result.get("qubits", 0),
                "gates": result.get("gate_count", 0),
                "parameters": result.get("parameters", 0),
            }
        
        return {
            "problem": problem,
            "algorithms": comparison,
        }
    
    def pareto_analysis(
        self,
        problem: str,
        metrics: list[str] = ["accuracy", "time", "cost"],
    ) -> dict[str, Any]:
        """Perform Pareto analysis of algorithms.
        
        Identifies Pareto-optimal algorithms that are not
        dominated in all metrics.
        
        Args:
            problem: Problem name
            metrics: Metrics to consider
            
        Returns:
            Pareto analysis
        """
        if problem not in self.results:
            return {"error": "No results for problem"}
        
        algorithms = list(self.results[problem].keys())
        n_algs = len(algorithms)
        
        # Build score matrix (lower is better)
        scores = np.zeros((n_algs, len(metrics)))
        
        for i, alg in enumerate(algorithms):
            result = self.results[problem][alg]
            for j, metric in enumerate(metrics):
                value = result.get(metric, 0)
                # Invert accuracy (higher is better)
                if metric == "accuracy":
                    value = -value
                scores[i, j] = value
        
        # Find Pareto front
        pareto_optimal = []
        for i in range(n_algs):
            dominated = False
            for j in range(n_algs):
                if i != j:
                    # j dominates i if j is better or equal in all metrics
                    # and strictly better in at least one
                    if all(scores[j] <= scores[i]) and any(scores[j] < scores[i]):
                        dominated = True
                        break
            if not dominated:
                pareto_optimal.append(algorithms[i])
        
        return {
            "problem": problem,
            "metrics": metrics,
            "pareto_optimal": pareto_optimal,
            "total_algorithms": n_algs,
        }


class MaterialComparison:
    """Compare material formulations.
    
    Provides comparison of:
    - Property profiles
    - Cost vs performance
    - Environmental impact
    - Application suitability
    """
    
    def __init__(self) -> None:
        """Initialize material comparison."""
        self.materials: dict[str, dict[str, Any]] = {}
    
    def add_material(
        self,
        name: str,
        properties: dict[str, float],
        cost: float = 0.0,
        environmental_score: float = 0.0,
    ) -> None:
        """Add material for comparison.
        
        Args:
            name: Material name
            properties: Property values
            cost: Cost per kg
            environmental_score: Environmental impact score (0-100)
        """
        self.materials[name] = {
            "properties": properties,
            "cost": cost,
            "environmental_score": environmental_score,
        }
    
    def compare_property(
        self,
        property_name: str,
    ) -> ComparisonResult:
        """Compare materials on a property.
        
        Args:
            property_name: Property to compare
            
        Returns:
            Comparison result
        """
        items = []
        values = []
        
        for name, data in self.materials.items():
            if property_name in data["properties"]:
                items.append(name)
                values.append(data["properties"][property_name])
        
        return ComparisonResult(
            metric=ComparisonMetric.ACCURACY,  # Higher is better
            items=items,
            values=values,
        )
    
    def spider_chart_data(
        self,
        material_names: list[str],
        properties: list[str],
    ) -> dict[str, Any]:
        """Generate data for spider/radar chart.
        
        Args:
            material_names: Materials to compare
            properties: Properties to include
            
        Returns:
            Spider chart data
        """
        data = {}
        
        # Find max values for normalization
        max_values = {prop: 0.0 for prop in properties}
        for name in material_names:
            if name in self.materials:
                props = self.materials[name]["properties"]
                for prop in properties:
                    if prop in props:
                        max_values[prop] = max(max_values[prop], props[prop])
        
        # Normalize values
        for name in material_names:
            if name in self.materials:
                props = self.materials[name]["properties"]
                normalized = {}
                for prop in properties:
                    if prop in props and max_values[prop] > 0:
                        normalized[prop] = props[prop] / max_values[prop]
                    else:
                        normalized[prop] = 0.0
                data[name] = normalized
        
        return {
            "materials": material_names,
            "properties": properties,
            "data": data,
            "max_values": max_values,
        }
    
    def rank_for_application(
        self,
        requirements: dict[str, tuple[float, float, float]],
    ) -> list[tuple[str, float]]:
        """Rank materials for an application.
        
        Args:
            requirements: {property: (min, target, weight)}
            
        Returns:
            Ranked list of (material, score)
        """
        scores = {}
        
        for name, data in self.materials.items():
            props = data["properties"]
            score = 0.0
            total_weight = 0.0
            
            for prop, (min_val, target_val, weight) in requirements.items():
                if prop in props:
                    value = props[prop]
                    
                    # Score based on how well it meets target
                    if value >= target_val:
                        prop_score = 1.0
                    elif value >= min_val:
                        prop_score = (value - min_val) / (target_val - min_val)
                    else:
                        prop_score = 0.0
                    
                    score += prop_score * weight
                    total_weight += weight
            
            if total_weight > 0:
                scores[name] = score / total_weight
        
        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return ranked


class CostBenefitAnalyzer:
    """Analyze cost vs benefit tradeoffs.
    
    Provides:
    - ROI calculations
    - Break-even analysis
    - Total cost of ownership
    - Value optimization
    """
    
    def __init__(self) -> None:
        """Initialize cost-benefit analyzer."""
        self.options: dict[str, dict[str, Any]] = {}
    
    def add_option(
        self,
        name: str,
        initial_cost: float,
        recurring_cost: float,  # Per unit/time
        benefit_value: float,
        implementation_time: float = 0.0,  # Time to value
    ) -> None:
        """Add option for analysis.
        
        Args:
            name: Option name
            initial_cost: Upfront investment
            recurring_cost: Ongoing cost per period
            benefit_value: Benefit value per period
            implementation_time: Time to realize benefits
        """
        self.options[name] = {
            "initial_cost": initial_cost,
            "recurring_cost": recurring_cost,
            "benefit_value": benefit_value,
            "implementation_time": implementation_time,
        }
    
    def compute_roi(
        self,
        option_name: str,
        periods: int,
    ) -> dict[str, Any]:
        """Compute Return on Investment.
        
        ROI = (Gain - Cost) / Cost × 100%
        
        Args:
            option_name: Option to analyze
            periods: Number of periods
            
        Returns:
            ROI analysis
        """
        if option_name not in self.options:
            return {"error": "Option not found"}
        
        opt = self.options[option_name]
        
        total_cost = opt["initial_cost"] + opt["recurring_cost"] * periods
        total_benefit = opt["benefit_value"] * max(0, periods - opt["implementation_time"])
        
        net_gain = total_benefit - total_cost
        roi = (net_gain / total_cost * 100) if total_cost > 0 else 0
        
        return {
            "option": option_name,
            "periods": periods,
            "total_cost": total_cost,
            "total_benefit": total_benefit,
            "net_gain": net_gain,
            "roi_percent": roi,
        }
    
    def break_even_analysis(
        self,
        option_name: str,
    ) -> dict[str, Any]:
        """Compute break-even point.
        
        Args:
            option_name: Option to analyze
            
        Returns:
            Break-even analysis
        """
        if option_name not in self.options:
            return {"error": "Option not found"}
        
        opt = self.options[option_name]
        
        net_periodic_benefit = opt["benefit_value"] - opt["recurring_cost"]
        
        if net_periodic_benefit <= 0:
            return {
                "option": option_name,
                "break_even_periods": float("inf"),
                "message": "Never breaks even",
            }
        
        break_even = opt["initial_cost"] / net_periodic_benefit + opt["implementation_time"]
        
        return {
            "option": option_name,
            "break_even_periods": break_even,
            "net_periodic_benefit": net_periodic_benefit,
            "initial_investment": opt["initial_cost"],
        }
    
    def compare_options(
        self,
        periods: int,
    ) -> dict[str, Any]:
        """Compare all options.
        
        Args:
            periods: Analysis period
            
        Returns:
            Comparison of all options
        """
        rois = {}
        break_evens = {}
        
        for name in self.options:
            rois[name] = self.compute_roi(name, periods)
            break_evens[name] = self.break_even_analysis(name)
        
        # Rank by ROI
        ranked = sorted(
            rois.items(),
            key=lambda x: x[1].get("roi_percent", 0),
            reverse=True,
        )
        
        return {
            "analysis_periods": periods,
            "roi_comparison": rois,
            "break_even_comparison": break_evens,
            "ranking_by_roi": [r[0] for r in ranked],
            "best_option": ranked[0][0] if ranked else None,
        }
    
    def tco_analysis(
        self,
        option_name: str,
        years: int,
        discount_rate: float = 0.05,
    ) -> dict[str, Any]:
        """Total Cost of Ownership analysis.
        
        Args:
            option_name: Option to analyze
            years: Analysis horizon
            discount_rate: Annual discount rate
            
        Returns:
            TCO analysis
        """
        if option_name not in self.options:
            return {"error": "Option not found"}
        
        opt = self.options[option_name]
        
        # Present value calculation
        pv_recurring = 0.0
        for year in range(years):
            pv_recurring += opt["recurring_cost"] / ((1 + discount_rate) ** year)
        
        tco = opt["initial_cost"] + pv_recurring
        
        return {
            "option": option_name,
            "years": years,
            "discount_rate": discount_rate,
            "initial_cost": opt["initial_cost"],
            "pv_recurring_cost": pv_recurring,
            "total_cost_of_ownership": tco,
        }
