"""Benchmarking and Analytics Package.

Provides comprehensive performance evaluation for:
- Quantum algorithm benchmarks
- Material simulation accuracy
- Multi-backend performance comparison
- Hardware utilization metrics
- Predictive accuracy validation
"""

from .accuracy import (
    AccuracyValidator,
    ExperimentalComparison,
    MaterialValidation,
    PredictionMetrics,
)
from .analytics import (
    AnalyticsDashboard,
    MetricsCollector,
    ReportGenerator,
    TrendAnalyzer,
)
from .comparison import (
    AlgorithmComparison,
    BackendComparison,
    CostBenefitAnalyzer,
    MaterialComparison,
)
from .performance import (
    ClassicalBenchmark,
    PerformanceBenchmark,
    QuantumBenchmark,
    ScalingAnalyzer,
    SpeedupAnalyzer,
)

__all__ = [
    # Performance benchmarks
    "PerformanceBenchmark",
    "QuantumBenchmark",
    "ClassicalBenchmark",
    "SpeedupAnalyzer",
    "ScalingAnalyzer",
    # Accuracy validation
    "AccuracyValidator",
    "ExperimentalComparison",
    "PredictionMetrics",
    "MaterialValidation",
    # Analytics
    "AnalyticsDashboard",
    "ReportGenerator",
    "MetricsCollector",
    "TrendAnalyzer",
    # Comparison tools
    "BackendComparison",
    "AlgorithmComparison",
    "MaterialComparison",
    "CostBenefitAnalyzer",
]
