"""Benchmarking and Analytics Package.

Provides comprehensive performance evaluation for:
- Quantum algorithm benchmarks
- Material simulation accuracy
- Multi-backend performance comparison
- Hardware utilization metrics
- Predictive accuracy validation
"""

from .performance import (
    PerformanceBenchmark,
    QuantumBenchmark,
    ClassicalBenchmark,
    SpeedupAnalyzer,
    ScalingAnalyzer,
)
from .accuracy import (
    AccuracyValidator,
    ExperimentalComparison,
    PredictionMetrics,
    MaterialValidation,
)
from .analytics import (
    AnalyticsDashboard,
    ReportGenerator,
    MetricsCollector,
    TrendAnalyzer,
)
from .comparison import (
    BackendComparison,
    AlgorithmComparison,
    MaterialComparison,
    CostBenefitAnalyzer,
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
