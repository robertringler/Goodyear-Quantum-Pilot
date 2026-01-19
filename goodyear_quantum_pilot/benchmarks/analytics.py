"""Analytics Dashboard and Reporting Module.

Provides analytics, visualization, and reporting tools for:
- Real-time simulation monitoring
- Performance dashboards
- Automated report generation
- Trend analysis and forecasting
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics tracked."""

    PERFORMANCE = auto()
    ACCURACY = auto()
    COST = auto()
    QUALITY = auto()
    RELIABILITY = auto()


class TrendDirection(Enum):
    """Trend direction indicators."""

    INCREASING = auto()
    DECREASING = auto()
    STABLE = auto()
    VOLATILE = auto()


@dataclass
class Metric:
    """Single metric data point.

    Attributes:
        name: Metric name
        value: Current value
        unit: Unit of measurement
        timestamp: Recording timestamp
        metric_type: Type of metric
        target: Target value (optional)
    """

    name: str
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.now)
    metric_type: MetricType = MetricType.PERFORMANCE
    target: float | None = None

    @property
    def on_target(self) -> bool | None:
        """Check if metric is on target."""
        if self.target is None:
            return None
        return self.value >= self.target


@dataclass
class MetricSeries:
    """Time series of metrics.

    Attributes:
        name: Metric name
        values: Value array
        timestamps: Timestamp array
        unit: Unit of measurement
    """

    name: str
    values: list[float] = field(default_factory=list)
    timestamps: list[datetime] = field(default_factory=list)
    unit: str = ""

    def add(self, value: float, timestamp: datetime | None = None) -> None:
        """Add a data point."""
        self.values.append(value)
        self.timestamps.append(timestamp or datetime.now())

    @property
    def current(self) -> float:
        """Get current (latest) value."""
        return self.values[-1] if self.values else 0.0

    @property
    def mean(self) -> float:
        """Get mean value."""
        return np.mean(self.values) if self.values else 0.0

    @property
    def std(self) -> float:
        """Get standard deviation."""
        return np.std(self.values) if self.values else 0.0


class MetricsCollector:
    """Collect and manage metrics.

    Provides centralized metrics collection with:
    - Real-time tracking
    - Historical storage
    - Aggregation and statistics
    - Export capabilities

    Example:
        >>> collector = MetricsCollector()
        >>>
        >>> # Track simulation performance
        >>> collector.record("simulation_time", 45.2, "seconds")
        >>> collector.record("accuracy", 98.5, "%")
        >>>
        >>> # Get summary
        >>> print(collector.get_summary())
    """

    def __init__(
        self,
        max_history: int = 10000,
    ) -> None:
        """Initialize metrics collector.

        Args:
            max_history: Maximum history per metric
        """
        self.max_history = max_history
        self.metrics: dict[str, MetricSeries] = {}
        self.metadata: dict[str, dict[str, Any]] = {}

        logger.info(f"MetricsCollector initialized (max_history={max_history})")

    def record(
        self,
        name: str,
        value: float,
        unit: str = "",
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a metric value.

        Args:
            name: Metric name
            value: Metric value
            unit: Unit of measurement
            timestamp: Optional timestamp
            metadata: Optional metadata
        """
        if name not in self.metrics:
            self.metrics[name] = MetricSeries(name=name, unit=unit)

        self.metrics[name].add(value, timestamp)

        # Trim history
        if len(self.metrics[name].values) > self.max_history:
            self.metrics[name].values = self.metrics[name].values[-self.max_history :]
            self.metrics[name].timestamps = self.metrics[name].timestamps[-self.max_history :]

        if metadata:
            self.metadata[name] = metadata

    def get_metric(self, name: str) -> MetricSeries | None:
        """Get metric series by name."""
        return self.metrics.get(name)

    def get_current(self, name: str) -> float:
        """Get current value of metric."""
        series = self.metrics.get(name)
        return series.current if series else 0.0

    def get_statistics(self, name: str) -> dict[str, float]:
        """Get statistics for metric.

        Args:
            name: Metric name

        Returns:
            Statistics dictionary
        """
        series = self.metrics.get(name)
        if not series or not series.values:
            return {}

        values = np.array(series.values)

        return {
            "current": float(values[-1]),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
            "count": len(values),
        }

    def get_summary(self) -> dict[str, dict[str, float]]:
        """Get summary of all metrics."""
        return {name: self.get_statistics(name) for name in self.metrics}

    def export_json(self, filepath: str) -> None:
        """Export metrics to JSON file.

        Args:
            filepath: Output file path
        """
        data = {
            name: {
                "values": series.values,
                "timestamps": [t.isoformat() for t in series.timestamps],
                "unit": series.unit,
            }
            for name, series in self.metrics.items()
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)


class TrendAnalyzer:
    """Analyze trends in metrics.

    Provides trend detection and forecasting:
    - Linear trend detection
    - Seasonality analysis
    - Anomaly detection
    - Forecasting
    """

    def __init__(
        self,
        window_size: int = 10,
    ) -> None:
        """Initialize trend analyzer.

        Args:
            window_size: Analysis window size
        """
        self.window_size = window_size

    def detect_trend(
        self,
        values: list[float] | NDArray[np.float64],
    ) -> TrendDirection:
        """Detect trend direction.

        Args:
            values: Value series

        Returns:
            Trend direction
        """
        if len(values) < 2:
            return TrendDirection.STABLE

        values_arr = np.array(values)

        # Calculate slope using linear regression
        x = np.arange(len(values_arr))
        slope = np.polyfit(x, values_arr, 1)[0]

        # Calculate volatility
        std = np.std(values_arr)
        mean = np.mean(values_arr)
        cv = std / abs(mean) if mean != 0 else 0

        if cv > 0.5:
            return TrendDirection.VOLATILE

        threshold = std * 0.1  # 10% of std as threshold

        if slope > threshold:
            return TrendDirection.INCREASING
        elif slope < -threshold:
            return TrendDirection.DECREASING
        else:
            return TrendDirection.STABLE

    def compute_moving_average(
        self,
        values: list[float] | NDArray[np.float64],
        window: int | None = None,
    ) -> NDArray[np.float64]:
        """Compute moving average.

        Args:
            values: Value series
            window: Window size (default: self.window_size)

        Returns:
            Moving average series
        """
        window = window or self.window_size
        values_arr = np.array(values)

        if len(values_arr) < window:
            return values_arr

        return np.convolve(values_arr, np.ones(window) / window, mode="valid")

    def detect_anomalies(
        self,
        values: list[float] | NDArray[np.float64],
        threshold: float = 3.0,
    ) -> list[int]:
        """Detect anomalies using Z-score.

        Args:
            values: Value series
            threshold: Z-score threshold

        Returns:
            Indices of anomalies
        """
        values_arr = np.array(values)

        mean = np.mean(values_arr)
        std = np.std(values_arr)

        if std == 0:
            return []

        z_scores = np.abs((values_arr - mean) / std)
        anomaly_indices = np.where(z_scores > threshold)[0]

        return anomaly_indices.tolist()

    def forecast(
        self,
        values: list[float] | NDArray[np.float64],
        steps: int = 5,
    ) -> NDArray[np.float64]:
        """Forecast future values.

        Simple linear extrapolation.

        Args:
            values: Historical values
            steps: Steps to forecast

        Returns:
            Forecasted values
        """
        values_arr = np.array(values)
        x = np.arange(len(values_arr))

        # Linear fit
        coeffs = np.polyfit(x, values_arr, 1)

        # Extrapolate
        future_x = np.arange(len(values_arr), len(values_arr) + steps)
        return np.polyval(coeffs, future_x)

    def analyze(
        self,
        series: MetricSeries,
    ) -> dict[str, Any]:
        """Comprehensive trend analysis.

        Args:
            series: Metric series to analyze

        Returns:
            Analysis results
        """
        values = series.values

        if len(values) < 3:
            return {"error": "Insufficient data"}

        return {
            "metric": series.name,
            "current": series.current,
            "mean": series.mean,
            "std": series.std,
            "trend": self.detect_trend(values).name,
            "anomaly_count": len(self.detect_anomalies(values)),
            "forecast_5": self.forecast(values, 5).tolist(),
        }


class ReportGenerator:
    """Generate analytics reports.

    Creates formatted reports in various formats:
    - Markdown
    - HTML
    - JSON
    - Executive summary

    Example:
        >>> generator = ReportGenerator(collector)
        >>>
        >>> # Generate executive summary
        >>> report = generator.executive_summary()
        >>>
        >>> # Export as markdown
        >>> generator.export_markdown("report.md")
    """

    def __init__(
        self,
        collector: MetricsCollector,
        analyzer: TrendAnalyzer | None = None,
    ) -> None:
        """Initialize report generator.

        Args:
            collector: Metrics collector
            analyzer: Trend analyzer (optional)
        """
        self.collector = collector
        self.analyzer = analyzer or TrendAnalyzer()

    def executive_summary(self) -> str:
        """Generate executive summary.

        Returns:
            Formatted summary string
        """
        summary = self.collector.get_summary()

        report = f"""
================================================================================
                        QUANTUM TIRE SIMULATION
                         EXECUTIVE SUMMARY
================================================================================

Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

METRICS OVERVIEW
----------------
Total Metrics Tracked: {len(summary)}

"""

        for name, stats in summary.items():
            series = self.collector.get_metric(name)
            trend = self.analyzer.detect_trend(series.values) if series else TrendDirection.STABLE

            report += f"""
{name.upper()}
  Current: {stats.get('current', 0):.4f} {series.unit if series else ''}
  Mean: {stats.get('mean', 0):.4f}
  Std Dev: {stats.get('std', 0):.4f}
  Range: [{stats.get('min', 0):.4f}, {stats.get('max', 0):.4f}]
  Trend: {trend.name}
  Data Points: {stats.get('count', 0)}
"""

        report += """
================================================================================
                             END OF REPORT
================================================================================
"""

        return report

    def performance_report(self) -> str:
        """Generate performance report."""
        report = f"""
# Performance Report

Generated: {datetime.now().isoformat()}

## Summary Statistics

| Metric | Current | Mean | Std | Trend |
|--------|---------|------|-----|-------|
"""

        for name, series in self.collector.metrics.items():
            stats = self.collector.get_statistics(name)
            trend = self.analyzer.detect_trend(series.values)

            report += f"| {name} | {stats.get('current', 0):.2f} | "
            report += f"{stats.get('mean', 0):.2f} | {stats.get('std', 0):.2f} | "
            report += f"{trend.name} |\n"

        return report

    def accuracy_report(
        self,
        validation_results: list[dict[str, Any]],
    ) -> str:
        """Generate accuracy validation report.

        Args:
            validation_results: Validation results

        Returns:
            Formatted report
        """
        report = f"""
# Accuracy Validation Report

Generated: {datetime.now().isoformat()}

## Validation Summary

Total Validations: {len(validation_results)}
Passed: {sum(1 for r in validation_results if r.get('passed', False))}
Failed: {sum(1 for r in validation_results if not r.get('passed', True))}

## Detailed Results

| Property | Predicted | Actual | Error (%) | Status |
|----------|-----------|--------|-----------|--------|
"""

        for result in validation_results:
            status = "✓ PASS" if result.get("passed", False) else "✗ FAIL"
            report += f"| {result.get('property', 'N/A')} | "
            report += f"{result.get('predicted', 0):.4f} | "
            report += f"{result.get('actual', 0):.4f} | "
            report += f"{result.get('error_pct', 0):.2f} | "
            report += f"{status} |\n"

        return report

    def export_markdown(self, filepath: str) -> None:
        """Export report as markdown file.

        Args:
            filepath: Output file path
        """
        report = self.performance_report()

        with open(filepath, "w") as f:
            f.write(report)

    def export_json(self, filepath: str) -> None:
        """Export report as JSON file.

        Args:
            filepath: Output file path
        """
        data = {
            "generated": datetime.now().isoformat(),
            "metrics": self.collector.get_summary(),
            "analysis": {
                name: self.analyzer.analyze(series)
                for name, series in self.collector.metrics.items()
            },
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)


class AnalyticsDashboard:
    """Analytics dashboard for real-time monitoring.

    Provides dashboard data generation for:
    - Real-time metrics display
    - Historical charts
    - KPI scorecards
    - Alert status

    Note: This generates data suitable for frontend
    visualization frameworks.
    """

    def __init__(
        self,
        collector: MetricsCollector,
        analyzer: TrendAnalyzer | None = None,
    ) -> None:
        """Initialize analytics dashboard.

        Args:
            collector: Metrics collector
            analyzer: Trend analyzer
        """
        self.collector = collector
        self.analyzer = analyzer or TrendAnalyzer()

        # KPI definitions
        self.kpis: dict[str, dict[str, Any]] = {}
        self.alerts: list[dict[str, Any]] = []

    def define_kpi(
        self,
        name: str,
        metric: str,
        target: float,
        warning_threshold: float,
        critical_threshold: float,
        higher_is_better: bool = True,
    ) -> None:
        """Define a KPI.

        Args:
            name: KPI name
            metric: Source metric name
            target: Target value
            warning_threshold: Warning threshold
            critical_threshold: Critical threshold
            higher_is_better: Direction of improvement
        """
        self.kpis[name] = {
            "metric": metric,
            "target": target,
            "warning": warning_threshold,
            "critical": critical_threshold,
            "higher_is_better": higher_is_better,
        }

    def get_kpi_status(self, name: str) -> dict[str, Any]:
        """Get KPI status.

        Args:
            name: KPI name

        Returns:
            KPI status
        """
        if name not in self.kpis:
            return {"error": f"KPI {name} not defined"}

        kpi = self.kpis[name]
        current = self.collector.get_current(kpi["metric"])

        if kpi["higher_is_better"]:
            if current >= kpi["target"]:
                status = "GREEN"
            elif current >= kpi["warning"]:
                status = "YELLOW"
            elif current >= kpi["critical"]:
                status = "ORANGE"
            else:
                status = "RED"
        else:
            if current <= kpi["target"]:
                status = "GREEN"
            elif current <= kpi["warning"]:
                status = "YELLOW"
            elif current <= kpi["critical"]:
                status = "ORANGE"
            else:
                status = "RED"

        return {
            "name": name,
            "current": current,
            "target": kpi["target"],
            "status": status,
            "on_target": (current >= kpi["target"]) == kpi["higher_is_better"],
        }

    def get_dashboard_data(self) -> dict[str, Any]:
        """Get complete dashboard data.

        Returns:
            Dashboard data structure
        """
        kpi_statuses = {name: self.get_kpi_status(name) for name in self.kpis}

        metrics_summary = {
            name: {
                "current": series.current,
                "mean": series.mean,
                "unit": series.unit,
                "trend": self.analyzer.detect_trend(series.values).name,
            }
            for name, series in self.collector.metrics.items()
        }

        return {
            "timestamp": datetime.now().isoformat(),
            "kpis": kpi_statuses,
            "metrics": metrics_summary,
            "alerts": self.alerts,
            "overall_health": self._compute_health_score(),
        }

    def _compute_health_score(self) -> dict[str, Any]:
        """Compute overall health score."""
        if not self.kpis:
            return {"score": 100, "status": "GREEN"}

        green_count = sum(
            1 for name in self.kpis if self.get_kpi_status(name).get("status") == "GREEN"
        )

        score = green_count / len(self.kpis) * 100

        if score >= 80:
            status = "GREEN"
        elif score >= 60:
            status = "YELLOW"
        elif score >= 40:
            status = "ORANGE"
        else:
            status = "RED"

        return {
            "score": score,
            "status": status,
            "kpis_on_target": green_count,
            "total_kpis": len(self.kpis),
        }

    def add_alert(
        self,
        message: str,
        severity: str = "INFO",
        metric: str | None = None,
    ) -> None:
        """Add an alert.

        Args:
            message: Alert message
            severity: Severity level (INFO, WARNING, ERROR, CRITICAL)
            metric: Related metric (optional)
        """
        self.alerts.append(
            {
                "timestamp": datetime.now().isoformat(),
                "message": message,
                "severity": severity,
                "metric": metric,
            }
        )

    def check_thresholds(self) -> list[dict[str, Any]]:
        """Check all thresholds and generate alerts.

        Returns:
            Generated alerts
        """
        new_alerts = []

        for name in self.kpis:
            status = self.get_kpi_status(name)

            if status["status"] == "RED":
                alert = {
                    "message": f"KPI {name} is CRITICAL",
                    "severity": "CRITICAL",
                    "metric": self.kpis[name]["metric"],
                }
                new_alerts.append(alert)
                self.alerts.append(
                    {
                        **alert,
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            elif status["status"] == "ORANGE":
                alert = {
                    "message": f"KPI {name} needs attention",
                    "severity": "WARNING",
                    "metric": self.kpis[name]["metric"],
                }
                new_alerts.append(alert)
                self.alerts.append(
                    {
                        **alert,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        return new_alerts
