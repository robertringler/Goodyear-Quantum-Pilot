"""Accuracy Validation Module.

Provides accuracy validation and comparison tools for:
- Quantum simulation accuracy vs experiment
- Material property predictions
- Tire performance predictions
- Statistical uncertainty quantification
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation accuracy levels."""

    QUALITATIVE = auto()  # Correct trends
    SEMI_QUANTITATIVE = auto()  # Within 20%
    QUANTITATIVE = auto()  # Within 5%
    HIGH_FIDELITY = auto()  # Within 1%
    EXACT = auto()  # Numerical precision


@dataclass
class ValidationResult:
    """Validation comparison result.

    Attributes:
        property_name: Property being validated
        predicted: Predicted value(s)
        experimental: Experimental value(s)
        error_absolute: Absolute error
        error_relative: Relative error (%)
        r_squared: R² for series data
        rmse: Root mean square error
        validation_level: Achieved accuracy level
    """

    property_name: str
    predicted: float | NDArray[np.float64]
    experimental: float | NDArray[np.float64]
    error_absolute: float = 0.0
    error_relative: float = 0.0
    r_squared: float = 0.0
    rmse: float = 0.0
    validation_level: ValidationLevel = ValidationLevel.QUALITATIVE

    @property
    def passed(self) -> bool:
        """Check if validation passed (within 10%)."""
        return self.error_relative < 10.0


@dataclass
class UncertaintyEstimate:
    """Uncertainty quantification result.

    Attributes:
        mean: Mean prediction
        std: Standard deviation
        ci_lower: Lower confidence bound (95%)
        ci_upper: Upper confidence bound (95%)
        samples: Number of samples
    """

    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    samples: int = 0

    @property
    def relative_uncertainty(self) -> float:
        """Relative uncertainty (%)."""
        if self.mean != 0:
            return self.std / abs(self.mean) * 100
        return float("inf")


class AccuracyValidator:
    """General accuracy validation framework.

    Compares predictions against experimental data with
    comprehensive statistical analysis.

    Example:
        >>> validator = AccuracyValidator()
        >>>
        >>> # Compare tensile strength predictions
        >>> result = validator.validate_property(
        ...     "tensile_strength",
        ...     predicted=[20.1, 18.5, 22.3],
        ...     experimental=[19.8, 18.2, 21.9],
        ... )
        >>> print(f"RMSE: {result.rmse:.2f} MPa")
    """

    # Thresholds for validation levels
    THRESHOLDS = {
        ValidationLevel.EXACT: 0.001,
        ValidationLevel.HIGH_FIDELITY: 0.01,
        ValidationLevel.QUANTITATIVE: 0.05,
        ValidationLevel.SEMI_QUANTITATIVE: 0.20,
        ValidationLevel.QUALITATIVE: 0.50,
    }

    def __init__(self) -> None:
        """Initialize accuracy validator."""
        self.results: list[ValidationResult] = []

    def validate_property(
        self,
        property_name: str,
        predicted: float | list[float] | NDArray[np.float64],
        experimental: float | list[float] | NDArray[np.float64],
    ) -> ValidationResult:
        """Validate a property prediction.

        Args:
            property_name: Property being validated
            predicted: Predicted value(s)
            experimental: Experimental value(s)

        Returns:
            Validation result
        """
        pred_arr = np.atleast_1d(np.array(predicted))
        exp_arr = np.atleast_1d(np.array(experimental))

        # Compute errors
        abs_errors = np.abs(pred_arr - exp_arr)
        rel_errors = abs_errors / np.abs(exp_arr) * 100

        mean_abs = float(np.mean(abs_errors))
        mean_rel = float(np.mean(rel_errors))
        rmse = float(np.sqrt(np.mean(abs_errors**2)))

        # R² for arrays
        if len(pred_arr) > 1:
            ss_res = np.sum((exp_arr - pred_arr) ** 2)
            ss_tot = np.sum((exp_arr - np.mean(exp_arr)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        else:
            r_squared = 1.0 - mean_rel / 100

        # Determine validation level
        rel_error_fraction = mean_rel / 100
        level = ValidationLevel.QUALITATIVE
        for lvl, threshold in sorted(self.THRESHOLDS.items(), key=lambda x: x[1]):
            if rel_error_fraction <= threshold:
                level = lvl
                break

        result = ValidationResult(
            property_name=property_name,
            predicted=pred_arr if len(pred_arr) > 1 else float(pred_arr[0]),
            experimental=exp_arr if len(exp_arr) > 1 else float(exp_arr[0]),
            error_absolute=mean_abs,
            error_relative=mean_rel,
            r_squared=float(r_squared),
            rmse=rmse,
            validation_level=level,
        )

        self.results.append(result)
        return result

    def validate_time_series(
        self,
        property_name: str,
        predicted: NDArray[np.float64],
        experimental: NDArray[np.float64],
        time: NDArray[np.float64] | None = None,
    ) -> ValidationResult:
        """Validate time series prediction.

        Args:
            property_name: Property name
            predicted: Predicted time series
            experimental: Experimental time series
            time: Time points (optional)

        Returns:
            Validation result
        """
        return self.validate_property(property_name, predicted, experimental)

    def get_summary(self) -> dict[str, Any]:
        """Get validation summary."""
        if not self.results:
            return {}

        passed = sum(1 for r in self.results if r.passed)

        return {
            "total_validations": len(self.results),
            "passed": passed,
            "failed": len(self.results) - passed,
            "pass_rate": passed / len(self.results) * 100,
            "average_error_pct": np.mean([r.error_relative for r in self.results]),
            "max_error_pct": max(r.error_relative for r in self.results),
            "average_r_squared": np.mean([r.r_squared for r in self.results]),
        }


class ExperimentalComparison:
    """Compare predictions with experimental databases.

    Provides access to experimental data for validation:
    - Mechanical properties
    - Thermal properties
    - Durability test results
    - Performance metrics
    """

    # Sample experimental database (in production, from actual tests)
    EXPERIMENTAL_DATA = {
        "NR_tensile_strength": {"mean": 25.0, "std": 2.5, "unit": "MPa"},
        "NR_elongation": {"mean": 600.0, "std": 50.0, "unit": "%"},
        "NR_hardness": {"mean": 55.0, "std": 3.0, "unit": "Shore A"},
        "SBR_tensile_strength": {"mean": 18.0, "std": 2.0, "unit": "MPa"},
        "SBR_elongation": {"mean": 450.0, "std": 40.0, "unit": "%"},
        "silica_rolling_resistance": {"mean": 0.008, "std": 0.001, "unit": "N/N"},
        "tire_wear_rate": {"mean": 0.05, "std": 0.01, "unit": "mm/1000km"},
    }

    def __init__(self) -> None:
        """Initialize experimental comparison."""
        self.data = self.EXPERIMENTAL_DATA.copy()

    def get_experimental_value(
        self,
        property_key: str,
    ) -> dict[str, Any] | None:
        """Get experimental value for property.

        Args:
            property_key: Property identifier

        Returns:
            Experimental data or None
        """
        return self.data.get(property_key)

    def compare_prediction(
        self,
        property_key: str,
        predicted_value: float,
    ) -> dict[str, Any]:
        """Compare prediction with experimental data.

        Args:
            property_key: Property identifier
            predicted_value: Predicted value

        Returns:
            Comparison result
        """
        exp = self.get_experimental_value(property_key)

        if exp is None:
            return {"error": f"No experimental data for {property_key}"}

        error = abs(predicted_value - exp["mean"])
        error_relative = error / exp["mean"] * 100
        z_score = error / exp["std"]

        return {
            "property": property_key,
            "predicted": predicted_value,
            "experimental_mean": exp["mean"],
            "experimental_std": exp["std"],
            "unit": exp["unit"],
            "error": error,
            "error_relative_pct": error_relative,
            "z_score": z_score,
            "within_1sigma": z_score <= 1.0,
            "within_2sigma": z_score <= 2.0,
        }

    def add_experimental_data(
        self,
        property_key: str,
        mean: float,
        std: float,
        unit: str,
    ) -> None:
        """Add experimental data point.

        Args:
            property_key: Property identifier
            mean: Mean value
            std: Standard deviation
            unit: Unit of measurement
        """
        self.data[property_key] = {"mean": mean, "std": std, "unit": unit}


class PredictionMetrics:
    """Statistical metrics for predictions.

    Computes comprehensive metrics:
    - Mean Absolute Error (MAE)
    - Root Mean Square Error (RMSE)
    - Mean Absolute Percentage Error (MAPE)
    - Coefficient of Determination (R²)
    - Concordance Correlation Coefficient (CCC)
    """

    @staticmethod
    def mae(
        predicted: NDArray[np.float64],
        actual: NDArray[np.float64],
    ) -> float:
        """Compute Mean Absolute Error.

        Args:
            predicted: Predicted values
            actual: Actual values

        Returns:
            MAE value
        """
        return float(np.mean(np.abs(predicted - actual)))

    @staticmethod
    def rmse(
        predicted: NDArray[np.float64],
        actual: NDArray[np.float64],
    ) -> float:
        """Compute Root Mean Square Error.

        Args:
            predicted: Predicted values
            actual: Actual values

        Returns:
            RMSE value
        """
        return float(np.sqrt(np.mean((predicted - actual) ** 2)))

    @staticmethod
    def mape(
        predicted: NDArray[np.float64],
        actual: NDArray[np.float64],
    ) -> float:
        """Compute Mean Absolute Percentage Error.

        Args:
            predicted: Predicted values
            actual: Actual values

        Returns:
            MAPE value (%)
        """
        # Avoid division by zero
        mask = actual != 0
        return float(np.mean(np.abs((predicted[mask] - actual[mask]) / actual[mask])) * 100)

    @staticmethod
    def r_squared(
        predicted: NDArray[np.float64],
        actual: NDArray[np.float64],
    ) -> float:
        """Compute Coefficient of Determination (R²).

        Args:
            predicted: Predicted values
            actual: Actual values

        Returns:
            R² value
        """
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)

        if ss_tot > 0:
            return float(1 - ss_res / ss_tot)
        return 0.0

    @staticmethod
    def ccc(
        predicted: NDArray[np.float64],
        actual: NDArray[np.float64],
    ) -> float:
        """Compute Concordance Correlation Coefficient.

        CCC = 2 * r * σ_x * σ_y / (σ_x² + σ_y² + (μ_x - μ_y)²)

        Args:
            predicted: Predicted values
            actual: Actual values

        Returns:
            CCC value (-1 to 1)
        """
        mean_pred = np.mean(predicted)
        mean_act = np.mean(actual)
        var_pred = np.var(predicted)
        var_act = np.var(actual)
        cov = np.mean((predicted - mean_pred) * (actual - mean_act))

        denominator = var_pred + var_act + (mean_pred - mean_act) ** 2

        if denominator > 0:
            return float(2 * cov / denominator)
        return 0.0

    @classmethod
    def all_metrics(
        cls,
        predicted: NDArray[np.float64],
        actual: NDArray[np.float64],
    ) -> dict[str, float]:
        """Compute all prediction metrics.

        Args:
            predicted: Predicted values
            actual: Actual values

        Returns:
            Dictionary of all metrics
        """
        return {
            "mae": cls.mae(predicted, actual),
            "rmse": cls.rmse(predicted, actual),
            "mape": cls.mape(predicted, actual),
            "r_squared": cls.r_squared(predicted, actual),
            "ccc": cls.ccc(predicted, actual),
        }


class MaterialValidation:
    """Material property validation.

    Validates quantum-predicted material properties
    against experimental measurements.

    Categories:
    - Mechanical properties
    - Thermal properties
    - Dynamic properties
    - Aging properties
    """

    # Validation thresholds by property type
    THRESHOLDS = {
        "tensile_strength": 10.0,  # 10% error acceptable
        "elongation": 15.0,
        "hardness": 5.0,  # Shore units
        "modulus": 10.0,
        "tear_strength": 15.0,
        "rebound": 10.0,
        "hysteresis": 10.0,
        "thermal_conductivity": 15.0,
        "glass_transition": 3.0,  # °C
    }

    def __init__(self) -> None:
        """Initialize material validation."""
        self.validator = AccuracyValidator()
        self.comparison = ExperimentalComparison()

    def validate_mechanical(
        self,
        material_id: str,
        predicted: dict[str, float],
        experimental: dict[str, float] | None = None,
    ) -> dict[str, ValidationResult]:
        """Validate mechanical properties.

        Args:
            material_id: Material identifier
            predicted: Predicted properties
            experimental: Experimental properties (or use database)

        Returns:
            Validation results by property
        """
        results = {}

        properties = [
            "tensile_strength",
            "elongation",
            "hardness",
            "modulus_100",
            "modulus_300",
            "tear_strength",
        ]

        for prop in properties:
            if prop in predicted:
                # Get experimental value
                if experimental and prop in experimental:
                    exp_val = experimental[prop]
                else:
                    exp_data = self.comparison.get_experimental_value(f"{material_id}_{prop}")
                    if exp_data:
                        exp_val = exp_data["mean"]
                    else:
                        continue

                result = self.validator.validate_property(
                    f"{material_id}.{prop}",
                    predicted[prop],
                    exp_val,
                )
                results[prop] = result

        return results

    def validate_dynamic(
        self,
        material_id: str,
        predicted_storage_modulus: NDArray[np.float64],
        predicted_loss_modulus: NDArray[np.float64],
        frequencies: NDArray[np.float64],
        experimental_storage: NDArray[np.float64] | None = None,
        experimental_loss: NDArray[np.float64] | None = None,
    ) -> dict[str, Any]:
        """Validate dynamic mechanical properties.

        Args:
            material_id: Material identifier
            predicted_storage_modulus: Predicted E' (MPa)
            predicted_loss_modulus: Predicted E" (MPa)
            frequencies: Frequency points (Hz)
            experimental_storage: Experimental E' (optional)
            experimental_loss: Experimental E" (optional)

        Returns:
            Validation results
        """
        results = {}

        # Tan delta
        predicted_tan_delta = predicted_loss_modulus / predicted_storage_modulus

        if experimental_storage is not None:
            result = self.validator.validate_time_series(
                f"{material_id}.storage_modulus",
                predicted_storage_modulus,
                experimental_storage,
            )
            results["storage_modulus"] = result

        if experimental_loss is not None:
            result = self.validator.validate_time_series(
                f"{material_id}.loss_modulus",
                predicted_loss_modulus,
                experimental_loss,
            )
            results["loss_modulus"] = result

            experimental_tan_delta = experimental_loss / experimental_storage
            result = self.validator.validate_time_series(
                f"{material_id}.tan_delta",
                predicted_tan_delta,
                experimental_tan_delta,
            )
            results["tan_delta"] = result

        return results

    def validate_aging(
        self,
        material_id: str,
        predicted_retention: dict[str, float],
        aging_conditions: dict[str, Any],
        experimental_retention: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Validate aging predictions.

        Args:
            material_id: Material identifier
            predicted_retention: Predicted property retention (%)
            aging_conditions: Aging conditions (temp, time, etc.)
            experimental_retention: Experimental retention (optional)

        Returns:
            Validation results
        """
        results = {
            "aging_conditions": aging_conditions,
            "predictions": predicted_retention,
        }

        if experimental_retention:
            validations = {}
            for prop in predicted_retention:
                if prop in experimental_retention:
                    result = self.validator.validate_property(
                        f"{material_id}.aging.{prop}",
                        predicted_retention[prop],
                        experimental_retention[prop],
                    )
                    validations[prop] = result

            results["validation"] = validations

        return results

    def generate_validation_report(
        self,
        material_id: str,
    ) -> str:
        """Generate validation report for material.

        Args:
            material_id: Material identifier

        Returns:
            Formatted report string
        """
        summary = self.validator.get_summary()

        report = f"""
MATERIAL VALIDATION REPORT
==========================

Material: {material_id}
Date: 2025-01-13

SUMMARY
-------
Total Validations: {summary.get('total_validations', 0)}
Passed: {summary.get('passed', 0)}
Failed: {summary.get('failed', 0)}
Pass Rate: {summary.get('pass_rate', 0):.1f}%

Average Error: {summary.get('average_error_pct', 0):.2f}%
Maximum Error: {summary.get('max_error_pct', 0):.2f}%
Average R²: {summary.get('average_r_squared', 0):.4f}

DETAILED RESULTS
----------------
"""

        for result in self.validator.results:
            if material_id in result.property_name:
                status = "✓ PASS" if result.passed else "✗ FAIL"
                report += f"""
{result.property_name}:
  Predicted: {result.predicted}
  Experimental: {result.experimental}
  Error: {result.error_relative:.2f}%
  Level: {result.validation_level.name}
  Status: {status}
"""

        return report
