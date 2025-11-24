from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from flowcean.core import Report
from flowcean.core.report import ReportEntry

if TYPE_CHECKING:
    from flowcean.core.model import Model


class TestSelectBestModelBasics:
    """Test basic functionality of select_best_model."""

    def test_maximize_with_single_metric_simple_case(self) -> None:
        """Test selecting model with highest accuracy value."""
        # Arrange
        model_a = Mock(spec=["predict"])
        model_b = Mock(spec=["predict"])
        model_c = Mock(spec=["predict"])

        report = Report(
            {
                "model_a": ReportEntry({"accuracy": 0.85}),
                "model_b": ReportEntry({"accuracy": 0.92}),
                "model_c": ReportEntry({"accuracy": 0.78}),
            },
        )

        # Manually attach models as the method expects
        models_dict: dict[str, Model] = {
            "model_a": model_a,
            "model_b": model_b,
            "model_c": model_c,
        }
        object.__setattr__(report, "models_by_name", models_dict)

        # Act
        best_model = report.select_best_model("accuracy", mode="maximize")

        # Assert
        assert best_model is model_b, (
            "Should return model with highest accuracy (0.92)"
        )

    def test_minimize_with_single_metric_simple_case(self) -> None:
        """Test selecting model with lowest loss value."""
        # Arrange
        model_a = Mock(spec=["predict"])
        model_b = Mock(spec=["predict"])
        model_c = Mock(spec=["predict"])

        report = Report(
            {
                "model_a": ReportEntry({"loss": 0.5}),
                "model_b": ReportEntry({"loss": 0.3}),
                "model_c": ReportEntry({"loss": 0.7}),
            },
        )

        models_dict: dict[str, Model] = {
            "model_a": model_a,
            "model_b": model_b,
            "model_c": model_c,
        }
        object.__setattr__(report, "models_by_name", models_dict)

        # Act
        best_model = report.select_best_model("loss", mode="minimize")

        # Assert
        assert best_model is model_b, (
            "Should return model with lowest loss (0.3)"
        )


class TestSelectBestModelModeVariations:
    """Test different mode parameter variations."""

    def test_mode_max_shorthand(self) -> None:
        """Test that 'max' works as shorthand for 'maximize'."""
        # Arrange
        model_a = Mock(spec=["predict"])
        model_b = Mock(spec=["predict"])

        report = Report(
            {
                "model_a": ReportEntry({"score": 0.5}),
                "model_b": ReportEntry({"score": 0.8}),
            },
        )

        models_dict: dict[str, Model] = {
            "model_a": model_a,
            "model_b": model_b,
        }
        object.__setattr__(report, "models_by_name", models_dict)

        # Act
        best_model = report.select_best_model("score", mode="max")

        # Assert
        assert best_model is model_b

    def test_mode_min_shorthand(self) -> None:
        """Test that 'min' works as shorthand for 'minimize'."""
        # Arrange
        model_a = Mock(spec=["predict"])
        model_b = Mock(spec=["predict"])

        report = Report(
            {
                "model_a": ReportEntry({"error": 0.5}),
                "model_b": ReportEntry({"error": 0.2}),
            },
        )

        models_dict: dict[str, Model] = {
            "model_a": model_a,
            "model_b": model_b,
        }
        object.__setattr__(report, "models_by_name", models_dict)

        # Act
        best_model = report.select_best_model("error", mode="min")

        # Assert
        assert best_model is model_b

    def test_mode_case_insensitivity(self) -> None:
        """Test that mode parameter is case-insensitive."""
        # Arrange
        model_a = Mock(spec=["predict"])
        model_b = Mock(spec=["predict"])

        report = Report(
            {
                "model_a": ReportEntry({"score": 0.5}),
                "model_b": ReportEntry({"score": 0.8}),
            },
        )

        models_dict: dict[str, Model] = {
            "model_a": model_a,
            "model_b": model_b,
        }
        object.__setattr__(report, "models_by_name", models_dict)

        # Act & Assert
        result_upper = report.select_best_model("score", mode="MAXIMIZE")
        result_mixed = report.select_best_model("score", mode="MaXiMiZe")
        result_lower = report.select_best_model("score", mode="maximize")

        assert result_upper is model_b
        assert result_mixed is model_b
        assert result_lower is model_b


class TestSelectBestModelNestedMetrics:
    """Test handling of nested (hierarchical) metrics."""

    def test_nested_metric_uses_average(self) -> None:
        """Test that nested metrics are averaged correctly."""
        # Arrange
        model_a = Mock(spec=["predict"])
        model_b = Mock(spec=["predict"])
        model_c = Mock(spec=["predict"])

        report = Report(
            {
                "model_a": ReportEntry(
                    {"f1": {"class_0": 0.8, "class_1": 0.9}},
                ),  # avg: 0.85
                "model_b": ReportEntry(
                    {"f1": {"class_0": 0.7, "class_1": 0.8}},
                ),  # avg: 0.75
                "model_c": ReportEntry(
                    {"f1": {"class_0": 0.9, "class_1": 0.95}},
                ),  # avg: 0.925
            },
        )

        models_dict: dict[str, Model] = {
            "model_a": model_a,
            "model_b": model_b,
            "model_c": model_c,
        }
        object.__setattr__(report, "models_by_name", models_dict)

        # Act
        best_model = report.select_best_model("f1", mode="maximize")

        # Assert
        assert best_model is model_c, (
            "Should select model_c with average f1 of 0.925"
        )

    def test_nested_metric_minimize(self) -> None:
        """Test minimizing with nested metrics."""
        # Arrange
        model_a = Mock(spec=["predict"])
        model_b = Mock(spec=["predict"])

        report = Report(
            {
                "model_a": ReportEntry(
                    {"mae": {"feature_x": 0.5, "feature_y": 0.6}},
                ),  # avg: 0.55
                "model_b": ReportEntry(
                    {"mae": {"feature_x": 0.2, "feature_y": 0.3}},
                ),  # avg: 0.25
            },
        )

        models_dict: dict[str, Model] = {
            "model_a": model_a,
            "model_b": model_b,
        }
        object.__setattr__(report, "models_by_name", models_dict)

        # Act
        best_model = report.select_best_model("mae", mode="minimize")

        # Assert
        assert best_model is model_b, (
            "Should select model_b with lower average mae"
        )

    def test_mixed_simple_and_nested_metrics(self) -> None:
        """Test when some models have simple metrics and others have nested."""
        # Arrange
        model_a = Mock(spec=["predict"])
        model_b = Mock(spec=["predict"])
        model_c = Mock(spec=["predict"])

        report = Report(
            {
                "model_a": ReportEntry({"score": 0.80}),
                "model_b": ReportEntry(
                    {"score": {"sub1": 0.7, "sub2": 0.9}},
                ),  # avg: 0.8
                "model_c": ReportEntry({"score": 0.85}),
            },
        )

        models_dict: dict[str, Model] = {
            "model_a": model_a,
            "model_b": model_b,
            "model_c": model_c,
        }
        object.__setattr__(report, "models_by_name", models_dict)

        # Act
        best_model = report.select_best_model("score", mode="maximize")

        # Assert
        assert best_model is model_c, "Should select model_c with score 0.85"


class TestSelectBestModelErrorCases:
    """Test error handling and edge cases."""

    def test_invalid_mode_raises_valueerror(self) -> None:
        """Test that invalid mode parameter raises ValueError."""
        # Arrange
        model_a = Mock(spec=["predict"])
        report = Report({"model_a": ReportEntry({"accuracy": 0.85})})
        models_dict: dict[str, Model] = {"model_a": model_a}
        object.__setattr__(report, "models_by_name", models_dict)

        # Act & Assert
        with pytest.raises(ValueError, match="mode must be one of"):
            report.select_best_model("accuracy", mode="invalid")

    def test_metric_not_found_raises_valueerror(self) -> None:
        """Test that missing metric raises ValueError."""
        # Arrange
        model_a = Mock(spec=["predict"])
        report = Report({"model_a": ReportEntry({"accuracy": 0.85})})
        models_dict: dict[str, Model] = {"model_a": model_a}
        object.__setattr__(report, "models_by_name", models_dict)

        # Act & Assert
        with pytest.raises(ValueError, match="Metric 'nonexistent' not found"):
            report.select_best_model("nonexistent", mode="maximize")

    def test_no_models_attached_raises_valueerror(self) -> None:
        """Test that missing models_by_name attribute raises ValueError."""
        # Arrange
        report = Report({"model_a": ReportEntry({"accuracy": 0.85})})
        # Intentionally don't attach models_by_name

        # Act & Assert
        with pytest.raises(
            ValueError,
            match="does not contain attached models",
        ):
            report.select_best_model("accuracy", mode="maximize")

    def test_best_model_not_in_models_dict_raises_valueerror(self) -> None:
        """Test when best model name is not in models_by_name dict."""
        # Arrange
        model_a = Mock(spec=["predict"])
        model_c = Mock(spec=["predict"])

        report = Report(
            {
                "model_a": ReportEntry({"accuracy": 0.85}),
                "model_b": ReportEntry({"accuracy": 0.92}),
                "model_c": ReportEntry({"accuracy": 0.78}),
            },
        )

        # models_by_name is missing model_b (which has best accuracy)
        models_dict: dict[str, Model] = {
            "model_a": model_a,
            "model_c": model_c,
        }
        object.__setattr__(report, "models_by_name", models_dict)

        # Act & Assert
        with pytest.raises(
            ValueError,
            match="does not contain attached models",
        ):
            report.select_best_model("accuracy", mode="maximize")

    def test_models_by_name_not_a_dict_raises_valueerror(self) -> None:
        """Test when models_by_name is not a dict."""
        # Arrange
        report = Report({"model_a": ReportEntry({"accuracy": 0.85})})
        object.__setattr__(report, "models_by_name", "not a dict")

        # Act & Assert
        with pytest.raises(
            ValueError,
            match="does not contain attached models",
        ):
            report.select_best_model("accuracy", mode="maximize")

    def test_empty_nested_metric_skipped(self) -> None:
        """Test that models with empty nested metrics are skipped."""
        # Arrange
        model_a = Mock(spec=["predict"])
        model_b = Mock(spec=["predict"])

        report = Report(
            {
                "model_a": ReportEntry({"f1": {}}),  # Empty nested dict
                "model_b": ReportEntry({"f1": {"class_0": 0.9}}),
            },
        )

        models_dict: dict[str, Model] = {
            "model_a": model_a,
            "model_b": model_b,
        }
        object.__setattr__(report, "models_by_name", models_dict)

        # Act
        best_model = report.select_best_model("f1", mode="maximize")

        # Assert
        assert best_model is model_b, (
            "Should skip model_a with empty nested metric"
        )

    def test_all_models_missing_metric_raises_valueerror(self) -> None:
        """Test when metric exists in some entries but not the one searched for."""
        # Arrange
        model_a = Mock(spec=["predict"])
        model_b = Mock(spec=["predict"])

        report = Report(
            {
                "model_a": ReportEntry({"accuracy": 0.85}),
                "model_b": ReportEntry({"precision": 0.90}),
            },
        )

        models_dict: dict[str, Model] = {
            "model_a": model_a,
            "model_b": model_b,
        }
        object.__setattr__(report, "models_by_name", models_dict)

        # Act & Assert
        with pytest.raises(ValueError, match="Metric 'recall' not found"):
            report.select_best_model("recall", mode="maximize")


class TestSelectBestModelEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_model_returns_that_model(self) -> None:
        """Test with only one model in report."""
        # Arrange
        model_a = Mock(spec=["predict"])
        report = Report({"model_a": ReportEntry({"accuracy": 0.85})})
        models_dict: dict[str, Model] = {"model_a": model_a}
        object.__setattr__(report, "models_by_name", models_dict)

        # Act
        best_model = report.select_best_model("accuracy", mode="maximize")

        # Assert
        assert best_model is model_a

    def test_tied_scores_returns_first_encountered(self) -> None:
        """Test behavior when multiple models have identical best scores."""
        # Arrange
        model_a = Mock(spec=["predict"])
        model_b = Mock(spec=["predict"])
        model_c = Mock(spec=["predict"])

        report = Report(
            {
                "model_a": ReportEntry({"accuracy": 0.90}),
                "model_b": ReportEntry({"accuracy": 0.90}),
                "model_c": ReportEntry({"accuracy": 0.80}),
            },
        )

        models_dict: dict[str, Model] = {
            "model_a": model_a,
            "model_b": model_b,
            "model_c": model_c,
        }
        object.__setattr__(report, "models_by_name", models_dict)

        # Act
        best_model = report.select_best_model("accuracy", mode="maximize")

        # Assert - should return first encountered in iteration order
        assert best_model is model_a, "Should return first model when tied"

    def test_negative_metric_values(self) -> None:
        """Test with negative metric values."""
        # Arrange
        model_a = Mock(spec=["predict"])
        model_b = Mock(spec=["predict"])
        model_c = Mock(spec=["predict"])

        report = Report(
            {
                "model_a": ReportEntry({"loss": -0.5}),
                "model_b": ReportEntry({"loss": -0.3}),
                "model_c": ReportEntry({"loss": -0.8}),
            },
        )

        models_dict: dict[str, Model] = {
            "model_a": model_a,
            "model_b": model_b,
            "model_c": model_c,
        }
        object.__setattr__(report, "models_by_name", models_dict)

        # Act & Assert
        best_max = report.select_best_model("loss", mode="maximize")
        assert best_max is model_b, "When maximizing, -0.3 > -0.5 > -0.8"

        best_min = report.select_best_model("loss", mode="minimize")
        assert best_min is model_c, "When minimizing, -0.8 < -0.5 < -0.3"

    def test_zero_values(self) -> None:
        """Test with zero metric values."""
        # Arrange
        model_a = Mock(spec=["predict"])
        model_b = Mock(spec=["predict"])

        report = Report(
            {
                "model_a": ReportEntry({"error": 0.0}),
                "model_b": ReportEntry({"error": 0.1}),
            },
        )

        models_dict: dict[str, Model] = {
            "model_a": model_a,
            "model_b": model_b,
        }
        object.__setattr__(report, "models_by_name", models_dict)

        # Act
        best_model = report.select_best_model("error", mode="minimize")

        # Assert
        assert best_model is model_a, "Should handle zero values correctly"

    def test_numpy_float_types(self) -> None:
        """Test with numpy float types."""
        import numpy as np

        # Arrange
        model_a = Mock(spec=["predict"])
        model_b = Mock(spec=["predict"])

        report = Report(
            {
                "model_a": ReportEntry({"accuracy": np.float64(0.85)}),
                "model_b": ReportEntry({"accuracy": np.float32(0.92)}),
            },
        )

        models_dict: dict[str, Model] = {
            "model_a": model_a,
            "model_b": model_b,
        }
        object.__setattr__(report, "models_by_name", models_dict)

        # Act
        best_model = report.select_best_model("accuracy", mode="maximize")

        # Assert
        assert best_model is model_b, "Should handle numpy float types"

    def test_very_large_number_of_nested_values(self) -> None:
        """Test averaging with many nested values."""
        # Arrange
        model_a = Mock(spec=["predict"])
        model_b = Mock(spec=["predict"])

        # Create many nested values
        many_values_a = {f"metric_{i}": 0.5 for i in range(100)}
        many_values_b = {f"metric_{i}": 0.6 for i in range(100)}

        report = Report(
            {
                "model_a": ReportEntry({"score": many_values_a}),  # avg: 0.5
                "model_b": ReportEntry({"score": many_values_b}),  # avg: 0.6
            },
        )

        models_dict: dict[str, Model] = {
            "model_a": model_a,
            "model_b": model_b,
        }
        object.__setattr__(report, "models_by_name", models_dict)

        # Act
        best_model = report.select_best_model("score", mode="maximize")

        # Assert
        assert best_model is model_b, (
            "Should handle many nested values correctly"
        )
