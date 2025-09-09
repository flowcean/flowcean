from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
import polars as pl
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from great_tables import GT
    from rich.style import StyleType


class Reportable(Protocol):
    @abstractmethod
    def __str__(self) -> str:
        """Return a string representation."""


class ReportEntry(dict[str, Reportable | Mapping[str, Reportable]]):
    def flatten(self, delimiter: str = ": ") -> dict[str, Reportable]:
        row: dict[str, Reportable] = {}
        for metric, value in self.items():
            if isinstance(value, Mapping):
                value = cast("Mapping[str, Reportable]", value)
                row.update(
                    {
                        f"{metric}{delimiter}{submetric}": subvalue
                        for submetric, subvalue in value.items()
                    },
                )
            else:
                row[metric] = value
        return row


class Report(dict[str, ReportEntry]):
    """A structured container for evaluation results of multiple models.

    The `Report` maps model names to their metric results. For each model:

      * top-level keys are metric names,
      * values are either:
          - a single `Reportable` (e.g., scalar metric result), or
          - a nested mapping from submetric names to `Reportable` objects
            (e.g., per-class F1 scores, per-feature regression errors, or
            multi-output results).

    This hierarchical structure allows uniform representation of both
    simple metrics and complex hierarchical metrics across multiple models.

    Example:
    >>> report = Report(
    ...     {
    ...         "model_a": {
    ...             "accuracy": 0.95,
    ...             "f1": {"class_0": 0.91, "class_1": 0.89},
    ...         },
    ...         "model_b": {
    ...             "mae": {"feature_x": 0.2, "feature_y": 0.3},
    ...         },
    ...     }
    ... )
    """

    def great_table(self) -> GT:
        rows = [
            {"model": model} | entry.flatten() for model, entry in self.items()
        ]
        table = pl.DataFrame(rows)
        return (
            table.style.tab_spanner_delim(": ")  # pyright: ignore[reportUnknownMemberType] until https://github.com/posit-dev/great-tables/pull/770 is merged
            .tab_header(
                title="Model Evaluation Report",
                subtitle="Metrics for each trained model",
            )
            .tab_stub(rowname_col="model")
            .tab_stubhead(label="Model")
            .fmt_number(decimals=2)
            .opt_stylize(style=3)
        )

    def __repr__(self) -> str:
        return f"Report(models={list(self.keys())})"

    def __str__(self) -> str:
        lines: list[str] = []
        for model, entry in self.items():
            lines.append(f"=== {model} ===")
            for metric, value in entry.items():
                if isinstance(value, Mapping):
                    value = cast("Mapping[str, Reportable]", value)
                    for submetric, subvalue in value.items():
                        reportable = _format_value(subvalue)
                        lines.append(f"{metric} -> {submetric} {reportable}")
                else:
                    reportable = _format_value(value)
                    lines.append(f"{metric} {reportable}")
            lines.append("")
        return "\n".join(lines)

    def pretty_print(
        self,
        header_style: StyleType = "bold magenta",
        metric_style: StyleType = "cyan",
        value_style: StyleType = "green",
        title_style: StyleType = "bold yellow",
    ) -> None:
        """Pretty print the report to the terminal."""
        console = Console()
        for model, entry in self.items():
            table = Table(show_header=True, header_style=header_style)
            table.add_column("Metric", style=metric_style, no_wrap=True)
            table.add_column("Value", style=value_style)

            for metric, value in entry.items():
                if isinstance(value, Mapping):
                    value = cast("Mapping[str, Reportable]", value)
                    for submetric, subvalue in value.items():
                        table.add_row(
                            f"{metric} → {submetric}",
                            _format_value(subvalue),
                        )
                else:
                    table.add_row(metric, _format_value(value))

            panel = Panel(
                table,
                title=f"[{title_style}]{model}[/]",
                expand=False,
            )
            console.print(panel)


def _format_value(value: Reportable) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{value:.4f}"
    return str(value)
