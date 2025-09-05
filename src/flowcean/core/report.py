from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, cast

import polars as pl

if TYPE_CHECKING:
    from great_tables import GT


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
