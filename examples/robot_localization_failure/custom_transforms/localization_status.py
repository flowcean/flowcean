import logging
from math import nan

import polars as pl
from scipy.spatial.transform import Rotation
from typing_extensions import override

from flowcean.core.transform import Transform

logger = logging.getLogger(__name__)


class LocalizationStatus(Transform):
    """Detects localization status based on position and heading errors."""

    def __init__(
        self,
        time_series: str,
        position_threshold: float,
        heading_threshold: float,
        *,
        ground_truth: str = "/momo/pose",
        estimation: str = "/amcl_pose",
    ) -> None:
        """Initialize the localization status transform.

        Args:
            time_series: Feature name containing time series data.
            position_threshold: Position error to consider delocalized.
            heading_threshold: Heading error to consider delocalized.
            ground_truth: Name of the ground truth pose feature.
            estimation: Name of the estimated pose feature.
        """
        self.time_series = time_series
        self.ground_truth = ground_truth
        self.estimation = estimation
        self.position_threshold = position_threshold
        self.heading_threshold = heading_threshold

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        logger.debug(
            "Calculating localization status for time series '%s' with "
            "position threshold %.2f and heading threshold %.2f",
            self.time_series,
            self.position_threshold,
            self.heading_threshold,
        )
        return data.with_columns(
            pl.col(self.time_series).list.eval(
                pl.element().struct.with_fields(
                    pl.field("value")
                    .struct.with_fields(
                        self._position_error(),
                        self._heading_error(),
                    )
                    .struct.with_fields(self._is_delocalized()),
                ),
            ),
        )

    def _position_error(self) -> pl.Expr:
        return (
            (
                (
                    pl.field(
                        f"{self.ground_truth}/pose.position.x",
                    )
                    - pl.field(
                        f"{self.estimation}/pose.pose.position.x",
                    )
                )
                ** 2
                + (
                    pl.field(
                        f"{self.ground_truth}/pose.position.y",
                    )
                    - pl.field(
                        f"{self.estimation}/pose.pose.position.y",
                    )
                )
                ** 2
            )
            .sqrt()
            .alias("position_error")
        )

    def _heading_error(self) -> pl.Expr:
        return (
            pl.struct(
                pl.field(
                    f"{self.ground_truth}/pose.orientation.x",
                ).alias("ground_truth_x"),
                pl.field(
                    f"{self.ground_truth}/pose.orientation.y",
                ).alias("ground_truth_y"),
                pl.field(
                    f"{self.ground_truth}/pose.orientation.z",
                ).alias("ground_truth_z"),
                pl.field(
                    f"{self.ground_truth}/pose.orientation.w",
                ).alias("ground_truth_w"),
                pl.field(
                    f"{self.estimation}/pose.pose.orientation.x",
                ).alias("estimated_x"),
                pl.field(
                    f"{self.estimation}/pose.pose.orientation.y",
                ).alias("estimated_y"),
                pl.field(
                    f"{self.estimation}/pose.pose.orientation.z",
                ).alias("estimated_z"),
                pl.field(
                    f"{self.estimation}/pose.pose.orientation.w",
                ).alias("estimated_w"),
            )
            .map_elements(_calculate_heading_error, return_dtype=pl.Float32)
            .alias("heading_error")
        )

    def _is_delocalized(self) -> pl.Expr:
        return (
            (pl.field("position_error") > self.position_threshold)
            | (pl.field("heading_error") > self.heading_threshold)
        ).alias("is_delocalized")


def _calculate_heading_error(sample: dict[str, float]) -> float:
    try:
        ground_truth = Rotation.from_quat(
            [
                sample["ground_truth_x"],
                sample["ground_truth_y"],
                sample["ground_truth_z"],
                sample["ground_truth_w"],
            ],
        )
        estimated = Rotation.from_quat(
            [
                sample["estimated_x"],
                sample["estimated_y"],
                sample["estimated_z"],
                sample["estimated_w"],
            ],
        )
        error = ground_truth.inv() * estimated
        _, _, yaw = error.as_euler("xyz")
    except ValueError as e:
        msg = (
            "Invalid quaternion encountered. "
            f"Ground truth: {sample}, "
            f"Estimated: {sample}. Error: {e}"
        )
        logger.debug(msg)
        return nan
    else:
        return yaw
