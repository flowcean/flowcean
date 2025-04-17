import logging
from math import nan

import polars as pl
from scipy.spatial.transform import Rotation

from flowcean.core.transform import Transform

logger = logging.getLogger(__name__)


class LocalizationStatus(Transform):
    """Checks if the robot is delocalized based on position and heading errors.

    Compares the estimated pose with the ground truth pose to determine
    the position error (i.e. euclidean distance) and the heading error robot is
    delocalized. The robot is considered delocalized if any error is above
    a certain threshold regarding either heading or position.
    The `isDelocalized` column is computed based on the position and heading
    errors. If both errors are below the threshold, the robot is considered
    localized. Otherwise, it is considered delocalized.
    """

    def __init__(
        self,
        ground_truth_pose: str = "/momo/pose",
        estimated_pose: str = "/amcl_pose",
        position_threshold: float = 0.7,
        heading_threshold: float = 0.7,
    ) -> None:
        """Initialize the LocalizationStatus transform.

        Args:
            ground_truth_pose: Topic name for the ground truth pose.
            estimated_pose: Topic name for the estimated pose.
            position_threshold: Max position error to be considered localized.
            heading_threshold: Max heading error to be considered localized.
        """
        self.ground_truth_pose = ground_truth_pose
        self.estimated_pose = estimated_pose
        self.position_threshold = position_threshold
        self.heading_threshold = heading_threshold

    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """Apply the localization status transformation.

        Args:
            data: Input LazyFrame containing pose timeseries data.

        Returns:
            data: LazyFrame with added columns: position_error, heading_error, isDelocalized.
        """
        logger.debug("Computing localization status")
        topics = [self.ground_truth_pose, self.estimated_pose]
        joined = self._join_topics(data, topics)
        filled = joined.select(
            pl.all().forward_fill().over("index"),
        ).drop_nulls()

        # Compute the position error (euclidean distance)
        position_error = filled.with_columns(
            (
                (
                    pl.col(f"{self.ground_truth_pose}/pose.position.x")
                    - pl.col(f"{self.estimated_pose}/pose.pose.position.x")
                )
                ** 2
                + (
                    pl.col(f"{self.ground_truth_pose}/pose.position.y")
                    - pl.col(f"{self.estimated_pose}/pose.pose.position.y")
                )
                ** 2
            )
            .sqrt()
            .alias("position_error"),
        )
        # Compute heading error
        heading_error = position_error.with_columns(
            pl.struct(
                [
                    f"{self.ground_truth_pose}/pose.orientation.x",
                    f"{self.ground_truth_pose}/pose.orientation.y",
                    f"{self.ground_truth_pose}/pose.orientation.z",
                    f"{self.ground_truth_pose}/pose.orientation.w",
                    f"{self.estimated_pose}/pose.pose.orientation.x",
                    f"{self.estimated_pose}/pose.pose.orientation.y",
                    f"{self.estimated_pose}/pose.pose.orientation.z",
                    f"{self.estimated_pose}/pose.pose.orientation.w",
                ],
            )
            .map_elements(
                lambda s: self._calculate_heading_error(
                    [
                        s[f"{self.ground_truth_pose}/pose.orientation.x"],
                        s[f"{self.ground_truth_pose}/pose.orientation.y"],
                        s[f"{self.ground_truth_pose}/pose.orientation.z"],
                        s[f"{self.ground_truth_pose}/pose.orientation.w"],
                    ],
                    [
                        s[f"{self.estimated_pose}/pose.pose.orientation.x"],
                        s[f"{self.estimated_pose}/pose.pose.orientation.y"],
                        s[f"{self.estimated_pose}/pose.pose.orientation.z"],
                        s[f"{self.estimated_pose}/pose.pose.orientation.w"],
                    ],
                ),
                return_dtype=pl.Float64,
            )
            .alias("heading_error"),
        )
        # Determine if the robot is delocalized
        delocalized = heading_error.with_columns(
            (
                (pl.col("position_error") > self.position_threshold)
                | (pl.col("heading_error") > self.heading_threshold)
            ).alias("isDelocalized"),
        )
        # put isDelocalized, position_error and heading_error back into the original format
        localization_status_features = {
            "isDelocalized": "time",
            "position_error": "time",
            "heading_error": "time",
        }
        if isinstance(localization_status_features, str):
            time_feature = {
                feature_name: localization_status_features
                for feature_name in data.collect_schema().names()
                if feature_name != localization_status_features
            }
        else:
            time_feature = localization_status_features

        nested = delocalized.select(
            [
                pl.struct(
                    pl.col(t_feature).alias("time"),
                    pl.col(value_feature).alias("value"),
                )
                .implode()
                .alias(value_feature)
                for value_feature, t_feature in time_feature.items()
            ],
        )

        return pl.concat([data, nested], how="horizontal")

    def _explode_timeseries(
        self,
        df: pl.LazyFrame,
        column: str,
    ) -> pl.LazyFrame:
        exploded = df.select(column).with_row_index().explode(column)
        unnested = exploded.unnest(column)
        renamed = unnested.with_columns(
            pl.col("value").name.map_fields(lambda x: f"{column}/{x}"),
        )
        return renamed.unnest("value")

    def _join_topics(
        self,
        df: pl.LazyFrame,
        topics: list[str],
    ) -> pl.LazyFrame:
        it = iter(topics)
        joined = self._explode_timeseries(df, next(it))

        for topic in it:
            joined = joined.join(
                self._explode_timeseries(df, topic),
                on=["index", "time"],
                how="full",
                coalesce=True,
            )

        return joined.sort(["index", "time"])

    def _calculate_heading_error(
        self,
        ground_truth_quat: list[float],
        estimated_quat: list[float],
    ) -> float:
        """Calculate the heading error between two quaternions.

        Args:
            ground_truth_quat: Ground truth quaternion [x, y, z, w].
            estimated_quat: Estimated quaternion [x, y, z, w].

        Returns:
            Heading error in radians, or nan if invalid.
        """
        try:
            ground_truth_rot = Rotation.from_quat(ground_truth_quat)
            estimated_rot = Rotation.from_quat(estimated_quat)
            relative_rot = ground_truth_rot.inv() * estimated_rot
            return abs(relative_rot.as_euler("zyx")[0])  # Yaw
        except ValueError as e:
            msg = (
                "Invalid quaternion encountered. "
                f"Ground truth: {ground_truth_quat}, "
                f"Estimated: {estimated_quat}. Error: {e}"
            )
            logger.debug(msg)
            return nan
