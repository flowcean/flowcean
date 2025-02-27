import logging

import polars as pl

from flowcean.core.transform import Transform

logger = logging.getLogger(__name__)


class LocalizationStatus(Transform):
    """Checks if the robot is delocalized based on position and heading errors.

    Compares the position and heading errors to determine if the robot is
    delocalized. The robot is considered delocalized if any error is above
    a certain threshold.

    As an example, consider the following time series input data with position
    and heading error thresholds as 1.2 and 1.2 respectively:

    ┌────────────────────┬─────────────────┐
    │ /position_error    ┆ /heading_error  │
    │ list[struct[2]]    ┆ list[struct[2]] |
    ╞════════════════════╪═════════════════|
    │ [                  ┆ [               │
    │   {1000, {0.9}},   ┆   {1000, {1.1}},│
    │   {2000, {1.5}}    ┆   {2000, {0.8}} │
    │ ]                  ┆ ]               │
    └────────────────────┴─────────────────┘

    After applying the transform, the output will be:

    ┌────────────────────┬─────────────────┬─────────────────┐
    │ /position_error    ┆ /heading_error  ┆ isDelocalized   │
    │ list[struct[2]]    ┆ list[struct[2]] ┆ list[struct[2]] |
    ╞════════════════════╪═════════════════╪═════════════════|
    │ [                  ┆ [               ┆ [               │
    │   {1000, {0.9}},   ┆   {1000, {1.1}},┆   {1000, {0}},  │
    │   {2000, {1.5}}    ┆   {2000, {0.8}} ┆   {2000, {1}}   │
    │ ]                  ┆ ]               ┆ ]               │
    └────────────────────┴─────────────────┴─────────────────┘

    The `isDelocalized` column is computed based on the position and heading
    errors. If both errors are below the threshold, the robot is considered
    localized. Otherwise, it is considered delocalized.

    Note:
        Align the time stamps of the position and heading errors using
        MatchSamplingRate transform before applying this transform.
    """

    def __init__(
        self,
        position_error_feature_name: str = "/position_error",
        heading_error_feature_name: str = "/heading_error",
        position_threshold: float = 1.2,
        heading_threshold: float = 1.2,
    ) -> None:
        """Initialize the ParticleCloudImage transform.

        Args:
            position_error_feature_name: Name of the position error feature.
            heading_error_feature_name: Name of the heading error feature.
            position_threshold: Max position error to be considered localized.
            heading_threshold: Max heading error to be considered localized.
        """
        self.position_error_feature_name = position_error_feature_name
        self.heading_error_feature_name = heading_error_feature_name
        self.position_threshold = position_threshold
        self.heading_threshold = heading_threshold

    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        logger.debug("Computing localization status")

        data = data.with_columns(
            pl.struct(
                [
                    self.position_error_feature_name,
                    self.heading_error_feature_name,
                ],
            )
            .map_elements(
                lambda s: self.compute_localization_status(
                    s[self.position_error_feature_name],
                    s[self.heading_error_feature_name],
                ),
            )
            .alias("isDelocalized"),
        )

        logger.debug("Localization status computation completed")
        return data.collect().lazy()

    def compute_localization_status(
        self,
        pos_list: list,
        head_list: list,
    ) -> list:
        """Compute localization status based on position and heading errors."""
        return [
            {
                "time": pos["time"],
                "value": {
                    "data": 0
                    if (
                        pos["value"]["data"] < self.position_threshold
                        and head["value"]["data"] < self.heading_threshold
                    )
                    else 1,
                },
            }
            for pos, head in zip(pos_list, head_list, strict=False)
        ]
