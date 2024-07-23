import logging
from typing import override

import polars as pl
import polars.selectors as cs

from flowcean.core import Transform

logger = logging.getLogger(__name__)


class EuclideanDistance(Transform):
    """Computes the Euclidean distance between two time series.

    Computes the Euclidean distance between two time series. The below example
    shows the usage of an `EuclideanDistance` transform in a `run.py` file.
    Assuming the loaded data is represented by the table:

    | feature_a            | feature_b            |
    | ---                  | ---                  |
    | [{1.0, {1.0, 1.0}},  | [{1.0, {1.0, 2.0}},  |
    |  {2.0, {2.0, 2.0}},  |  {2.0, {1.0, 2.0}},  |
    |  {3.0, {3.0, 3.0}},  |  {3.0, {2.0, 3.0}},  |
    |  {4.0, {4.0, 4.0}}]  |  {4.0, {3.0, 4.0}}]  |

    The following transform can be used to compute the Euclidean distance
    between the time series `feature_a` and `feature_b`.

    ```
        ...
        environment.load()
        data = environment.get_data()
        transform = EuclideanDistance(
            feature_a_name="feature_a",
            feature_b_name="feature_b",
            output_feature_name="euclidean_distance",
        )
        transformed_data = transform.transform(data)
        ...
    ```

    The resulting Dataframe after the transform is:

    | feature_a            | feature_b            | euclidean_distance        |
    | ---                  | ---                  | ---                       |
    | [{1.0, {1.0, 1.0}},  | [{1.0, {1.0, 2.0}},  | [{1.0, {1.414213562371}}, |
    |  {2.0, {2.0, 2.0}},  |  {2.0, {1.0, 2.0}},  |  {2.0, {1.414213562371}}, |
    |  {3.0, {3.0, 3.0}},  |  {3.0, {2.0, 3.0}},  |  {3.0, {1.414213562371}}, |
    |  {4.0, {4.0, 4.0}}]  |  {4.0, {3.0, 4.0}}]  |  {4.0, {1.414213562371}}] |

    """

    def __init__(
        self,
        feature_a_name: str,
        feature_b_name: str,
        output_feature_name: str,
    ) -> None:
        """Initializes the `EuclideanDistance` transform.

        Args:
            feature_a_name (str): The name of the first feature.
            feature_b_name (str): The name of the second feature.
            output_feature_name (str): The name of the output feature.

        """
        self.feature_a_name = feature_a_name
        self.feature_b_name = feature_b_name
        self.output_feature_name = output_feature_name

    @override
    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        logger.info(
            "Computing the Euclidean distance between %s and %s.",
            self.feature_a_name,
            self.feature_b_name,
        )

        time = (
            data.select(self.feature_a_name)
            .explode(self.feature_a_name)
            .unnest(self.feature_a_name)
            .select("time")
        )

        feature_a = (
            data.select(self.feature_a_name)
            .explode(self.feature_a_name)
            .unnest(self.feature_a_name)
            .select("value")
            .unnest("value")
        )
        feature_b = (
            data.select(self.feature_b_name)
            .explode(self.feature_b_name)
            .unnest(self.feature_b_name)
            .select("value")
            .unnest("value")
        )

        euclidean_distance = (
            (feature_a - feature_b)
            .select(cs.all() ** 2)
            .with_columns(value=pl.sum_horizontal(cs.all()).sqrt())
            .select("value")
        )

        return data.hstack(
            data.explode(cs.all())
            .hstack(euclidean_distance)
            .hstack(time)
            .select(
                [
                    pl.struct(
                        pl.col("time"),
                        pl.struct(pl.col("value")),
                    ).alias(self.output_feature_name)
                ]
            )
            .select(cs.all().implode())
        )
