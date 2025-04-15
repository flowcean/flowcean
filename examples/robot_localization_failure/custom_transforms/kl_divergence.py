import logging

import numpy as np
import polars as pl
from scipy.special import kl_div
from tqdm import tqdm

from flowcean.core.transform import Transform

logger = logging.getLogger(__name__)


class KLDivergence(Transform):
    """Computes KL divergence for multiple features based on a target column.

    For each feature specified, this transform computes KL divergence between
    the distribution of values when the target column (e.g., "isDelocalized")
    equals 0 (localized) and when it equals 1 (delocalized). The histograms for
    each feature are built using a fixed bin size (provided per feature).

    Additionally, if top_n is provided, only the top n features (those with the
    highest KL divergence values) are retained in the output.
    The corresponding feature columns are dropped from the DataFrame if they
    are not among the selected ones.

    The output is a new column "kl_divergence" that is a struct (dictionary)
    with keys as the feature names and values as the computed KL divergence.

    Example Input DataFrame (each column is a list of structs):
        ┌────────────────────┬───────────────────────┬──────────────────────┐
        │ isDelocalized      ┆ cog_max_distance      ┆ cog_mean_dist        │
        │ list[struct[2]]    ┆ list[struct[2]]       ┆ list[struct[2]]      │
        ├────────────────────┼───────────────────────┼──────────────────────┤
        │ [{time, {data:0}}, ┆[{time, {value:8.12}}, │[{time, {value:8.12}},│
        │  {time, {data:1}}, │ {time, {value:9.26}}, │ {time, {value:9.26}},│
        │   ...]             │  ...]                 │  ...]                │
        └────────────────────┴───────────────────────┴──────────────────────┘

    After applying KLDivergence with target "isDelocalized", features
    {"cog_max_distance": 0.01, "cog_mean_dist": 0.01},
    and top_n (e.g., top_n=2),
    the output DataFrame will include an additional column "kl_divergence"
    that looks like:

        ┌────────────────────────────────────────────────────┐
        │               kl_divergence                        │
        │                struct[top_n]                       │
        ├────────────────────────────────────────────────────┤
        │ { "cog_max_distance": 18.5, "cog_mean_dist": 3.2 } │
        └────────────────────────────────────────────────────┘

    and the feature columns not among the top_n will be dropped.

    Note: The timestamps of the features must be aligned with the target column
        (e.g., "isDelocalized") using the MatchSamplingRate transform
        before applying this transform.
    """

    def __init__(
        self,
        target_column: str,
        features: dict[str, float],
        top_n: int = -1,
    ) -> None:
        """Initialize the KLDivergence transform.

        Args:
            target_column: Name of the column used for grouping
                (e.g., "isDelocalized").
                Each element is expected to be a struct with a key "value"
                containing a dict with key "data".
            features: A dictionary mapping feature column names to
                their desired bin sizes.
                For example: {"cog_max_distance": 0.01, "cog_mean_dist": 0.01}
            top_n: Optional; if provided, only keep the top n features
                (by KL divergence) in the output.
                Features not selected will be dropped from the DataFrame.
        """
        self.target_column = target_column
        self.features = features
        self.top_n = top_n

    def compute_kl_divergence(
        self,
        statuses: list,
        feature_values: list,
        bin_size: float,
    ) -> float:
        """Compute the KL divergence for a feature using a fixed bin size.

        Args:
            statuses: List of status values
            (0 for localized, 1 for delocalized).
            feature_values: List of numeric values for the feature.
            bin_size: Fixed bin size
            (e.g. 0.01 for 1 cm, 1 for percent or degrees).

        Returns:
            The KL divergence value as a float.
        """
        values_localized = [
            v for s, v in zip(statuses, feature_values, strict=False) if s == 0
        ]
        values_delocalized = [
            v for s, v in zip(statuses, feature_values, strict=False) if s == 1
        ]

        if not values_localized or not values_delocalized:
            return float("nan")

        min_val = min(feature_values)
        max_val = max(feature_values)
        bins = np.arange(min_val, max_val + bin_size, bin_size)
        if len(bins) < 2:  # noqa: PLR2004
            return 0.0

        counts_localized, _ = np.histogram(values_localized, bins=bins)
        counts_delocalized, _ = np.histogram(values_delocalized, bins=bins)

        p_localized = counts_localized.astype(float) / counts_localized.sum()
        q_delocalized = (
            counts_delocalized.astype(float) / counts_delocalized.sum()
        )

        epsilon = 1e-10
        p_localized += epsilon
        q_delocalized += epsilon
        p_localized /= p_localized.sum()
        q_delocalized /= q_delocalized.sum()

        kl_elements = kl_div(p_localized, q_delocalized)
        return kl_elements.sum()

    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        logger.debug("Computing KL divergence for specified features")
        df = data.collect()
        row = df[0]

        # Extract statuses from the target column.
        # Each element is expected to be a dict:
        # {"time": ..., "value": {"data": 0 or 1}}
        target_list = row[self.target_column]
        statuses = [item["value"]["data"] for item in target_list[0]]

        # Compute KL divergence for each feature.
        kl_dict = {}
        for feature, bin_size in tqdm(
            self.features.items(),
            desc="Computing KL divergence",
        ):
            if feature not in row:
                continue
            feature_list = row[feature]
            feature_values = [item["value"] for item in feature_list[0]]
            kl_val = self.compute_kl_divergence(
                statuses,
                feature_values,
                bin_size,
            )
            kl_dict[feature] = kl_val

        logger.debug("KL divergence computation completed: %s", kl_dict)

        # If top_n is specified, sort and keep only the top n features.
        if self.top_n != -1:
            sorted_items = sorted(
                kl_dict.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            kl_dict = dict(sorted_items[: self.top_n])

            # drop from the DataFrame feature columns not in top_n selection.
            features_to_drop = [f for f in self.features if f not in kl_dict]
            df = df.drop(features_to_drop)

        # Create a new column with the KL divergence results.
        # The dictionary is converted to a struct.
        kl_column = pl.lit(kl_dict)
        df = df.with_columns(kl_column.alias("kl_divergence"))
        return df.lazy()
