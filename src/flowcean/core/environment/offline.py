from __future__ import annotations

from typing import TYPE_CHECKING, override

import polars as pl

from flowcean.core.environment.incremental import IncrementalEnvironment
from flowcean.core.environment.observable import Observable
from flowcean.core.environment.stepable import Finished

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from flowcean.core.transform import Transform


class OfflineEnvironment(Observable[pl.DataFrame]):
    def join(self, other: OfflineEnvironment) -> OfflineEnvironment:
        return JoinedOfflineEnvironment(self, other)

    def __or__(self, other: OfflineEnvironment) -> OfflineEnvironment:
        return self.join(other)

    def chain(self, *other: OfflineEnvironment) -> ChainedOfflineEnvironments:
        return ChainedOfflineEnvironments([self, *other])

    def __add__(self, other: OfflineEnvironment) -> ChainedOfflineEnvironments:
        return self.chain(other)

    def with_transform(
        self,
        transform: Transform,
    ) -> TransformedOfflineEnvironment:
        return TransformedOfflineEnvironment(self, transform)

    def __rshift__(
        self,
        transform: Transform,
    ) -> TransformedOfflineEnvironment:
        return self.with_transform(transform)


class JoinedOfflineEnvironment(OfflineEnvironment):
    environments: Sequence[OfflineEnvironment]

    def __init__(self, *environments: OfflineEnvironment) -> None:
        self.environments = environments

    @override
    def observe(self) -> pl.DataFrame:
        return pl.concat(
            [environment.observe() for environment in self.environments],
            how="horizontal",
        )


class ChainedOfflineEnvironments(IncrementalEnvironment):
    environments: Iterator[OfflineEnvironment]
    element: OfflineEnvironment

    def __init__(self, environments: Iterable[OfflineEnvironment]) -> None:
        self.environments = iter(environments)
        self.element = next(self.environments)

    @override
    def observe(self) -> pl.DataFrame:
        return self.element.observe()

    @override
    def step(self) -> None:
        try:
            self.element = next(self.environments)
        except StopIteration:
            raise Finished from StopIteration


class TransformedOfflineEnvironment(OfflineEnvironment):
    environment: OfflineEnvironment
    transform: Transform

    def __init__(
        self,
        environment: OfflineEnvironment,
        transform: Transform,
    ) -> None:
        self.environment = environment
        self.transform = transform

    @override
    def observe(self) -> pl.DataFrame:
        data = self.environment.observe()
        return self.transform(data)


#
# def chain(self, other: OfflineEnvironment) -> OfflineEnvironment:
#     """Combine this environment with another one vertically.
#
#     Args:
#         other: The environment to append vertically.
#
#     Returns:
#         The combined environment.
#     """
#     # prevent circular imports
#     from .chain import ChainEnvironment
#
#     return ChainEnvironment(self, other)
#
# def join(self, other: OfflineEnvironment) -> OfflineEnvironment:
#     """Joins this environment with another one horizontally.
#
#     Args:
#         other: The environment to join horizontally.
#
#     Returns:
#         The joined environment.
#     """
#     # prevent circular imports
#     from .joined import JoinedEnvironment
#
#     return JoinedEnvironment(self, other)
#
# def to_time_series(
#     self, time_feature: str | dict[str, str]
# ) -> OfflineEnvironment:
#     """Convert this environment to a time series.
#
#     Args:
#         time_feature: The feature in this environment that represents the
#             time vector. Either a string if all series share a common time
#             vector, or a dictionary where the keys are the value features
#             and the values are the corresponding time vector feature names.
#
#     Returns:
#         A OfflineEnvironment with exactly one sample containing the source
#         environment as a time series.
#     """
#     from flowcean.environments.dataset import Dataset
#
#     # Get the underlying dataframe
#     data = self.get_data()
#     # Create the time feature mapping
#     if isinstance(time_feature, str):
#         time_feature = {
#             feature_name: time_feature
#             for feature_name in data.columns
#             if feature_name != time_feature
#         }
#
#     # Convert the features into a time series
#     return Dataset(
#         data.select(
#             [
#                 pl.struct(
#                     pl.col(t_feature).alias("time"),
#                     pl.col(value_feature).alias("value"),
#                 )
#                 .implode()
#                 .alias(value_feature)
#                 for value_feature, t_feature in time_feature.items()
#             ]
#         )
#     )
