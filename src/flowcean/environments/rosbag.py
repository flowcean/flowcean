"""Loading data from rosbag topics."""

from collections.abc import Sequence
from pathlib import Path
from typing import Self, override

import polars as pl
from rosbags.dataframe import get_dataframe
from rosbags.highlevel import AnyReader

from flowcean.core.environment import OfflineEnvironment
from flowcean.core.environment.base import NotLoadedError


class RosbagLoader(OfflineEnvironment):
    """Environment to load data from a rosbag file.

    The RosbagEnvironment is used to load data from a rosbag file. The
    environment is initialized with the path to the rosbag file and a
    dictionary of topics to load.

    Example:
        ```python
        from flowcean.environments.rosbag import RosbagLoader

        environment = RosbagLoader(
            path="example_rosbag",
            topics={
                "/j100_0000/amcl_pose": [
                    "pose.pose.position.x",
                    "pose.pose.position.y",
                ],
                "/j100_0000/odometry": [
                    "pose.pose.position.x",
                    "pose.pose.position.y",
                ],
            },
        )
        environment.load()
        data = environment.get_data()
        print(data)
        ```
    """

    def __init__(
        self,
        path: str | Path,
        topics: dict[str, list[str]],
    ) -> None:
        """Initialize the RosbagEnvironment.

        Args:
            path: Path to the rosbag.
            topics: Dictionary of topics to load (`topic: [keys]`).
        """
        self.path = Path(path)
        self.topics = topics
        self.data = None

    @override
    def load(self) -> Self:
        with AnyReader([self.path]) as reader:
            features = [
                read_timeseries(reader, topic, keys)
                for topic, keys in self.topics.items()
            ]
            self.data = pl.concat(features, how="horizontal")
        return self

    @override
    def get_data(self) -> pl.DataFrame:
        if self.data is None:
            raise NotLoadedError
        return self.data


def read_timeseries(
    reader: AnyReader,
    topic: str,
    keys: Sequence[str],
) -> pl.DataFrame:
    """Read a timeseries from a rosbag topic.

    Args:
        reader: Rosbag reader.
        topic: Topic name.
        keys: Keys to read from the topic.

    Returns:
        Timeseries DataFrame.
    """
    data = pl.from_pandas(
        get_dataframe(reader, topic, keys).reset_index(names="time"),
    )
    nest_into_timeseries = pl.struct(
        [
            pl.col("time"),
            pl.struct(pl.exclude("time")).alias("value"),
        ]
    )
    return data.select(nest_into_timeseries.implode().alias(topic))
