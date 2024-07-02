"""Loading data from rosbag topics."""

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
            self.data = pl.concat(
                [
                    (
                        pl.from_pandas(
                            get_dataframe(reader, topic, keys).reset_index(
                                names="time"
                            ),
                        ).select(pl.struct([pl.all()]).implode().alias(topic))
                    )
                    for topic, keys in self.topics.items()
                ],
                how="horizontal",
            )
        return self

    @override
    def get_data(self) -> pl.DataFrame:
        if self.data is None:
            raise NotLoadedError
        return self.data
