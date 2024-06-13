from os import listdir
from pathlib import Path
from typing import Self, override

import polars as pl
from rosbags.dataframe import get_dataframe
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

from flowcean.core.environment import OfflineEnvironment


def guess_msgtype(path: Path) -> str:
    """Guess message type name from path."""
    name = path.relative_to(path.parents[2]).with_suffix("")
    if "msg" not in name.parts:
        name = name.parent / "msg" / name.name
    return str(name)


class RosbagEnvironment(OfflineEnvironment):
    """Environment for rosbags."""

    def __init__(
        self,
        path: str | Path,
        topics: dict[str, list[str]],
        custom_msgs_path: str | Path | None = None,
    ) -> None:
        """Initialize the RosbagEnvironment.

        Args:
            path: Path to the rosbag.
            topics: Dictionary of topics to load. The keys are the topic names
                and the values are lists of keys inside each message that
                shall be extracted.
            custom_msgs_path: Path to folder that contains custom ros message
                descriptions.
        """
        self.path = Path(path)
        self.topics = topics
        self.data = pl.DataFrame()
        self.typestore = get_typestore(Stores.ROS2_HUMBLE)

        if custom_msgs_path:
            add_types = {}
            for pathstr in listdir(custom_msgs_path):
                msgpath = custom_msgs_path / Path(pathstr)
                msgdef = msgpath.read_text(encoding="utf-8")
                add_types.update(
                    get_types_from_msg(msgdef, guess_msgtype(msgpath))
                )
            self.typestore.register(add_types)

    @override
    def load(self) -> Self:
        with AnyReader(
            [self.path], default_typestore=self.typestore
        ) as reader:
            joined_df = pl.DataFrame()
            for topic, keys in self.topics.items():
                df = pl.from_pandas(
                    get_dataframe(reader, topic, keys),
                    include_index=True,
                )
                df = df.rename({"None": "time"})
                df = df.rename(
                    lambda column_name, topic=topic: str(
                        f"{topic[1:]}." + column_name
                    )
                )
                df = pl.DataFrame(
                    {col: [df[col].to_list()] for col in df.columns}
                )
                joined_df = pl.concat(
                    [joined_df, df],
                    how="horizontal",
                )
            self.data = joined_df
        return self

    @override
    def get_data(self) -> pl.DataFrame:
        return self.data
