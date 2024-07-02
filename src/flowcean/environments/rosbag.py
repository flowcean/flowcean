from os import listdir
from pathlib import Path
from typing import Self, override

import polars as pl
from rosbags.dataframe import get_dataframe
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

from flowcean.core.environment import OfflineEnvironment

pl.Config.set_tbl_rows(100)
pl.Config.set_fmt_str_lengths(100)
pl.Config.set_fmt_table_cell_list_len(100)


class RosbagEnvironment(OfflineEnvironment):
    """Environment for rosbags."""

    def __init__(
        self,
        path: str | Path,
        topics: dict[str, list[str]],
        custom_msgs_path: str | Path | None = None,
        store: Stores = Stores.ROS2_HUMBLE,
    ) -> None:
        """Initialize the RosbagEnvironment.

        Args:
            path: Path to the rosbag.
            topics: Dictionary of topics to load. The keys are the topic names
                and the values are lists of keys inside each message that
                shall be extracted.
            custom_msgs_path: Path to folder that contains custom ROS message
                descriptions.
            store: Typestore to use for message parsing.
        """
        self.path = Path(path)
        self.topics = topics
        self.data = None
        self.typestore = get_typestore(store)

        if custom_msgs_path is not None:
            add_types = {}
            for pathstr in listdir(custom_msgs_path):
                msgpath = custom_msgs_path / Path(pathstr)
                msgdef = msgpath.read_text(encoding="utf-8")
                add_types.update(
                    get_types_from_msg(msgdef, self.guess_msgtype(msgpath))
                )
            self.typestore.register(add_types)

    @override
    def load(self) -> Self:
        with AnyReader(
            [self.path], default_typestore=self.typestore
        ) as reader:
            self.data = pl.concat(
                [
                    (
                        pl.from_pandas(
                            get_dataframe(reader, topic, keys),
                            include_index=True,
                        )
                        .rename({"None": "time"})
                        .select(pl.struct([pl.all()]).implode().alias(topic))
                    )
                    for topic, keys in self.topics.items()
                ],
                how="horizontal",
            )
        return self

    @override
    def get_data(self) -> pl.DataFrame:
        if self.data is None:
            msg = "data not loaded yet"
            raise ValueError(msg)
        return self.data

    @staticmethod
    def guess_msgtype(path: Path) -> str:
        """Guess message type name from path.

        Example usage:
        path = Path("/home/user/project/src/package/subpackage/file.msg")
        print(guess_msgtype(path))  # Output: package/subpackage/msg/file
        """
        name = path.relative_to(path.parents[2]).with_suffix("")
        if "msg" not in name.parts:
            name = name.parent / "msg" / name.name
        return str(name)
