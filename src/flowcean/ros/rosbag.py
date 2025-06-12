from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from rosbags.highlevel import AnyReader
from rosbags.interfaces import Msgdef, Nodetype
from rosbags.typesys import Stores, get_types_from_msg, get_typestore
from tqdm import tqdm

from flowcean.polars import DataFrame

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from os import PathLike

    from rosbags.interfaces.typing import FieldDesc, Typesdict

    AttrValue = str | bool | int | float | object


class RosbagError(Exception):
    """Dataframe conversion error."""


logger = logging.getLogger(__name__)


class RosbagLoader(DataFrame):
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
                "/amcl_pose": [
                    "pose.pose.position.x",
                    "pose.pose.position.y",
                ],
                "/odometry": [
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
        path: PathLike,
        topics: dict[str, list[str]],
        message_paths: Iterable[PathLike] | None = None,
    ) -> None:
        """Initialize the RosbagEnvironment.

        The structure of the data is inferred from the message definitions.
        If a message definition is not found in the ROS2 Humble typestore,
        it is added from the provided paths. Once all
        the message definitions are added, the data is loaded from the
        rosbag file.

        Args:
            path: Path to the rosbag.
            topics: Dictionary of topics to load (`topic: [keys]`).
            message_paths: List of paths to additional message definitions.
        """
        typestore = get_typestore(Stores.ROS2_HUMBLE)
        if message_paths is not None:
            additional_types = _collect_type_definitions(message_paths)
            typestore.register(additional_types)

        with AnyReader(
            [Path(path)],
            default_typestore=typestore,
        ) as reader:
            features = [
                _get_dataframe(reader, topic, keys)
                for topic, keys in tqdm(topics.items(), "Loading topics")
            ]

        super().__init__(pl.concat(features, how="horizontal"))


def _get_dataframe(
    reader: AnyReader,
    topicname: str,
    keys: Sequence[str],
) -> pl.DataFrame:
    if topicname not in reader.topics:
        msg = f"Requested unknown topic {topicname!r}."
        raise RosbagError(msg)
    topic = reader.topics[topicname]
    msgdef = reader.typestore.get_msgdef(str(topic.msgtype))

    getters = []
    for key in keys:
        subkeys = key.split(".")

        subdef = msgdef
        for subkey in subkeys[:-1]:
            subfield = next(
                (x for x in subdef.fields if x[0] == subkey),
                None,
            )
            subdef = _get_subdef(reader, subdef, subkey, subfield)

        if subkeys[-1] not in {x[0] for x in subdef.fields}:
            msg = f"Field {subkeys[-1]!r} does not exist on {subdef.name!r}."
            raise RosbagError(msg)
        getters.append(_create_getter(subkeys))

    timestamps = []
    data = []
    for connection, timestamp, rawdata in reader.messages(
        connections=topic.connections,
    ):
        dmsg = reader.deserialize(rawdata, connection.msgtype)
        timestamps.append(timestamp)
        row = []

        for x in getters:
            value = x(dmsg)
            if isinstance(value, list):
                row.append([_ros_msg_to_dict(i) for i in value])
            elif hasattr(value, "__dict__"):
                row.append(_ros_msg_to_dict(value))
            else:
                row.append(value)
        data.append(row)

    # data = [
    #     [x.tolist() if isinstance(x, np.ndarray) else x for x in row]
    #     for row in data
    # ]
    df = pl.DataFrame(data, schema=keys, orient="row")
    breakpoint()
    time = pl.Series("time", timestamps)
    nest_into_timeseries = pl.struct(
        [
            pl.col("time"),
            pl.struct(pl.exclude("time")).alias("value"),
        ],
    )
    return df.with_columns(time).select(
        nest_into_timeseries.implode().alias(topicname),
    )


def _get_subdef(
    reader: AnyReader,
    subdef: Msgdef[object],
    subkey: str,
    subfield: tuple[str, FieldDesc] | None,
) -> Msgdef[object]:
    if not subfield:
        msg = f"Field {subkey!r} does not exist on {subdef.name!r}."
        raise RosbagError(msg)
    if subfield[1][0] != Nodetype.NAME:
        msg = f"Field {subkey!r} of {subdef.name!r} is not a message."
        raise RosbagError(msg)
    return reader.typestore.get_msgdef(subfield[1][1])


def _create_getter(keys: list[str]) -> Callable[[object], AttrValue]:
    """Create getter for nested lookups."""

    def getter(msg: object) -> AttrValue:
        value = msg
        for key in keys:
            value = getattr(value, key)
        return value

    return getter


def _ros_msg_to_dict(obj: dict) -> dict:
    """Recursively convert a ROS message object into a dictionary.

    Args:
        obj: A ROS message object represented as a dictionary where keys
            are field names and values are their corresponding data.

    Returns:
        A dictionary representation of the ROS message, with all nested
        fields converted to dictionaries.
    """
    if not hasattr(obj, "__dict__"):
        return obj

    result = {}
    for key, value in obj.__dict__.items():
        if key == "__msgtype__":
            continue
        result[key] = _ros_msg_to_dict(value)
    return result


def _collect_type_definitions(message_paths: Iterable[PathLike]) -> Typesdict:
    return {
        k: v
        for path in message_paths
        for k, v in get_types_from_msg(
            Path(path).read_text(),
            _guess_msgtype(Path(path)),
        ).items()
    }


def _guess_msgtype(path: Path) -> str:
    name = path.relative_to(path.parents[2]).with_suffix("")
    if "msg" not in name.parts:
        name = name.parent / "msg" / name.name
    return str(name)
