"""Conversion of rosbag topics to data frames."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from rosbags.interfaces import Nodetype

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Union

    from rosbags.highlevel import AnyReader

    AttrValue = Union[str, bool, int, float, object]  # noqa: UP007


class DataframeError(Exception):
    """Dataframe conversion error."""


def get_dataframe(  # noqa: C901, PLR0912, PLR0915
    reader: AnyReader, topicname: str, keys: Sequence[str]
) -> pl.DataFrame:
    """Convert messages from a topic into a polars dataframe.

    Read all messages from a topic and extract referenced keys into
    a polars dataframe. The message timestamps are automatically added
    as the dataframe index.

    Keys support a dotted syntax to traverse nested messages.

    Args:
        reader: Opened rosbags reader.
        topicname: Topic name of messages to process.
        keys: Field names to get from each message.

    Raises:
        DataframeError: Reader not opened or topic or field does not exist.

    Returns:
        Polars dataframe.

    """
    # pylint: disable=too-many-locals
    if not reader.isopen:
        msg = "RosbagReader needs to be opened before accessing messages."
        raise DataframeError(msg)

    if topicname not in reader.topics:
        msg = f"Requested unknown topic {topicname!r}."
        raise DataframeError(msg)

    topic = reader.topics[topicname]
    assert topic.msgtype  # noqa: S101

    msgdef = reader.typestore.get_msgdef(topic.msgtype)

    def create_plain_getter(key: str) -> Callable[[object], AttrValue]:
        """Create getter for plain attribute lookups."""

        def getter(msg: object) -> AttrValue:
            return getattr(msg, key)  # type: ignore[no-any-return]

        return getter

    def create_nested_getter(keys: list[str]) -> Callable[[object], AttrValue]:
        """Create getter for nested lookups."""

        def getter(msg: object) -> AttrValue:
            value = msg
            for key in keys:
                value = getattr(value, key)
            return value

        return getter

    getters = []
    for key in keys:
        subkeys = key.split(".")
        subdef = msgdef
        for subkey in subkeys[:-1]:
            subfield = next((x for x in subdef.fields if x[0] == subkey), None)
            if not subfield:
                msg = f"Field {subkey!r} does not exist on {subdef.name!r}."
                raise DataframeError(msg)
            if subfield[1][0] != Nodetype.NAME:
                msg = f"Field {subkey!r} of {subdef.name!r} is not a message."
                raise DataframeError(msg)

            subdef = reader.typestore.get_msgdef(subfield[1][1])

        if subkeys[-1] not in {x[0] for x in subdef.fields}:
            msg = f"Field {subkeys[-1]!r} does not exist on {subdef.name!r}."
            raise DataframeError(msg)

        if len(subkeys) == 1:
            getters.append(create_plain_getter(subkeys[0]))
        else:
            getters.append(create_nested_getter(subkeys))

    timestamps = []
    data = []
    is_nested = False
    for _, timestamp, rawdata in reader.messages(
        connections=topic.connections
    ):
        dmsg = reader.deserialize(rawdata, topic.msgtype)
        timestamps.append(timestamp)
        row = []
        for x in getters:
            if isinstance(x(dmsg), list):
                is_nested = True
                list_data = []
                list_data.extend([ros_msg_to_dict(i) for i in x(dmsg)])
                data.append(list_data)
            else:
                row.append(x(dmsg))
        if not is_nested:
            data.append(row)

    if is_nested:
        df = pl.DataFrame([data], schema=tuple(keys))
    else:
        # convert numpy arrays to lists
        data = [
            [x.tolist() if hasattr(x, "tolist") else x for x in row]
            for row in data
        ]
        df = pl.DataFrame(data, schema=tuple(keys))
    time = pl.Series("time", timestamps)
    return df.with_columns(time)


def ros_msg_to_dict(obj: dict) -> dict:
    """Recursively convert ROS messages to a dictionary."""
    if hasattr(obj, "__dict__"):  # Check if the object has attributes
        result = {}
        for key, value in obj.__dict__.items():
            if isinstance(value, (float | int | str | bool)):  # Base types
                result[key] = value
            else:  # Nested object
                result[key] = ros_msg_to_dict(value)
        return result
    return obj  # Return the base value if it's not an object
