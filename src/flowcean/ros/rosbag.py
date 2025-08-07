from __future__ import annotations

import logging
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from os import PathLike

    from rosbags.interfaces.typing import Typesdict


class RosbagError(Exception):
    """Dataframe conversion error."""


logger = logging.getLogger(__name__)


def load_rosbag(
    path: str | PathLike,
    topics: dict[str, list[str]],
    *,
    message_paths: Iterable[str | PathLike] | None = None,
    cache: bool = True,
    cache_path: str | PathLike | None = None,
) -> pl.LazyFrame:
    """Load a ROS2 Humble rosbag file and convert it to a Polars LazyFrame.

    The structure of the data is inferred from the message definitions. If
    a message definition is not found in the ROS2 Humble typestore, it is
    added from the provided paths. Once all the message definitions are
    added, the data is loaded from the rosbag file.

    Args:
        path: Path to the rosbag.
        topics: Dictionary of topics to load (`topic: [paths]`).
        message_paths: List of paths to additional message definitions.
        cache: Whether to cache the data to a Parquet file.
        cache_path: Path to the cache file. If None, defaults to the same
            directory as the rosbag file with a .parquet extension.
    """
    path = Path(path)
    cache_path = (
        Path(cache_path)
        if cache_path is not None
        else path.with_suffix(".parquet")
    )

    if cache and cache_path.exists():
        logger.info("Loading data from cache...")
        return pl.scan_parquet(cache_path)

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    if message_paths is not None:
        additional_types = _collect_type_definitions(message_paths)
        typestore.register(additional_types)

    with AnyReader(
        [path],
        default_typestore=typestore,
    ) as reader:
        data = pl.concat(
            _generate_features(reader, topics),
            how="horizontal",
        )

    if cache:
        logger.info("Caching ROS data to Parquet file...")
        data.sink_parquet(Path(cache_path))

    return data


def _generate_features(
    reader: AnyReader,
    topics: dict[str, list[str]],
) -> Iterable[pl.LazyFrame]:
    bar = tqdm(topics.items(), "Loading topics")
    for topic, paths in bar:
        bar.set_description(f"Loading topic {topic!r}")
        yield _get_dataframe(reader, topic, paths)


def _get_dataframe(
    reader: AnyReader,
    topic_name: str,
    paths: Sequence[str],
) -> pl.LazyFrame:
    if topic_name not in reader.topics:
        msg = f"Requested unknown topic {topic_name!r}."
        raise RosbagError(msg)

    topic = reader.topics[topic_name]
    getters = [_create_getter(path.split(".")) for path in paths]

    timestamps = []
    data = []
    for connection, timestamp, rawdata in tqdm(
        reader.messages(
            connections=topic.connections,
        ),
        position=1,
        leave=False,
        total=topic.msgcount,
    ):
        dmsg = reader.deserialize(rawdata, connection.msgtype)
        timestamps.append(timestamp)
        row = [_message_to_dict(x(dmsg)) for x in getters]
        data.append(row)

    df = pl.LazyFrame(data, schema=paths, orient="row")
    time = pl.Series("time", timestamps)
    nest_into_timeseries = pl.struct(
        [
            pl.col("time"),
            pl.struct(pl.exclude("time")).alias("value"),
        ],
    )
    return df.with_columns(time).select(
        nest_into_timeseries.implode().alias(topic_name),
    )


def _collect_type_definitions(
    message_paths: Iterable[str | PathLike],
) -> Typesdict:
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


def _create_getter(keys: list[str]) -> Callable[[object], object]:
    """Create getter for nested lookups."""

    def getter(msg: object) -> object:
        value = msg
        for key in keys:
            value = getattr(value, key)
        return value

    return getter


def _message_to_dict(obj: object) -> object:
    if isinstance(obj, (tuple, list)):
        return type(obj)(_message_to_dict(item) for item in obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: _message_to_dict(getattr(obj, f.name))
            for f in fields(obj)
            if f.name != "__msgtype__"
        }
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
