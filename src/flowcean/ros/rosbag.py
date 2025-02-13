from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from rosbags.highlevel import AnyReader as AnyRosbagReader
from rosbags.interfaces import Msgdef, Nodetype
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

from flowcean.polars import DataFrame

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from rosbags.interfaces.typing import FieldDesc

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
        from flowcean.ros import RosbagLoader

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
        path: str | Path,
        topics: dict[str, list[str]],
        msgpaths: list[str],
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
            msgpaths: List of paths to additional message definitions.
        """
        if msgpaths is None:
            msgpaths = []
        self.path = Path(path)
        self.topics = topics
        self.typestore = get_typestore(Stores.ROS2_HUMBLE)
        add_types = {}
        for pathstr in msgpaths:
            msgpath = Path(pathstr)
            msgdef = msgpath.read_text(encoding="utf-8")
            add_types.update(
                get_types_from_msg(msgdef, self.guess_msgtype(msgpath)),
            )
            debug_msg = f"Added message type: {self.guess_msgtype(msgpath)}"
            logger.debug(debug_msg)
        self.typestore.register(add_types)

        with AnyRosbagReader(
            [self.path],
            default_typestore=self.typestore,
        ) as reader:
            features = [
                self.get_dataframe(reader, topic, keys)
                for topic, keys in self.topics.items()
            ]
            super().__init__(pl.concat(features, how="horizontal"))

    def guess_msgtype(self, path: Path) -> str:
        """Guess message type name from path.

        Args:
            path: Path to the message file.

        Returns:
            The message definition string.
        """
        name = path.relative_to(path.parents[2]).with_suffix("")
        if "msg" not in name.parts:
            name = name.parent / "msg" / name.name
        return str(name)

    def get_dataframe(
        self,
        reader: AnyRosbagReader,
        topicname: str,
        keys: Sequence[str],
    ) -> pl.DataFrame:
        """Convert messages from a topic into a polars dataframe.

        Read all messages from a topic and extract referenced keys into
        a polars dataframe. The timestamps of messages are automatically added
        as the dataframe index.

        Keys support a dotted syntax to traverse nested messages. Here is an
        example of a nested ROS message structure:
        /amcl_pose (geometry_msgs/PoseWithCovarianceStamped)
        ├── pose (PoseWithCovariance)
        │   ├── pose (Pose)
        │   │   ├── position (Point)
        │   │   │   ├── x (float)
        │   │   │   ├── y (float)
        │   │   │   └── z (float)
        │   │   └── orientation (Quaternion)
        │   └── covariance (array[36])

        The first key is 'pose.pose.position.x'. The subkeys are
        separated by dots. So, in this case, the subkeys are
        ['pose', 'pose', 'position', 'x']. Each subkey is used to
        traverse the nested message structure. If a subkey matches
        a field name, the next subkey is used to traverse deeper
        into the nested structure.

        Args:
            reader: Opened rosbags reader.
            topicname: Topic name of messages to process.
            keys: Field names to get from each message.

        Raises:
            DataframeError: Reader not opened or topic or field does not exist.

        Returns:
            Polars dataframe.

        """
        self.verify_topics(reader, topicname)
        topic = reader.topics[topicname]
        msgdef = reader.typestore.get_msgdef(str(topic.msgtype))

        getters = []
        # Iterate through each key provided by the user
        # e.g., "pose.pose.position.x")
        for key in keys:
            # Split the key into subkeys at dots
            # (e.g., ["pose", "pose", "position", "x"])
            subkeys = key.split(".")

            # Start with the top-level message definition
            subdef = msgdef

            # Process all subkeys except the last one
            # (e.g., ["pose", "pose", "position"])
            for subkey in subkeys[:-1]:
                # Find the field in the current message definition that matches
                # the subkey. x[0] is the field name, returns None if not found
                subfield = next(
                    (x for x in subdef.fields if x[0] == subkey),
                    None,
                )

                # Get the message definition for this subfield to continue
                # traversing e.g., get definition of a 'pose' message
                subdef = self.get_subdef(reader, subdef, subkey, subfield)

            # Verify the final subkey exists in the last message definition
            # e.g. check that 'x' exists in the 'position' message
            if subkeys[-1] not in {x[0] for x in subdef.fields}:
                msg = (
                    f"Field {subkeys[-1]!r} does not exist on {subdef.name!r}."
                )
                raise RosbagError(msg)
            # Create a getter function to extract the value from the message
            getters.append(self.create_getter(subkeys))

        timestamps = []
        data = []
        for _, timestamp, rawdata in reader.messages(
            connections=topic.connections,
        ):
            dmsg = reader.deserialize(rawdata, str(topic.msgtype))
            timestamps.append(timestamp)
            row = []

            for x in getters:
                value = x(dmsg)
                if isinstance(value, list):
                    # Convert list items to dicts but keep them in the row
                    row.append([self.ros_msg_to_dict(i) for i in value])
                elif hasattr(value, "__dict__"):
                    row.append(self.ros_msg_to_dict(value))
                else:
                    row.append(value)
            data.append(row)

        # Handle any numpy arrays
        data = [
            [x.tolist() if isinstance(x, np.ndarray) else x for x in row]
            for row in data
        ]
        df = pl.DataFrame(data, schema=tuple(keys), orient="row")
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

    def get_subdef(
        self,
        reader: AnyRosbagReader,
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

    def verify_topics(
        self,
        reader: AnyRosbagReader,
        topicname: str,
    ) -> None:
        if not reader.isopen:
            msg = "RosbagReader needs to be opened before accessing messages."
            raise RosbagError(msg)

        if topicname not in reader.topics:
            msg = f"Requested unknown topic {topicname!r}."
            raise RosbagError(msg)

    def create_getter(self, keys: list[str]) -> Callable[[object], AttrValue]:
        """Create getter for nested lookups."""

        def getter(msg: object) -> AttrValue:
            value = msg
            for key in keys:
                value = getattr(value, key)
            return value

        return getter

    def ros_msg_to_dict(self, obj: dict) -> dict:
        """Recursively convert a ROS message object into a dictionary.

        Args:
            obj (dict): A ROS message object represented as a dictionary where
                keys are field names and values are their corresponding data.

        Returns:
            dict: A dictionary representation of the ROS message, with all
                nested fields converted to dictionaries.
        """
        if hasattr(obj, "__dict__"):  # Check if the object has attributes
            result = {}
            for key, value in obj.__dict__.items():
                result[key] = self.ros_msg_to_dict(value)
            return result
        return obj  # Return the base value if it's not an object
