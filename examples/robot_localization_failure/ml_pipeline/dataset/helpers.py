import numpy as np
import polars as pl

def get_topics():
    return {
        "/amcl_pose": [
            "pose.pose.position.x",
            "pose.pose.position.y",
            "pose.pose.orientation.x",
            "pose.pose.orientation.y",
            "pose.pose.orientation.z",
            "pose.pose.orientation.w",
        ],
        "/momo/pose": [
            "pose.position.x",
            "pose.position.y",
            "pose.orientation.x",
            "pose.orientation.y",
            "pose.orientation.z",
            "pose.orientation.w",
        ],
        "/scan": [
            "ranges", "angle_min", "angle_max",
            "angle_increment", "range_min", "range_max"
        ],
        "/map": [
            "data", "info.resolution", "info.width", "info.height",
            "info.origin.position.x", "info.origin.position.y",
            "info.origin.position.z",
            "info.origin.orientation.x",
            "info.origin.orientation.y",
            "info.origin.orientation.z",
            "info.origin.orientation.w",
        ],
        "/delocalizations": ["data"],
        "/particle_cloud": ["particles"],
    }


def timeseries_to_df(row_df: pl.DataFrame, column: str, value_name: str):
    """
    Convert a nested time series stored in a single DataFrame cell into
    a flat two-column DataFrame.

    Example
    -------
    Input (row_df[column][0]):
        [
            {"time": 0.0, "value": 10.0},
            {"time": 1.0, "value": 10.5},
            {"time": 2.0, "value": 11.0},
        ]

    Output:
        shape: (3, 2)
        ┌──────┬─────────┐
        │ time │ reading │   <-- value_name="reading"
        ├──────┼─────────┤
        │ 0.0  │ 10.0    │
        │ 1.0  │ 10.5    │
        │ 2.0  │ 11.0    │
        └──────┴─────────┘

    `value_name` becomes the name of the second column.
    """

    ts = row_df[column][0]
    if len(ts) == 0:
        return pl.DataFrame({"time": [], value_name: []})
    return pl.DataFrame({
        "time": [e["time"] for e in ts],
        value_name: [e["value"] for e in ts],
    })


def yaw_from_quat(qx, qy, qz, qw):
    return float(np.arctan2(
        2.0 * (qw*qz + qx*qy),
        qw*qw + qx*qx - qy*qy - qz*qz
    ))
