import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from custom_transforms.collapse import Collapse
from custom_transforms.detect_delocalizations import DetectDelocalizations
from custom_transforms.localization_status import LocalizationStatus

from flowcean.core.transform import Lambda
from flowcean.polars import SliceTimeSeries, ZeroOrderHold
from flowcean.polars.transforms.drop import Drop
from flowcean.ros import load_rosbag

# -----------------------------------------------------------------------------
# Topics and message definitions (constant)
# -----------------------------------------------------------------------------
TOPICS = {
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
        "ranges",
        "angle_min",
        "angle_max",
        "angle_increment",
        "range_min",
        "range_max",
    ],
    "/map": [
        "data",
        "info.resolution",
        "info.width",
        "info.height",
        "info.origin.position.x",
        "info.origin.position.y",
        "info.origin.position.z",
        "info.origin.orientation.x",
        "info.origin.orientation.y",
        "info.origin.orientation.z",
        "info.origin.orientation.w",
    ],
    "/delocalizations": ["data"],
    "/particle_cloud": ["particles"],
}

MESSAGE_PATHS = [
    "ros_msgs/sensor_msgs/msg/LaserScan.msg",
    "ros_msgs/nav2_msgs/msg/Particle.msg",
    "ros_msgs/nav2_msgs/msg/ParticleCloud.msg",
]


# -----------------------------------------------------------------------------
# Helper transform
# -----------------------------------------------------------------------------
def convert_map_to_bool(df: pl.LazyFrame) -> pl.LazyFrame:
    """Convert /map.data from integers to booleans."""
    return df.with_columns(
        pl.col("/map").struct.with_fields(
            pl.field("data").list.eval(pl.element() != 0),
        ),
    )


# -----------------------------------------------------------------------------
# Process a single rosbag (one bag → one LazyFrame)
# -----------------------------------------------------------------------------
def process_single_bag(bag_path: str) -> pl.LazyFrame:
    print(f"Loading bag: {bag_path}")
    raw = load_rosbag(bag_path, TOPICS, message_paths=MESSAGE_PATHS)

    df = Collapse("/map").apply(raw)
    df = Lambda(convert_map_to_bool).apply(df)

    df = ZeroOrderHold(
        features=["/scan", "/particle_cloud", "/momo/pose", "/amcl_pose"],
        name="measurements",
    ).apply(df)

    df = DetectDelocalizations("/delocalizations", name="slice_points").apply(
        df
    )
    df = Drop("/delocalizations").apply(df)

    df = SliceTimeSeries("measurements", "slice_points").apply(df)
    df = Drop("slice_points").apply(df)

    df = LocalizationStatus(
        time_series="measurements",
        ground_truth="/momo/pose",
        estimation="/amcl_pose",
        position_threshold=0.4,
        heading_threshold=0.4,
    ).apply(df)

    return df


# -----------------------------------------------------------------------------
# Process multiple bags (list of bag paths → one *combined* DataFrame)
# -----------------------------------------------------------------------------
def process_multiple_bags(bag_paths: list[str]) -> pl.DataFrame:
    processed_frames = []

    for path in bag_paths:
        df_lazy = process_single_bag(path)
        processed_frames.append(df_lazy.collect())  # eager df

    combined_df = pl.concat(processed_frames, how="vertical")

    print(f"Combined {len(bag_paths)} bags → {combined_df.height} rows")
    return combined_df


# -----------------------------------------------------------------------------
# Plot histogram of position error (BIN WIDTH INSTEAD OF BIN COUNT)
# -----------------------------------------------------------------------------
def plot_position_error(df: pl.DataFrame, bin_width: float = 0.05):
    """Plot histogram where *bin width* is specified instead of number of bins."""
    flat = df.explode("measurements")

    errors = (
        flat.select(
            pl.col("measurements")
            .struct.field("value")
            .struct.field("position_error")
            .alias("position_error"),
        )
        .drop_nulls()
        .to_series()
        .to_list()
    )

    # Define bin edges using bin width
    bins = np.arange(0, max(errors) + bin_width, bin_width)

    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=bins, edgecolor="black")
    plt.xlabel("Position Error (m)")
    plt.ylabel("Number of Samples")
    plt.title(f"Position Error Histogram (bin width = {bin_width} m)")
    plt.grid(True)
    plt.show()


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    BAG_FILES = [
        "recordings/momo_real_world_data/rosbag2_2025_11_12-16_38_21",
        # Add more bag files here...
    ]

    df = process_multiple_bags(BAG_FILES)

    # You can change bin width here (example: 0.02 meters)
    plot_position_error(df, bin_width=0.05)

    print("Finished!")
