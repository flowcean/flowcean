#!/usr/bin/env python
# /// script
# dependencies = [
#     "flowcean",
#     "matplotlib",
#     "opencv-python",
# ]
#
# [tool.uv.sources]
# flowcean = { path = "../../", editable = true }
# ///

import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
from custom_transforms.localization_status import LocalizationStatus
from custom_transforms.zero_order_hold_matching import ZeroOrderHoldMatching

import flowcean.cli
from flowcean.core.strategies.offline import evaluate_offline, learn_offline
from flowcean.polars.environments.dataframe import DataFrame
from flowcean.polars.environments.train_test_split import TrainTestSplit
from flowcean.polars.transforms.drop import Drop
from flowcean.polars.transforms.match_sampling_rate import MatchSamplingRate
from flowcean.ros.rosbag import RosbagLoader
from flowcean.sklearn.adaboost_classifier import AdaBoost
from flowcean.sklearn.metrics.classification import Accuracy

USE_ROSBAG = False
WS = Path(__file__).resolve().parent
CACHE_FILE = WS / "cached_ros_data.parquet"
ROS_BAG_PATH = WS / "rec_20241021_152106"


def load_or_cache_ros_data(*, force_refresh: bool = False) -> pl.LazyFrame:
    cache_exists = CACHE_FILE.exists()
    if cache_exists and not force_refresh:
        print("Loading data from cache.")
        data = pl.read_parquet(CACHE_FILE).lazy()
        if data.collect().height > 0:
            return data
        print("Cache invalid; reloading from ROS bag.")
    print("Loading data from ROS bag.")
    environment = RosbagLoader(
        path=ROS_BAG_PATH,
        topics={
            "/amcl_pose": [
                "pose.pose.position.x",
                "pose.pose.position.y",
                "pose.pose.orientation.x",
                "pose.pose.orientation.y",
                "pose.pose.orientation.z",
                "pose.pose.orientation.w",
            ],
            "/momo/pose": ["pose.position.x", "pose.position.y"],
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
            "/position_error": ["data"],
            "/heading_error": ["data"],
        },
        msgpaths=[
            str(WS / "ros_msgs/LaserScan.msg"),
            str(WS / "ros_msgs/nav2_msgs/msg/Particle.msg"),
            str(WS / "ros_msgs/nav2_msgs/msg/ParticleCloud.msg"),
        ],
    )
    data = environment.observe()
    print("Caching data to Parquet.")
    collected_data = data.collect()
    collected_data.write_parquet(CACHE_FILE, compression="snappy")
    print(f"Cache created/updated at {CACHE_FILE}")
    return data


def get_reduced_timestamps(
    data: pl.LazyFrame,
    columns: list[str],
    start_after_all: bool = True,
) -> pl.LazyFrame:
    """Precompute a reduced set of timestamps starting after all columns have data."""
    # Explode and unnest all columns to get their timestamps
    exploded_dfs = [
        data.select(col)
        .lazy()
        .explode(col)
        .unnest(col)
        .with_columns(pl.col("time").cast(pl.Int64))
        for col in columns
    ]

    # Get first timestamp for each column
    first_times = [
        df.select(pl.col("time").min().alias("first_time")).collect()[
            0,
            "first_time",
        ]
        for df in exploded_dfs
    ]

    # Get all unique timestamps
    all_times = (
        pl.concat([df.select("time") for df in exploded_dfs], how="vertical")
        .unique()
        .sort("time")
    )

    # Start after all topics have published
    if start_after_all:
        start_time = max(first_times) if first_times else 0
        all_times = all_times.filter(pl.col("time") >= start_time)

    return all_times


def extract_map_data(data: pl.LazyFrame) -> dict:
    """Extract the first non-null /map value as a dictionary."""
    map_df = (
        data.select("/map")
        .lazy()
        .explode("/map")
        .unnest("/map")
        .drop_nulls("value")
        .select("value")
        .collect()
    )
    if map_df.height > 0:
        return map_df[0, "value"]
    return {}


def main() -> None:
    flowcean.cli.initialize_logging()
    data = load_or_cache_ros_data(force_refresh=USE_ROSBAG)

    initial_transform = (
        MatchSamplingRate(
            reference_feature_name="/heading_error",
            feature_interpolation_map={"/position_error": "linear"},
        )
        | LocalizationStatus(
            position_error_feature_name="/position_error",
            heading_error_feature_name="/heading_error",
            position_threshold=1.2,
            heading_threshold=1.2,
        )
        | Drop(
            features=["/heading_error", "/position_error", "/momo/pose"],
        )
    )
    intermediate_data = initial_transform(data)
    print("Intermediate data schema:", intermediate_data.collect_schema())

    # Extract map data and remove /map from columns to process
    map_data = extract_map_data(intermediate_data)
    print(f"Extracted map data: {map_data}")
    columns_to_batch = [
        col
        for col in intermediate_data.collect_schema().names()
        if col != "/map"
    ]
    batch_size = 1
    batches = [
        columns_to_batch[i : i + batch_size]
        for i in range(0, len(columns_to_batch), batch_size)
    ]

    # Precompute reduced timestamps without /map
    all_times = get_reduced_timestamps(
        intermediate_data,
        columns_to_batch,
        start_after_all=True,  # Start after all remaining topics have data
    )
    print(f"Reduced all_times length: {all_times.collect().height}")

    temp_dir = WS / "temp_batch_results"
    temp_dir.mkdir(exist_ok=True)

    # Process and save each batch with reduced all_times
    for i, batch in enumerate(batches):
        batch_transform = ZeroOrderHoldMatching(
            columns=batch,
            all_times=all_times,
        )
        batch_result = batch_transform.apply(intermediate_data.select(batch))
        batch_path = temp_dir / f"batch_{i}.parquet"
        batch_result.sink_parquet(batch_path, compression="snappy")
        print(f"Batch {batch} saved to {batch_path}")

    # Combine batches incrementally
    batch_files = sorted(temp_dir.glob("batch_*.parquet"))
    transformed_data = pl.scan_parquet(batch_files[0])
    for batch_file in batch_files[1:]:
        next_batch = pl.scan_parquet(batch_file)
        added_columns = [
            c for c in next_batch.collect_schema().names() if c != "time"
        ]
        existing_columns = [
            c for c in transformed_data.collect_schema().names() if c != "time"
        ]
        transformed_data = transformed_data.join(
            next_batch,
            on="time",
            how="full",
        ).select(
            pl.col("time"),
            *[pl.col(c) for c in existing_columns],
            *[pl.col(c) for c in added_columns],
        )

    output_path = WS / "transformed_data.parquet"
    transformed_data.sink_parquet(output_path, compression="snappy")

    sample_data = pl.read_parquet(output_path)
    print(f"Sample data: {sample_data}")
    sample_data = sample_data.drop_nulls()
    print(f"Sample data after dropping nulls: {sample_data}")

    # clean up temporary files
    shutil.rmtree(temp_dir)
    return
    # loop over all columns and unnest them
    for column in collected_transformed_data.columns:
        collected_transformed_data = collected_transformed_data.unnest(
            column,
        ).rename({"time": column + "_time", "value": column + "_value"})
        # drop time columns
        collected_transformed_data = collected_transformed_data.drop(
            column + "_time",
        )
    # convert dict to value for isDelocalized_value
    collected_transformed_data = collected_transformed_data.unnest(
        "isDelocalized_value",
    ).rename(
        {"data": "isDelocalized_value"},
    )
    print(f"collected transformed data: {collected_transformed_data}")
    data_environment = DataFrame(data=collected_transformed_data)
    train, test = TrainTestSplit(ratio=0.8, shuffle=True).split(
        data_environment,
    )
    # inputs are all features except "isDelocalized_value"
    inputs = collected_transformed_data.columns
    print(f"inputs: {inputs}")
    inputs.remove("isDelocalized_value")
    outputs = ["isDelocalized_value"]
    learners = [
        AdaBoost(),
    ]
    for learner in learners:
        t_start = datetime.now(tz=timezone.utc)
        model = learn_offline(
            train,
            learner,
            inputs,
            outputs,
        )
        delta_t = datetime.now(tz=timezone.utc) - t_start
        print(f"Learning took {np.round(delta_t.microseconds / 1000, 1)} ms")

        report = evaluate_offline(
            model,
            test,
            inputs,
            outputs,
            [Accuracy()],
        )
        print(report)


if __name__ == "__main__":
    main()
