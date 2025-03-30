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

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import torch
from architectures.complex_cnn import ComplexCNN
from architectures.medium_complex_cnn import MediumComplexCNN
from architectures.simple_cnn import SimpleCNN
from custom_transforms.localization_status import LocalizationStatus
from custom_transforms.particle_cloud_image import ParticleCloudImage
from custom_transforms.particle_cloud_statistics import ParticleCloudStatistics

import flowcean.cli
from flowcean.core.strategies.offline import evaluate_offline, learn_offline
from flowcean.polars.environments.dataframe import DataFrame
from flowcean.polars.environments.train_test_split import TrainTestSplit
from flowcean.polars.transforms.explode import Explode
from flowcean.polars.transforms.match_sampling_rate import MatchSamplingRate
from flowcean.polars.transforms.select import Select
from flowcean.polars.transforms.time_window import TimeWindow
from flowcean.ros.rosbag import RosbagLoader
from flowcean.sklearn.adaboost import AdaBoost
from flowcean.sklearn.metrics.classification import Accuracy
from flowcean.torch import LightningLearner

torch.set_float32_matmul_precision("medium")

USE_ROSBAG = False
WS = Path(__file__).resolve().parent
CACHE_FILE = WS / "cached_ros_data.parquet"
ROS_BAG_PATH = WS / "rec_20241021_152106"


def load_or_cache_ros_data(
    *,
    force_refresh: bool = False,
) -> pl.LazyFrame:
    """Load data from ROS bag or cache, with optional refresh.

    Args:
        force_refresh: If True, reload from ROS bag and overwrite cache.

    Returns:
        LazyFrame containing the ROS bag data.
    """
    # Check if cache exists and is valid
    cache_exists = CACHE_FILE.exists()

    if cache_exists and not force_refresh:
        # Load cached data
        print("Loading data from cache.")
        data = pl.read_parquet(CACHE_FILE).lazy()
        # Optional: Validate cache (e.g., check metadata or row count)
        if data.collect().height > 0:
            return data
        print("Cache invalid; reloading from ROS bag.")

    # Load from ROS bag
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
            "/momo/pose": [
                "pose.position.x",
                "pose.position.y",
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

    # Cache the data
    print("Caching data to Parquet.")
    collected_data = data.collect()
    collected_data.write_parquet(CACHE_FILE, compression="snappy")
    print(f"Cache created/updated at {CACHE_FILE}")
    return data


def main() -> None:
    flowcean.cli.initialize_logging()

    # Load data with caching (set force_refresh=True to always reload)
    data = load_or_cache_ros_data(force_refresh=USE_ROSBAG)

    pixel_size = 36
    transform = (
        TimeWindow(
            time_start=1729516868012553090,
            time_end=1729516968012553090,
        )
        | ParticleCloudImage(
            particle_cloud_feature_name="/particle_cloud",
            save_images=False,
            cutting_area=15.0,
            image_pixel_size=pixel_size,
        )
        # timestamps need to be aligned before applying LocalizationStatus
        | MatchSamplingRate(
            reference_feature_name="/heading_error",
            feature_interpolation_map={"/position_error": "linear"},
        )
        | LocalizationStatus(
            position_error_feature_name="/position_error",
            heading_error_feature_name="/heading_error",
        )
        | Select(
            [
                "/particle_cloud",
                "/particle_cloud_image",
                "/position_error",
                "/heading_error",
                "isDelocalized",
            ],
        )
        | MatchSamplingRate(
            reference_feature_name="/particle_cloud_image",
            feature_interpolation_map={
                "/position_error": "linear",
                "/heading_error": "linear",
                "isDelocalized": "nearest",
            },
        )
        | ParticleCloudStatistics()
        | Select(
            [
                "/particle_cloud_image",
                "/position_error",
                "/heading_error",
                "isDelocalized",
                "num_clusters",
                "main_cluster_variance_x",
                "main_cluster_variance_y",
            ],
        )
        | Explode(
            features=[
                "/particle_cloud_image",
                "/position_error",
                "/heading_error",
                "isDelocalized",
                "num_clusters",
                "main_cluster_variance_x",
                "main_cluster_variance_y",
            ],
        )
    )
    transformed_data = transform(data)

    print(f"transformed data: {transformed_data.collect()}")
    # unnest data
    collected_transformed_data = transformed_data.collect()
    # loop over all columns and unnest them
    for column in collected_transformed_data.columns:
        collected_transformed_data = collected_transformed_data.unnest(
            column,
        ).rename({"time": column + "_time", "value": column + "_value"})
    # convert dict to value for isDelocalized_value
    collected_transformed_data = collected_transformed_data.unnest(
        "isDelocalized_value",
    ).rename(
        {"data": "isDelocalized_value"},
    )
    # print(collected_transformed_data)
    # print(collected_transformed_data.shape)
    data_environment = DataFrame(data=collected_transformed_data)
    train, test = TrainTestSplit(ratio=0.8, shuffle=True).split(
        data_environment,
    )

    # Define inputs and outputs
    base_inputs = ["/particle_cloud_image_value"]  # For CNN
    boost_features = [
        "num_clusters_value",
        "main_cluster_variance_x_value",
        "main_cluster_variance_y_value",
    ]
    all_inputs = base_inputs + boost_features
    outputs = ["isDelocalized_value"]

    base_learners = [
        # CNN (simple)
        LightningLearner(
            module=SimpleCNN(image_size=pixel_size),
            max_epochs=2,
            batch_size=4,
            num_workers=15,
        ),
        # CNN (medium)
        LightningLearner(
            module=MediumComplexCNN(image_size=pixel_size),
            max_epochs=2,
            batch_size=4,
            num_workers=15,
        ),
        # CNN (complex)
        LightningLearner(
            module=ComplexCNN(image_size=pixel_size),
            max_epochs=2,
            batch_size=4,
            num_workers=15,
        ),
    ]

    # Wrap base learners with AdaBoost
    learners = [
        AdaBoost(
            base_learner=learner,
            base_input_features=base_inputs,
            boost_features=boost_features,
            n_estimators=50,
            learning_rate=1.0,
        )
        for learner in base_learners
    ]

    for i, learner in enumerate(learners):
        print(f"\nTraining learner {i + 1}")
        t_start = datetime.now(tz=timezone.utc)
        model = learn_offline(
            train,
            learner,
            all_inputs,
            outputs,
        )
        delta_t = datetime.now(tz=timezone.utc) - t_start
        print(f"Learning took {np.round(delta_t.microseconds / 1000, 1)} ms")

        # Compute base predictions for evaluation
        base_inputs_test = test.data.select(base_inputs)
        base_predictions = (
            learner.base_model.predict(base_inputs_test)
            .collect()
            .to_numpy()
            .ravel()
        )

        eval_data = test.data.select(boost_features + outputs).with_columns(
            pl.Series("base_pred", base_predictions),
        )

        print(f"Evaluating learner {i + 1}")
        report = evaluate_offline(
            model,
            DataFrame(eval_data),
            [*boost_features, "base_pred"],
            outputs,
            [Accuracy()],
        )
        print(report)


if __name__ == "__main__":
    main()
