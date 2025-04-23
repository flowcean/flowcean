#!/usr/bin/env python

import logging
from pathlib import Path

from custom_transforms.collapse import Collapse
from custom_transforms.zero_order_hold_matching import ZeroOrderHoldMatching
from rosbag import load_or_cache_ros_data

import flowcean.cli
from flowcean.polars.transforms.drop import Drop

logger = logging.getLogger(__name__)

WS = Path(__file__).resolve().parent
ROSBAG_NAME = "rec_20241021_152106"
ROSBAG_PATH = WS / ROSBAG_NAME
ROS_MESSAGE_TYPES = [
    WS / "ros_msgs/LaserScan.msg",
    WS / "ros_msgs/nav2_msgs/msg/Particle.msg",
    WS / "ros_msgs/nav2_msgs/msg/ParticleCloud.msg",
]

SAVE_IMAGES = True
IMAGE_PIXEL_SIZE = 100
CROP_REGION_SIZE = 10.0


flowcean.cli.initialize_logging()

dataframes = []
data = load_or_cache_ros_data(
    ROSBAG_PATH,
    message_definitions=ROS_MESSAGE_TYPES,
)
logger.info("Loaded data from ROS bag")


transform = (
    Collapse("/map", element=1)
    | ZeroOrderHoldMatching(
        topics=[
            "/scan",
            "/particle_cloud",
            "/momo/pose",
            "/amcl_pose",
        ],
    )
    | Drop(["/scan", "/particle_cloud", "/momo/pose", "/amcl_pose"])
    | Drop(["/map"])
    # | SliceTimeSeries(
    #     time_series="aligned",
    #     counter_feature="/delocalizations",
    # )
    # | LocalizationStatus(
    #     ground_truth_pose="/momo/pose",
    #     estimated_pose="/amcl_pose",
    #     position_threshold=0.4,
    #     heading_threshold=0.4,
    # )
    # | ShiftTimestamps(
    #     shift=1.0,
    #     feature="isDelocalized",
    # )
    # | Drop(["position_error", "heading_error", "/delocalizations"])
    # | ZeroOrderHoldMatching(
    #     topics=[
    #         "/scan",
    #         "/particle_cloud",
    #         "/momo/pose",
    #         "/amcl_pose",
    #         "isDelocalized",
    #     ],
    # )
    # | MapImage(
    #     occupancy_map=occupancy_map,
    #     crop_region_size=CROP_REGION_SIZE,
    #     image_pixel_size=IMAGE_PIXEL_SIZE,
    #     save_images=SAVE_IMAGES,
    # )
    # | ParticleCloudStatistics()
    # | ScanMapStatistics(occupancy_map=occupancy_map)
    # | ScanImage(
    #     crop_region_size=CROP_REGION_SIZE,
    #     image_pixel_size=IMAGE_PIXEL_SIZE,
    #     save_images=SAVE_IMAGES,
    # )
    # | ParticleCloudImage(
    #     crop_region_size=CROP_REGION_SIZE,
    #     image_pixel_size=IMAGE_PIXEL_SIZE,
    #     save_images=SAVE_IMAGES,
    # )
    # | Drop(
    #     features=[
    #         "/particle_cloud",
    #         "/scan",
    #         "/momo/pose",
    #         "/amcl_pose",
    #         "position_error",
    #         "heading_error",
    #         "scan_points",
    #         "scan_points_sensor",
    #     ],
    # )
)

transformed_data = transform(data)


# transformed_dataframes = []
# for row_df in dataframes:
#     # Apply the transform to each slice
#     transformed_slice = zoh_transform.apply(row_df.lazy())
#     transformed_dataframes.append(transformed_slice)
# # Concatenate all transformed slices
# transformed_data = pl.concat(transformed_dataframes).lazy()
# print("Data after ZeroOrderHoldMatching:")
# print(transformed_data.collect())
# tensor_transform = ImagesToTensor(
#     image_columns=[
#         "/scan_image_value",
#         "/map_image_value",
#         "/particle_cloud_image_value",
#     ],
#     height=IMAGE_PIXEL_SIZE,
#     width=IMAGE_PIXEL_SIZE,
# )
# transformed_data = tensor_transform.apply(transformed_data)
# print("Data after ImagesToTensor:")
# print(transformed_data.collect())
#
# transformed_data = (
#     transformed_data.with_columns(
#         pl.col("isDelocalized_value")
#         .struct[0]
#         .alias("isDelocalized_value_scalar"),
#     )
#     .drop("isDelocalized_value")
#     .rename({"isDelocalized_value_scalar": "isDelocalized_value"})
# )
# collected_transformed_data = transformed_data.collect()
# data_environment = DataFrame(data=collected_transformed_data)
# train, test = TrainTestSplit(ratio=0.8, shuffle=True).split(
#     data_environment,
# )
# cnn_inputs = [
#     "image_tensor",
# ]
# outputs = ["isDelocalized_value"]
# # adaboost inputs are all features except outputs and tensors
# adaboost_inputs = collected_transformed_data.columns
# adaboost_inputs.remove("isDelocalized_value")
# adaboost_inputs.remove("/scan_image_value")
# adaboost_inputs.remove("/map_image_value")
# adaboost_inputs.remove("image_tensor")
# adaboost_inputs.remove("time")
# adaboost_inputs.remove("scan_points_sensor_value")
# adaboost_inputs.remove("slice_id")
# print(f"adaboost inputs: {adaboost_inputs}")
# print(f"cnn inputs: {cnn_inputs}")
# print(f"outputs: {outputs}")
# cnn_learners = [
#     LightningLearner(
#         module=CNN(
#             image_size=IMAGE_PIXEL_SIZE,
#             learning_rate=1e-3,
#             in_channels=2,
#         ),
#         batch_size=4,
#         max_epochs=5,
#     ),
# ]
# adaboost_learners = [
#     AdaBoost(),
# ]
# for learner in cnn_learners:
#     t_start = datetime.now(tz=timezone.utc)
#     model = learn_offline(
#         environment=train,
#         learner=learner,
#         inputs=cnn_inputs,
#         outputs=outputs,
#     )
#     delta_t = datetime.now(tz=timezone.utc) - t_start
#     print(f"Learning took {np.round(delta_t.microseconds / 1000, 1)} ms")
#
#     report = evaluate_offline(
#         model,
#         test,
#         cnn_inputs,
#         outputs,
#         [Accuracy()],
#     )
#     print(report)
#
# for learner in adaboost_learners:
#     t_start = datetime.now(tz=timezone.utc)
#     model = learn_offline(
#         train,
#         learner,
#         adaboost_inputs,
#         outputs,
#     )
#     delta_t = datetime.now(tz=timezone.utc) - t_start
#     print(f"Learning took {np.round(delta_t.microseconds / 1000, 1)} ms")
#
#     report = evaluate_offline(
#         model,
#         test,
#         adaboost_inputs,
#         outputs,
#         [Accuracy()],
#     )
#     print(report)
