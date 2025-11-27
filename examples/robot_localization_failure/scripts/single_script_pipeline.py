from pathlib import Path

import polars as pl
import torch
from architectures.cnn import CNN
from custom_learners.image_based_lightning_learner import (
    ImageBasedLightningLearner,
)
from custom_transforms.collapse import Collapse
from custom_transforms.detect_delocalizations import DetectDelocalizations
from custom_transforms.localization_status import LocalizationStatus
from feature_images import DiskCaching, FeatureImagesData

import flowcean.cli
from flowcean.core import evaluate_offline
from flowcean.core.transform import Lambda
from flowcean.polars import DataFrame, SliceTimeSeries, ZeroOrderHold
from flowcean.polars.transforms.drop import Drop
from flowcean.ros import load_rosbag
from flowcean.sklearn import (
    Accuracy,
    ClassificationReport,
    FBetaScore,
    PrecisionScore,
    Recall,
)


def explode_and_collect_samples(data: pl.LazyFrame) -> pl.LazyFrame:
    return (
        data.explode("measurements")
        .unnest("measurements")
        .unnest("value")
        .select(
            pl.col("/map"),
            pl.struct(
                [
                    pl.col("/scan/ranges").alias("ranges"),
                    pl.col("/scan/angle_min").alias("angle_min"),
                    pl.col("/scan/angle_max").alias("angle_max"),
                    pl.col("/scan/angle_increment").alias(
                        "angle_increment",
                    ),
                    pl.col("/scan/range_min").alias("range_min"),
                    pl.col("/scan/range_max").alias("range_max"),
                ],
            ).alias("/scan"),
            pl.col("/particle_cloud/particles").alias(
                "/particle_cloud",
            ),
            pl.struct(
                [
                    pl.struct(
                        [
                            pl.col(
                                "/amcl_pose/pose.pose.position.x",
                            ).alias("position.x"),
                            pl.col(
                                "/amcl_pose/pose.pose.position.y",
                            ).alias("position.y"),
                            pl.col(
                                "/amcl_pose/pose.pose.orientation.x",
                            ).alias("orientation.x"),
                            pl.col(
                                "/amcl_pose/pose.pose.orientation.y",
                            ).alias("orientation.y"),
                            pl.col(
                                "/amcl_pose/pose.pose.orientation.z",
                            ).alias("orientation.z"),
                            pl.col(
                                "/amcl_pose/pose.pose.orientation.w",
                            ).alias("orientation.w"),
                        ],
                    ).alias("pose"),
                ],
            ).alias("/amcl_pose"),
            pl.col("is_delocalized"),
        )
    )


def convert_map_to_bool(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(
        pl.col("/map").struct.with_fields(
            pl.field("data").list.eval(pl.element() != 0),
        ),
    )


config = flowcean.cli.initialize()

# print("Config type: ",type(config))
# print("Config: ", config)

# print(config.rosbag.training_paths)           # ['toy_data/rec_20250922_152613_id_01', 'toy_data/rec_20250922_152613_id_02']
# print(type(config.rosbag.training_paths))     # <class 'omegaconf.listconfig.ListConfig'>

# print(config.rosbag.training_paths[0])        # toy_data/rec_20250922_152613_id_01
# print(type(config.rosbag.training_paths[0]))  # <class 'str'>

topics = {
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

message_paths = config.rosbag.message_paths

# print(message_paths)      # ['ros_msgs/sensor_msgs/msg/LaserScan.msg', 'ros_msgs/nav2_msgs/msg/Particle.msg', 'ros_msgs/nav2_msgs/msg/ParticleCloud.msg']
# print(type(message_paths))  # <class 'omegaconf.listconfig.ListConfig'>

##### first bag file ###############

first_bag_path = config.rosbag.training_paths[0]
second_bag_path = config.rosbag.training_paths[1]
third_bag_path = config.rosbag.evaluation_paths[0]
fourth_bag_path = config.rosbag.evaluation_paths[1]

data_bag_1 = load_rosbag(
    first_bag_path,
    topics,
    message_paths=message_paths,
)  # lazy frame
data_bag_2 = load_rosbag(second_bag_path, topics, message_paths=message_paths)
data_bag_3 = load_rosbag(third_bag_path, topics, message_paths=message_paths)
data_bag_4 = load_rosbag(fourth_bag_path, topics, message_paths=message_paths)

# print(data_bag_1.collect()) # This is a lazy df with mentioned columns (raw data)

# print(type(data_bag_1.collect()["/map"][0][0]["value"]["info.width"]))
# print(data_bag_1.collect()["/map"][0][0]["value"]["info.width"])

Collapse_map_transform = Collapse("/map")  # from list[struct[2]] -> struct[11]
collapsed_data_bag_1 = Collapse_map_transform.apply(data_bag_1)
collapsed_data_bag_2 = Collapse_map_transform.apply(data_bag_2)
collapsed_data_bag_3 = Collapse_map_transform.apply(data_bag_3)
collapsed_data_bag_4 = Collapse_map_transform.apply(data_bag_4)

# print(collapsed_data_bag_1.collect())
# print(collapsed_data_bag_1.collect()["/map"][0]["info.width"])

convert_map_transform = Lambda(convert_map_to_bool)
lambda_data_bag_1 = convert_map_transform.apply(collapsed_data_bag_1)
lambda_data_bag_2 = convert_map_transform.apply(collapsed_data_bag_2)
lambda_data_bag_3 = convert_map_transform.apply(collapsed_data_bag_3)
lambda_data_bag_4 = convert_map_transform.apply(collapsed_data_bag_4)

# print(lambda_data_bag_1) # lazy frame
# print(lambda_data_bag_1.collect())    # bool values in map data

zoh_transform = ZeroOrderHold(
    features=["/scan", "/particle_cloud", "/momo/pose", "/amcl_pose"],
    name="measurements",
)
zoh_data_bag_1 = zoh_transform.apply(lambda_data_bag_1)
zoh_data_bag_2 = zoh_transform.apply(lambda_data_bag_2)
zoh_data_bag_3 = zoh_transform.apply(lambda_data_bag_3)
zoh_data_bag_4 = zoh_transform.apply(lambda_data_bag_4)

# print(zoh_data_bag_1.collect())
# print(zoh_data_bag_1.schema)

#########################################
# df = zoh_data_bag_1.collect()
# measurements = df["measurements"][0]  # list[struct[time, value]]
# print(f"Total measurements: {len(measurements)}")

# for i in range(0, 6888):
#     amcl_x = measurements[i]["value"]["/amcl_pose/pose.pose.position.x"]
#     amcl_y = measurements[i]["value"]["/amcl_pose/pose.pose.position.y"]
#     print(f"t={measurements[i]['time']}: AMCL ({amcl_x:.3f}, {amcl_y:.3f})")
#########################################

drop_data_transform = Drop(
    "/scan",
    "/particle_cloud",
    "/momo/pose",
    "/amcl_pose",
)
drop_data_bag_1 = drop_data_transform.apply(zoh_data_bag_1)
drop_data_bag_2 = drop_data_transform.apply(zoh_data_bag_2)
drop_data_bag_3 = drop_data_transform.apply(zoh_data_bag_3)
drop_data_bag_4 = drop_data_transform.apply(zoh_data_bag_4)

# print(drop_data_bag_1.collect())

detect_delocalizations_transform = DetectDelocalizations(
    "/delocalizations",
    name="slice_points",
)
detect_deloc_data_bag_1 = detect_delocalizations_transform.apply(
    drop_data_bag_1,
)
detect_deloc_data_bag_2 = detect_delocalizations_transform.apply(
    drop_data_bag_2,
)
detect_deloc_data_bag_3 = detect_delocalizations_transform.apply(
    drop_data_bag_3,
)
detect_deloc_data_bag_4 = detect_delocalizations_transform.apply(
    drop_data_bag_4,
)

# print(detect_deloc_data_bag_1.collect())

drop_deloc_transform = Drop("/delocalizations")
drop_deloc_bag_1 = drop_deloc_transform.apply(detect_deloc_data_bag_1)
drop_deloc_bag_2 = drop_deloc_transform.apply(detect_deloc_data_bag_2)
drop_deloc_bag_3 = drop_deloc_transform.apply(detect_deloc_data_bag_3)
drop_deloc_bag_4 = drop_deloc_transform.apply(detect_deloc_data_bag_4)

# print(drop_deloc_bag_1.collect())

slice_time_series_transform = SliceTimeSeries(
    time_series="measurements",
    slice_points="slice_points",
)
slice_data_bag_1 = slice_time_series_transform.apply(drop_deloc_bag_1)
slice_data_bag_2 = slice_time_series_transform.apply(drop_deloc_bag_2)
slice_data_bag_3 = slice_time_series_transform.apply(drop_deloc_bag_3)
slice_data_bag_4 = slice_time_series_transform.apply(drop_deloc_bag_4)

# print(slice_data_bag_1.collect())

drop_slice_points_transform = Drop("slice_points")
drop_slice_bag_1 = drop_slice_points_transform.apply(slice_data_bag_1)
drop_slice_bag_2 = drop_slice_points_transform.apply(slice_data_bag_2)
drop_slice_bag_3 = drop_slice_points_transform.apply(slice_data_bag_3)
drop_slice_bag_4 = drop_slice_points_transform.apply(slice_data_bag_4)

# print(drop_slice_bag_1.collect())

localization_status_transform = LocalizationStatus(
    time_series="measurements",
    ground_truth="/momo/pose",
    estimation="/amcl_pose",
    position_threshold=config.localization.position_threshold,
    heading_threshold=config.localization.heading_threshold,
)
loc_status_data_bag_1 = localization_status_transform.apply(drop_slice_bag_1)
loc_status_data_bag_2 = localization_status_transform.apply(drop_slice_bag_2)
loc_status_data_bag_3 = localization_status_transform.apply(drop_slice_bag_3)
loc_status_data_bag_4 = localization_status_transform.apply(drop_slice_bag_4)

# print(loc_status_data_bag_1.collect())
# print(loc_status_data_bag_2.collect())

runs_train_lf = [loc_status_data_bag_1, loc_status_data_bag_2]
concatenated_runs_train_lf = pl.concat(runs_train_lf, how="vertical")
samples_train_lf = explode_and_collect_samples(concatenated_runs_train_lf)

runs_eval_lf = [loc_status_data_bag_3, loc_status_data_bag_4]
concatenated_runs_eval_lf = pl.concat(runs_eval_lf, how="vertical")
samples_eval_lf = explode_and_collect_samples(concatenated_runs_eval_lf)

samples_train = samples_train_lf.collect(engine="streaming")
samples_eval = samples_eval_lf.collect(engine="streaming")

print(samples_train)
print(samples_eval)

## samples_train and samples_eval has all the data pre-processed and time aligned

# print("Schema: ")
# print(samples_train.schema)

################################

# train_counts = samples_train["is_delocalized"].value_counts()
# eval_counts = samples_eval["is_delocalized"].value_counts()

# print("Training set:")
# print(train_counts)

# print("\nEvaluation set:")
# print(eval_counts)

################################

true_counts_train = (
    samples_train["is_delocalized"]
    .value_counts()
    .filter(pl.col("is_delocalized") == True)
    .select("count")
    .item()
)
false_counts_train = (
    samples_train["is_delocalized"]
    .value_counts()
    .filter(pl.col("is_delocalized") == False)
    .select("count")
    .item()
)
ratio = false_counts_train / true_counts_train

print("True counts in training data: ", true_counts_train)
print("False counts in training data: ", false_counts_train)
print("Ratio: ", ratio)


# ####Debugging###

# true_counts_eval = (
#     samples_eval["is_delocalized"]
#     .value_counts()
#     .filter(pl.col("is_delocalized") == True)
#     .select("count")
#     .item()
# )
# false_counts_eval = (
#     samples_eval["is_delocalized"]
#     .value_counts()
#     .filter(pl.col("is_delocalized") == False)
#     .select("count")
#     .item()
# )
# ratio = false_counts_eval / true_counts_eval

# print("True counts in eval data: ", true_counts_eval)
# print("False counts in eval data: ", false_counts_eval)
# print("Ratio: ", ratio)

# ################

### Learning parameters ###

image_size = config.architecture.image_size
in_channels = 3
learning_rate = config.learning.learning_rate
pos_weight = torch.tensor(
    [ratio],
    dtype=torch.float32,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
batch_size = config.learning.batch_size
max_epochs = config.learning.epochs
width_meters = config.architecture.width_meters
preload = config.learning.preload
disk_cache_dir = config.learning.disk_cache_dir

#############################

module = CNN(
    in_channels=in_channels,
    learning_rate=learning_rate,
    pos_weight=None,  # pos_weight
)

# === STEP 1 — Build FeatureImagesData ===

inputs_train = samples_train.drop(["is_delocalized"])
outputs_train = samples_train.select(["is_delocalized"])

dataset = FeatureImagesData(
    inputs=inputs_train,
    outputs=outputs_train,
    image_size=image_size,  # from config
    width_meters=width_meters,  # from config
)

print("STEP 1 OK: Created FeatureImagesData")
print("Dataset length:", len(dataset))

# === STEP 2 — Wrap dataset in DiskCaching ===

cache_dir = Path(disk_cache_dir)  # e.g. "learning_cache"
cache_dir.mkdir(exist_ok=True, parents=True)

cached_dataset = DiskCaching(
    dataset,  # the FeatureImagesData we just created
    cache_dir,  # the cache directory
)

print("STEP 2 OK: DiskCaching initialized")
print("Cache directory:", cache_dir.resolve())
print("Cached dataset length:", len(cached_dataset))

# === STEP 3 — Trigger caching and generate .pt files ===

print("STEP 3: Starting cache warmup… (this will take time)")

cached_dataset.warmup(show_progress=True)

print("STEP 3 OK: Cache warmup complete")

# Show some cached files
cached_files = list(cache_dir.glob("*.pt"))
print(f"Number of cached .pt files: {len(cached_files)}")
if len(cached_files) > 0:
    print("Sample cached file:", cached_files[0])

# === STEP 4 — Train with ImageBasedLightningLearner ===

print("STEP 4: Initializing ImageBasedLightningLearner...")

learner = ImageBasedLightningLearner(
    module=module,  # CNN
    batch_size=batch_size,
    max_epochs=max_epochs,
    image_size=image_size,
    width_meters=width_meters,
    preload=preload,  # True → load cached .pt files
    disk_cache_dir=disk_cache_dir,  # "learning_cache"
)

print("STEP 4: Starting training...")

model = learner.learn(
    inputs=samples_train.drop(["is_delocalized"]),
    outputs=samples_train.select(["is_delocalized"]),
)

print("STEP 4 OK: Training complete.")
print("Trained model:", model)

# === STEP 5 — Evaluate the trained model ===

print("STEP 5: Starting evaluation...")

# 1. Prepare validation dataset
eval_inputs = samples_eval.drop(["is_delocalized"])
eval_outputs = samples_eval.select(["is_delocalized"])

eval_dataset = FeatureImagesData(
    inputs=eval_inputs,
    outputs=eval_outputs,
    image_size=image_size,
    width_meters=width_meters,
)

# 2. Use disk caching for evaluation too (optional but recommended)
eval_cache_dir = "eval_cache"
eval_cache_path = Path(eval_cache_dir)
eval_cache_path.mkdir(exist_ok=True)

eval_cached_dataset = DiskCaching(eval_dataset, eval_cache_path)


print("STEP 5: Precomputing evaluation cache...")
eval_cached_dataset.warmup(show_progress=True)

# 3. Convert to Flowcean DataFrame wrapper for evaluate_offline

eval_df = DataFrame(samples_eval)

metrics = [
    Accuracy(),
    ClassificationReport(),
    FBetaScore(beta=0.5),
    PrecisionScore(),
    Recall(),
]

print("STEP 5: Running evaluation metrics...")

report = evaluate_offline(
    model,
    eval_df,
    inputs=["/map", "/scan", "/particle_cloud", "/amcl_pose"],
    outputs=["is_delocalized"],
    metrics=metrics,
)

print("STEP 5 OK: Evaluation complete.")
print(report)
