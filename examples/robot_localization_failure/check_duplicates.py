import polars as pl
from custom_transforms.collapse import Collapse
from custom_transforms.detect_delocalizations import DetectDelocalizations
from custom_transforms.localization_status import LocalizationStatus
from custom_transforms.slice_time_series import SliceTimeSeries
from custom_transforms.zero_order_hold_matching import ZeroOrderHold

import flowcean.cli
from flowcean.core.transform import Lambda
from flowcean.polars.transforms.drop import Drop
from flowcean.ros import load_rosbag


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

# message_paths = config.rosbag.message_paths

# print(message_paths)      # ['ros_msgs/sensor_msgs/msg/LaserScan.msg', 'ros_msgs/nav2_msgs/msg/Particle.msg', 'ros_msgs/nav2_msgs/msg/ParticleCloud.msg']
# print(type(message_paths))  # <class 'omegaconf.listconfig.ListConfig'>

##### first bag file ###############

first_bag_path = config.rosbag.training_paths[0]
second_bag_path = config.rosbag.training_paths[1]
third_bag_path = config.rosbag.evaluation_paths[0]
fourth_bag_path = config.rosbag.evaluation_paths[1]

data_bag_1 = load_rosbag(first_bag_path, topics)  # lazy frame
data_bag_2 = load_rosbag(second_bag_path, topics)
data_bag_3 = load_rosbag(third_bag_path, topics)
data_bag_4 = load_rosbag(fourth_bag_path, topics)

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
    "/scan", "/particle_cloud", "/momo/pose", "/amcl_pose"
)
drop_data_bag_1 = drop_data_transform.apply(zoh_data_bag_1)
drop_data_bag_2 = drop_data_transform.apply(zoh_data_bag_2)
drop_data_bag_3 = drop_data_transform.apply(zoh_data_bag_3)
drop_data_bag_4 = drop_data_transform.apply(zoh_data_bag_4)

# print(drop_data_bag_1.collect())

detect_delocalizations_transform = DetectDelocalizations(
    "/delocalizations", name="slice_points"
)
detect_deloc_data_bag_1 = detect_delocalizations_transform.apply(
    drop_data_bag_1
)
detect_deloc_data_bag_2 = detect_delocalizations_transform.apply(
    drop_data_bag_2
)
detect_deloc_data_bag_3 = detect_delocalizations_transform.apply(
    drop_data_bag_3
)
detect_deloc_data_bag_4 = detect_delocalizations_transform.apply(
    drop_data_bag_4
)

# print(detect_deloc_data_bag_1.collect())

drop_deloc_transform = Drop("/delocalizations")
drop_deloc_bag_1 = drop_deloc_transform.apply(detect_deloc_data_bag_1)
drop_deloc_bag_2 = drop_deloc_transform.apply(detect_deloc_data_bag_2)
drop_deloc_bag_3 = drop_deloc_transform.apply(detect_deloc_data_bag_3)
drop_deloc_bag_4 = drop_deloc_transform.apply(detect_deloc_data_bag_4)

# print(drop_deloc_bag_1.collect())

slice_time_series_transform = SliceTimeSeries(
    time_series="measurements", slice_points="slice_points"
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
    position_threshold=0.4,
    heading_threshold=0.4,
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


############

samples_train = samples_train_lf.collect(engine="streaming")
samples_eval = samples_eval_lf.collect(engine="streaming")

# print(samples_train)
# print(samples_eval)


# Serialize each complex column to JSON-like string so Polars can compare them
df_flat = samples_train.with_columns(
    [
        pl.col(c).struct.json_encode().alias(c)
        if samples_train.schema[c] == pl.Struct
        else pl.col(c).list.eval(pl.element().struct.json_encode()).alias(c)
        if str(samples_train.schema[c]).startswith("List(Struct")
        else pl.col(c)
        for c in samples_train.columns
    ]
)


total_rows = df_flat.height
unique_rows = df_flat.unique().height

dupe_groups = (
    df_flat.group_by(df_flat.columns).count().filter(pl.col("count") > 1)
)

num_dupe_groups = dupe_groups.height
total_rows_in_dupes = dupe_groups["count"].sum()
extra_duplicate_rows = (dupe_groups["count"] - 1).sum()

print("🔍 Duplicate Statistics (Safe Mode)")
print("===================================")
print(f"Total rows:                       {total_rows}")
print(f"Unique rows:                      {unique_rows}")
print(f"Number of duplicate groups:       {num_dupe_groups}")
print(f"Total rows in duplicate groups:   {total_rows_in_dupes}")
print(f"Extra duplicate rows (beyond 1st): {extra_duplicate_rows}")
print(
    f"→ Check sum (unique + extras):    {unique_rows + extra_duplicate_rows} (should equal total rows)"
)
