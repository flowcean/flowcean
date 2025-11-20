from __future__ import annotations
from pathlib import Path
import numpy as np
import polars as pl

import flowcean.cli
from flowcean.ros import load_rosbag

from custom_transforms.particle_cloud_statistics import ParticleCloudStatistics
from custom_transforms.scan_map_statistics import ScanMapStatistics


# ==========================================================
# CONFIG / TOPICS
# ==========================================================

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


# ==========================================================
# HELPERS
# ==========================================================

def timeseries_to_df(row_df: pl.DataFrame, column: str, value_name: str) -> pl.DataFrame:
    ts = row_df[column][0]
    if len(ts) == 0:
        return pl.DataFrame({"time": [], value_name: []})

    return pl.DataFrame({
        "time": [e["time"] for e in ts],
        value_name: [e["value"] for e in ts],
    })


def yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    return float(np.arctan2(
        2.0 * (qw * qz + qx * qy),
        qw * qw + qx * qx - qy * qy - qz * qz,
    ))


# ==========================================================
# CORE PROCESSING FOR A SINGLE BAG
# ==========================================================

def process_single_bag(
    bag_path: str,
    topics: dict,
    message_paths: list,
    position_threshold: float,
    heading_threshold: float,
) -> pl.DataFrame:
    """Returns a fully processed ML-ready table for ONE rosbag."""
    print(f"\n=== Processing bag: {bag_path} ===")

    raw_lf = load_rosbag(bag_path, topics, message_paths=message_paths)
    full_schema = raw_lf.collect().schema

    # -----------------------------
    # Extract occupancy map
    # -----------------------------
    map_df = raw_lf.select("/map").collect()
    map_value = map_df["/map"][0][0]["value"]

    occupancy_map = {
        "data": map_value["data"],
        "info.width": map_value["info.width"],
        "info.height": map_value["info.height"],
        "info.resolution": map_value["info.resolution"],
        "info.origin.position.x": map_value["info.origin.position.x"],
        "info.origin.position.y": map_value["info.origin.position.y"],
        "info.origin.position.z": map_value["info.origin.position.z"],
        "info.origin.orientation.x": map_value["info.origin.orientation.x"],
        "info.origin.orientation.y": map_value["info.origin.orientation.y"],
        "info.origin.orientation.z": map_value["info.origin.orientation.z"],
        "info.origin.orientation.w": map_value["info.origin.orientation.w"],
    }

    # -----------------------------
    # Apply ScanMapStatistics
    # -----------------------------
    sms = ScanMapStatistics(
        occupancy_map=occupancy_map,
        scan_topic="/scan",
        sensor_pose_topic="/amcl_pose",
    )
    with_sms = sms.apply(raw_lf)

    # -----------------------------
    # Apply ParticleCloudStatistics
    # -----------------------------
    pcs = ParticleCloudStatistics(particle_cloud_feature_name="/particle_cloud")
    full_df = pcs.apply(with_sms).collect()

    # -----------------------------
    # Base time index = scan times
    # -----------------------------
    scan_ts = full_df["/scan"][0]
    base = pl.DataFrame({"time": [e["time"] for e in scan_ts]}).sort("time")

    # -----------------------------
    # Scan-map features
    # -----------------------------
    scanmap_cols = [
        "point_distance", "point_fitting", "point_inlier", "point_quality",
        "ray_inlier", "ray_inlier_percent", "ray_matching_percent",
        "ray_outlier_percent", "ray_quality",
        "angle_inlier", "angle_quality",
        "line_angle", "line_distance", "line_fitting", "line_length",
    ]
    for col in scanmap_cols:
        ts = timeseries_to_df(full_df, col, col)
        base = base.join(ts, on="time", how="inner")

    # -----------------------------
    # Particle-cloud features
    # -----------------------------
    pcs_cols = [
        "cog_max_distance", "cog_mean_dist", "cog_mean_absolute_deviation",
        "cog_median", "cog_median_absolute_deviation",
        "cog_min_distance", "cog_standard_deviation",
        "circle_radius", "circle_mean", "circle_mean_absolute_deviation",
        "circle_median", "circle_median_absolute_deviation",
        "circle_min_distance", "circle_standard_deviation",
        "num_clusters",
        "main_cluster_variance_x", "main_cluster_variance_y",
    ]
    for col in pcs_cols:
        ts = timeseries_to_df(full_df, col, col)
        if ts.height > 0:
            base = base.join_asof(ts.sort("time"), on="time", strategy="backward")

    # -----------------------------
    # AMCL & GT poses aligned
    # -----------------------------
    # AMCL
    amcl_rows = []
    for e in full_df["/amcl_pose"][0]:
        v = e["value"]
        amcl_rows.append({
            "time": e["time"],
            "amcl_x": v["pose.pose.position.x"],
            "amcl_y": v["pose.pose.position.y"],
            "amcl_qx": v["pose.pose.orientation.x"],
            "amcl_qy": v["pose.pose.orientation.y"],
            "amcl_qz": v["pose.pose.orientation.z"],
            "amcl_qw": v["pose.pose.orientation.w"],
        })
    base = base.join_asof(pl.DataFrame(amcl_rows).sort("time"),
                          on="time", strategy="backward")

    # GT
    gt_rows = []
    for e in full_df["/momo/pose"][0]:
        v = e["value"]
        gt_rows.append({
            "time": e["time"],
            "gt_x": v["pose.position.x"],
            "gt_y": v["pose.position.y"],
            "gt_qx": v["pose.orientation.x"],
            "gt_qy": v["pose.orientation.y"],
            "gt_qz": v["pose.orientation.z"],
            "gt_qw": v["pose.orientation.w"],
        })
    base = base.join_asof(pl.DataFrame(gt_rows).sort("time"),
                          on="time", strategy="backward")

    base = base.drop_nulls(subset=["amcl_x", "gt_x"])

    # -----------------------------
    # Compute yaw, errors, labels
    # -----------------------------
    base = base.with_columns([
        pl.struct("amcl_qx", "amcl_qy", "amcl_qz", "amcl_qw")
        .map_elements(lambda s: yaw_from_quat(
            s["amcl_qx"], s["amcl_qy"], s["amcl_qz"], s["amcl_qw"]))
        .alias("amcl_yaw"),

        pl.struct("gt_qx", "gt_qy", "gt_qz", "gt_qw")
        .map_elements(lambda s: yaw_from_quat(
            s["gt_qx"], s["gt_qy"], s["gt_qz"], s["gt_qw"]))
        .alias("gt_yaw"),
    ])

    base = base.with_columns([
        (((pl.col("gt_x") - pl.col("amcl_x")) ** 2 +
          (pl.col("gt_y") - pl.col("amcl_y")) ** 2).sqrt())
        .alias("position_error"),

        (pl.col("gt_yaw") - pl.col("amcl_yaw")).alias("heading_error_raw"),
    ])

    base = base.with_columns(
        ((pl.col("heading_error_raw") + np.pi) % (2 * np.pi) - np.pi)
        .alias("heading_error")
    )

    base = base.with_columns(
        ((pl.col("position_error") > position_threshold) |
         (pl.col("heading_error").abs() > heading_threshold))
        .alias("is_delocalized")
    )

    base = base.with_columns(
        (pl.col("position_error") +
         0.5 * pl.col("heading_error").abs())
        .alias("combined_error")
    )

    return base


# ==========================================================
# MAIN: Process all bags
# ==========================================================

def main() -> None:
    config = flowcean.cli.initialize()
    topics = get_topics()

    position_threshold = float(config.localization.position_threshold)
    heading_threshold = float(config.localization.heading_threshold)

    # ---------------------------
    # TRAINING SET
    # ---------------------------
    train_tables = []
    for bag in config.rosbag.training_paths:
        t = process_single_bag(
            bag_path=bag,
            topics=topics,
            message_paths=config.rosbag.message_paths,
            position_threshold=position_threshold,
            heading_threshold=heading_threshold,
        )
        train_tables.append(t)

    train_df = pl.concat(train_tables, how="vertical").sort("time")
    train_df.write_parquet("train.parquet")
    train_df.write_csv("train.csv")
    print("\nSaved TRAINING dataset")

    # ---------------------------
    # EVALUATION SET
    # ---------------------------
    eval_tables = []
    for bag in config.rosbag.evaluation_paths:
        t = process_single_bag(
            bag_path=bag,
            topics=topics,
            message_paths=config.rosbag.message_paths,
            position_threshold=position_threshold,
            heading_threshold=heading_threshold,
        )
        eval_tables.append(t)

    eval_df = pl.concat(eval_tables, how="vertical").sort("time")
    eval_df.write_parquet("eval.parquet")
    eval_df.write_csv("eval.csv")
    print("Saved EVALUATION dataset")


if __name__ == "__main__":
    main()
