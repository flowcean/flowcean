import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from utils import calculate_scan_points, euler_from_quaternion


def scan_points_transform(
    data: pl.DataFrame,
    scan_topic: str = "/scan",
    sensor_pose_topic: str = "/amcl_pose",
) -> pl.DataFrame:
    """Calculate the scan points of a LaserScan message."""
    scan_timeseries = data[scan_topic].to_list()[0]
    amcl_pose_timeseries = data[sensor_pose_topic].to_list()[0]
    scan_points_timeseries = []
    for scan in scan_timeseries:
        timestamp = scan["time"]
        print(timestamp)
        # get the latest pose
        poses_before_current_scan = [
            entry
            for entry in amcl_pose_timeseries
            if entry["time"] < timestamp
        ]
        if poses_before_current_scan:
            pose = poses_before_current_scan[-1]["value"]
            pose = (
                pose["pose.pose.position.x"],
                pose["pose.pose.position.y"],
                euler_from_quaternion(
                    pose["pose.pose.orientation.x"],
                    pose["pose.pose.orientation.y"],
                    pose["pose.pose.orientation.z"],
                    pose["pose.pose.orientation.w"],
                )[2],
            )
            scan_points = calculate_scan_points(
                pose=pose,
                ranges=scan["value"]["ranges"],
                angle_min=scan["value"]["angle_min"],
                angle_max=scan["value"]["angle_max"],
                angle_increment=scan["value"]["angle_increment"],
                range_max=scan["value"]["range_max"],
                range_min=scan["value"]["range_min"],
            )
            scan["value"] = scan_points.tolist()
            scan["time"] = timestamp
            scan_points_timeseries.append(scan)
        else:
            continue
    # plot the last scan points
    last_scan_points = np.array(scan_points_timeseries[-1]["value"])
    fig = plt.figure()
    fig.add_subplot(111)
    plt.scatter(last_scan_points[:, 0], last_scan_points[:, 1])
    plt.savefig("last_scan_points.png")
    return data.hstack(pl.DataFrame({"scan_points": [scan_points_timeseries]}))
