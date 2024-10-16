import logging

import flowcean.cli
from flowcean.environments.rosbag import RosbagLoader
from flowcean.transforms import EuclideanDistance, MatchSamplingRate

logger = logging.getLogger(__name__)


def main() -> None:
    flowcean.cli.initialize_logging()

    environment = RosbagLoader(
        path="example_rosbag",
        topics={
            "/j100_0000/amcl_pose": [
                "pose.pose.position.x",
                "pose.pose.position.y",
            ],
            "/j100_0000/odometry": [
                "pose.pose.position.x",
                "pose.pose.position.y",
            ],
            "/j100_0000/sensors/imu_0/data_raw": [
                "orientation_covariance",
            ],
            "/j100_0000/sensors/lidar3d_0/scan": [
                "ranges",
            ],
            "/j100_0000/particle_cloud": ["particles"],
        },
        msgpaths=[
            "/opt/ros/humble/share/sensor_msgs/msg/Imu.msg",
            "/opt/ros/humble/share/sensor_msgs/msg/LaserScan.msg",
            "/opt/ros/humble/share/nav2_msgs/msg/ParticleCloud.msg",
            "/opt/ros/humble/share/nav2_msgs/msg/Particle.msg",
        ],
    )
    environment.load()
    data = environment.get_data()
    print(f"original data: {data}")

    return
    transform = MatchSamplingRate(
        reference_feature_name="/j100_0000/amcl_pose",
        feature_interpolation_map={
            "/j100_0000/odometry": "linear",
        },
    ) | EuclideanDistance(
        feature_a_name="/j100_0000/amcl_pose",
        feature_b_name="/j100_0000/odometry",
        output_feature_name="position_error",
    )
    transformed_data = transform.transform(data)

    print(
        transformed_data.explode(
            "/j100_0000/amcl_pose", "/j100_0000/odometry", "position_error"
        )
    )


if __name__ == "__main__":
    main()
