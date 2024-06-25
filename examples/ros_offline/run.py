import logging
from pathlib import Path

import matplotlib.pyplot as plt
import polars.selectors as cs
import polars as pl

import flowcean.cli
from flowcean.environments.rosbag import RosbagEnvironment
from flowcean.transforms import MatchSamplingRate, Select

logger = logging.getLogger(__name__)


def main() -> None:
    flowcean.cli.initialize_logging()

    environment = RosbagEnvironment(
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
        },
    )
    environment.load()
    data = environment.get_data()

    acml = (
        data.select(pl.col("/j100_0000/amcl_pose").explode())
        .unnest(cs.all())
        .rename(lambda name: "amcl_" + name if name != "time" else name)
    )
    ground_truth = (
        data.select(pl.col("/j100_0000/odometry").explode())
        .unnest(cs.all())
        .rename(
            lambda name: "ground_truth_" + name if name != "time" else name
        )
    )
    result = (
        pl.concat([acml, ground_truth], how="diagonal")
        .sort("time")
        .with_columns(
            pl.col(
                [
                    "ground_truth_pose.pose.position.x",
                    "ground_truth_pose.pose.position.y",
                ]
            ).interpolate()
        )
        .drop_nulls()
    )
    print(result)
    # return
    plt.scatter(
        result["amcl_pose.pose.position.x"],
        result["amcl_pose.pose.position.y"],
    )
    plt.scatter(
        result["ground_truth_pose.pose.position.x"],
        result["ground_truth_pose.pose.position.y"],
    )
    plt.quiver(
        result["amcl_pose.pose.position.x"],
        result["amcl_pose.pose.position.y"],
        result["ground_truth_pose.pose.position.x"]
        - result["amcl_pose.pose.position.x"],
        result["ground_truth_pose.pose.position.y"]
        - result["amcl_pose.pose.position.y"],
        scale_units="xy",
        angles="xy",
        scale=1,
    )
    plt.show()

    return
    ground_truth = data.select(
        [
            pl.col("/j100_0000/odometry")
            .list.eval(
                pl.struct(
                    [
                        pl.element()
                        .struct.field("pose.pose.position.x")
                        .alias("x"),
                        pl.element()
                        .struct.field("pose.pose.position.y")
                        .alias("y"),
                    ]
                )
            )
            .explode()
        ]
    ).unnest("/j100_0000/odometry")
    amcl = data.select(
        [
            pl.col("/j100_0000/amcl_pose")
            .list.eval(
                pl.struct(
                    [
                        pl.element()
                        .struct.field("pose.pose.position.x")
                        .alias("x"),
                        pl.element()
                        .struct.field("pose.pose.position.y")
                        .alias("y"),
                    ]
                )
            )
            .explode()
        ]
    ).unnest("/j100_0000/amcl_pose")

    plt.scatter(ground_truth["x"], ground_truth["y"])
    plt.scatter(amcl["x"], amcl["y"])
    plt.show()


if __name__ == "__main__":
    main()
