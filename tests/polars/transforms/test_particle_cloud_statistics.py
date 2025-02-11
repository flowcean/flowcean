import math
import unittest
from datetime import datetime, timezone

import polars as pl

from flowcean.polars import (
    ParticleCloudStatistics,
)


class TestParticleCloudStatistics(unittest.TestCase):
    """Unit tests for the ParticleCloudStatistics class.

    We define three particles:
      - A: (0, 0), weight=1
      - B: (2, 0), weight=2
      - C: (0, 2), weight=1

    The orientation is the same for all particles.

    We then check each feature method against known, hand-computed values.
    """

    def setUp(self) -> None:
        self.particle_cloud_transform = ParticleCloudStatistics(
            "particle_cloud",
        )

        self.list_of_particles = [
            {
                "pose": {
                    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
                "weight": 1.0,
            },
            {
                "pose": {
                    "position": {"x": 2.0, "y": 0.0, "z": 0.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
                "weight": 2.0,
            },
            {
                "pose": {
                    "position": {"x": 0.0, "y": 2.0, "z": 0.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
                "weight": 1.0,
            },
        ]
        self.data_frame = pl.DataFrame(
            {
                "particle_cloud": [
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                1,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"particles": self.list_of_particles},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                2,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"particles": self.list_of_particles},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                3,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"particles": self.list_of_particles},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                4,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"particles": self.list_of_particles},
                        },
                    ],
                ],
                "feature_b": [
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                0,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 1.0},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                5,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 2.0},
                        },
                    ],
                ],
                "const": [1],
            },
        )

        # DBSCAN parameters for clustering the particles
        self.eps = 2.0
        self.min_samples = 2

    #########################
    # Helper / direct method tests
    #########################

    def test_center_of_gravity(self) -> None:
        cog = self.particle_cloud_transform.center_of_gravity(
            self.list_of_particles,
        )
        # Expect (1.0, 0.5)
        assert math.isclose(cog["x"], 1.0, abs_tol=1e-2), cog["x"]
        assert math.isclose(cog["y"], 0.5, abs_tol=1e-2), cog["y"]

    def test_calculate_cog_mean(self) -> None:
        cmean = self.particle_cloud_transform.calculate_cog_mean(
            self.list_of_particles,
        )
        # Expect (2/3, 2/3)
        assert math.isclose(cmean["x"], 2.0 / 3.0, abs_tol=1e-2), cmean["x"]
        assert math.isclose(cmean["y"], 2.0 / 3.0, abs_tol=1e-2), cmean["y"]

    def test_dist(self) -> None:
        # Check distance
        assert math.isclose(
            self.particle_cloud_transform.dist((0, 0), (3, 4)),
            5.0,
            abs_tol=1e-2,
        )
        assert math.isclose(
            self.particle_cloud_transform.dist((1, 1), (2, 2)),
            math.sqrt(2),
            abs_tol=1e-2,
        )

    def test_is_in_circle(self) -> None:
        # Circle center = (1,1), radius = sqrt(2)
        center, radius = (1, 1), math.sqrt(2)
        # (0,0) should be inside
        assert self.particle_cloud_transform.is_in_circle(
            (0, 0),
            (center, radius),
        )
        # (3,3) is outside
        assert not self.particle_cloud_transform.is_in_circle(
            (3, 3),
            (center, radius),
        )

    def test_circle_from_two_points(self) -> None:
        center, radius = self.particle_cloud_transform.circle_from_two_points(
            (0, 0),
            (2, 0),
        )
        # Expect center => (1,0), radius => 1
        assert math.isclose(center[0], 1.0, abs_tol=1e-2), center[0]
        assert math.isclose(center[1], 0.0, abs_tol=1e-2), center[1]
        assert math.isclose(radius, 1.0, abs_tol=1e-2), radius

    def test_circle_from_three_points(self) -> None:
        center, radius = (
            self.particle_cloud_transform.circle_from_three_points(
                (0, 0),
                (2, 0),
                (0, 2),
            )
        )
        # Expect center => (1,1), radius => sqrt(2)
        assert math.isclose(center[0], 1.0, abs_tol=1e-2), center[0]
        assert math.isclose(center[1], 1.0, abs_tol=1e-2), center[1]
        assert math.isclose(radius, math.sqrt(2), abs_tol=1e-2), radius

    def test_welzl(self) -> None:
        points = [(0, 0), (2, 0), (0, 2)]
        center, radius = self.particle_cloud_transform.welzl(points)
        assert math.isclose(center[0], 1.0, abs_tol=1e-2), center[0]
        assert math.isclose(center[1], 1.0, abs_tol=1e-2), center[1]
        assert math.isclose(radius, math.sqrt(2), abs_tol=1e-2), radius

    #########################
    # Feature method tests
    #########################

    def test_cog_max_dist(self) -> None:
        # Weighted center => (1,0.5)
        max_distance, furthest_particle = (
            self.particle_cloud_transform.cog_max_dist(
                self.list_of_particles,
            )
        )
        # Expect ~1.8020, furthest => (0.0, 2.0)
        assert math.isclose(max_distance, 1.8020, abs_tol=1e-2), max_distance
        assert furthest_particle == (0.0, 2.0)

    def test_cog_mean_dist(self) -> None:
        # Unweighted center => (2/3, 2/3) => average dist ~1.3081
        result = self.particle_cloud_transform.cog_mean_dist(
            self.list_of_particles,
        )
        assert math.isclose(result, 1.3081, abs_tol=1e-2), result

    def test_cog_mean_absolute_deviation(self) -> None:
        # Expected ~0.2435
        mad = self.particle_cloud_transform.cog_mean_absolute_deviation(
            self.list_of_particles,
        )
        assert math.isclose(mad, 0.2435, abs_tol=1e-2), mad

    def test_cog_median(self) -> None:
        # Expected median ~1.4907
        median_dist = self.particle_cloud_transform.cog_median(
            self.list_of_particles,
        )
        assert math.isclose(median_dist, 1.4907, abs_tol=1e-2), median_dist

    def test_cog_median_absolute_deviation(self) -> None:
        # Expected ~0.0
        mad = self.particle_cloud_transform.cog_median_absolute_deviation(
            self.list_of_particles,
        )
        assert math.isclose(mad, 0.0, abs_tol=1e-2), mad

    def test_cog_min_dist(self) -> None:
        # Weighted center => (1,0.5); expected min dist ~1.1180
        min_dist, closest_particle = (
            self.particle_cloud_transform.cog_min_dist(
                self.list_of_particles,
            )
        )
        assert math.isclose(min_dist, 1.1180, abs_tol=1e-2), min_dist
        assert closest_particle == (0.0, 0.0)

    def test_cog_standard_deviation(self) -> None:
        # Distances => [0.9428, 1.4907, 1.4907]; stdev => ~0.2583
        std_dev = self.particle_cloud_transform.cog_standard_deviation(
            self.list_of_particles,
        )
        assert math.isclose(std_dev, 0.2583, abs_tol=1e-2), std_dev

    def test_smallest_enclosing_circle(self) -> None:
        points = [
            (p["pose"]["position"]["x"], p["pose"]["position"]["y"])
            for p in self.list_of_particles
        ]
        center, radius = (
            self.particle_cloud_transform.smallest_enclosing_circle(points)
        )
        # Expect (1.0, 1.0), sqrt(2)
        assert math.isclose(center[0], 1.0, abs_tol=1e-2), center[0]
        assert math.isclose(center[1], 1.0, abs_tol=1e-2), center[1]
        assert math.isclose(radius, math.sqrt(2), abs_tol=1e-2), radius

    def test_circle_mean(self) -> None:
        # All distances => sqrt(2). => mean => sqrt(2)
        mean_dist = self.particle_cloud_transform.circle_mean(
            self.list_of_particles,
        )
        assert math.isclose(mean_dist, math.sqrt(2), abs_tol=1e-2), mean_dist

    def test_circle_mean_absolute_deviation(self) -> None:
        # All distances => sqrt(2). => MAD => 0
        mad = self.particle_cloud_transform.circle_mean_absolute_deviation(
            self.list_of_particles,
        )
        assert math.isclose(mad, 0.0, abs_tol=1e-2), mad

    def test_circle_median(self) -> None:
        # All distances => sqrt(2). => median => sqrt(2)
        median_dist = self.particle_cloud_transform.circle_median(
            self.list_of_particles,
        )
        assert math.isclose(
            median_dist,
            math.sqrt(2),
            abs_tol=1e-2,
        ), median_dist

    def test_circle_median_absolute_deviation(self) -> None:
        # All distances => sqrt(2). => median => sqrt(2). => dev => 0
        mad = self.particle_cloud_transform.circle_median_absolute_deviation(
            self.list_of_particles,
        )
        assert math.isclose(mad, 0.0, abs_tol=1e-2), mad

    def test_circle_min_dist(self) -> None:
        # All distances => sqrt(2). => min => sqrt(2)
        cmin = self.particle_cloud_transform.circle_min_dist(
            self.list_of_particles,
        )
        assert math.isclose(cmin, math.sqrt(2), abs_tol=1e-2), cmin

    def test_circle_std_deviation(self) -> None:
        # All distances => sqrt(2). => stdev => 0
        std_dev = self.particle_cloud_transform.circle_std_deviation(
            self.list_of_particles,
        )
        assert math.isclose(std_dev, 0.0, abs_tol=1e-2), std_dev

    def test_count_clusters(self) -> None:
        # With eps=2.0, min_samples=2 => all 3 points form 1 cluster
        num = self.particle_cloud_transform.count_clusters(
            self.list_of_particles,
            self.eps,
            self.min_samples,
        )
        assert num == 1

    def test_main_cluster_variance_x(self) -> None:
        # The main cluster => x coords [0,2,0], mean=2/3 => variance ~0.8889
        var_x = self.particle_cloud_transform.main_cluster_variance_x(
            self.list_of_particles,
            self.eps,
            self.min_samples,
        )
        assert math.isclose(var_x, 0.8889, abs_tol=1e-2), var_x

    def test_main_cluster_variance_y(self) -> None:
        # The main cluster => y coords [0,0,2], mean=2/3 => variance ~0.8889
        var_y = self.particle_cloud_transform.main_cluster_variance_y(
            self.list_of_particles,
            self.eps,
            self.min_samples,
        )
        assert math.isclose(var_y, 0.8889, abs_tol=1e-2), var_y

    def test_transform(self) -> None:
        transformed_data = self.particle_cloud_transform.apply(
            self.data_frame.lazy(),
        ).collect()
        print(transformed_data)
        # check that there are 18 columns of type List(Struct()) types
        list_struct_columns = [
            dtype
            for dtype in transformed_data.dtypes
            if isinstance(dtype, pl.List)
            and isinstance(dtype.inner, pl.Struct)
        ]
        assert len(list_struct_columns) == 18


if __name__ == "__main__":
    unittest.main()
