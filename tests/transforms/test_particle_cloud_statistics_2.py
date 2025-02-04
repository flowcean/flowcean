import math
import unittest

from flowcean.transforms.particle_cloud_statistics import (
    ParticleCloudStatistics,
)


class TestParticleCloudStatistics(unittest.TestCase):
    """Unit tests for the ParticleCloudStatistics class.

    We define five particles with specific positions/orientations/weights.

    We then check each feature method against known, hand-computed values.
    """

    def setUp(self) -> None:
        # Instantiate the transform
        self.pcs = ParticleCloudStatistics()

        # Define a small set of particles (unchanged)
        self.list_of_particles = [
            {
                "pose": {
                    "position": {
                        "x": 2.7026532109185792,
                        "y": 1.3363095842400234,
                        "z": 0.0,
                        "__msgtype__": "geometry_msgs/msg/Point",
                    },
                    "orientation": {
                        "x": 0.0,
                        "y": 0.0,
                        "z": -0.6206412601751722,
                        "w": 0.7840946538321596,
                        "__msgtype__": "geometry_msgs/msg/Quaternion",
                    },
                    "__msgtype__": "geometry_msgs/msg/Pose",
                },
                "weight": 0.0005980861244019139,
                "__msgtype__": "nav2_msgs/msg/Particle",
            },
            {
                "pose": {
                    "position": {
                        "x": 2.9070964865479705,
                        "y": 3.0649213798266697,
                        "z": 0.0,
                        "__msgtype__": "geometry_msgs/msg/Point",
                    },
                    "orientation": {
                        "x": 0.0,
                        "y": 0.0,
                        "z": -0.45132518076103845,
                        "w": 0.8923595582560967,
                        "__msgtype__": "geometry_msgs/msg/Quaternion",
                    },
                    "__msgtype__": "geometry_msgs/msg/Pose",
                },
                "weight": 0.0005980861244019139,
                "__msgtype__": "nav2_msgs/msg/Particle",
            },
            {
                "pose": {
                    "position": {
                        "x": 2.80871858542121,
                        "y": 1.5363776884978138,
                        "z": 0.0,
                        "__msgtype__": "geometry_msgs/msg/Point",
                    },
                    "orientation": {
                        "x": 0.0,
                        "y": 0.0,
                        "z": -0.36432616851598243,
                        "w": 0.9312714120676442,
                        "__msgtype__": "geometry_msgs/msg/Quaternion",
                    },
                    "__msgtype__": "geometry_msgs/msg/Pose",
                },
                "weight": 0.0005980861244019139,
                "__msgtype__": "nav2_msgs/msg/Particle",
            },
            {
                "pose": {
                    "position": {
                        "x": 1.8221955477463578,
                        "y": 1.6169840054666116,
                        "z": 0.0,
                        "__msgtype__": "geometry_msgs/msg/Point",
                    },
                    "orientation": {
                        "x": 0.0,
                        "y": 0.0,
                        "z": -0.584478714347991,
                        "w": 0.8114090414052085,
                        "__msgtype__": "geometry_msgs/msg/Quaternion",
                    },
                    "__msgtype__": "geometry_msgs/msg/Pose",
                },
                "weight": 0.0005980861244019139,
                "__msgtype__": "nav2_msgs/msg/Particle",
            },
            {
                "pose": {
                    "position": {
                        "x": 2.12472141189225,
                        "y": 1.5361849999975508,
                        "z": 0.0,
                        "__msgtype__": "geometry_msgs/msg/Point",
                    },
                    "orientation": {
                        "x": 0.0,
                        "y": 0.0,
                        "z": -0.4347883702383812,
                        "w": 0.900532660765534,
                        "__msgtype__": "geometry_msgs/msg/Quaternion",
                    },
                    "__msgtype__": "geometry_msgs/msg/Pose",
                },
                "weight": 0.0005980861244019139,
                "__msgtype__": "nav2_msgs/msg/Particle",
            },
        ]

        # DBSCAN parameters
        self.eps = 2.0
        self.min_samples = 2

    #########################
    # Helper / direct method tests
    #########################

    def test_center_of_gravity(self) -> None:
        cog = self.pcs.center_of_gravity(self.list_of_particles)
        # Expect (2.473, 1.818) within 2 decimal places
        assert math.isclose(cog["x"], 2.473, abs_tol=1e-2), cog["x"]
        assert math.isclose(cog["y"], 1.818, abs_tol=1e-2), cog["y"]

    def test_calculate_cog_mean(self) -> None:
        cmean = self.pcs.calculate_cog_mean(self.list_of_particles)
        # Expect (2.473, 1.818)
        assert math.isclose(cmean["x"], 2.473, abs_tol=1e-2), cmean["x"]
        assert math.isclose(cmean["y"], 1.818, abs_tol=1e-2), cmean["y"]

    def test_dist(self) -> None:
        # (0, 0) to (3, 4) => 5.0
        # (1, 1) to (2, 2) => sqrt(2)
        assert math.isclose(self.pcs.dist((0, 0), (3, 4)), 5.0, abs_tol=1e-2)
        assert math.isclose(
            self.pcs.dist((1, 1), (2, 2)),
            math.sqrt(2),
            abs_tol=1e-2,
        )

    def test_is_in_circle(self) -> None:
        # Circle center = (1,1), radius = sqrt(2)
        center, radius = (1, 1), math.sqrt(2)
        # (0,0) inside
        assert self.pcs.is_in_circle((0, 0), (center, radius))
        # (3,3) outside
        assert not self.pcs.is_in_circle((3, 3), (center, radius))

    def test_circle_from_two_points(self) -> None:
        center, radius = self.pcs.circle_from_two_points((0, 0), (2, 0))
        # Expect center (1,0), radius 1
        assert math.isclose(center[0], 1.0, abs_tol=1e-2), center[0]
        assert math.isclose(center[1], 0.0, abs_tol=1e-2), center[1]
        assert math.isclose(radius, 1.0, abs_tol=1e-2), radius

    def test_circle_from_three_points(self) -> None:
        center, radius = self.pcs.circle_from_three_points(
            (0, 0),
            (2, 0),
            (0, 2),
        )
        # Expect center (1,1), radius sqrt(2)
        assert math.isclose(center[0], 1.0, abs_tol=1e-2), center[0]
        assert math.isclose(center[1], 1.0, abs_tol=1e-2), center[1]
        assert math.isclose(radius, math.sqrt(2), abs_tol=1e-2), radius

    def test_welzl(self) -> None:
        points = [(0, 0), (2, 0), (0, 2)]
        center, radius = self.pcs.welzl(points)
        # Expect (1,1), sqrt(2)
        assert math.isclose(center[0], 1.0, abs_tol=1e-2), center[0]
        assert math.isclose(center[1], 1.0, abs_tol=1e-2), center[1]
        assert math.isclose(radius, math.sqrt(2), abs_tol=1e-2), radius

    #########################
    # Feature method tests
    #########################

    def test_cog_max_dist(self) -> None:
        max_distance, furthest_particle = self.pcs.cog_max_dist(
            self.list_of_particles,
        )
        # Expect ~1.3201
        assert math.isclose(max_distance, 1.3201, abs_tol=1e-2), max_distance

    def test_cog_mean_dist(self) -> None:
        result = self.pcs.cog_mean_dist(self.list_of_particles)
        # Expect ~0.6843
        assert math.isclose(result, 0.6843, abs_tol=1e-2), result

    def test_cog_mean_absolute_deviation(self) -> None:
        mad = self.pcs.cog_mean_absolute_deviation(self.list_of_particles)
        # Expect ~0.2543
        assert math.isclose(mad, 0.2543, abs_tol=1e-2), mad

    def test_cog_median(self) -> None:
        median_dist = self.pcs.cog_median(self.list_of_particles)
        # Expect ~0.5337
        assert math.isclose(median_dist, 0.5337, abs_tol=1e-2), median_dist

    def test_cog_median_absolute_deviation(self) -> None:
        mad = self.pcs.cog_median_absolute_deviation(self.list_of_particles)
        # Expect ~0.0955
        assert math.isclose(mad, 0.0955, abs_tol=1e-2), mad

    def test_cog_min_dist(self) -> None:
        min_dist, closest_particle = self.pcs.cog_min_dist(
            self.list_of_particles,
        )
        # Expect ~0.4382
        assert math.isclose(min_dist, 0.4382, abs_tol=1e-2), min_dist

    def test_cog_standard_deviation(self) -> None:
        std_dev = self.pcs.cog_standard_deviation(self.list_of_particles)
        # Expect ~0.3296
        assert math.isclose(std_dev, 0.3296, abs_tol=1e-2), std_dev

    def test_smallest_enclosing_circle(self) -> None:
        points = [
            (p["pose"]["position"]["x"], p["pose"]["position"]["y"])
            for p in self.list_of_particles
        ]
        center, radius = self.pcs.smallest_enclosing_circle(points)
        # Expect ~0.9213
        assert math.isclose(radius, 0.9213, abs_tol=1e-2), radius

    def test_circle_mean(self) -> None:
        mean_dist = self.pcs.circle_mean(self.list_of_particles)
        # Expect ~0.8647
        assert math.isclose(mean_dist, 0.8647, abs_tol=1e-2), mean_dist

    def test_circle_mean_absolute_deviation(self) -> None:
        mad = self.pcs.circle_mean_absolute_deviation(self.list_of_particles)
        # Expect ~0.06801
        assert math.isclose(mad, 0.06801, abs_tol=1e-2), mad

    def test_circle_median(self) -> None:
        median_dist = self.pcs.circle_median(self.list_of_particles)
        # Expect ~0.9213
        assert math.isclose(median_dist, 0.9213, abs_tol=1e-2), median_dist

    def test_circle_median_absolute_deviation(self) -> None:
        mad = self.pcs.circle_median_absolute_deviation(self.list_of_particles)
        # Expect ~0.0
        assert math.isclose(mad, 0.0, abs_tol=1e-2), mad

    def test_circle_min_dist(self) -> None:
        cmin = self.pcs.circle_min_dist(self.list_of_particles)
        # Expect ~0.763
        assert math.isclose(cmin, 0.763, abs_tol=1e-2), cmin

    def test_circle_std_deviation(self) -> None:
        std_dev = self.pcs.circle_std_deviation(self.list_of_particles)
        # Expect ~0.0702
        assert math.isclose(std_dev, 0.0702, abs_tol=1e-2), std_dev

    def test_count_clusters(self) -> None:
        num = self.pcs.count_clusters(
            self.list_of_particles,
            self.eps,
            self.min_samples,
        )
        # Expect 1
        assert num == 1

    def test_main_cluster_variance_x(self) -> None:
        var_x = self.pcs.main_cluster_variance_x(
            self.list_of_particles,
            self.eps,
            self.min_samples,
        )
        # Expect ~0.1797
        assert math.isclose(var_x, 0.1797, abs_tol=1e-2), var_x

    def test_main_cluster_variance_y(self) -> None:
        var_y = self.pcs.main_cluster_variance_y(
            self.list_of_particles,
            self.eps,
            self.min_samples,
        )
        # Expect ~0.3971
        assert math.isclose(var_y, 0.3971, abs_tol=1e-2), var_y


if __name__ == "__main__":
    unittest.main()
