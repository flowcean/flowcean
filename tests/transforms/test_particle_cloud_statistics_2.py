import math
import unittest

from flowcean.transforms.particle_cloud_statistics import (
    ParticleCloudStatistics,
)


class TestParticleCloudStatistics(unittest.TestCase):
    """Unit tests for the ParticleCloudStatistics class.

    We define three particles with the following positions/weights:
      - A: (0, 0), weight=1
      - B: (2, 0), weight=2
      - C: (0, 2), weight=1

    We then check each feature method against known, hand-computed values.
    """

    def setUp(self)-> None:
        # Instantiate the transform
        self.pcs = ParticleCloudStatistics()

        # Define a small set of particles
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

    def test_center_of_gravity(self)-> None:
        cog = self.pcs.center_of_gravity(self.list_of_particles)
        self.assertAlmostEqual(cog["x"], 2.473, places=2)
        self.assertAlmostEqual(cog["y"], 1.818, places=2)

    def test_calculate_cog_mean(self)-> None:
        cmean = self.pcs.calculate_cog_mean(self.list_of_particles)
        self.assertAlmostEqual(cmean["x"], 2.473, places=2)
        self.assertAlmostEqual(cmean["y"], 1.818, places=2)

    def test_dist(self)-> None:
        self.assertAlmostEqual(self.pcs.dist((0, 0), (3, 4)), 5.0, places=2)
        self.assertAlmostEqual(
            self.pcs.dist((1, 1), (2, 2)),
            math.sqrt(2),
            places=2,
        )

    def test_is_in_circle(self)-> None:
        # Circle center = (1,1), radius = sqrt(2)
        center, radius = (1, 1), math.sqrt(2)
        # (0,0) should be inside
        self.assertTrue(self.pcs.is_in_circle((0, 0), (center, radius)))
        # (3,3) is outside
        self.assertFalse(self.pcs.is_in_circle((3, 3), (center, radius)))

    def test_circle_from_two_points(self)-> None:
        center, radius = self.pcs.circle_from_two_points((0, 0), (2, 0))
        # Center => (1,0), radius => 1
        self.assertAlmostEqual(center[0], 1.0, places=2)
        self.assertAlmostEqual(center[1], 0.0, places=2)
        self.assertAlmostEqual(radius, 1.0, places=2)

    def test_circle_from_three_points(self)-> None:
        # (0,0), (2,0), (0,2) => encl. circle center => (1,1),radius => sqrt(2)
        center, radius = self.pcs.circle_from_three_points(
            (0, 0),
            (2, 0),
            (0, 2),
        )
        self.assertAlmostEqual(center[0], 1.0, places=2)
        self.assertAlmostEqual(center[1], 1.0, places=2)
        self.assertAlmostEqual(radius, math.sqrt(2), places=2)

    def test_welzl(self)-> None:
        # smallest enclosing circle using Welzl's
        points = [(0, 0), (2, 0), (0, 2)]
        center, radius = self.pcs.welzl(points)
        self.assertAlmostEqual(center[0], 1.0, places=2)
        self.assertAlmostEqual(center[1], 1.0, places=2)
        self.assertAlmostEqual(radius, math.sqrt(2), places=2)

    #########################
    # Feature method tests
    #########################

    def test_cog_max_dist(self)-> None:
        max_distance, furthest_particle = self.pcs.cog_max_dist(
            self.list_of_particles,
        )
        self.assertAlmostEqual(max_distance, 1.3201, places=2)

    def test_cog_mean_dist(self)-> None:
        result = self.pcs.cog_mean_dist(self.list_of_particles)
        self.assertAlmostEqual(result, 0.6843, places=2)

    def test_cog_mean_absolute_deviation(self)-> None:
        mad = self.pcs.cog_mean_absolute_deviation(self.list_of_particles)
        self.assertAlmostEqual(mad, 0.2543, places=2)

    def test_cog_median(self)-> None:
        median_dist = self.pcs.cog_median(self.list_of_particles)
        self.assertAlmostEqual(median_dist, 0.5337, places=2)

    def test_cog_median_absolute_deviation(self)-> None:
        mad = self.pcs.cog_median_absolute_deviation(self.list_of_particles)
        self.assertAlmostEqual(mad, 0.0955, places=2)

    def test_cog_min_dist(self)-> None:
        min_dist, closest_particle = self.pcs.cog_min_dist(
            self.list_of_particles,
        )
        self.assertAlmostEqual(min_dist, 0.4382, places=2)

    def test_cog_standard_deviation(self)-> None:
        std_dev = self.pcs.cog_standard_deviation(self.list_of_particles)
        self.assertAlmostEqual(std_dev, 0.3296, places=2)

    def test_smallest_enclosing_circle(self)-> None:
        points = [
            (p["pose"]["position"]["x"], p["pose"]["position"]["y"])
            for p in self.list_of_particles
        ]
        center, radius = self.pcs.smallest_enclosing_circle(points)
        self.assertAlmostEqual(radius, 0.9213, places=2)

    def test_circle_mean(self)-> None:
        mean_dist = self.pcs.circle_mean(self.list_of_particles)
        self.assertAlmostEqual(mean_dist, 0.8647, places=2)

    def test_circle_mean_absolute_deviation(self)-> None:
        # All distances => sqrt(2). => MAD => 0
        mad = self.pcs.circle_mean_absolute_deviation(self.list_of_particles)
        self.assertAlmostEqual(mad, 0.06801, places=2)

    def test_circle_median(self)-> None:
        # All distances => sqrt(2). => median => sqrt(2)
        median_dist = self.pcs.circle_median(self.list_of_particles)
        self.assertAlmostEqual(median_dist, 0.9213, places=2)

    def test_circle_median_absolute_deviation(self)-> None:
        # All distances => sqrt(2). => median => sqrt(2). => dev => 0
        mad = self.pcs.circle_median_absolute_deviation(self.list_of_particles)
        self.assertAlmostEqual(mad, 0.0, places=2)

    def test_circle_min_dist(self)-> None:
        # All distances => sqrt(2). => min => sqrt(2)
        cmin = self.pcs.circle_min_dist(self.list_of_particles)
        self.assertAlmostEqual(cmin, 0.763, places=2)

    def test_circle_std_deviation(self)-> None:
        # All distances => sqrt(2). => stdev => 0
        std_dev = self.pcs.circle_std_deviation(self.list_of_particles)
        self.assertAlmostEqual(std_dev, 0.0702, places=2)

    def test_count_clusters(self)-> None:
        num = self.pcs.count_clusters(
            self.list_of_particles,
            self.eps,
            self.min_samples,
        )
        self.assertEqual(num, 1)

    def test_main_cluster_variance_x(self)-> None:
        # The main cluster => x coords [0,2,0], mean=2/3
        # => variance => ~0.8889
        var_x = self.pcs.main_cluster_variance_x(
            self.list_of_particles,
            self.eps,
            self.min_samples,
        )
        # Because DBSCAN lumps them together in one cluster, we should see ~0.8889
        self.assertAlmostEqual(var_x, 0.1797, places=2)

    def test_main_cluster_variance_y(self)-> None:
        # The main cluster => y coords [0,0,2], mean=2/3
        # => variance => ~0.8889
        var_y = self.pcs.main_cluster_variance_y(
            self.list_of_particles,
            self.eps,
            self.min_samples,
        )
        self.assertAlmostEqual(var_y, 0.3971, places=2)

if __name__ == "__main__":
    unittest.main()
