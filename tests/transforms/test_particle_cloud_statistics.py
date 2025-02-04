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

        # DBSCAN parameters
        self.eps = 2.0
        self.min_samples = 2

    #########################
    # Helper / direct method tests
    #########################

    def test_center_of_gravity(self)-> None:
        # Weighted COG with A(0,0, w=1), B(2,0, w=2), C(0,2, w=1):
        #   total_weight = 4
        #   x = (0*1 + 2*2 + 0*1) / 4 = 4/4 = 1
        #   y = (0*1 + 0*2 + 2*1) / 4 = 2/4 = 0.5
        cog = self.pcs.center_of_gravity(self.list_of_particles)
        self.assertAlmostEqual(cog["x"], 1.0, places=2)
        self.assertAlmostEqual(cog["y"], 0.5, places=2)

    def test_calculate_cog_mean(self)-> None:
        # Unweighted mean: ( (0+2+0)/3, (0+0+2)/3 ) => (2/3, 2/3)
        cmean = self.pcs.calculate_cog_mean(self.list_of_particles)
        self.assertAlmostEqual(cmean["x"], 2.0 / 3.0, places=2)
        self.assertAlmostEqual(cmean["y"], 2.0 / 3.0, places=2)

    def test_dist(self)-> None:
        # Simple distance checks
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
        # (0,0), (2,0), (0,2) => encl. circle center => (1,1), radius => sqrt(2)
        center, radius = self.pcs.circle_from_three_points(
            (0, 0), (2, 0), (0, 2)
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
        # Weighted center => (1,0.5)
        # Distances:
        #   A->COG ~1.1180..., B->COG ~1.1180..., C->COG ~1.8027...
        # => max => ~1.8027
        max_distance, furthest_particle = self.pcs.cog_max_dist(
            self.list_of_particles
        )
        self.assertAlmostEqual(max_distance, 1.8020, places=2)
        self.assertEqual(furthest_particle, (0.0, 2.0))

    def test_cog_mean_dist(self)-> None:
        # Unweighted center => (2/3, 2/3)
        # Distances:
        #   A->mean ~0.9428, B->mean ~1.4907, C->mean ~1.4907
        # => average => ~1.3081
        result = self.pcs.cog_mean_dist(self.list_of_particles)
        self.assertAlmostEqual(result, 1.3081, places=2)

    def test_cog_mean_absolute_deviation(self)-> None:
        # Distances => ~[0.9428, 1.4907, 1.4907], mean => 1.3081
        # abs dev => [0.3653, 0.1826, 0.1826], mean => 0.2435
        mad = self.pcs.cog_mean_absolute_deviation(self.list_of_particles)
        self.assertAlmostEqual(mad, 0.2435, places=2)

    def test_cog_median(self)-> None:
        # Distances => [0.9428, 1.4907, 1.4907], sorted => same
        # => median => middle => 1.4907
        median_dist = self.pcs.cog_median(self.list_of_particles)
        self.assertAlmostEqual(median_dist, 1.4907, places=2)

    def test_cog_median_absolute_deviation(self)-> None:
        # median distance => 1.4907
        # abs dev => [0.5479, 0, 0], sorted => [0, 0, 0.5479]
        # => median => 0
        mad = self.pcs.cog_median_absolute_deviation(self.list_of_particles)
        self.assertAlmostEqual(mad, 0.0, places=2)

    def test_cog_min_dist(self)-> None:
        # Weighted center => (1,0.5)
        # min distance => ~1.1180 from either A(0,0) or B(2,0)
        min_dist, closest_particle = self.pcs.cog_min_dist(
            self.list_of_particles
        )
        self.assertAlmostEqual(min_dist, 1.1180, places=2)
        self.assertEqual(closest_particle, (0.0, 0.0))

    def test_cog_standard_deviation(self)-> None:
        # Distances => [0.9428, 1.4907, 1.4907], mean =>1.3081
        # stdev => ~0.2583
        std_dev = self.pcs.cog_standard_deviation(self.list_of_particles)
        self.assertAlmostEqual(std_dev, 0.2583, places=2)

    def test_smallest_enclosing_circle(self)-> None:
        points = [
            (p["pose"]["position"]["x"], p["pose"]["position"]["y"])
            for p in self.list_of_particles
        ]
        center, radius = self.pcs.smallest_enclosing_circle(points)
        self.assertAlmostEqual(center[0], 1.0, places=2)
        self.assertAlmostEqual(center[1], 1.0, places=2)
        self.assertAlmostEqual(radius, math.sqrt(2), places=2)

    def test_circle_mean(self)-> None:
        # Circle center => (1,1), each point => distance sqrt(2).
        # => mean => sqrt(2)
        mean_dist = self.pcs.circle_mean(self.list_of_particles)
        self.assertAlmostEqual(mean_dist, math.sqrt(2), places=2)

    def test_circle_mean_absolute_deviation(self)-> None:
        # All distances => sqrt(2). => MAD => 0
        mad = self.pcs.circle_mean_absolute_deviation(self.list_of_particles)
        self.assertAlmostEqual(mad, 0.0, places=2)

    def test_circle_median(self)-> None:
        # All distances => sqrt(2). => median => sqrt(2)
        median_dist = self.pcs.circle_median(self.list_of_particles)
        self.assertAlmostEqual(median_dist, math.sqrt(2), places=2)

    def test_circle_median_absolute_deviation(self)-> None:
        # All distances => sqrt(2). => median => sqrt(2). => dev => 0
        mad = self.pcs.circle_median_absolute_deviation(self.list_of_particles)
        self.assertAlmostEqual(mad, 0.0, places=2)

    def test_circle_min_dist(self)-> None:
        # All distances => sqrt(2). => min => sqrt(2)
        cmin = self.pcs.circle_min_dist(self.list_of_particles)
        self.assertAlmostEqual(cmin, math.sqrt(2), places=2)

    def test_circle_std_deviation(self)-> None:
        # All distances => sqrt(2). => stdev => 0
        std_dev = self.pcs.circle_std_deviation(self.list_of_particles)
        self.assertAlmostEqual(std_dev, 0.0, places=2)

    def test_count_clusters(self)-> None:
        # With eps=2.0, min_samples=2 => all 3 points form 1 cluster
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
        self.assertAlmostEqual(var_x, 0.8889, places=2)

    def test_main_cluster_variance_y(self)-> None:
        # The main cluster => y coords [0,0,2], mean=2/3
        # => variance => ~0.8889
        var_y = self.pcs.main_cluster_variance_y(
            self.list_of_particles,
            self.eps,
            self.min_samples,
        )
        self.assertAlmostEqual(var_y, 0.8889, places=2)


if __name__ == "__main__":
    unittest.main()
