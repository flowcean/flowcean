import logging
import math
import random
import sys

import numpy as np
import polars as pl
from sklearn.cluster import DBSCAN

from flowcean.core import Transform

logger = logging.getLogger(__name__)


class ParticleCloudStatistics(Transform):
    """Calculates statistical features from particle cloud data.

    This class provides methods to calculate various statistical features from
    a particle cloud, such as distances to the center of gravity, clustering
    information, and distances to the smallest enclosing circle. These features
    can be used for further analysis or machine learning tasks.
    """

    def __init__(
        self,
        particle_cloud_feature_name: str = "/particle_cloud",
    ) -> None:
        """Initialize the ParticleCloudStatistics transform.

        Args:
            particle_cloud_feature_name: Name of the particle cloud feature.
        """
        self.particle_cloud_feature_name = particle_cloud_feature_name

    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        logger.debug("Matching sampling rate of time series.")

        sys.setrecursionlimit(1000000)

        particle_cloud = data.collect()[0, self.particle_cloud_feature_name]

        number_of_messages = len(particle_cloud)

        all_features = []

        for i in range(number_of_messages):
            message_dictionary = particle_cloud[i]

            message_time = message_dictionary["time"]
            particles_dict = message_dictionary["value"]
            list_of_particles = particles_dict["particles"]

            features = self.extract_features_from_message(
                list_of_particles,
                eps=0.3,
                min_samples=5,
            )

            features["time"] = message_time
            all_features.append(features)

        features_df = pl.DataFrame(all_features)
        time_values = features_df["time"].to_list()

        new_data = {}

        for col in features_df.columns:
            if col == "time":
                continue

            feature_values = features_df[col].to_list()

            # Combine feature value with corresponding time into a dictionary
            dict_list = [
                {"time": t, "value": val}
                for t, val in zip(time_values, feature_values, strict=False)
            ]

            # Put entire dict_list as a single entry - a list of all structs.
            new_data[col] = [dict_list]

        final_df = pl.DataFrame(new_data)

        return (
            data.collect()
            .drop(self.particle_cloud_feature_name)
            .hstack(final_df)
            .lazy()
        )

    def cog_max_dist(
        self,
        list_of_particles: list[dict],
    ) -> tuple[float, tuple | None]:
        """Calculates maximum distance from any particle to center of gravity.

        Args:
            list_of_particles: List of dictionaries representing particles.

        Returns:
            tuple: The maximum distance and coordinates of particle furthest
            from the COG.
        """
        cog = self.center_of_gravity(list_of_particles)
        cog_x, cog_y = cog["x"], cog["y"]

        max_distance = 0
        furthest_particle = None
        for particle in list_of_particles:
            px = particle["pose"]["position"]["x"]
            py = particle["pose"]["position"]["y"]
            distance = np.sqrt((px - cog_x) ** 2 + (py - cog_y) ** 2)
            if distance > max_distance:
                max_distance = distance
                furthest_particle = (px, py)

        return max_distance, furthest_particle

    def cog_mean_dist(self, list_of_particles: list[dict]) -> float:
        """Calculates mean distance of all particles from center of gravity.

        Args:
            list_of_particles: List of dictionaries representing particles.

        Returns:
            float: The mean distance of particles from the center of gravity.
        """
        num_particles = len(list_of_particles)
        if num_particles == 0:
            return 0.0

        cog = self.calculate_cog_mean(list_of_particles)
        cog_x, cog_y = cog["x"], cog["y"]

        distances = []
        for particle in list_of_particles:
            px = particle["pose"]["position"]["x"]
            py = particle["pose"]["position"]["y"]
            dist = math.sqrt((px - cog_x) ** 2 + (py - cog_y) ** 2)
            distances.append(dist)

        return sum(distances) / num_particles

    def cog_mean_absolute_deviation(
        self,
        list_of_particles: list[dict],
    ) -> float:
        """Get mean absolute deviation of distances to center of gravity mean.

        Args:
            list_of_particles: List of dictionaries representing particles.

        Returns:
            float: The mean absolute deviation of distances.
        """
        self.cog_mean = self.calculate_cog_mean(list_of_particles)
        mean_x, mean_y = self.cog_mean["x"], self.cog_mean["y"]

        distances = [
            np.sqrt(
                (particle["pose"]["position"]["x"] - mean_x) ** 2
                + (particle["pose"]["position"]["y"] - mean_y) ** 2,
            )
            for particle in list_of_particles
        ]

        # Calculate mean absolute deviation
        mean_distance = sum(distances) / len(distances)

        return sum(abs(d - mean_distance) for d in distances) / len(distances)

    def cog_median(self, list_of_particles: list[dict]) -> float:
        """Get median of distances from all particles to cog mean.

        Args:
            list_of_particles: List of dictionaries representing particles.

        Returns:
            float: The median distance from particles to the COG mean.
        """
        cog_mean = self.calculate_cog_mean(list_of_particles)
        mean_x, mean_y = cog_mean["x"], cog_mean["y"]

        distances = [
            np.sqrt(
                (particle["pose"]["position"]["x"] - mean_x) ** 2
                + (particle["pose"]["position"]["y"] - mean_y) ** 2,
            )
            for particle in list_of_particles
        ]

        distances.sort()
        n = len(distances)
        if n == 0:
            return 0

        if n % 2 == 1:
            return distances[n // 2]

        return (distances[n // 2 - 1] + distances[n // 2]) / 2

    def cog_median_absolute_deviation(
        self,
        list_of_particles: list[dict],
    ) -> float:
        """Calculates the median absolute deviation (MAD).

        Calculates the median absolute deviation (MAD) from the median distance
        of particles to the center of gravity mean.

        Args:
            list_of_particles: List of dictionaries representing particles.

        Returns:
            float: The median absolute deviation of distances.
        """
        # calculate_cog_median to get the median distance
        median_distance = self.cog_median(list_of_particles)

        cog_mean = self.calculate_cog_mean(list_of_particles)
        mean_x, mean_y = cog_mean["x"], cog_mean["y"]

        distances = [
            np.sqrt(
                (particle["pose"]["position"]["x"] - mean_x) ** 2
                + (particle["pose"]["position"]["y"] - mean_y) ** 2,
            )
            for particle in list_of_particles
        ]

        # Calculate absolute deviations from the median distance
        absolute_deviations = [abs(d - median_distance) for d in distances]

        # Calculate the median of the absolute deviations
        absolute_deviations.sort()
        n_dev = len(absolute_deviations)
        if n_dev == 0:
            return 0

        if n_dev % 2 == 1:
            return absolute_deviations[n_dev // 2]

        return (
            absolute_deviations[n_dev // 2 - 1]
            + absolute_deviations[n_dev // 2]
        ) / 2

    def cog_min_dist(
        self,
        list_of_particles: list[dict],
    ) -> tuple[float, tuple | None]:
        """Calculates minimum distance from any particle to center of gravity.

        Args:
            list_of_particles: List of dictionaries representing particles.

        Returns:
            tuple: The minimum distance and coordinates of particle closest to
            the COG.
        """
        cog = self.center_of_gravity(list_of_particles)
        cog_x, cog_y = cog["x"], cog["y"]

        min_distance = float("inf")
        closest_particle = None
        for particle in list_of_particles:
            px = particle["pose"]["position"]["x"]
            py = particle["pose"]["position"]["y"]
            distance = np.sqrt((px - cog_x) ** 2 + (py - cog_y) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_particle = (px, py)

        return min_distance, closest_particle

    def cog_standard_deviation(self, list_of_particles: list[dict]) -> float:
        """Calculates standard deviation of distances to cog mean.

        Args:
            list_of_particles: List of dictionaries representing particles.

        Returns:
            float: The standard deviation of distances.
        """
        cog_mean = self.calculate_cog_mean(list_of_particles)
        mean_x, mean_y = cog_mean["x"], cog_mean["y"]

        distances = [
            np.sqrt(
                (particle["pose"]["position"]["x"] - mean_x) ** 2
                + (particle["pose"]["position"]["y"] - mean_y) ** 2,
            )
            for particle in list_of_particles
        ]

        variance = sum((d - np.mean(distances)) ** 2 for d in distances) / len(
            distances,
        )

        return np.sqrt(variance)

    def smallest_enclosing_circle(self, points: list[tuple]) -> tuple:
        """Find the smallest enclosing circle for a set of points."""
        shuffled_points = points[:]
        random.shuffle(shuffled_points)
        return self.welzl(shuffled_points)

    def circle_mean(self, list_of_particles: list[dict]) -> float:
        """Calculate mean of distances from the circle center to the points."""
        points = [
            (
                particle["pose"]["position"]["x"],
                particle["pose"]["position"]["y"],
            )
            for particle in list_of_particles
        ]
        circle = self.smallest_enclosing_circle(points)
        center, _ = circle

        total_distance = sum(self.dist(center, point) for point in points)
        return total_distance / len(points)

    def circle_mean_absolute_deviation(
        self,
        list_of_particles: list[dict],
    ) -> float:
        """Get mean absolute deviation of distances from circle center."""
        points = [
            (
                particle["pose"]["position"]["x"],
                particle["pose"]["position"]["y"],
            )
            for particle in list_of_particles
        ]
        circle = self.smallest_enclosing_circle(points)
        center, _ = circle

        distances = [self.dist(center, point) for point in points]
        mean_distance = sum(distances) / len(points)

        return sum(abs(d - mean_distance) for d in distances) / len(points)

    def circle_median(self, list_of_particles: list[dict]) -> float:
        """Calculate median of distances from the circle center to points."""
        points = [
            (
                particle["pose"]["position"]["x"],
                particle["pose"]["position"]["y"],
            )
            for particle in list_of_particles
        ]
        circle = self.smallest_enclosing_circle(points)
        center, _ = circle

        distances = sorted(self.dist(center, point) for point in points)
        n = len(distances)

        if n == 0:
            return 0

        if n % 2 == 1:
            return distances[n // 2]
        return (distances[n // 2 - 1] + distances[n // 2]) / 2

    def circle_median_absolute_deviation(
        self,
        list_of_particles: list[dict],
    ) -> float:
        """Get median absolute deviation of distances from circle center."""
        points = [
            (
                particle["pose"]["position"]["x"],
                particle["pose"]["position"]["y"],
            )
            for particle in list_of_particles
        ]
        circle = self.smallest_enclosing_circle(points)
        center, _ = circle

        distances = [self.dist(center, point) for point in points]
        median_distance = self.circle_median(list_of_particles)

        # Compute median absolute deviation
        abs_deviation = sorted(abs(d - median_distance) for d in distances)
        n = len(abs_deviation)

        return (
            abs_deviation[n // 2]
            if n % 2 == 1
            else (abs_deviation[n // 2 - 1] + abs_deviation[n // 2]) / 2
        )

    def circle_min_dist(self, list_of_particles: list[dict]) -> float:
        """Get min distance between circle center and its closest particle."""
        points = [
            (
                particle["pose"]["position"]["x"],
                particle["pose"]["position"]["y"],
            )
            for particle in list_of_particles
        ]
        circle = self.smallest_enclosing_circle(points)
        center, _ = circle

        return min(self.dist(center, point) for point in points)

    def circle_std_deviation(self, list_of_particles: list[dict]) -> float:
        """Get standard deviation of distances from circle center to points."""
        points = [
            (
                particle["pose"]["position"]["x"],
                particle["pose"]["position"]["y"],
            )
            for particle in list_of_particles
        ]
        circle = self.smallest_enclosing_circle(points)
        center, _ = circle

        distances = [self.dist(center, point) for point in points]
        mean_distance = sum(distances) / len(points)

        variance = sum((d - mean_distance) ** 2 for d in distances) / len(
            points,
        )

        return math.sqrt(variance)

    def count_clusters(
        self,
        list_of_particles: list[dict],
        eps: float,
        min_samples: int,
    ) -> int:
        """Count the number of clusters in the particle cloud using DBSCAN."""
        positions = np.array(
            [
                (p["pose"]["position"]["x"], p["pose"]["position"]["y"])
                for p in list_of_particles
            ],
        )

        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(positions)

        # Count the number of unique clusters
        unique_labels = set(labels)

        return len(unique_labels) - (1 if -1 in unique_labels else 0)

    def main_cluster_variance_x(
        self,
        list_of_particles: list[dict],
        eps: float,
        min_samples: int,
    ) -> float:
        """Calculate the variance in the x-direction for the main cluster."""
        positions = np.array(
            [
                (p["pose"]["position"]["x"], p["pose"]["position"]["y"])
                for p in list_of_particles
            ],
        )

        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(positions)

        # Find the main cluster (largest non-noise cluster)
        unique_labels, counts = np.unique(
            labels[labels != -1],
            return_counts=True,
        )
        if len(unique_labels) == 0:
            return 0.0

        main_cluster_label = unique_labels[np.argmax(counts)]

        # Get points in the main cluster
        main_cluster_points = positions[labels == main_cluster_label]

        return float(np.var(main_cluster_points[:, 0]))

    def main_cluster_variance_y(
        self,
        list_of_particles: list[dict],
        eps: float,
        min_samples: int,
    ) -> float:
        """Calculate the variance in the y-direction for the main cluster."""
        positions = np.array(
            [
                (p["pose"]["position"]["x"], p["pose"]["position"]["y"])
                for p in list_of_particles
            ],
        )

        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(positions)

        # Find the main cluster (largest non-noise cluster)
        unique_labels, counts = np.unique(
            labels[labels != -1],
            return_counts=True,
        )
        if len(unique_labels) == 0:
            return 0.0

        main_cluster_label = unique_labels[np.argmax(counts)]

        # Get points in the main cluster
        main_cluster_points = positions[labels == main_cluster_label]

        return float(np.var(main_cluster_points[:, 1]))

    def extract_features_from_message(
        self,
        list_of_particles: list[dict],
        eps: float = 0.3,
        min_samples: int = 5,
    ) -> dict:
        """Extracts all features from a list of particles.

        Args:
            list_of_particles: List of dictionaries representing particles.
            eps: The maximum distance between two samples for one to be
                considered as in the neighborhood of the other.
            min_samples: The number of samples in a neighborhood for a point
                to be considered as a core point.

        Returns:
            dict: A dictionary containing all extracted features.
        """
        features = {}

        max_distance, furthest_particle = self.cog_max_dist(list_of_particles)
        features["cog_max_distance"] = max_distance

        features["cog_mean_dist"] = self.cog_mean_dist(list_of_particles)

        features["cog_mean_absolute_deviation"] = (
            self.cog_mean_absolute_deviation(
                list_of_particles,
            )
        )

        features["cog_median"] = self.cog_median(list_of_particles)

        features["cog_median_absolute_deviation"] = (
            self.cog_median_absolute_deviation(
                list_of_particles,
            )
        )

        min_distance, closest_particle = self.cog_min_dist(list_of_particles)
        features["cog_min_distance"] = min_distance

        features["cog_standard_deviation"] = self.cog_standard_deviation(
            list_of_particles,
        )

        points = [
            (
                particle["pose"]["position"]["x"],
                particle["pose"]["position"]["y"],
            )
            for particle in list_of_particles
        ]
        circle = self.smallest_enclosing_circle(points)
        features["circle_radius"] = circle[1]

        features["circle_mean"] = self.circle_mean(list_of_particles)

        features["circle_mean_absolute_deviation"] = (
            self.circle_mean_absolute_deviation(list_of_particles)
        )

        features["circle_median"] = self.circle_median(list_of_particles)

        features["circle_median_absolute_deviation"] = (
            self.circle_median_absolute_deviation(list_of_particles)
        )

        features["circle_min_distance"] = self.circle_min_dist(
            list_of_particles,
        )

        features["circle_standard_deviation"] = self.circle_std_deviation(
            list_of_particles,
        )

        features["num_clusters"] = self.count_clusters(
            list_of_particles,
            eps,
            min_samples,
        )

        features["main_cluster_variance_x"] = self.main_cluster_variance_x(
            list_of_particles,
            eps,
            min_samples,
        )

        features["main_cluster_variance_y"] = self.main_cluster_variance_y(
            list_of_particles,
            eps,
            min_samples,
        )

        return features

    ############### Helper Functions ################

    def center_of_gravity(self, list_of_particles: list[dict]) -> dict:
        """Get the center of gravity of particles based on their weights.

        Args:
        list_of_particles (list): List of dictionaries representing particles.

        Returns:
        dict: A dictionary with the x and y coordinates of center of gravity.
        """
        total_weight = sum(
            particle["weight"] for particle in list_of_particles
        )
        if total_weight == 0:
            return {"x": 0, "y": 0}  # Default to origin if no weight

        cog_x = (
            sum(
                particle["pose"]["position"]["x"] * particle["weight"]
                for particle in list_of_particles
            )
            / total_weight
        )
        cog_y = (
            sum(
                particle["pose"]["position"]["y"] * particle["weight"]
                for particle in list_of_particles
            )
            / total_weight
        )

        return {"x": cog_x, "y": cog_y}

    def calculate_cog_mean(self, list_of_particles: list[dict]) -> dict:
        """Calculates mean position (cog mean) over all particles.

        Args:
        list_of_particles (list): List of dictionaries representing particles.

        Returns:
        dict: A dictionary with the x and y coordinates of the mean position.
        """
        num_particles = len(list_of_particles)
        if num_particles == 0:
            return {"x": 0, "y": 0}  # Default to origin if no particles

        mean_x = (
            sum(
                particle["pose"]["position"]["x"]
                for particle in list_of_particles
            )
            / num_particles
        )
        mean_y = (
            sum(
                particle["pose"]["position"]["y"]
                for particle in list_of_particles
            )
            / num_particles
        )

        return {"x": mean_x, "y": mean_y}

    def welzl(  # noqa: PLR0911
        self,
        points: list[tuple],
        boundary: list[tuple[float, float]] | None = None,
    ) -> tuple:
        """Recursive Welzl's algorithm to find the minimum enclosing circle."""
        if boundary is None:
            boundary = []

        if not points or len(boundary) == 3:  # noqa: PLR2004
            if len(boundary) == 0:
                return ((0, 0), 0)
            if len(boundary) == 1:
                return (boundary[0], 0)
            if len(boundary) == 2:  # noqa: PLR2004
                return self.circle_from_two_points(boundary[0], boundary[1])
            if len(boundary) == 3:  # noqa: PLR2004
                try:
                    return self.circle_from_three_points(
                        boundary[0],
                        boundary[1],
                        boundary[2],
                    )
                except ValueError:
                    return self.circle_from_two_points(
                        boundary[0],
                        boundary[1],
                    )

        p = points.pop()
        circle = self.welzl(points.copy(), boundary)

        if self.is_in_circle(p, circle):
            points.append(p)
            return circle

        boundary.append(p)
        circle = self.welzl(points.copy(), boundary)
        boundary.pop()
        points.append(p)
        return circle

    def circle_from_two_points(
        self,
        p1: tuple[float, float],
        p2: tuple[float, float],
    ) -> tuple:
        """Return the smallest circle from two points."""
        center = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        radius = self.dist(p1, p2) / 2
        return (center, radius)

    def circle_from_three_points(
        self,
        p1: tuple[float, float],
        p2: tuple[float, float],
        p3: tuple[float, float],
    ) -> tuple:
        """Return the smallest circle from three points."""
        ax, ay = p1
        bx, by = p2
        cx, cy = p3

        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if d == 0:
            msg = "Collinear points"
            raise ValueError(msg)

        ux = (
            (ax**2 + ay**2) * (by - cy)
            + (bx**2 + by**2) * (cy - ay)
            + (cx**2 + cy**2) * (ay - by)
        ) / d
        uy = (
            (ax**2 + ay**2) * (cx - bx)
            + (bx**2 + by**2) * (ax - cx)
            + (cx**2 + cy**2) * (bx - ax)
        ) / d
        center = (ux, uy)
        radius = self.dist(center, p1)
        return (center, radius)

    def dist(self, p1: tuple[float, float], p2: tuple[float, float]) -> float:
        """Calculate the Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def is_in_circle(
        self,
        point: tuple[float, float],
        circle: tuple[tuple[float, float], float],
    ) -> bool:
        """Check if a point is inside or on the boundary of a circle."""
        center, radius = circle
        return self.dist(point, center) <= radius
