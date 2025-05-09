import bisect

import cv2
import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from numba import njit
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from typing_extensions import override

from flowcean.core.transform import Transform

# Constants for raycasting optimization
RAYCAST_BATCH_SIZE = (
    512  # Process rays in batches for better cache utilization
)
# Threshold for considering a cell occupied in the occupancy grid (0-100)
OCCUPANCY_THRESHOLD = 50

# Constants for line feature computation
MAX_LINE_DISTANCE = 0.1  # Maximum distance for matching lines (meters)
MAX_ANGLE_DIFF = 5  # Maximum angle difference for matching lines (degrees)

# Constants for raycasting feature computation
RAYCAST_TOLERANCE = 0.1  # meters
RAYCAST_EPSILON = 0.05  # meters


class ScanMapStatistics(Transform):
    """Computes features based on comparing a Laserscan to an occupancy map."""

    def __init__(
        self,
        occupancy_map: dict,
        scan_topic: str = "/scan",
        sensor_pose_topic: str = "/amcl_pose",
        *,
        plotting: bool = False,
    ) -> None:
        """Initializes the ScanMap transform.

        Args:
            occupancy_map: The occupancy map data.
            scan_topic: The name of the topic providing LaserScan data.
            sensor_pose_topic: The name of the topic providing the pose of the
                sensor.
            plotting: A flag to enable or disable plotting of the detected
                lines and scan points.
        """
        self.scan_topic = scan_topic
        self.sensor_pose_topic = sensor_pose_topic
        self.plotting = plotting
        self.precomputed_map_lines: dict[str, np.ndarray] | None = None
        self.occupancy_map = occupancy_map

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        collected_data = data.collect()
        collected_data = self.compute_lidar_scan_points(collected_data)

        # Precompute map line parameters once
        if self.precomputed_map_lines is None:
            self._precompute_map_line_parameters()

        collected_data = self.derive_scan_point_features(collected_data)
        return collected_data.lazy()

    def _precompute_map_line_parameters(self) -> dict:
        """Precompute all map-related line parameters once."""
        map_array = self.occupancy_map["data"]
        width = self.occupancy_map["info.width"]
        height = self.occupancy_map["info.height"]
        occupancy_grid = np.array(map_array).reshape((height, width))
        detected_lines = self.detect_lines_from_grid(occupancy_grid)
        lines_np = (
            np.array(detected_lines) if detected_lines else np.empty((0, 4))
        )
        if lines_np.size == 0:
            self.precomputed_map_lines = {
                "lines_np": np.empty((0, 4)),
                "x1": np.empty(0),
                "y1": np.empty(0),
                "x2": np.empty(0),
                "y2": np.empty(0),
                "line_vec": np.empty((0, 2)),
                "line_normals": np.empty((0, 2)),
                "line_start": np.empty((0, 2)),
                "line_length_squared": np.empty(0),
            }
        # Convert to float64 when extracting coordinates
        x1 = lines_np[:, 0].astype(np.float64)
        y1 = lines_np[:, 1].astype(np.float64)
        x2 = lines_np[:, 2].astype(np.float64)
        y2 = lines_np[:, 3].astype(np.float64)
        line_vec = np.column_stack((x2 - x1, y2 - y1))
        line_normals = np.column_stack((-line_vec[:, 1], line_vec[:, 0]))
        line_normals /= np.linalg.norm(line_normals, axis=1, keepdims=True)
        self.precomputed_map_lines = {
            "lines_np": lines_np,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "line_vec": line_vec,
            "line_normals": line_normals,
            "line_start": lines_np[:, :2],
            "line_length_squared": np.sum(line_vec**2, axis=1),
        }
        return self.precomputed_map_lines

    @staticmethod
    @njit(cache=True, fastmath=True)
    def _vectorized_distance_calculation(
        points: np.ndarray,
        line_start: np.ndarray,
        line_vec: np.ndarray,
        line_length_squared: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Numba-optimized distance calculation."""
        n_points = points.shape[0]
        n_lines = line_start.shape[0]
        min_distances = np.empty(n_points, dtype=np.float32)
        closest_indices = np.empty(n_points, dtype=np.int32)

        for i in range(n_points):
            point = points[i]
            min_dist = float("inf")
            min_idx = -1
            for j in range(n_lines):
                vec_to_point = point - line_start[j]
                t = np.dot(vec_to_point, line_vec[j]) / line_length_squared[j]
                t = max(0.0, min(1.0, t))
                projection = line_start[j] + t * line_vec[j]
                dx = point[0] - projection[0]
                dy = point[1] - projection[1]
                dist = np.sqrt(dx**2 + dy**2)

                if dist < min_dist:
                    min_dist = dist
                    min_idx = j

            min_distances[i] = min_dist
            closest_indices[i] = min_idx

        return min_distances, closest_indices

    def _get_closest_line_distance_details(
        self,
        scan_pts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Optimized version using precomputed map lines."""
        if self.precomputed_map_lines is None or scan_pts.size == 0:
            return np.array([]), np.array([])

        params = self.precomputed_map_lines
        return self._vectorized_distance_calculation(
            points=scan_pts.astype(np.float32),
            line_start=params["line_start"].astype(np.float32),
            line_vec=params["line_vec"].astype(np.float32),
            line_length_squared=params["line_length_squared"].astype(
                np.float32,
            ),
        )

    @staticmethod
    @njit(cache=True)
    def _batched_raycast(
        sensor_x: float,
        sensor_y: float,
        angles: np.ndarray,
        map_origin_x: float,
        map_origin_y: float,
        map_resolution: float,
        grid: np.ndarray,
        max_range: float,
    ) -> np.ndarray:
        """Numba-optimized batched raycasting."""
        ranges = np.empty(len(angles), dtype=np.float32)
        grid_height, grid_width = grid.shape

        for i in range(len(angles)):
            angle = angles[i]
            x0 = (sensor_x - map_origin_x) / map_resolution
            y0 = (sensor_y - map_origin_y) / map_resolution
            x1 = x0 + (max_range * np.cos(angle)) / map_resolution
            y1 = y0 + (max_range * np.sin(angle)) / map_resolution

            # Bresenham's algorithm implementation
            dx = abs(int(x1) - int(x0))
            dy = -abs(int(y1) - int(y0))
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx + dy
            current_x, current_y = int(x0), int(y0)
            found = False

            for _ in range(max(dx, dy) * 2):
                if (
                    0 <= current_x < grid_width
                    and 0 <= current_y < grid_height
                ) and (grid[current_y, current_x] > OCCUPANCY_THRESHOLD):
                    found = True
                    break
                if current_x == int(x1) and current_y == int(y1):
                    break
                e2 = 2 * err
                if e2 >= dy:
                    err += dy
                    current_x += sx
                if e2 <= dx:
                    err += dx
                    current_y += sy

            ranges[i] = (
                (
                    np.hypot(
                        current_x * map_resolution + map_origin_x - sensor_x,
                        current_y * map_resolution + map_origin_y - sensor_y,
                    )
                )
                if found
                else max_range
            )

        return ranges

    def compute_lidar_scan_points(self, data: pl.DataFrame) -> pl.DataFrame:
        self.scan_timeseries = data[self.scan_topic].to_list()[0]
        amcl_pose_timeseries = data[self.sensor_pose_topic].to_list()[0]

        # Precompute poses into (time, x, y, theta)
        pose_entries = []
        for entry in tqdm(amcl_pose_timeseries, "Precomputing poses"):
            pose_data = entry["value"]
            x = pose_data["pose.pose.position.x"]
            y = pose_data["pose.pose.position.y"]
            orientation_x = pose_data["pose.pose.orientation.x"]
            orientation_y = pose_data["pose.pose.orientation.y"]
            orientation_z = pose_data["pose.pose.orientation.z"]
            orientation_w = pose_data["pose.pose.orientation.w"]
            theta = Rotation.from_quat(
                [orientation_x, orientation_y, orientation_z, orientation_w],
            ).as_euler("xyz", degrees=False)[2]
            pose_entries.append((entry["time"], x, y, theta))
        pose_times = [entry[0] for entry in pose_entries]

        scan_points_timeseries = []  # Map-frame scan points
        scan_points_sensor_timeseries = []  # Sensor-frame scan points
        self.synced_sensor_poses = []
        for scan in tqdm(self.scan_timeseries, "Computing scan points"):
            timestamp = scan["time"]
            # Binary search for the latest pose before timestamp
            idx = bisect.bisect_right(pose_times, timestamp) - 1
            if idx >= 0:
                _, x, y, theta = pose_entries[idx]
                # Compute scan points in both frames
                scan_points_map, scan_points_sensor = (
                    self.calculate_scan_points(
                        pose=(x, y, theta),
                        ranges=scan["value"]["ranges"],
                        angle_min=scan["value"]["angle_min"],
                        angle_max=scan["value"]["angle_max"],
                        angle_increment=scan["value"]["angle_increment"],
                        range_min=scan["value"]["range_min"],
                        range_max=scan["value"]["range_max"],
                    )
                )
                scan_points_timeseries.append(
                    {"time": timestamp, "value": scan_points_map.tolist()},
                )
                scan_points_sensor_timeseries.append(
                    {"time": timestamp, "value": scan_points_sensor.tolist()},
                )
                self.synced_sensor_poses.append((x, y, theta))
        return data.hstack(
            pl.DataFrame(
                {
                    "scan_points": [scan_points_timeseries],
                    "scan_points_sensor": [scan_points_sensor_timeseries],
                },
            ),
        )

    def _get_closest_line_distance(
        self,
        lines_np: np.ndarray,
        line_length_squared: np.ndarray,
        line_vec_expanded: np.ndarray,
        line_start_expanded: np.ndarray,
        has_zero_length: np.ndarray,
        scan_pts: np.ndarray,
    ) -> float:
        if scan_pts.size == 0:
            distance = 0.0
        elif lines_np.size == 0:
            distance = float("inf") if scan_pts.shape[0] > 0 else 0.0
        else:
            n_points = scan_pts.shape[0]
            points = scan_pts.reshape(n_points, 1, 2)  # (n_points, 1, 2)

            # Compute point_vec (n_points, M, 2)
            point_vec = points - line_start_expanded

            # Calculate numerator and t
            numerator = np.sum(point_vec * line_vec_expanded, axis=2)
            denominator = line_length_squared + 1e-12
            t = numerator / denominator
            t = np.clip(t, 0, 1)

            # Projection points (n_points, M, 2)
            proj = line_start_expanded + t[..., np.newaxis] * line_vec_expanded

            # Calculate distances
            dx = points[:, :, 0] - proj[:, :, 0]
            dy = points[:, :, 1] - proj[:, :, 1]
            distances = np.sqrt(dx**2 + dy**2)

            # Handle zero-length lines
            if np.any(has_zero_length):
                zero_line_dist = np.linalg.norm(
                    point_vec[:, has_zero_length, :],
                    axis=2,
                )
                distances[:, has_zero_length] = zero_line_dist

            min_distances = np.min(distances, axis=1)
            distance = np.mean(min_distances) if n_points > 0 else 0.0
        return distance

    def detect_lines_from_grid(
        self,
        occupancy_grid: np.ndarray,
        threshold1: int = 50,
        threshold2: int = 150,
        hough_threshold: int = 100,
        min_line_length: int = 30,
        max_line_gap: int = 5,
    ) -> list[tuple[float, float, float, float]]:
        """Perform the Hough Transform to find lines from an occupancy grid.

        Steps:
        1. Apply edge detection to the image.
        2. Map image points into Hough space using an accumulator.
        3. Detect lines (local maxima in the Hough space) with thresholds.
        4. Convert infinite lines into finite ones using Progressive
        Probabilistic Hough Transform.

        Parameters:
            occupancy_grid: A 2D binary occupancy grid.
            threshold1: First threshold for the Canny edge detector.
            threshold2: Second threshold for the Canny edge detector.
            hough_threshold: Accumulator threshold for Hough Transform.
            min_line_length: Minimum length of detected lines.
            max_line_gap: Maximum allowed gap between line segments.

        Returns:
            list: Detected finite lines represented as (x1, y1, x2, y2).
        """
        # Detect edges using Canny Edge Detector
        image = np.uint8(
            occupancy_grid * 2.55,
        )  # Scale grid to 0-255 for edge detection
        edges = cv2.Canny(image, threshold1, threshold2)  # type: ignore  # noqa: PGH003

        # OpenCV creates an accumulator internally with `cv2.HoughLinesP`

        # Detect lines using accumulator with thresholds
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=hough_threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap,
        )

        # Convert infinite lines into finite ones
        detected_lines = []
        if lines is not None:
            for line in tqdm(lines, desc="Detecting lines"):
                x1, y1, x2, y2 = line.ravel()
                detected_lines.append((x1, y1, x2, y2))

        return detected_lines

    def point_distance(self, map_lines: list, scan: np.ndarray) -> float:
        """Calculates the average distance of a scan point to a map line.

        Args:
            map_lines: A list of lines, where each line is represented
                as (x1, y1, x2, y2) (endpoints of the line).
            scan : An Nx2 array of points with x, y coordinates.

        Returns:
            float: The average distance of all scan points to the nearest line.
        """

        def distance_point_to_line(
            px: float,
            py: float,
            x1: float,
            y1: float,
            x2: float,
            y2: float,
        ) -> float:
            """Calculate the shortest distance from a point to a line."""
            line_vec = np.array([x2 - x1, y2 - y1])
            point_vec = np.array([px - x1, py - y1])
            line_length_squared = np.dot(line_vec, line_vec)

            # Handle the case where the line length is 0
            if line_length_squared == 0:
                return float(np.linalg.norm(point_vec))

            # Projection scalar (normalized distance along the line segment)
            t = max(
                0,
                min(1, np.dot(point_vec, line_vec) / line_length_squared),
            )

            # Projection point on the line segment
            projection = np.array([x1, y1]) + np.array(t) * line_vec
            return float(np.linalg.norm(np.array([px, py]) - projection))

        total_distance = 0.0
        num_points = scan.shape[0]

        for px, py in scan:
            min_distance = float("inf")
            # Calculate the distance to each map line and take the minimum
            for line in map_lines:
                x1, y1, x2, y2 = line
                dist = distance_point_to_line(px, py, x1, y1, x2, y2)
                min_distance = min(min_distance, dist)
            total_distance += min_distance

        return total_distance / num_points if num_points > 0 else 0.0

    def calculate_scan_points(
        self,
        pose: tuple[float, float, float],
        angle_min: float,
        angle_max: float,
        angle_increment: float,
        range_min: float,
        range_max: float,
        ranges: list[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculates the coordinates of scan points from a LaserScan message.

        Args:
            pose: The (x, y, theta) pose of the sensor in the map frame.
            angle_min: The start angle of the scan [rad].
            angle_max: The end angle of the scan [rad].
            angle_increment: Angular distance between measurements [rad].
            range_min: Minimum range value [m].
            range_max: Maximum range value [m].
            ranges: Measured distances for each scan [m].

        Returns:
            np.ndarray: Nx2 array of scan points in the map frame.
        """
        x_sensor, y_sensor, theta_sensor = pose

        # Compute the angles for each scan point
        angles = np.arange(angle_min, angle_max, angle_increment)

        # Filter ranges to exclude invalid readings
        valid_mask = (np.array(ranges) >= range_min) & (
            np.array(ranges) <= range_max
        )
        valid_ranges = np.array(ranges)[valid_mask]
        valid_angles = angles[valid_mask]

        # Sensor-frame coordinates (local to the sensor)
        x_sensor_frame = valid_ranges * np.cos(valid_angles)
        y_sensor_frame = valid_ranges * np.sin(valid_angles)
        scan_points_sensor = np.column_stack((x_sensor_frame, y_sensor_frame))

        # Map-frame coordinates (transformed from sensor frame)
        x_map = (
            x_sensor
            + x_sensor_frame * np.cos(theta_sensor)
            - y_sensor_frame * np.sin(theta_sensor)
        )
        y_map = (
            y_sensor
            + x_sensor_frame * np.sin(theta_sensor)
            + y_sensor_frame * np.cos(theta_sensor)
        )
        scan_points_map = np.column_stack((x_map, y_map))

        return scan_points_map, scan_points_sensor

    def plot_detected_lines(
        self,
        occupancy_grid: np.ndarray,
        lines: list[tuple[float, float, float, float]],
    ) -> None:
        """Plot the occupancy grid and overlay the detected lines."""
        plt.figure(figsize=(8, 6))

        plt.imshow(occupancy_grid, cmap="gray")

        # Overlay detected lines
        for line in lines:
            plt.plot(
                [line[0], line[2]],
                [line[1], line[3]],
                color="red",
                linewidth=2,
            )

        plt.title("Detected Lines Over Occupancy Grid")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("detected_lines.png")

    def derive_scan_point_features(self, data: pl.DataFrame) -> pl.DataFrame:  # noqa: PLR0915 this cannot be split
        """Derive features based on the scan points and the occupancy map."""
        map_array = self.occupancy_map["data"]
        width = self.occupancy_map["info.width"]
        height = self.occupancy_map["info.height"]
        map_resolution = self.occupancy_map["info.resolution"]
        map_origin_x = self.occupancy_map["info.origin.position.x"]
        map_origin_y = self.occupancy_map["info.origin.position.y"]
        occupancy_grid = np.array(map_array).reshape((height, width))
        detected_lines = self.detect_lines_from_grid(
            occupancy_grid,
            threshold1=50,
            threshold2=150,
            hough_threshold=75,
            min_line_length=10,
            max_line_gap=10,
        )
        if self.precomputed_map_lines is None:
            lines_np = self._precompute_map_line_parameters()["lines_np"]
        else:
            lines_np = self.precomputed_map_lines["lines_np"]
        complete_scan_timeseries = data[self.scan_topic].to_list()[0]
        scan_points_timeseries = data["scan_points"][0].to_list()
        point_distance_timeseries = []
        point_fitting_timeseries = []
        point_inlier_timeseries = []
        point_quality_timeseries = []
        ray_inlier_timeseries = []
        ray_inlier_percent_timeseries = []
        ray_matching_percent_timeseries = []
        ray_outlier_percent_timeseries = []
        ray_quality_timeseries = []
        angle_inlier_timeseries = []
        angle_quality_timeseries = []
        line_angle_timeseries = []
        line_distance_timeseries = []
        line_fitting_timeseries = []
        line_length_timeseries = []
        for i, scan in enumerate(
            tqdm(
                scan_points_timeseries,
                desc="Computing scan-map features",
            ),
        ):
            scan_pts = np.array(scan["value"])

            valid_mask = complete_scan_timeseries[i]["value"]["ranges"] != 0
            valid_angles = np.arange(
                complete_scan_timeseries[i]["value"]["angle_min"],
                complete_scan_timeseries[i]["value"]["angle_max"],
                complete_scan_timeseries[i]["value"]["angle_increment"],
            )
            sensor_x, sensor_y, theta_sensor = self.synced_sensor_poses[i]
            min_distances, closest_line_indices = (
                self._get_closest_line_distance_details(
                    scan_pts,
                )
            )
            n_points = len(min_distances)

            avg_distance = np.mean(min_distances) if n_points > 0 else 0.0
            # Feature 15: Point distance
            point_distance_timeseries.append(
                {"time": scan["time"], "value": avg_distance},
            )

            # Feature 16: Point fitting percentage
            threshold = np.median(min_distances) if n_points > 0 else 0.0
            fitting_percent = (
                np.sum(min_distances <= threshold) / n_points * 100
                if n_points > 0
                else 0.0
            )
            point_fitting_timeseries.append(
                {"time": scan["time"], "value": fitting_percent},
            )

            # Feature 17: Point inlier percentage
            inlier_count = self._compute_point_inliers(
                scan_pts,
                lines_np,
                closest_line_indices,
                sensor_x,
                sensor_y,
            )
            inlier_percent = (
                (inlier_count / n_points * 100) if n_points > 0 else 0.0
            )
            point_inlier_timeseries.append(
                {"time": scan["time"], "value": inlier_percent},
            )

            # Feature 18: Point quality
            qualities = 1.0 / (1.0 + min_distances)
            avg_quality = np.mean(qualities) if n_points > 0 else 0.0
            point_quality_timeseries.append(
                {"time": scan["time"], "value": avg_quality},
            )

            # Raycasting features (19-23)
            raycasted_ranges = self._perform_raycasting(
                sensor_x,
                sensor_y,
                theta_sensor,
                valid_angles,
                map_resolution,
                map_origin_x,
                map_origin_y,
                occupancy_grid,
                complete_scan_timeseries[i]["value"]["range_max"],
            )
            actual_ranges = np.array(
                complete_scan_timeseries[i]["value"]["ranges"],
            )[valid_mask]

            if len(actual_ranges) == 0:
                ray_inlier = ray_inlier_percent = ray_matching = (
                    ray_outlier
                ) = ray_quality = 0.0
            else:
                # Feature 19: Raycasting inlier
                inlier_mask = (
                    actual_ranges >= (raycasted_ranges - RAYCAST_TOLERANCE)
                ) & (actual_ranges <= (raycasted_ranges + RAYCAST_TOLERANCE))
                ray_inlier = np.sum(inlier_mask) / len(actual_ranges) * 100
                # Feature 20: Raycasting inlier percentage (actual < expected)
                ray_inlier_percent = (
                    np.sum(actual_ranges < raycasted_ranges)
                    / len(actual_ranges)
                    * 100
                )
                # Feature 21: Raycasting matching percentage
                matching_mask = (
                    np.abs(actual_ranges - raycasted_ranges) < RAYCAST_EPSILON
                )
                ray_matching = np.sum(matching_mask) / len(actual_ranges) * 100
                # Feature 22: Raycasting outlier percentage
                outlier_mask = actual_ranges > (
                    raycasted_ranges + RAYCAST_TOLERANCE
                )
                ray_outlier = np.sum(outlier_mask) / len(actual_ranges) * 100
                # Feature 23: Raycasting quality
                ray_quality = (
                    (np.sum(inlier_mask) + np.sum(matching_mask))
                    / len(actual_ranges)
                    * 100
                )

            ray_inlier_timeseries.append(
                {"time": scan["time"], "value": ray_inlier},
            )
            ray_inlier_percent_timeseries.append(
                {"time": scan["time"], "value": ray_inlier_percent},
            )
            ray_matching_percent_timeseries.append(
                {"time": scan["time"], "value": ray_matching},
            )
            ray_outlier_percent_timeseries.append(
                {"time": scan["time"], "value": ray_outlier},
            )
            ray_quality_timeseries.append(
                {"time": scan["time"], "value": ray_quality},
            )

            # Line-based features (24-29)
            detected_scan_lines = self.detect_lines_from_points(
                scan_pts,
                map_origin_x,
                map_origin_y,
                map_resolution,
                width,
                height,
            )

            # Feature 24: Angle inliers
            angle_inlier = self._compute_angle_inliers(
                detected_scan_lines,
                detected_lines,
                sensor_x,
                sensor_y,
            )
            angle_inlier_timeseries.append(
                {"time": scan["time"], "value": angle_inlier},
            )

            # Feature 25: Angle quality (simplified as average angle match)
            angle_quality = self._compute_angle_quality(
                detected_scan_lines,
                detected_lines,
            )
            angle_quality_timeseries.append(
                {"time": scan["time"], "value": angle_quality},
            )

            # Features 26-29
            line_angle, line_distance, line_fitting, line_length = (
                self._compute_line_features(
                    detected_scan_lines,
                    detected_lines,
                )
            )
            line_angle_timeseries.append(
                {"time": scan["time"], "value": line_angle},
            )
            line_distance_timeseries.append(
                {"time": scan["time"], "value": line_distance},
            )
            line_fitting_timeseries.append(
                {"time": scan["time"], "value": line_fitting},
            )
            line_length_timeseries.append(
                {"time": scan["time"], "value": line_length},
            )

        return data.hstack(
            pl.DataFrame(
                {
                    "point_distance": [point_distance_timeseries],
                    "point_fitting": [point_fitting_timeseries],
                    "point_inlier": [point_inlier_timeseries],
                    "point_quality": [point_quality_timeseries],
                    "ray_inlier": [ray_inlier_timeseries],
                    "ray_inlier_percent": [ray_inlier_percent_timeseries],
                    "ray_matching_percent": [ray_matching_percent_timeseries],
                    "ray_outlier_percent": [ray_outlier_percent_timeseries],
                    "ray_quality": [ray_quality_timeseries],
                    "angle_inlier": [angle_inlier_timeseries],
                    "angle_quality": [angle_quality_timeseries],
                    "line_angle": [line_angle_timeseries],
                    "line_distance": [line_distance_timeseries],
                    "line_fitting": [line_fitting_timeseries],
                    "line_length": [line_length_timeseries],
                },
            ),
        )

    def _compute_point_inliers(
        self,
        scan_pts: np.ndarray,
        lines_np: np.ndarray,
        closest_line_indices: np.ndarray,
        sensor_x: float,
        sensor_y: float,
    ) -> int:
        """Compute the number of inliers for each scan point."""
        if lines_np.size == 0 or scan_pts.size == 0:
            return 0
        # Convert to float64 when creating line_normals
        line_normals = np.column_stack(
            (
                -(lines_np[:, 3] - lines_np[:, 1]),
                lines_np[:, 2] - lines_np[:, 0],
            ),
        ).astype(np.float64)  # Add this explicit conversion

        line_normals /= np.linalg.norm(line_normals, axis=1, keepdims=True)
        sensor_vec = np.array(
            [sensor_x, sensor_y],
            dtype=np.float64,
        ) - lines_np[:, :2].astype(np.float64)
        sensor_signs = np.einsum("ij,ij->i", sensor_vec, line_normals)
        point_vec = scan_pts.astype(np.float64) - lines_np[
            closest_line_indices,
            :2,
        ].astype(np.float64)
        point_signs = np.einsum(
            "ij,ij->i",
            point_vec,
            line_normals[closest_line_indices],
        )
        inlier_mask = np.sign(point_signs) == np.sign(
            sensor_signs[closest_line_indices],
        )
        return np.sum(inlier_mask)

    def _perform_raycasting(
        self,
        sensor_x: float,
        sensor_y: float,
        theta_sensor: float,
        valid_angles: np.ndarray,
        map_resolution: float,
        map_origin_x: float,
        map_origin_y: float,
        occupancy_grid: np.ndarray,
        range_max: float,
    ) -> np.ndarray:
        """Optimized raycasting using numba-accelerated batch processing."""
        global_angles = theta_sensor + valid_angles
        return self._batched_raycast(
            sensor_x=sensor_x,
            sensor_y=sensor_y,
            angles=global_angles.astype(np.float32),
            map_origin_x=map_origin_x,
            map_origin_y=map_origin_y,
            map_resolution=map_resolution,
            grid=occupancy_grid.astype(np.uint8),
            max_range=range_max,
        )

    def _raycast(
        self,
        sensor_x: float,
        sensor_y: float,
        angle: float,
        map_origin_x: float,
        map_origin_y: float,
        map_resolution: float,
        grid: np.ndarray,
        max_range: float,
    ) -> float:
        """Raycast from the sensor to the map."""
        x0 = (sensor_x - map_origin_x) / map_resolution
        y0 = (sensor_y - map_origin_y) / map_resolution
        x1 = (
            sensor_x + max_range * np.cos(angle) - map_origin_x
        ) / map_resolution
        y1 = (
            sensor_y + max_range * np.sin(angle) - map_origin_y
        ) / map_resolution
        x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])

        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        current_x, current_y = x0, y0

        while True:
            if (
                0 <= current_x < grid.shape[1]
                and 0 <= current_y < grid.shape[0]
            ) and grid[current_y, current_x] > OCCUPANCY_THRESHOLD:
                return np.hypot(
                    (current_x * map_resolution + map_origin_x - sensor_x),
                    (current_y * map_resolution + map_origin_y - sensor_y),
                )
            if current_x == x1 and current_y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                current_x += sx
            if e2 <= dx:
                err += dx
                current_y += sy
        return max_range

    def detect_lines_from_points(
        self,
        points: np.ndarray,
        map_origin_x: float,
        map_origin_y: float,
        map_resolution: float,
        width: int,
        height: int,
    ) -> list[tuple[float, float, float, float]]:
        if len(points) == 0:
            return []
        image = np.zeros((height, width), dtype=np.uint8)
        xs = ((points[:, 0] - map_origin_x) / map_resolution).astype(int)
        ys = ((points[:, 1] - map_origin_y) / map_resolution).astype(int)
        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        np.add.at(image, (ys[valid], xs[valid]), 255)
        lines = cv2.HoughLinesP(
            image,
            1,
            np.pi / 180,
            50,
            minLineLength=10,
            maxLineGap=5,
        )
        if lines is None:
            return []
        detected = []
        for line in lines:
            x1, y1, x2, y2 = line.ravel()
            detected.append(
                (
                    x1 * map_resolution + map_origin_x,
                    y1 * map_resolution + map_origin_y,
                    x2 * map_resolution + map_origin_x,
                    y2 * map_resolution + map_origin_y,
                ),
            )
        return detected

    def _compute_angle_inliers(
        self,
        scan_lines: list[tuple[float, float, float, float]],
        map_lines: list[tuple[float, float, float, float]],
        sensor_x: float,
        sensor_y: float,
    ) -> float:
        if not scan_lines or not map_lines:
            return 0.0
        inlier_count = 0
        for s_line in scan_lines:
            s_mid = ((s_line[0] + s_line[2]) / 2, (s_line[1] + s_line[3]) / 2)
            s_dist = np.hypot(s_mid[0] - sensor_x, s_mid[1] - sensor_y)
            closest_dist = np.inf
            for m_line in map_lines:
                m_mid = (
                    (m_line[0] + m_line[2]) / 2,
                    (m_line[1] + m_line[3]) / 2,
                )
                m_dist = np.hypot(m_mid[0] - sensor_x, m_mid[1] - sensor_y)
                closest_dist = min(m_dist, closest_dist)
            if s_dist < closest_dist:
                inlier_count += 1
        return (inlier_count / len(scan_lines)) * 100

    def _compute_angle_quality(
        self,
        scan_lines: list[tuple[float, float, float, float]],
        map_lines: list[tuple[float, float, float, float]],
    ) -> float:
        if not scan_lines or not map_lines:
            return 0.0
        angles = []
        for s_line in scan_lines:
            s_angle = np.arctan2(s_line[3] - s_line[1], s_line[2] - s_line[0])
            closest_angle_diff = np.inf
            for m_line in map_lines:
                m_angle = np.arctan2(
                    m_line[3] - m_line[1],
                    m_line[2] - m_line[0],
                )
                angle_diff = np.abs(s_angle - m_angle)
                closest_angle_diff = min(angle_diff, closest_angle_diff)
            angles.append(closest_angle_diff)
        return np.degrees(np.mean(angles)) if angles else 0.0

    def _compute_line_features(
        self,
        scan_lines: list,
        map_lines: list,
    ) -> tuple:
        if not scan_lines:
            return 0.0, 0.0, 0.0, 0.0
        angle_diffs = []
        line_distances = []
        matching = 0
        lengths = []
        for s_line in scan_lines:
            s_angle = np.arctan2(s_line[3] - s_line[1], s_line[2] - s_line[0])
            s_length = np.hypot(s_line[2] - s_line[0], s_line[3] - s_line[1])
            lengths.append(s_length)
            min_dist = np.inf
            min_angle_diff = np.inf
            for m_line in map_lines:
                m_angle = np.arctan2(
                    m_line[3] - m_line[1],
                    m_line[2] - m_line[0],
                )
                angle_diff = np.abs(s_angle - m_angle)
                dist = self._line_distance(s_line, m_line)
                if dist < min_dist:
                    min_dist = dist
                    min_angle_diff = angle_diff
            angle_diffs.append(min_angle_diff)
            line_distances.append(min_dist)
            if (
                min_dist < MAX_LINE_DISTANCE
                and np.degrees(min_angle_diff) < MAX_ANGLE_DIFF
            ):
                matching += 1
        avg_angle = np.degrees(np.mean(angle_diffs)) if angle_diffs else 0.0
        avg_distance = np.mean(line_distances) if line_distances else 0.0
        fitting = (matching / len(scan_lines)) * 100 if scan_lines else 0.0
        avg_length = np.mean(lengths) if lengths else 0.0
        return avg_angle, avg_distance, fitting, avg_length

    def _line_distance(
        self,
        line1: tuple[float, float, float, float],
        line2: tuple[float, float, float, float],
    ) -> float:
        def dist(p1: tuple[float, float], p2: tuple[float, float]) -> float:
            return np.hypot(p1[0] - p2[0], p1[1] - p2[1])

        return min(
            dist(line1[:2], line2[:2]),
            dist(line1[:2], line2[2:]),
            dist(line1[2:], line2[:2]),
            dist(line1[2:], line2[2:]),
        )
