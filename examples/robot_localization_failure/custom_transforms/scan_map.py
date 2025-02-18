import bisect

import cv2
import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from typing_extensions import override

from flowcean.core.transform import Transform


class ScanMap(Transform):
    """Computes features based on comparing a Laserscan to an occupancy map.

    The ScanMap class is responsible for transforming and analyzing LaserScan
    data in conjunction with an occupancy grid map. By combining pose
    information from the sensor and scan data, this class computes scan point
    coordinates in the map frame and calculates distances between scan points
    and detected lines in the occupancy grid.
    """

    def __init__(
        self,
        scan_topic: str = "/scan",
        sensor_pose_topic: str = "/amcl_pose",
        *,
        plotting: bool = False,
    ) -> None:
        """Initializes the ScanMap transform.

        Args:
            scan_topic: The name of the topic providing LaserScan data.
            sensor_pose_topic: The name of the topic providing the pose of the
                sensor.
            plotting: A flag to enable or disable plotting of the detected
                lines and scan points.
        """
        self.scan_topic = scan_topic
        self.sensor_pose_topic = sensor_pose_topic
        self.plotting = plotting

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        collected_data = data.collect()
        collected_data = self.compute_lidar_scan_points(collected_data)
        collected_data = self.compute_scan_point_distances(collected_data)
        return collected_data.lazy()

    def compute_lidar_scan_points(
        self,
        data: pl.DataFrame,
    ) -> pl.DataFrame:
        scan_timeseries = data[self.scan_topic].to_list()[0]
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
                [
                    orientation_x,
                    orientation_y,
                    orientation_z,
                    orientation_w,
                ],
            ).as_euler("xyz", degrees=False)[2]
            pose_entries.append((entry["time"], x, y, theta))
        pose_times = [entry[0] for entry in pose_entries]

        scan_points_timeseries = []
        for scan in tqdm(scan_timeseries, "Computing scan points"):
            timestamp = scan["time"]
            # Binary search for the latest pose before timestamp
            idx = bisect.bisect_right(pose_times, timestamp) - 1
            if idx >= 0:
                _, x, y, theta = pose_entries[idx]
                scan_points = self.calculate_scan_points(
                    pose=(x, y, theta),
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
        return data.hstack(
            pl.DataFrame({"scan_points": [scan_points_timeseries]}),
        )

    def compute_scan_point_distances(self, data: pl.DataFrame) -> pl.DataFrame:
        map_entry = data["/map"].to_list()[0][1]["value"]
        map_array = map_entry["data"]
        width = map_entry["info.width"]
        height = map_entry["info.height"]
        occupancy_grid = np.array(map_array).reshape((height, width))
        detected_lines = self.detect_lines_from_grid(
            occupancy_grid,
            threshold1=50,
            threshold2=150,
            hough_threshold=75,
            min_line_length=10,
            max_line_gap=10,
        )
        lines_np = (
            np.array(detected_lines) if detected_lines else np.empty((0, 4))
        )

        # Predefine arrays to avoid unbound errors
        line_length_squared = np.array([])
        line_vec_expanded = np.array([])
        line_start_expanded = np.array([])

        # Precompute line parameters
        if lines_np.size > 0:
            x1 = lines_np[:, 0]
            y1 = lines_np[:, 1]
            x2 = lines_np[:, 2]
            y2 = lines_np[:, 3]
            line_vec = np.column_stack((x2 - x1, y2 - y1))
            line_length_squared = np.sum(line_vec**2, axis=1)
            line_start = lines_np[:, :2]
            has_zero_length = line_length_squared == 0
            line_vec_expanded = line_vec[np.newaxis, :, :]
            line_start_expanded = line_start[np.newaxis, :, :]
        else:
            has_zero_length = np.zeros(0, dtype=bool)
            line_vec_expanded = np.zeros((1, 0, 2))
            line_start_expanded = np.zeros((1, 0, 2))
            line_length_squared = np.array([])

        if self.plotting:
            self.plot_detected_lines(occupancy_grid, detected_lines)

        scan_points_timeseries = data["scan_points"][0].to_list()
        point_distance_timeseries = []

        for scan in tqdm(
            scan_points_timeseries,
            desc="Computing point distances",
        ):
            scan_pts = np.array(scan["value"])
            distance = self._get_closest_line_distance(
                lines_np,
                line_length_squared,
                line_vec_expanded,
                line_start_expanded,
                has_zero_length,
                scan_pts,
            )
            point_distance_timeseries.append(
                {"time": scan["time"], "value": distance},
            )

        if self.plotting:
            last_scan_pts = np.array(scan_points_timeseries[-1]["value"])
            plt.figure()
            plt.scatter(last_scan_pts[:, 0], last_scan_pts[:, 1])
            plt.savefig("last_scan_points.png")

        return data.hstack(
            pl.DataFrame({"point_distance": [point_distance_timeseries]}),
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
            for line in lines:
                x1, y1, x2, y2 = line[0]
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
            projection = np.array([x1, y1]) + t * line_vec
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
    ) -> np.ndarray:
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

        # Convert polar coordinates to Cartesian coordinates in sensor frame
        x_sensor_frame = valid_ranges * np.cos(valid_angles)
        y_sensor_frame = valid_ranges * np.sin(valid_angles)

        # Transform points from sensor frame to map frame
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

        return np.column_stack((x_map, y_map))

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
