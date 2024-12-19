import math

import cv2
import matplotlib.pyplot as plt
import numpy as np


def euler_from_quaternion(x: float, y: float, z: float, w: float) -> tuple:
    """Converts a quaternion into Euler angles (roll, pitch, yaw).

    Args:
        x (float): The x component of the quaternion.
        y (float): The y component of the quaternion.
        z (float): The z component of the quaternion.
        w (float): The w component of the quaternion.

    Returns:
        tuple: A tuple containing (roll, pitch, yaw) in radians.
    """
    # Roll (x-axis rotation)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    # Pitch (y-axis rotation)
    t2 = 2.0 * (w * y - z * x)
    t2 = max(-1.0, min(1.0, t2))  # Clamp to handle numerical errors
    pitch = math.asin(t2)

    # Yaw (z-axis rotation)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    return roll, pitch, yaw


def hough_line_transform(
    occupancy_grid: np.ndarray,
    threshold1: int = 50,
    threshold2: int = 150,
    hough_threshold: int = 100,
    min_line_length: int = 30,
    max_line_gap: int = 5,
) -> list[tuple[float, float, float, float]]:
    """Perform the Hough Transform to find finite lines from an occupancy grid.

    Steps:
    1. Apply edge detection to the image.
    2. Map image points into Hough space using an accumulator.
    3. Detect lines (local maxima in the Hough space) with thresholds.
    4. Convert infinite lines into finite ones using Progressive Probabilistic
    Hough Transform.

    Parameters:
        occupancy_grid (numpy.ndarray): A 2D binary occupancy grid.
        threshold1 (int): First threshold for the Canny edge detector.
        threshold2 (int): Second threshold for the Canny edge detector.
        hough_threshold (int): Accumulator threshold for Hough Transform.
        min_line_length (int): Minimum length of detected lines.
        max_line_gap (int): Maximum allowed gap between line segments.

    Returns:
        list: Detected finite lines represented as (x1, y1, x2, y2).
    """
    # Detect edges using Canny Edge Detector
    image = np.uint8(
        occupancy_grid * 2.55
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


def plot_detected_lines(
    occupancy_grid: np.ndarray, lines: list[tuple[float, float, float, float]]
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


def point_distance(map_lines: list, scan: np.ndarray) -> float:
    """Calculates the average distance of a scan point to a map line.

    Args:
        map_lines (list): A list of lines, where each line is represented as
        (x1, y1, x2, y2) (endpoints of the line).
        scan (np.ndarray): An Nx2 array of scan points with x, y coordinates.

    Returns:
        float: The average distance of all scan points to the nearest map line.
    """

    def distance_point_to_line(
        px: float, py: float, x1: float, y1: float, x2: float, y2: float
    ) -> float:
        """Calculate the shortest distance from a point to a line segment."""
        line_vec = np.array([x2 - x1, y2 - y1])
        point_vec = np.array([px - x1, py - y1])
        line_length_squared = np.dot(line_vec, line_vec)

        # Handle the case where the line length is 0
        if line_length_squared == 0:
            return float(np.linalg.norm(point_vec))

        # Projection scalar (normalized distance along the line segment)
        t = max(0, min(1, np.dot(point_vec, line_vec) / line_length_squared))

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
        pose (tuple): The (x, y, theta) pose of the sensor in the map frame.
        angle_min (float): The start angle of the scan [rad].
        angle_max (float): The end angle of the scan [rad].
        angle_increment (float): Angular distance between measurements [rad].
        range_min (float): Minimum range value [m].
        range_max (float): Maximum range value [m].
        ranges (list): Measured distances for each scan [m].

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

    # Convert polar coordinates to Cartesian coordinates in the sensor frame
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
