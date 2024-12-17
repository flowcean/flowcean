import cv2
import matplotlib.pyplot as plt
import numpy as np


def hough_line_transform(
    occupancy_grid: np.ndarray,
    threshold1: int = 50,
    threshold2: int = 150,
    hough_threshold: int = 100,
    min_line_length: int = 30,
    max_line_gap: int = 5,
) -> tuple:
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
    # Edge Detection using Canny Edge Detector
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

    return detected_lines, edges


def plot_detected_lines(
    occupancy_grid: np.ndarray, edges: np.ndarray, lines: list
) -> None:
    """Plot the occupancy grid and overlay the detected lines."""
    plt.figure(figsize=(8, 6))  # Single plot

    # Show the occupancy grid
    plt.imshow(occupancy_grid, cmap="gray")

    # Overlay detected lines
    for line in lines:
        plt.plot(
            [line[0], line[2]],  # x-coordinates
            [line[1], line[3]],  # y-coordinates
            color="red",  # Line color
            linewidth=2,  # Line thickness
        )

    # Set title and show the plot
    plt.title("Detected Lines Over Occupancy Grid")
    plt.axis("off")  # Remove axes for cleaner output
    plt.tight_layout()
    plt.savefig("detected_lines.png")  # Save the image
    plt.show()
