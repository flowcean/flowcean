import numpy as np
import polars as pl
from utils import hough_line_transform, plot_detected_lines, point_distance


def point_distance_transform(data: pl.DataFrame) -> pl.DataFrame:
    """Calculate the mean distance of the detected lines to the scan points."""
    map_array = data["/map"].to_list()[0][1]["value"]["data"]
    width = data["/map"].to_list()[0][1]["value"]["info.width"]
    height = data["/map"].to_list()[0][1]["value"]["info.height"]
    occupancy_grid = np.array(map_array).reshape((height, width))
    detected_lines = hough_line_transform(
        occupancy_grid,
        threshold1=50,
        threshold2=150,
        hough_threshold=75,
        min_line_length=10,
        max_line_gap=10,
    )
    print(data)
    plot_detected_lines(occupancy_grid=occupancy_grid, lines=detected_lines)
    scan_points_timeseries = data["scan_points"][0].to_list()
    point_distance_timeseries = []
    for scan in scan_points_timeseries:
        scan_points = np.array(scan["value"])
        distance = point_distance(detected_lines, scan_points)
        timestamp = scan["time"]
        point_distance_timeseries.append(
            {"time": timestamp, "value": distance}
        )
    return data.hstack(
        pl.DataFrame({"point_distance": [point_distance_timeseries]})
    )
