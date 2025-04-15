import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
from custom_transforms.scan_map_statistics import ScanMapStatistics


class TestScanMap(unittest.TestCase):
    def setUp(self) -> None:
        # Initialize ScanMap with default parameters
        self.scan_map = ScanMapStatistics(
            scan_topic="/scan",
            sensor_pose_topic="/amcl_pose",
            plotting=False,
        )

    def test_init(self) -> None:
        assert self.scan_map.scan_topic == "/scan"
        assert self.scan_map.sensor_pose_topic == "/amcl_pose"
        assert self.scan_map.plotting is False
        assert self.scan_map.precomputed_map_lines is None

    @patch("custom_transforms.scan_map.pl.LazyFrame.collect")
    def test_apply(self, mock_collect: MagicMock) -> None:
        # Mock LazyFrame and its collect method
        mock_df = pl.DataFrame({"/scan": [], "/amcl_pose": [], "/map": []})
        mock_lazy = pl.LazyFrame(mock_df)
        mock_collect.return_value = mock_df

        with (
            patch.object(
                self.scan_map,
                "compute_lidar_scan_points",
                return_value=mock_df,
            ) as mock_compute,
            patch.object(
                self.scan_map,
                "_precompute_map_line_parameters",
            ) as mock_precompute,
            patch.object(
                self.scan_map,
                "derive_scan_point_features",
                return_value=mock_df,
            ) as mock_derive,
        ):
            result = self.scan_map.apply(mock_lazy)
            assert isinstance(result, pl.LazyFrame)
            assert mock_compute.call_count == 1
            assert mock_compute.call_args.args[0] is mock_df
            assert mock_precompute.call_count == 1
            assert mock_precompute.call_args.args[0] is mock_df
            assert mock_derive.call_count == 1
            assert mock_derive.call_args.args[0] is mock_df

    def test_precompute_map_line_parameters_empty(self) -> None:
        # Mock DataFrame with an empty map
        mock_df = pl.DataFrame(
            {
                "/map": [
                    [
                        {
                            "value": {
                                "data": [],
                                "info.width": 0,
                                "info.height": 0,
                            },
                        },
                    ],
                ],
            },
        )
        result = self.scan_map._precompute_map_line_parameters(mock_df)  # noqa: SLF001
        assert np.array_equal(result["lines_np"], np.empty((0, 4)))
        assert len(result["x1"]) == 0

    def test_precompute_map_line_parameters_valid(self) -> None:
        # Mock DataFrame with a simple map
        mock_df = pl.DataFrame(
            {
                "/map": [
                    [
                        {
                            "value": {
                                "data": [0, 100, 100, 0],
                                "info.width": 2,
                                "info.height": 2,
                            },
                        },
                    ],
                ],
            },
        )
        with patch.object(
            self.scan_map,
            "detect_lines_from_grid",
            return_value=[(0, 0, 1, 1)],
        ) as mock_detect:
            _ = mock_detect
            result = self.scan_map._precompute_map_line_parameters(mock_df)  # noqa: SLF001
            assert np.array_equal(result["lines_np"], np.array([[0, 0, 1, 1]]))
            assert np.array_equal(result["line_vec"], np.array([[1, 1]]))

    def test_vectorized_distance_calculation(self) -> None:
        # Setup mock precomputed_map_lines for this test
        self.scan_map.precomputed_map_lines = {
            "lines_np": np.array(
                [[0, 0, 1, 1], [2, 2, 3, 3]],
                dtype=np.float64,
            ),
            "line_start": np.array([[0, 0], [2, 2]], dtype=np.float32),
            "line_vec": np.array([[1, 1], [1, 1]], dtype=np.float32),
            "line_length_squared": np.array([2, 2], dtype=np.float32),
        }
        points = np.array([[0.5, 0.5], [2.5, 2.5]], dtype=np.float32)
        distances, indices = (
            ScanMapStatistics._vectorized_distance_calculation(  # noqa: SLF001
                points,
                self.scan_map.precomputed_map_lines["line_start"],
                self.scan_map.precomputed_map_lines["line_vec"],
                self.scan_map.precomputed_map_lines["line_length_squared"],
            )
        )
        assert abs(distances[0] - 0.0) < 1e-5  # Regular assert with tolerance
        assert indices[0] == 0
        assert abs(distances[1] - 0.0) < 1e-5
        assert indices[1] == 1

    def test_get_closest_line_distance_details_empty(self) -> None:
        self.scan_map.precomputed_map_lines = {
            "line_start": np.array([], dtype=np.float32),
            "line_vec": np.array([], dtype=np.float32),
            "line_length_squared": np.array([], dtype=np.float32),
        }
        scan_pts = np.array([], dtype=np.float64)
        distances, indices = self.scan_map._get_closest_line_distance_details(  # noqa: SLF001
            scan_pts,
        )
        assert len(distances) == 0
        assert len(indices) == 0

    def test_batched_raycast(self) -> None:
        sensor_x, sensor_y = 0.0, 0.0
        angles = np.array([0, np.pi / 2], dtype=np.float32)
        grid = np.array([[0, 100], [0, 0]], dtype=np.uint8)
        ranges = ScanMapStatistics._batched_raycast(  # noqa: SLF001
            sensor_x,
            sensor_y,
            angles,
            0.0,
            0.0,
            1.0,
            grid,
            10.0,
        )
        assert abs(ranges[0] - 1.0) < 1e-5  # Hits obstacle at (1, 0)
        assert abs(ranges[1] - 10.0) < 1e-5  # No obstacle

    def test_calculate_scan_points(self) -> None:
        pose = (0.0, 0.0, 0.0)  # x, y, theta
        ranges = [1.0, 2.0]
        angle_min, angle_max, angle_increment = (
            0.0,
            np.pi / 2 + 0.001,
            np.pi / 2,
        )  # Ensure 2 angles
        range_min, range_max = 0.0, 10.0
        points = self.scan_map.calculate_scan_points(
            pose,
            angle_min,
            angle_max,
            angle_increment,
            range_min,
            range_max,
            ranges,
        )
        expected = np.array([[1.0, 0.0], [0.0, 2.0]])
        assert np.all(np.abs(points - expected) < 1e-5)

    @patch("custom_transforms.scan_map.cv2.HoughLinesP")
    def test_detect_lines_from_grid(self, mock_hough: MagicMock) -> None:
        grid = np.array([[0, 100], [100, 0]], dtype=np.uint8)
        mock_hough.return_value = np.array([[0, 0, 1, 1]])
        lines = self.scan_map.detect_lines_from_grid(grid)
        assert lines == [(0, 0, 1, 1)]

    def test_point_distance(self) -> None:
        map_lines = [(0, 0, 1, 1)]
        scan = np.array([[0.5, 0.5], [1.5, 1.5]])
        distance = self.scan_map.point_distance(map_lines, scan)
        assert (
            abs(distance - np.sqrt(2) / 4) < 1e-5
        )  # Correct perpendicular distance


if __name__ == "__main__":
    unittest.main()
