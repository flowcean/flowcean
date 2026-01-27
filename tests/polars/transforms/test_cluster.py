import unittest

from cv2 import exp
import polars as pl
from polars.testing import assert_frame_equal
from sklearn.cluster import KMeans

from flowcean.polars import Cluster


class ClusterTransform(unittest.TestCase):
    def test_cluster_default(self) -> None:
        transform = Cluster(KMeans(n_clusters=2))

        data_frame = pl.DataFrame(
            {
                "a": [1.0, 1.1, 2.0, 8.0, 8.1, 9.0],
            },
        ).lazy()

        transform.fit(data_frame)

        transformed_data = transform(data_frame).collect()

        if transformed_data["cluster_label"][0] == 0:
            expected_labels = [0, 0, 0, 1, 1, 1]
        else:
            expected_labels = [1, 1, 1, 0, 0, 0]

        assert_frame_equal(
            transformed_data,
            pl.DataFrame(
                {
                    "a": [1.0, 1.1, 2.0, 8.0, 8.1, 9.0],
                    "cluster_label": expected_labels,
                },
            ),
        )


if __name__ == "__main__":
    unittest.main()
