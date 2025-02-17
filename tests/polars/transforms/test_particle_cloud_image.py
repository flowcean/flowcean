import unittest
from datetime import datetime, timezone

import numpy as np
import polars as pl

from flowcean.polars.transforms.particle_cloud_image import ParticleCloudImage


class TestParticleCloudImage(unittest.TestCase):
    def setUp(self) -> None:
        self.transform = ParticleCloudImage(
            particle_cloud_feature_name="/particle_cloud",
            cutting_area=15.0,
            image_pixel_size=300,
            save_images=False,
        )

        self.list_of_particles = [
            {
                "pose": {
                    "position": {"x": 1.0, "y": 2.0, "z": 0.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
            },
            {
                "pose": {
                    "position": {"x": -2.0, "y": -3.0, "z": 0.0},
                    "orientation": {
                        "x": 0.0,
                        "y": 0.0,
                        "z": 0.707,
                        "w": 0.707,
                    },
                },
            },
        ]

        self.data_frame = pl.DataFrame(
            {
                "/particle_cloud": [
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                1,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"particles": self.list_of_particles},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                2,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"particles": self.list_of_particles},
                        },
                    ],
                ],
                "feature_b": [
                    [
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                0,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 1.0},
                        },
                        {
                            "time": datetime(
                                2024,
                                6,
                                25,
                                12,
                                26,
                                5,
                                0,
                                tzinfo=timezone.utc,
                            ),
                            "value": {"x": 2.0},
                        },
                    ],
                ],
                "const": [1],
            },
        )

    def test_apply(self) -> None:
        transformed_data = self.transform.apply(
            self.data_frame.lazy(),
        ).collect()
        assert "/particle_cloud_image" in transformed_data.columns
        assert len(transformed_data["/particle_cloud_image"][0]) == 2

        # Check image size
        images = transformed_data["/particle_cloud_image"][0]
        for img_record in images:
            img_array = np.array(img_record["image"], dtype=np.uint8)
            assert img_array.shape == (300, 300)


if __name__ == "__main__":
    unittest.main()
