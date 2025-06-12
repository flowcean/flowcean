import matplotlib.pyplot as plt
import numpy as np
from custom_transforms.map_image import crop_map_image
from PIL import Image

bitmap = Image.open("bitmap.png")
robot_theta = np.deg2rad(30)
crop_width = 100
crop_height = 100

cropped_map_image = crop_map_image(
    map_image=np.array(bitmap),
    robot_position=np.array([1, 0]),
    robot_orientation=np.array(
        [
            [np.cos(robot_theta), -np.sin(robot_theta)],
            [np.sin(robot_theta), np.cos(robot_theta)],
        ],
    ),
    map_resolution=10.0 / 1000,
    map_origin=np.array([-5, -7]),
    width=crop_width,
    height=crop_height,
    width_meters=2,
)

plt.imshow(cropped_map_image, cmap="gray", origin="lower")
plt.scatter(
    crop_width / 2,
    crop_height / 2,
    color="red",
    s=50,
    marker="x",
    label="Robot Position",
)
plt.quiver(
    crop_width / 2,
    crop_height / 2,
    crop_width / 4,
    0,
    angles="xy",
    scale_units="xy",
    scale=1,
    color="red",
    label="Robot Orientation",
)
plt.title("Cropped Map Image")
plt.show()
