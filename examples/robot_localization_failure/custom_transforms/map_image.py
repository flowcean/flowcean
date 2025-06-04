import numpy as np
import cv2
from numpy.typing import NDArray


def crop_map(
    map_image: NDArray,
    robot_position: NDArray,  # [x, y] in global meters
    robot_rotation: NDArray,  # 2×2 rotation from robot→global (world coords)
    map_origin: NDArray,  # [x0, y0] of the map’s top-left, in meters
    map_resolution: float,  # meters per pixel
    crop_width: int,  # output width (px)
    crop_height: int,  # output height (px)
    width_meters: float,  # meters spanned horizontally
    border_mode: int = cv2.BORDER_REPLICATE,
) -> NDArray:
    map_width, map_height = map_image.shape[:2]
    robot_px = (robot_position - map_origin) / map_resolution

    src_px_span = width_meters / map_resolution
    scale = src_px_span / crop_width
    print(f"scale: {scale}")
    RS = robot_rotation * scale

    cx, cy = crop_width / 2, crop_height / 2
    t = robot_px - RS @ np.array([cx, cy])
    M = np.zeros((2, 3))
    M[:, :2] = RS
    M[:, 2] = t

    return cv2.warpAffine(
        map_image,
        M,
        (crop_width, crop_height),
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderMode=border_mode,
    )


# import matplotlib.pyplot as plt
# import numpy as np
#
# H, W = 600, 800
# map_img = np.zeros((H, W), dtype=np.uint8)
# cv2.rectangle(map_img, (150, 100), (700, 500), color=1, thickness=10)
# cv2.rectangle(map_img, (200, 150), (50, 450), color=1, thickness=10)
# cv2.circle(map_img, (140, 100), 50, color=1, thickness=10)
#
# theta = np.deg2rad(30)
# R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
# t = np.array([3.0, 1.5])
#
# map_res = 0.02
# crop_res = 0.01
# crop_w, crop_h = 400, 400
#
# patch = crop_map(
#     map_image=map_img,
#     translation=t,
#     rotation_matrix=R,
#     map_resolution=map_res,
#     cropped_resolution=crop_res,
#     crop_width=crop_w,
#     crop_height=crop_h,
# )
#
# # 5) Visualize:
# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(map_img, cmap="gray_r")
# ax[1].imshow(patch, cmap="gray_r")
# plt.title("Robot-centric occupancy patch")
# plt.show()
