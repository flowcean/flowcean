import cv2
import numpy as np


def crop_map(
    map_image: np.ndarray,
    translation: np.ndarray,
    rotation_matrix: np.ndarray,
    map_resolution: float,
    cropped_resolution: float,
    crop_width: int,
    crop_height: int,
    border_mode: int = cv2.BORDER_REPLICATE,
) -> np.ndarray:
    map_height, _map_width = map_image.shape[:2]

    u = translation[0] / map_resolution
    v = map_height - (translation[1] / map_resolution)
    p_robot = np.array([u, v])

    scale = map_resolution / cropped_resolution
    rotation = rotation_matrix * scale

    center_out = np.array([crop_width / 2.0, crop_height / 2.0])
    t_pix = center_out - rotation @ p_robot

    affine_transform = np.zeros((2, 3), dtype=np.float32)
    affine_transform[:, :2] = rotation
    affine_transform[:, 2] = t_pix

    return cv2.warpAffine(
        map_image,
        affine_transform,
        dsize=(crop_width, crop_height),
        flags=cv2.INTER_NEAREST,
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
