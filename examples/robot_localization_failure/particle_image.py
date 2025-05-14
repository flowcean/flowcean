import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


class ScaleArgumentError(ValueError):
    """Error raised when an invalid number of scale arguments are provided."""

    def __init__(self) -> None:
        message = (
            "Specify exactly one of"
            "meter_per_pixel, width_meters, or height_meters"
        )
        super().__init__(message)


def particles_to_image(
    particles: np.ndarray,
    width: int,
    height: int,
    *,
    meter_per_pixel: float | None = None,
    width_meters: float | None = None,
    height_meters: float | None = None,
    isometry: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    specified_scales = sum(
        argument is not None
        for argument in (meter_per_pixel, width_meters, height_meters)
    )
    if specified_scales != 1:
        raise ScaleArgumentError
    if meter_per_pixel is None:
        if width_meters is not None:
            meter_per_pixel = width_meters / width
        elif height_meters is not None:
            meter_per_pixel = height_meters / height
        else:
            raise ScaleArgumentError

    if isometry is None:
        rotation = np.eye(2)
        translation = np.zeros(2)
    else:
        rotation, translation = isometry
        rotation = np.asarray(rotation, float).reshape(2, 2)
        translation = np.asarray(translation, float).reshape(
            2,
        )

    image = np.zeros((height, width), dtype=float)

    pts = particles[:, :2].T
    x_trans, y_trans = rotation @ pts + translation[:, None]
    ws = particles[:, 2]

    cols = np.round((y_trans / meter_per_pixel) + (width / 2)).astype(int)
    rows = np.round(-(x_trans / meter_per_pixel) + (height / 2)).astype(int)

    mask = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
    np.add.at(image, (rows[mask], cols[mask]), ws[mask])

    return image


def compute_image(x: dict) -> NDArray:
    particles = np.array(
        [
            [particle["x"], particle["y"], particle["weight"]]
            for particle in x["particles"]
        ],
    )
    position = np.array([x["position"]["x"], x["position"]["y"]])
    rotation = (
        Rotation.from_quat(
            (
                x["orientation"]["x"],
                x["orientation"]["y"],
                x["orientation"]["z"],
                x["orientation"]["w"],
            ),
        )
        .inv()
        .as_matrix()[:2, :2]
    )
    map_to_robot = (rotation, rotation @ -position)
    return particles_to_image(
        particles,
        width=100,
        height=100,
        width_meters=5.0,
        isometry=map_to_robot,
    )


data = pl.scan_parquet("zoh.parquet")

values = pl.col("measurements").explode().struct.field("value")
particles = (
    values.struct.field("/particle_cloud/particles")
    .list.eval(
        pl.struct(
            pl.element()
            .struct.field("pose")
            .struct.field("position")
            .struct.field("x"),
            pl.element()
            .struct.field("pose")
            .struct.field("position")
            .struct.field("y"),
            pl.element().struct.field("weight"),
        ),
    )
    .alias("particles")
)
position = pl.struct(
    values.struct.field("/amcl_pose/pose.pose.position.x").alias("x"),
    values.struct.field("/amcl_pose/pose.pose.position.y").alias("y"),
).alias("position")
orientation = pl.struct(
    values.struct.field("/amcl_pose/pose.pose.orientation.x").alias(
        "x",
    ),
    values.struct.field("/amcl_pose/pose.pose.orientation.y").alias(
        "y",
    ),
    values.struct.field("/amcl_pose/pose.pose.orientation.z").alias(
        "z",
    ),
    values.struct.field("/amcl_pose/pose.pose.orientation.w").alias(
        "w",
    ),
).alias("orientation")
images = data.select(
    pl.struct(
        particles,
        position,
        orientation,
    ).map_elements(compute_image, return_dtype=pl.Object),
)

c = images.collect(engine="streaming")
arst

from matplotlib import animation

fig, ax = plt.subplots()
imshow = ax.imshow(
    p["particles"][0] * 255.0,
    cmap="gray",
)
ani = animation.FuncAnimation(
    fig,
    lambda i: imshow.set_array(p["particles"][i] * 255.0),
    frames=len(p),
    interval=50,
)
ani.save(filename="/tmp/pillow_example.gif", writer="pillow")
# plt.show()
art

time_i = 0
values = data.select(
    pl.col("measurements")
    .explode()
    .get(time_i)
    .struct.field("value")
    .struct.unnest(),
)
particles = (
    values.select(
        pl.col(
            "/particle_cloud/particles",
        )
        .explode()
        .struct.unnest(),
    )
    .select(pl.col("pose").struct.field("position").struct.unnest(), "weight")
    .drop("z")
    .collect(engine="streaming")
    .to_numpy()
)
pose = (
    values.select(
        pl.col(
            "/amcl_pose/pose.pose.position.x",
            "/amcl_pose/pose.pose.position.y",
            "/amcl_pose/pose.pose.orientation.x",
            "/amcl_pose/pose.pose.orientation.y",
            "/amcl_pose/pose.pose.orientation.z",
            "/amcl_pose/pose.pose.orientation.w",
        ),
    )
    .collect(engine="streaming")
    .row(0)
)
position = np.array(pose[:2])
rotation = Rotation.from_quat(pose[2:]).inv().as_matrix()[:2, :2]
map_to_robot = (rotation, rotation @ -position)

img = particles_to_image(
    particles,
    width=100,
    height=100,
    width_meters=5.0,
    isometry=map_to_robot,
)

plt.imshow(img * 255.0, cmap="gray")
plt.title("Accumulated particle weights")
plt.axis("off")
plt.show()
