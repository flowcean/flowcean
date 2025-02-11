import os
import math
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

# -------------------------------
# Configuration Variables
# -------------------------------
SAVE_IMAGES = False           # Set to True to save images to disk, False to only create the DataFrame
region_size = 15.0           # Square region side length in meters (e.g., can be 1.5 or 15)
image_pixel_size = 300       # Output image resolution (e.g., 300x300 pixels)
half_region = region_size / 2.0

# Create an output folder name that adapts to non-integer region sizes.
# For example, region_size=1.5 becomes "1_5"
region_str = str(region_size).replace('.', '_')
output_dir = f"particle_images_{region_str}m_{image_pixel_size}p"
if SAVE_IMAGES:
    os.makedirs(output_dir, exist_ok=True)

# -------------------------------
# Read the JSON ROS Bag Data
# -------------------------------
data = pl.read_json("cached_ros_data.json")  # JSON file with all topics as loaded by RosbagLoader
particle_cloud = data[0, 3]  # Polars Series; each entry is a dict representing one message

# -------------------------------
# Prepare a List for the Image Records
# -------------------------------
# Each record will be a dict with keys "time" and "image"
# The image field will be a nested list (converted from a grayscale NumPy array) so that the DataFrame column type is list[struct(2)]
image_records = []

# -------------------------------
# Helper Function: Rotate a Point
# -------------------------------
def rotate_point(x, y, angle, origin):
    """
    Rotate a point (x, y) about the given origin by the specified angle (in radians).
    """
    ox, oy = origin
    dx = x - ox
    dy = y - oy
    rotated_x = dx * math.cos(angle) - dy * math.sin(angle)
    rotated_y = dx * math.sin(angle) + dy * math.cos(angle)
    return rotated_x + ox, rotated_y + oy

# -------------------------------
# Process given messages
# -------------------------------
for message_number, message_data in enumerate(particle_cloud[3300:3303], start=3300):
    time_stamp = message_data["time"]
    print(f"Processing message {message_number} at time {time_stamp}")

    # Extract particle data from the message.
    particles_dict = message_data["value"]  # Dict with key "particles"
    list_of_particles = particles_dict["particles"]  # List of particle dicts

    if not list_of_particles:
        print(f"Message {message_number} has no particles. Skipping.")
        continue

    # -------------------------------
    # Step 1: Compute Mean Position and Mean Orientation
    # -------------------------------
    sum_x, sum_y = 0.0, 0.0
    sum_sin, sum_cos = 0.0, 0.0
    num_particles = len(list_of_particles)
    particles_data = []  # To store each particle's x, y, and yaw

    for particle in list_of_particles:
        pos = particle["pose"]["position"]
        x = pos["x"]
        y = pos["y"]
        quat = particle["pose"]["orientation"]
        # Compute yaw (rotation about z-axis) assuming the quaternion represents 2D rotation
        yaw = 2 * math.atan2(quat["z"], quat["w"])
        particles_data.append({"x": x, "y": y, "yaw": yaw})
        sum_x += x
        sum_y += y
        sum_sin += math.sin(yaw)
        sum_cos += math.cos(yaw)

    mean_x = sum_x / num_particles
    mean_y = sum_y / num_particles
    mean_yaw = math.atan2(sum_sin, sum_cos)

    # -------------------------------
    # Step 2: Rotate Particle Data by -mean_yaw About the Mean Position
    # -------------------------------
    rotated_particles = []
    for p in particles_data:
        x, y, yaw = p["x"], p["y"], p["yaw"]
        new_x, new_y = rotate_point(x, y, -mean_yaw, (mean_x, mean_y))
        new_yaw = yaw - mean_yaw  # Adjust orientation accordingly
        rotated_particles.append({"x": new_x, "y": new_y, "yaw": new_yaw})

    # -------------------------------
    # Step 3: Filter Out Particles Outside the Square Region
    # -------------------------------
    filtered_particles = [
        p for p in rotated_particles
        if abs(p["x"] - mean_x) <= half_region and abs(p["y"] - mean_y) <= half_region
    ]

    # -------------------------------
    # Step 4: Create the Image
    # -------------------------------
    # Create a figure that is 1x1 inch at dpi=image_pixel_size (yielding image_pixel_size x image_pixel_size pixels)
    fig, ax = plt.subplots(figsize=(1, 1), dpi=image_pixel_size)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(mean_x - half_region, mean_x + half_region)
    ax.set_ylim(mean_y - half_region, mean_y + half_region)
    ax.axis("off")  # Remove axes, grid, ticks, etc.

    # Plot each filtered particle as a black dot (adjust markersize as needed)
    for p in filtered_particles:
        ax.plot(p["x"], p["y"], "ko", markersize=2)

    # Save the image if SAVE_IMAGES is True
    if SAVE_IMAGES:
        filename = os.path.join(output_dir, f"particle_region_{message_number:04d}.png")
        plt.savefig(filename, dpi=image_pixel_size)

    # Convert the figure to a NumPy array using the RGBA buffer
    fig.canvas.draw()
    img_array = np.asarray(fig.canvas.buffer_rgba())  # shape: (image_pixel_size, image_pixel_size, 4)
    img_array = img_array[..., :3]  # Remove the alpha channel

    # Convert the RGB image to grayscale using luminance conversion
    gray_image = np.dot(img_array, [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    plt.close(fig)
    
    # Append a record with the timestamp and the grayscale image (converted to nested lists)
    image_records.append({"time": time_stamp, "image": gray_image.tolist()})

# -------------------------------
# Create a Polars DataFrame with One Column and One Row
# -------------------------------
# The column '/particle_cloud_image' will contain a list of structs,
# each struct has 2 fields: 'time' (Int64) and 'image' (list of lists of Int64)
df_images = pl.DataFrame({"/particle_cloud_image": [image_records]})
print(df_images)
print(df_images.schema)
