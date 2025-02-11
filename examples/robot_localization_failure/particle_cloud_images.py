import os
import math
import polars as pl
import matplotlib.pyplot as plt

# --- Read the JSON ROS bag data (example) ---
data = pl.read_json("cached_ros_data.json")
particle_cloud = data[0, 3]  # polars Series of dictionaries (one per message)

# Create an output directory for the images
output_dir = "particle_images"
os.makedirs(output_dir, exist_ok=True)

# --- Helper Function ---
def rotate_point(x, y, angle, origin):
    """Rotate point (x, y) about origin by a given angle (in radians)."""
    ox, oy = origin
    dx = x - ox
    dy = y - oy
    rotated_x = dx * math.cos(angle) - dy * math.sin(angle)
    rotated_y = dx * math.sin(angle) + dy * math.cos(angle)
    return rotated_x + ox, rotated_y + oy

# Define half-length for the 1.5m x 1.5m region (0.75m)
half_region = 1.5 / 2.0

for message_number, message_data in enumerate(particle_cloud):
    # Each message_data is a dict with "time" and "value"
    time = message_data["time"]
    print(f"Processing message {message_number} at time {time}")
    
    particles_dict = message_data["value"]  # dict with key "particles"
    list_of_particles = particles_dict["particles"]  # list of particle dicts
    
    if not list_of_particles:
        print(f"Message {message_number} has no particles. Skipping.")
        continue
    
    # --- Compute Mean Position and Mean Orientation ---
    sum_x, sum_y = 0.0, 0.0
    sum_sin, sum_cos = 0.0, 0.0
    num_particles = len(list_of_particles)
    particles_data = []
    
    for particle in list_of_particles:
        pos = particle["pose"]["position"]
        x = pos["x"]
        y = pos["y"]
        quat = particle["pose"]["orientation"]
        # Compute yaw (assuming rotation only about z-axis)
        yaw = 2 * math.atan2(quat["z"], quat["w"])
        particles_data.append({"x": x, "y": y, "yaw": yaw})
        sum_x += x
        sum_y += y
        sum_sin += math.sin(yaw)
        sum_cos += math.cos(yaw)
    
    mean_x = sum_x / num_particles
    mean_y = sum_y / num_particles
    mean_yaw = math.atan2(sum_sin, sum_cos)
    
    # --- Rotate Particle Data by -mean_yaw About the Mean ---
    rotated_particles = []
    for p in particles_data:
        x, y, yaw = p["x"], p["y"], p["yaw"]
        new_x, new_y = rotate_point(x, y, -mean_yaw, (mean_x, mean_y))
        new_yaw = yaw - mean_yaw
        rotated_particles.append({"x": new_x, "y": new_y, "yaw": new_yaw})
    
    # --- Filter Particles Outside the 1.5m x 1.5m Region ---
    filtered_particles = [
        p for p in rotated_particles
        if abs(p["x"] - mean_x) <= half_region and abs(p["y"] - mean_y) <= half_region
    ]
    
    # --- Save the Region as a 36x36 Pixel Image ---
    # Create a 1x1 inch figure with 36 dpi (36x36 pixels)
    fig, ax = plt.subplots(figsize=(1, 1), dpi=36)
    # Remove all margins so that the axes fill the entire figure
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Set the axis limits to show the 1.5m x 1.5m region centered at the mean
    ax.set_xlim(mean_x - half_region, mean_x + half_region)
    ax.set_ylim(mean_y - half_region, mean_y + half_region)
    ax.axis("off")  # Turn off axes, grid, ticks, etc.
    
    # Plot each filtered particle as a black dot (adjust markersize as needed)
    for p in filtered_particles:
        ax.plot(p["x"], p["y"], "ko", markersize=2)
    
    # Save the figure without extra padding or tight bounding boxes
    filename = os.path.join(output_dir, f"particle_region_{message_number:04d}.png")
    plt.savefig(filename, dpi=36)  
    plt.close(fig)
    
    print(f"Saved image for message {message_number} as {filename}")
