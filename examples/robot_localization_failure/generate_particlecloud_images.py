#!/usr/bin/env python3
# /// script
# dependencies = [
#     "flowcean",
#     "matplotlib"
# ]
#
# ///

import os

import matplotlib.pyplot as plt
import polars as pl


def plot_particles(
    list_of_particles,
    output_path,
    image_width=800,
    image_height=600,
):
    """Plots all particles in a 2D plot using their positions and saves the image without number scales.

    Parameters:
    list_of_particles (list): List of dictionaries representing particles.
    output_path (str): Path to save the image.
    image_width (int): Width of the saved image in pixels.
    image_height (int): Height of the saved image in pixels.
    """
    # Determine plot bounds to avoid cutting off dots
    all_x = [
        particle["pose"]["position"]["x"] for particle in list_of_particles
    ]
    all_y = [
        particle["pose"]["position"]["y"] for particle in list_of_particles
    ]
    margin = 1.0  # Add some margin around the particles
    min_x, max_x = min(all_x) - margin, max(all_x) + margin
    min_y, max_y = min(all_y) - margin, max(all_y) + margin

    # Convert pixel dimensions to inches for matplotlib
    dpi = 100  # Dots per inch
    figsize = (image_width / dpi, image_height / dpi)

    plt.figure(figsize=figsize)
    # Plot particle positions as dots
    plt.scatter(all_x, all_y, color="black")

    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.axis("off")  # Turn off axis scales and labels

    plt.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0,
    )  # Save the image
    plt.close()


# Load data
data = pl.read_json("cached_ros_data.json")
particle_cloud = data[0, 3]

# Ensure the output directory exists
output_dir = "particle_cloud_imagess"
os.makedirs(output_dir, exist_ok=True)

# Iterate over all messages and save images
for message_number in range(len(particle_cloud)):
    print(f"Processing message {message_number + 1}/{len(particle_cloud)}...")
    message_data = particle_cloud[message_number]
    particles_dict = message_data["value"]
    list_of_particles = particles_dict["particles"]
    output_path = os.path.join(
        output_dir,
        f"particles_plot_{message_number}.png",
    )
    plot_particles(
        list_of_particles,
        output_path,
        image_width=800,
        image_height=600,
    )

print("All messages processed and images saved.")
