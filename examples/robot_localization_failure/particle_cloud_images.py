import math
import matplotlib.pyplot as plt

# ============================
# Step 1: Define the Particle List
# ============================
list_of_particles = [
    {'pose': {'position': {'x': 2.7026532109185792, 'y': 1.3363095842400234, 'z': 0.0,
                             '__msgtype__': 'geometry_msgs/msg/Point'},
              'orientation': {'x': 0.0, 'y': 0.0, 'z': -0.6206412601751722, 'w': 0.7840946538321596,
                              '__msgtype__': 'geometry_msgs/msg/Quaternion'},
              '__msgtype__': 'geometry_msgs/msg/Pose'},
     'weight': 0.0005980861244019139, '__msgtype__': 'nav2_msgs/msg/Particle'},
    {'pose': {'position': {'x': 2.9070964865479705, 'y': 3.0649213798266697, 'z': 0.0,
                             '__msgtype__': 'geometry_msgs/msg/Point'},
              'orientation': {'x': 0.0, 'y': 0.0, 'z': -0.45132518076103845, 'w': 0.8923595582560967,
                              '__msgtype__': 'geometry_msgs/msg/Quaternion'},
              '__msgtype__': 'geometry_msgs/msg/Pose'},
     'weight': 0.0005980861244019139, '__msgtype__': 'nav2_msgs/msg/Particle'},
    {'pose': {'position': {'x': 2.80871858542121, 'y': 1.5363776884978138, 'z': 0.0,
                             '__msgtype__': 'geometry_msgs/msg/Point'},
              'orientation': {'x': 0.0, 'y': 0.0, 'z': -0.36432616851598243, 'w': 0.9312714120676442,
                              '__msgtype__': 'geometry_msgs/msg/Quaternion'},
              '__msgtype__': 'geometry_msgs/msg/Pose'},
     'weight': 0.0005980861244019139, '__msgtype__': 'nav2_msgs/msg/Particle'},
    {'pose': {'position': {'x': 1.8221955477463578, 'y': 1.6169840054666116, 'z': 0.0,
                             '__msgtype__': 'geometry_msgs/msg/Point'},
              'orientation': {'x': 0.0, 'y': 0.0, 'z': -0.584478714347991, 'w': 0.8114090414052085,
                              '__msgtype__': 'geometry_msgs/msg/Quaternion'},
              '__msgtype__': 'geometry_msgs/msg/Pose'},
     'weight': 0.0005980861244019139, '__msgtype__': 'nav2_msgs/msg/Particle'},
    {'pose': {'position': {'x': 2.12472141189225, 'y': 1.5361849999975508, 'z': 0.0,
                             '__msgtype__': 'geometry_msgs/msg/Point'},
              'orientation': {'x': 0.0, 'y': 0.0, 'z': -0.4347883702383812, 'w': 0.900532660765534,
                              '__msgtype__': 'geometry_msgs/msg/Quaternion'},
              '__msgtype__': 'geometry_msgs/msg/Pose'},
     'weight': 0.0005980861244019139, '__msgtype__': 'nav2_msgs/msg/Particle'}
]

# ============================
# Step 2: Compute Mean Position and Mean Orientation
# ============================
sum_x, sum_y = 0.0, 0.0
sum_sin, sum_cos = 0.0, 0.0
num_particles = len(list_of_particles)
particles_data = []  # List to store each particle's (x, y, yaw)

for particle in list_of_particles:
    pos = particle['pose']['position']
    x = pos['x']
    y = pos['y']
    quat = particle['pose']['orientation']
    # Compute yaw assuming rotation only about the z-axis:
    yaw = 2 * math.atan2(quat['z'], quat['w'])
    
    particles_data.append({'x': x, 'y': y, 'yaw': yaw})
    sum_x += x
    sum_y += y
    sum_sin += math.sin(yaw)
    sum_cos += math.cos(yaw)

mean_x = sum_x / num_particles
mean_y = sum_y / num_particles
mean_yaw = math.atan2(sum_sin, sum_cos)

print("Mean Position: ({:.3f}, {:.3f})".format(mean_x, mean_y))
print("Mean Orientation (radians): {:.3f}".format(mean_yaw))

# ============================
# Step 3: Rotate the Particle Data by -mean_yaw About the Mean Position
# ============================
def rotate_point(x, y, angle, origin):
    """Rotate a point (x, y) about the given origin by the specified angle (in radians)."""
    ox, oy = origin
    dx = x - ox
    dy = y - oy
    rotated_x = dx * math.cos(angle) - dy * math.sin(angle)
    rotated_y = dx * math.sin(angle) + dy * math.cos(angle)
    return rotated_x + ox, rotated_y + oy

rotated_particles = []
for p in particles_data:
    x, y, yaw = p['x'], p['y'], p['yaw']
    new_x, new_y = rotate_point(x, y, -mean_yaw, (mean_x, mean_y))
    new_yaw = yaw - mean_yaw  # Adjust orientation accordingly
    rotated_particles.append({'x': new_x, 'y': new_y, 'yaw': new_yaw})

# ============================
# Step 4: Filter Particles Outside the 1.5m x 1.5m Region Centered at the Mean
# ============================
# Define half-length of the square (1.5m x 1.5m gives half-length = 0.75m)
half_region = 1.5 / 2.0  # 0.75 m

filtered_particles = [
    p for p in rotated_particles
    if abs(p['x'] - mean_x) <= half_region and abs(p['y'] - mean_y) <= half_region
]

# ============================
# Step 5: Save the Region as an Image (36x36 pixels) with Only Black Dots
# ============================
# We want an image that is 36x36 pixels. One way is to create a figure of 1 inch x 1 inch with dpi=36.
fig, ax = plt.subplots(figsize=(1, 1), dpi=36)

# Set the limits to match the 1.5m x 1.5m region centered at the mean
ax.set_xlim(mean_x - half_region, mean_x + half_region)
ax.set_ylim(mean_y - half_region, mean_y + half_region)

# Turn off the axis (which also removes the grid, ticks, and labels)
ax.axis('off')

# Plot each filtered particle as a black dot
# (Adjust markersize; here markersize=2 is chosen for visibility)
for p in filtered_particles:
    ax.plot(p['x'], p['y'], 'ko', markersize=2)

# Save the figure to a file (e.g., 'particle_region.png')
# bbox_inches='tight' and pad_inches=0 remove any extra white space.
plt.savefig("particle_region.png", bbox_inches='tight', pad_inches=0)
plt.close()

print("Saved 36x36 pixel image as 'particle_region.png'")
