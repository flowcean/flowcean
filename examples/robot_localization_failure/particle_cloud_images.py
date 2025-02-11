import math
import matplotlib.pyplot as plt

list_of_particles = [
    {'pose': {'position': {'x': 2.7026532109185792, 'y': 1.3363095842400234, 'z': 0.0, '__msgtype__': 'geometry_msgs/msg/Point'},
              'orientation': {'x': 0.0, 'y': 0.0, 'z': -0.6206412601751722, 'w': 0.7840946538321596, '__msgtype__': 'geometry_msgs/msg/Quaternion'},
              '__msgtype__': 'geometry_msgs/msg/Pose'},
     'weight': 0.0005980861244019139, '__msgtype__': 'nav2_msgs/msg/Particle'},
    {'pose': {'position': {'x': 2.9070964865479705, 'y': 3.0649213798266697, 'z': 0.0, '__msgtype__': 'geometry_msgs/msg/Point'},
              'orientation': {'x': 0.0, 'y': 0.0, 'z': -0.45132518076103845, 'w': 0.8923595582560967, '__msgtype__': 'geometry_msgs/msg/Quaternion'},
              '__msgtype__': 'geometry_msgs/msg/Pose'},
     'weight': 0.0005980861244019139, '__msgtype__': 'nav2_msgs/msg/Particle'},
    {'pose': {'position': {'x': 2.80871858542121, 'y': 1.5363776884978138, 'z': 0.0, '__msgtype__': 'geometry_msgs/msg/Point'},
              'orientation': {'x': 0.0, 'y': 0.0, 'z': -0.36432616851598243, 'w': 0.9312714120676442, '__msgtype__': 'geometry_msgs/msg/Quaternion'},
              '__msgtype__': 'geometry_msgs/msg/Pose'},
     'weight': 0.0005980861244019139, '__msgtype__': 'nav2_msgs/msg/Particle'},
    {'pose': {'position': {'x': 1.8221955477463578, 'y': 1.6169840054666116, 'z': 0.0, '__msgtype__': 'geometry_msgs/msg/Point'},
              'orientation': {'x': 0.0, 'y': 0.0, 'z': -0.584478714347991, 'w': 0.8114090414052085, '__msgtype__': 'geometry_msgs/msg/Quaternion'},
              '__msgtype__': 'geometry_msgs/msg/Pose'},
     'weight': 0.0005980861244019139, '__msgtype__': 'nav2_msgs/msg/Particle'},
    {'pose': {'position': {'x': 2.12472141189225, 'y': 1.5361849999975508, 'z': 0.0, '__msgtype__': 'geometry_msgs/msg/Point'},
              'orientation': {'x': 0.0, 'y': 0.0, 'z': -0.4347883702383812, 'w': 0.900532660765534, '__msgtype__': 'geometry_msgs/msg/Quaternion'},
              '__msgtype__': 'geometry_msgs/msg/Pose'},
     'weight': 0.0005980861244019139, '__msgtype__': 'nav2_msgs/msg/Particle'}
]

# ----- Step 1: Compute the Mean Position and Circular Mean Orientation -----
sum_x = 0.0
sum_y = 0.0
sum_sin = 0.0
sum_cos = 0.0
num_particles = len(list_of_particles)
particles_data = []  # store each particle's x, y, and yaw

for particle in list_of_particles:
    pos = particle['pose']['position']
    x = pos['x']
    y = pos['y']
    quat = particle['pose']['orientation']
    # Compute yaw assuming only a rotation about the z-axis:
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

# ----- Step 2: Rotate the Data by -mean_yaw About the Mean Position -----
def rotate_point(x, y, angle, origin):
    """Rotate a point (x, y) about an origin by a given angle (in radians)."""
    ox, oy = origin
    # Translate point to origin
    dx = x - ox
    dy = y - oy
    # Rotate the point
    rotated_x = dx * math.cos(angle) - dy * math.sin(angle)
    rotated_y = dx * math.sin(angle) + dy * math.cos(angle)
    # Translate back
    return rotated_x + ox, rotated_y + oy

rotated_particles = []
for p in particles_data:
    x, y, yaw = p['x'], p['y'], p['yaw']
    # Rotate each particle's position about the mean by -mean_yaw
    new_x, new_y = rotate_point(x, y, -mean_yaw, (mean_x, mean_y))
    # Adjust the particle's orientation accordingly
    new_yaw = yaw - mean_yaw
    rotated_particles.append({'x': new_x, 'y': new_y, 'yaw': new_yaw})

# ----- Step 3: Filter Out Particles Outside the 1.5m x 1.5m Region -----
# Define the half-length of the region (1.5m square has half-length 0.75m)
half_region = 1.5 / 2.0  # 0.75 m
# Keep only particles whose x and y are within half_region of the mean position.
filtered_particles = [
    p for p in rotated_particles
    if abs(p['x'] - mean_x) <= half_region and abs(p['y'] - mean_y) <= half_region
]

# ----- Step 4: Plot the Filtered, Rotated Data -----
plt.figure()

# Plot each filtered particle (position and orientation)
for p in filtered_particles:
    x, y, yaw = p['x'], p['y'], p['yaw']
    plt.plot(x, y, 'bo')  # Blue dot for position
    arrow_length = 0.3
    plt.arrow(x, y,
              arrow_length * math.cos(yaw),
              arrow_length * math.sin(yaw),
              head_width=0.1, head_length=0.1, fc='r', ec='r')

# Plot the mean position as a green star
plt.plot(mean_x, mean_y, 'g*', markersize=15, label='Mean Position')

# Plot the mean orientation arrow (after rotation, it points horizontally to the right)
arrow_length_mean = 0.5
plt.arrow(mean_x, mean_y, arrow_length_mean, 0,
          head_width=0.1, head_length=0.1, fc='k', ec='k', label='Mean Orientation')

# Optional: Draw a dashed magenta rectangle representing the 1.5m x 1.5m region
region_rect = plt.Rectangle((mean_x - half_region, mean_y - half_region), 1.5, 1.5,
                            fill=False, color='magenta', linestyle='--', label='1.5m x 1.5m Region')
plt.gca().add_patch(region_rect)

plt.xlabel('X (rotated)')
plt.ylabel('Y (rotated)')
plt.title('Filtered Particle Positions (within 1.5m x 1.5m region centered at the mean)')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()
