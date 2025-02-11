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

sum_x, sum_y = 0.0, 0.0
sum_sin, sum_cos = 0.0, 0.0
num_particles = len(list_of_particles)
particles_data = []  # to store each particle's x, y, and yaw

for particle in list_of_particles:
    pos = particle['pose']['position']
    x = pos['x']
    y = pos['y']
    quat = particle['pose']['orientation']
    # Compute yaw assuming a 2D rotation (only z and w are nonzero)
    yaw = 2 * math.atan2(quat['z'], quat['w'])
    
    particles_data.append({'x': x, 'y': y, 'yaw': yaw})
    
    sum_x += x
    sum_y += y
    sum_sin += math.sin(yaw)
    sum_cos += math.cos(yaw)

mean_x = sum_x / num_particles
mean_y = sum_y / num_particles
# Compute circular (angular) mean of orientations
mean_yaw = math.atan2(sum_sin, sum_cos)

print("Mean Position: ({:.3f}, {:.3f})".format(mean_x, mean_y))
print("Mean Orientation (radians): {:.3f}".format(mean_yaw))

# --- Rotating the Particle Data by the Mean Orientation ---

# Define a helper function to rotate a point (x,y) about a given origin by an angle (radians)
def rotate_point(x, y, angle, origin=(0, 0)):
    ox, oy = origin
    # Translate point back to origin:
    dx = x - ox
    dy = y - oy
    # Rotate the point
    rotated_x = dx * math.cos(angle) - dy * math.sin(angle)
    rotated_y = dx * math.sin(angle) + dy * math.cos(angle)
    # Translate back
    return rotated_x + ox, rotated_y + oy

# We rotate every particle by -mean_yaw about the mean position so that the mean orientation becomes zero.
rotated_particles = []
for pdata in particles_data:
    x, y, yaw = pdata['x'], pdata['y'], pdata['yaw']
    # Rotate the position about the mean position
    new_x, new_y = rotate_point(x, y, -mean_yaw, origin=(mean_x, mean_y))
    # Adjust the orientation by subtracting the mean yaw
    new_yaw = yaw - mean_yaw
    rotated_particles.append({'x': new_x, 'y': new_y, 'yaw': new_yaw})

# In the rotated data, the mean position remains the same and the mean orientation is now zero.
rotated_mean_x, rotated_mean_y = mean_x, mean_y
rotated_mean_yaw = 0.0

# --- Plotting the Rotated Data ---

plt.figure()

# Plot each rotated particle (position and orientation)
for pdata in rotated_particles:
    x = pdata['x']
    y = pdata['y']
    yaw = pdata['yaw']
    # Plot the position as a blue dot
    plt.plot(x, y, 'bo')
    # Draw an arrow to show the particle's orientation
    arrow_length = 0.3
    plt.arrow(x, y,
              arrow_length * math.cos(yaw),
              arrow_length * math.sin(yaw),
              head_width=0.1, head_length=0.1, fc='r', ec='r')

# Plot the mean position (green star)
plt.plot(rotated_mean_x, rotated_mean_y, 'g*', markersize=15, label='Mean Position')
# Plot the mean orientation arrow (black arrow). Since the data is rotated,
# the mean orientation now points horizontally to the right (angle 0).
arrow_length_mean = 0.5
plt.arrow(rotated_mean_x, rotated_mean_y, arrow_length_mean, 0,
          head_width=0.1, head_length=0.1, fc='k', ec='k', label='Mean Orientation (0 rad)')

plt.xlabel('X (rotated)')
plt.ylabel('Y (rotated)')
plt.title('Rotated Particle Positions and Orientations')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()
