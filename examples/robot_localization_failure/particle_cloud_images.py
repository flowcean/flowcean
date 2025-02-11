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

# -------------------------------------
# Step 1: Compute Mean Position & Orientation
# -------------------------------------
sum_x, sum_y = 0.0, 0.0
sum_sin, sum_cos = 0.0, 0.0
particles_data = []  # List to store each particle's (x, y, yaw)
num_particles = len(list_of_particles)

for particle in list_of_particles:
    pos = particle['pose']['position']
    x = pos['x']
    y = pos['y']
    quat = particle['pose']['orientation']
    # Compute yaw assuming a 2D rotation (only z and w components are nonzero)
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

# -------------------------------------
# Step 2: Rotate the Particle Data by -mean_yaw
# -------------------------------------
# Helper function to rotate a point (x, y) about a given origin by an angle (radians)
def rotate_point(x, y, angle, origin=(0, 0)):
    ox, oy = origin
    dx = x - ox
    dy = y - oy
    rotated_x = dx * math.cos(angle) - dy * math.sin(angle)
    rotated_y = dx * math.sin(angle) + dy * math.cos(angle)
    return rotated_x + ox, rotated_y + oy

rotated_particles = []
for pdata in particles_data:
    x, y, yaw = pdata['x'], pdata['y'], pdata['yaw']
    # Rotate the position about the mean position
    new_x, new_y = rotate_point(x, y, -mean_yaw, origin=(mean_x, mean_y))
    # Adjust the orientation by subtracting the mean yaw
    new_yaw = yaw - mean_yaw
    rotated_particles.append({'x': new_x, 'y': new_y, 'yaw': new_yaw})

# In the rotated frame, the mean orientation becomes 0.
rotated_mean_x, rotated_mean_y = mean_x, mean_y
rotated_mean_yaw = 0.0

# -------------------------------------
# Step 3: Create Side-by-Side Plots (Before and After Rotation)
# -------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# Plot settings for both subplots
arrow_length = 0.3
arrow_length_mean = 0.5

# --- Plot Before Rotation (Original Data) ---
for pdata in particles_data:
    x, y, yaw = pdata['x'], pdata['y'], pdata['yaw']
    ax1.plot(x, y, 'bo')
    ax1.arrow(x, y, arrow_length * math.cos(yaw), arrow_length * math.sin(yaw),
              head_width=0.1, head_length=0.1, fc='r', ec='r')
ax1.plot(mean_x, mean_y, 'g*', markersize=15, label='Mean Position')
ax1.arrow(mean_x, mean_y, arrow_length_mean * math.cos(mean_yaw),
          arrow_length_mean * math.sin(mean_yaw),
          head_width=0.1, head_length=0.1, fc='k', ec='k', label='Mean Orientation')
ax1.set_title("Before Rotation")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.axis("equal")
ax1.grid(True)
ax1.legend()

# --- Plot After Rotation (Rotated Data) ---
for pdata in rotated_particles:
    x, y, yaw = pdata['x'], pdata['y'], pdata['yaw']
    ax2.plot(x, y, 'bo')
    ax2.arrow(x, y, arrow_length * math.cos(yaw), arrow_length * math.sin(yaw),
              head_width=0.1, head_length=0.1, fc='r', ec='r')
ax2.plot(rotated_mean_x, rotated_mean_y, 'g*', markersize=15, label='Mean Position')
# In the rotated frame, the mean orientation is aligned horizontally (angle 0)
ax2.arrow(rotated_mean_x, rotated_mean_y, arrow_length_mean, 0,
          head_width=0.1, head_length=0.1, fc='k', ec='k', label='Mean Orientation (0 rad)')
ax2.set_title("After Rotation")
ax2.set_xlabel("X (rotated)")
ax2.set_ylabel("Y (rotated)")
ax2.axis("equal")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()
