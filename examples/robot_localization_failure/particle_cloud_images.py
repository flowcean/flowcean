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

plt.figure()

sum_x = 0
sum_y = 0
num_particles = len(list_of_particles)

for particle in list_of_particles:
    pos = particle['pose']['position']
    x = pos['x']
    y = pos['y']

    sum_x += x
    sum_y += y
    
    quat = particle['pose']['orientation']
    # Calculate yaw (rotation around the z-axis) using 2D conversion
    yaw = 2 * math.atan2(quat['z'], quat['w'])
    
    # Plot the position as a blue dot
    plt.plot(x, y, 'bo')
    
    arrow_length = 0.3
    
    # Draw an arrow representing the orientation
    plt.arrow(x, y, arrow_length * math.cos(yaw), arrow_length * math.sin(yaw),
              head_width=0.1, head_length=0.1, fc='r', ec='r')

mean_x = sum_x / num_particles
mean_y = sum_y / num_particles

# Plot the mean position as a green star
plt.plot(mean_x, mean_y, 'g*', markersize=15, label='Mean Position')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Particle Positions, Orientations, and Mean Position')
plt.axis('equal')
plt.grid(True)
plt.legend()

plt.show()
