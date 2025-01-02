

def center_of_gravity_maximum_distance(particle_cloud) -> Float:
"""
Calculate the center of gravity of the particle cloud and the maximum distance
of a particle to the center of gravity.
"""

# Calculate the center of gravity
center_of_gravity = particle_cloud.mean()
