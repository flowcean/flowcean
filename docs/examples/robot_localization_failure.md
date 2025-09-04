# Robot Localization Failure

Highly dynamic environments as the one depicted below still remain a challenge for localization systems of mobile robots.
Predictively monitoring the performance of robot localization systems improves the reliability and efficiency of autonomous navigation.
![Predictively monitoring the performance of a robot localization system in a challenging environment](./images/localization_example.png)

This example (which can be found in [flowcean/examples/ros_offline](https://github.com/flowcean/flowcean/tree/main/examples/robot_localization_failure)
) is assuming the robot utilizes the Robot Operating System(ROS) which is a software framework that is widely used in the robotics community. We are using the version [ROS 2 Humble](https://docs.ros.org/en/humble/index.html).
ROS allows users to record robotics data, such as sensor data or internal data of running processes in so called *rosbags*. Flowcean supports rosbag data using its `RosbagLoader` environment. Here is a code snippet from the `run.py`.

```python
from flowcean.environments.rosbag import RosbagLoader

environment = RosbagLoader(
            path="rec_20241021_152106",
            topics={
                "/amcl_pose": [
                    "pose.pose.position.x",
                    "pose.pose.position.y",
                ],
                "/momo/pose": [
                    "pose.position.x",
                    "pose.position.y",
                ],
                "/scan": [
                    "ranges",
                ],
                "/particle_cloud": ["particles"],
                "/position_error": ["data"],
                "/heading_error": ["data"],
            },
            msgpaths=[
                "/opt/ros/humble/share/sensor_msgs/msg/LaserScan.msg",
                "/opt/ros/humble/share/nav2_msgs/msg/ParticleCloud.msg",
                "/opt/ros/humble/share/nav2_msgs/msg/Particle.msg",
            ],
        )
```

The `RosbagLoader` requires a list of ROS topics to be loaded along with a specification which fields of the ROS message should be extracted. Here is a brief explanation of the topics that are loaded in this example:

- `/amcl_pose`: this is the pose that the localization algorithm of the robot estimates. AMCL stands for Adaptive Monte Carlo Localization and is a commonly used localization approach using a range sensor, e.g. a LiDAR sensor.
- `/momo/pose`: this is the pose provided by a motion capturing system that is installed in our lab. This data is used to determine the error of the data provided by AMCL.
- `/scan`: this is the data from the LiDAR sensor.
- `/particle_cloud`: AMCL is a probabilistic localization approach that uses a set of particles (possible poses of the robot) to describe the probability distribution for the robot in space.
- `/position_error`: The position error calculated using the euclidean distance.
- `/heading_error`: The heading error in degrees.

## Learning Approach

The goal of this example is to predict localization failures in robotic systems by analyzing sensor data and localization information. To achieve this, we process rosbag recordings into visual representations and train a Convolutional Neural Network (CNN) to classify whether a robot is experiencing localization issues.

### Data Processing

The localization data is processed in several steps:

1. The raw rosbag data is loaded and transformed using a series of custom transforms:
   - `Collapse`: Handles map data from multiple messages
   - `DetectDelocalizations`: Identifies points where localization quality drops
   - `SliceTimeSeries`: Divides data into segments without containing localization reset events
   - `LocalizationStatus`: Determines if the robot is properly localized based on position and heading thresholds

2. The transformed data is converted into image-based features for the CNN cropped around the robot's position:
   - **Map Image**: A top-down view of the occupancy grid map
   - **Scan Image**: A visualization of the LiDAR scan data
   - **Particle Cloud Image**: A representation of the AMCL particle distribution

These three images are stacked together as input channels for the CNN model.

### Model Architecture

The model uses a CNN architecture specifically designed to analyze the spatial relationships in the image data:

```python
CNN(
    image_size=config.architecture.image_size,
    in_channels=3,  # map, scan, particles
    learning_rate=config.learning.learning_rate,
)
```

The hyperparameters for training are defined in the configuration file, including:

- Image size: 150x150 pixels
- Physical width: 15 meters
- Learning rate: 0.0001
- Batch size: 128
- Training epochs: 50

### Evaluation

The model is evaluated on a separate test dataset using several classification metrics:

- Accuracy
- F1-Score
- Precision
- Recall
- Detailed classification report

## Data Exploration

The example also provides a visualization tool using Dash to explore the dataset interactively:

```sh
cd examples/robot_localization_failure
uv run explore.py
```

This interactive dashboard allows you to:

- View the full map with robot positions
- See the cropped map, scan, and particle images used for training
- Track the position error over time
- Analyze individual data points throughout the recording

## Run this example

To run this example first make sure you followed the [installation instructions](../getting_started/prerequisites.md) to setup python and `uv`.
Now you can run the example using

```sh
cd examples/robot_localization_failure
uv run run.py
```
