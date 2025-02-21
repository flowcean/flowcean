# flowcean_ros package

This package provides an interface for using Flowcean in ROS 2 Humble.

## Pre-requisites

To use this package, you need to have ROS 2 installed on your system. You can follow the instructions on the [ROS 2 installation page](https://docs.ros.org/en/humble/Installation.html) to install ROS 2 on your system. You will also need to [create a workspace](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Creating-A-Workspace/Creating-A-Workspace.html) to build the package.

## Installation

To be able to use the latest features of Flowcean, you need to clone the Flowcean repository and install it from a local project path. Then build the flowcean_ros package, as some features may not be included in the latest PyPi release.

Here are the exact steps to install the package:

```bash
cd ~/catkin_ws/src
git clone https://github.com/flowcean/flowcean
# create a virtual environment inside the flowcean_ros package
cd flowcean/flowcean_ros
uv venv --python=3.10
source venv/bin/activate
pip install -e ~/catkin_ws/src/flowcean

# build the package
cd ~/catkin_ws
colcon build --packages-select flowcean_ros
source install/setup.bash # or setup.zsh
```

## Usage

Specify which topics to subscribe to in `config/topic_info.yaml` file. Specify the transforms in the data_preprocessor.py file.

If you make changes, don't forget to build the package and source the setup file:

```bash
cd ~/catkin_ws
colcon build --packages-select flowcean_ros
source install/setup.bash # or setup.zsh
```

Then run the following command to start the node:

```bash
ros2 run flowcean_ros flowcean_ros_node
```
