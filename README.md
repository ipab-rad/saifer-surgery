# Safe AI for surgical assistance

Developing methods for assistive robots to learn correct actions from human experts, in the domain of surgical assistance in operating rooms. As AI based decision-making methods make their way from internet applications to more safety-critical physical systems, questions about the robustness of the models and policies become increasingly more important. This project is developing methods to address this through novel methods for learning specifications from human experts and synthesising policies that are correct by construction. These developments are grounded in the domain of surgical assistance with autonomous robots in the operating room.

## Requirements:

Ubuntu 16.04, ROS Kinetic or Ubuntu 18.04, ROS Melodic

## Installation:

- Clone this repository
```
cd saifer-surgery
catkin_make
source devel/setup.bash
```

## Current functionality:

Visualise dual arm setup and test moveit planning configuration
```
roslaunch ur10_moveit demo.launch
```
