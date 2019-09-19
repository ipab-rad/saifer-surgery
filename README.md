# Safe AI for surgical assistance

<img align="right" alt="" src="https://github.com/ipab-rad/saifer-surgery/blob/master/docs/images/front.png" width="400" />  As AI based decision-making methods make their way from internet applications to more safety-critical physical systems, questions about the robustness of the models and policies become increasingly more important. This project is developing methods to address this through novel methods for learning specifications from human experts and synthesising policies that are correct by construction. These developments are grounded in the domain of surgical assistance with autonomous robots in the operating room.

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

### Motion planning and control

<img align="right" alt="" src="https://github.com/ipab-rad/saifer-surgery/blob/master/docs/images/arms.gif" width="200" /> Launch arms, plan and execute using MoveIt!
```
roslaunch saifer_launch dual.launch
```

### User defined contour following

<img align="right" alt="" src="https://github.com/ipab-rad/saifer-surgery/blob/master/src/saif_ui/contour_launch/ims/surface.gif" width="200" /> Select a pointcloud region in Rviz and follow this surface using MoveIt! Cartesian waypoint following and position control. This requires calibrated offsets depending on the tool used for contour following. See the [contour launch](./src/saif_ui/contour_launch) node for more detail.
```
roslaunch contour_launch contour.launch
```





