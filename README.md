# Safe AI for surgical assistance

<img align="right" alt="" src="https://github.com/ipab-rad/saifer-surgery/blob/master/docs/images/front.png" width="400" />  As AI based decision-making methods make their way from internet applications to more safety-critical physical systems, questions about the robustness of the models and policies become increasingly more important. This project is developing methods to address this through novel methods for learning specifications from human experts and synthesising policies that are correct by construction. These developments are grounded in the domain of surgical assistance with autonomous robots in the operating room.

## Requirements:

Ubuntu 16.04, ROS Kinetic or Ubuntu 18.04, ROS Melodic

## Installation:

- Install the [required packages](https://github.com/ipab-rad/saifer-surgery/wiki/Required-packages), clone this repository, and
```
cd saifer-surgery
git submodule update --init --recursive
catkin_make
source devel/setup.bash
```
___
## Current functionality (Warning: this is semi-functional research code under active development):

### Motion planning and control

<img align="right" alt="" src="https://github.com/ipab-rad/saifer-surgery/blob/master/docs/images/arms.gif" width="150" /> Launch arms, plan and execute using MoveIt!
```
roslaunch saifer_launch dual.launch
```

___

### 3D mouse control with spacenav

<img align="right" alt="" src="http://wiki.ros.org/spacenav_node?action=AttachFile&do=get&target=spacenav.png" width="120" /> Inverse dynamics on the red arm, with gripper opening and closing. See the [spacenav_teleop](./src/saif_control/spacenav_teleop) node for more detail.
```
roslaunch saifer_launch dual.launch
roslaunch spacenav_teleop teleop.launch
```

___
### User defined contour following

<img align="right" alt="" src="https://github.com/ipab-rad/saifer-surgery/blob/master/src/saif_ui/contour_launch/ims/surface.gif" width="150" /> Select a pointcloud region in Rviz and follow this surface using MoveIt! Cartesian waypoint following and position control. This requires calibrated offsets depending on the tool used for contour following. See the [contour launch](./src/saif_ui/contour_launch) node for more detail.
```
roslaunch contour_launch contour.launch
```





