# Safe AI for surgical assistance

<img align="right" alt="" src="https://github.com/ipab-rad/saifer-surgery/blob/master/docs/images/front.png" width="400" />  As AI based decision-making methods make their way from internet applications to more safety-critical physical systems, questions about the robustness of the models and policies become increasingly more important. This project is developing methods to address this through novel methods for learning specifications from human experts and synthesising policies that are correct by construction. These developments are grounded in the domain of surgical assistance with autonomous robots in the operating room.

This repository is under an MIT license, but pulls in git submodules that may be subject to different licenses. ### NB While still being prototyped, some submodules may be private - the public branch should remain accessible at all times.

## Requirements:

Ubuntu 16.04, ROS Kinetic
## Installation:

- Install the [required packages](https://github.com/ipab-rad/saifer-surgery/wiki/Required-packages), clone this repository, and
```
cd saifer-surgery
git submodule update --init --recursive
catkin_make
source devel/setup.bash
```

## How to contribute:

- Bug fixes, code cleanups and functionality improvements are encouraged, along with testing and documenting
- Help fight the hard coded parameters
- At the moment, the preferred integration strategy is: 
  - Build standalone library/ tool with core functionality (eg. [visual IRL](https://github.com/ipab-rad/visual_irl/tree/4bd514caab754971353f7e77a481f564f747c311)) and link using a git submodule
  - Develop a small app showing how to use this functionality (eg. [autonomous ultrasound scanner](https://github.com/ipab-rad/saifer-surgery/tree/master/src/saif_apps/src/ultrasound_imager))
  - Document both with READMEs
___
## Current functionality 
### (Warning: this is semi-functional research code under active development):

___
### Motion planning and control with MoveIt!

<img align="right" alt="" src="https://github.com/ipab-rad/saifer-surgery/blob/master/docs/images/arms.gif" width="150" /> [Launch arms](https://github.com/ipab-rad/saifer-surgery/tree/master/src/saif_ui/saifer_launch), plan and execute using MoveIt!
```
roslaunch saifer_launch dual_arm.launch
```

___

### 3D mouse control with spacenav

<img align="right" alt="" src="http://wiki.ros.org/spacenav_node?action=AttachFile&do=get&target=spacenav.png" width="120" /> Inverse dynamics on the red arm, with gripper opening and closing. See the [spacenav_teleop](https://github.com/ipab-rad/spacenav_teleop) node for more detail.
```
roslaunch saifer_launch dual_arm.launch
roslaunch spacenav_teleop teleop.launch
```

___
### User defined contour following

<img align="right" alt="" src="https://github.com/ipab-rad/saifer-surgery/blob/master/src/saif_apps/contour_launch/ims/surface.gif" width="150" /> Select a pointcloud region in Rviz and follow this surface using MoveIt! Cartesian waypoint following and position control. This requires calibrated offsets depending on the tool used for contour following. See the [contour launch](./src/saif_apps/contour_launch) node for more detail.
```
roslaunch contour_launch contour.launch
```

___
### Kinesthetic demonstration

<img align="right" alt="" src="https://github.com/ipab-rad/ft_compliance/blob/master/ims/demo.gif" width="150" />Physically move red arm (only red has this functionality at present, to avoid risk of crushing blue camera. Make sure to zero the ft sensor before using. See the [ft_compliance](https://github.com/ipab-rad/ft_compliance) node for more detail.
```
rosservice call /red/robotiq_ft_sensor_acc "command_id: 0 command: 'SET ZRO'"
roslaunch ft_compliance compliance.launch
```

___
### Autonomous ultrasound scanning

<img align="right" alt="" src="https://github.com/ipab-rad/saifer-surgery/blob/master/docs/images/scan.gif" width="150"/>This app uses Bayesian optimisation to move an ultrasound scanner over an ultrasound imaging phantom in search of a tumour like object. The app relies on a reward model learned from demonstration ultrasound image sequences using the [visual IRL](https://github.com/ipab-rad/visual_irl/tree/4bd514caab754971353f7e77a481f564f747c311) package. 

Launch robot and ultrasound image streamer
```
roslaunch saifer_launch dual_arm.launch
roslaunch ultrasound_epiphan us.launch
rosrun ultrasound_imager pairwise_ultrasound_scanner.py
```


