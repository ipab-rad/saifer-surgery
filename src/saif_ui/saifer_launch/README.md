# Launch files

Three launch files are available, for red, blue and dual arms. I recommend launching both arms at once, as this is more stable code.
We assume familiarity with ros - so make sure you work through the ros tutorials first. This launch file launches robot drivers, with moveit planning and jointtrajectoryaction control interfaces.

### Motion planning and control

<img align="right" alt="" src="https://github.com/ipab-rad/saifer-surgery/blob/master/docs/images/arms.gif" width="150" /> Launch arms, plan and execute using MoveIt!
```
roslaunch saifer_launch dual.launch
```

In terms of planning, we have two primary moveit planning groups available - 'red_arm' and 'blue_arm'.

The tutorial [here](http://docs.ros.org/kinetic/api/moveit_tutorials/html/doc/move_group_python_interface/move_group_python_interface_tutorial.html) shows how to interface with moveit planning groups in python.

