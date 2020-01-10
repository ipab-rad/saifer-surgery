# Spacenav_teleop

![spacenav image](http://wiki.ros.org/spacenav_node?action=AttachFile&do=get&target=spacenav.png)

`roslaunch spacenav_teleop teleop.launch` 

launches the [spacenav_node](http://wiki.ros.org/spacenav_node) joystick driver, and the spacenav_teleop node provided by this package, which provides inverse dynamics motion control of the red arm. Buttons are mapped to open and close the Robotiq 3f gripper. 

To be used alongside

`roslaunch saifer_launch dual_arm.launch` 

