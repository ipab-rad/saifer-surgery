#!/usr/bin/env python  
import copy
import sys
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import numpy as np
from robotiq_ft_sensor.srv import sensor_accessor

moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('resetter',anonymous=True)
robot = moveit_commander.RobotCommander()

group_name = "red_arm"
group = moveit_commander.MoveGroupCommander(group_name)

print('Moving to start pose')

#start_pose = [1.870887041091919,-2.2503507773028772,1.8966856002807617,-1.222773853932516,1.6179306507110596,-0.34370404878725225]
start_pose = [2.0953848361968994, -2.162729565297262, 2.03804349899292, -1.4732831160174769, 1.612236738204956, -0.5673425833331507]
group.go(start_pose, wait=True)
group.stop()

print('Zeroing ft sensor')
rospy.wait_for_service('/red/robotiq_ft_sensor_acc')
zero_sensor = rospy.ServiceProxy('/red/robotiq_ft_sensor_acc',sensor_accessor)
print(zero_sensor(8,'SET ZRO'))
#rosservice.call_service("/red/robotiq_ft_sensor_acc",[0,'SET ZRO'])



