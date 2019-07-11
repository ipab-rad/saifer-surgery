#!/usr/bin/env python
import sys, time, collections
import copy, math
import rospy
import tf
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
import numpy as np
import argparse

import actionlib

def wait_for_state_update(box_name="table", box_is_known=False, box_is_attached=False, timeout=4):
   
   start = rospy.get_time()
   seconds = rospy.get_time()
   while (seconds - start < timeout) and not rospy.is_shutdown():
      # Test if the box is in attached objects
      attached_objects = scene.get_attached_objects([box_name])
      is_attached = len(attached_objects.keys()) > 0

      # Test if the box is in the scene.
      # Note that attaching the box will remove it from known_objects
      is_known = box_name in scene.get_known_object_names()

      # Test if we are in the expected state
      if (box_is_attached == is_attached) and (box_is_known == is_known):
         return True

      # Sleep so that we give other threads time on the processor
      rospy.sleep(0.1)
      seconds = rospy.get_time()

   # If we exited the while loop without returning then we timed out
   return False

def add_collision_table(pose,size,box_name="table", timeout=4):
   rospy.sleep(0.1)

   p = geometry_msgs.msg.PoseStamped()
   p.header.frame_id = robot.get_planning_frame()
   p.pose.position.x = pose[0] # fwd
   p.pose.position.y = pose[1] # left/right
   p.pose.position.z = pose[2]
   p.pose.orientation.w = 1.
   scene.add_box(box_name, p, size)
   
   return wait_for_state_update(box_name=box_name, box_is_known=True)

def remove_collision_table(box_name="table", timeout=4):

   scene.remove_world_object(box_name)
   return wait_for_state_update(box_name=box_name, box_is_known=False)


if __name__ == "__main__":

   parser = argparse.ArgumentParser(description='Add/remove objects from the planing scene.')
   parser.add_argument('--add', action='store_true', default=False, help='Render the env')

   rospy.init_node('config_scene',anonymous=True)

   robot = moveit_commander.RobotCommander()
   scene = moveit_commander.PlanningSceneInterface()

   rospy.sleep(1)

   success = add_collision_table([0,0,1.9],[6,2,1],box_name='table')
   print("Add object success:", success)
   success = add_collision_table([0,0,-0.555555],[6,6,1],box_name='ceiling')
   print("Add object success:", success)
   success = add_collision_table([0.75+0.7,0.6,0.25],[0.3,0.3,0.5],box_name='kinect1')
   print("Add object success:", success)
   success = add_collision_table([0.75+0.7,-0.6,0.25],[0.3,0.3,0.5],box_name='kinect2')
   print("Add object success:", success)
   success = add_collision_table([-0.75-0.7,0.6,0.25],[0.3,0.3,0.5],box_name='kinect3')
   print("Add object success:", success)
   success = add_collision_table([-0.75-0.7,-0.6,0.25],[0.3,0.3,0.5],box_name='kinect4')
   print("Add object success:", success)





   # success = remove_collision_table(box_name="table")
   
print("Add object success:", success)
