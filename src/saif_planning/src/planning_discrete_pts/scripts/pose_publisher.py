#!/usr/bin/env python
from cv_bridge import CvBridge, CvBridgeError
from sklearn.gaussian_process import GaussianProcessRegressor 
from keras.applications.inception_v3 import InceptionV3
import tensorflow as tf
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') # append back in order to import rospy
import rospy 
import message_filters
import argparse
from add_pts import PlanningGraph
from sensor_msgs.msg import Image
import std_msgs
import path_plan as pp 
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseArray, Pose, PoseStamped 
import numpy as np
import moveit_commander
import random
import threading 
import matplotlib.pyplot as plt
import pylab as pl
from IPython import display



if __name__ == "__main__":

    rospy.init_node('pose_publisher', anonymous=True)
    
    group = moveit_commander.MoveGroupCommander("blue_arm")
    
    msg = PoseArray()
    poses = np.load("data/cycle_stitch3_poses.npy")
    pub = rospy.Publisher('all_poses2', PoseArray, queue_size=1)
    pub_current = rospy.Publisher('current_pose2', PoseStamped, queue_size=1)
    pose = Pose()
    pose_list = []
    #for node in self.PG.getNodes():
    
    #print("poses: {}".format(poses))
    for p in poses:
        pose = Pose()
        #geom_msg.Point(x=rt.tvec[0], y=rt.tvec[1], z=rt.tvec[2])
        pose.position.x = p[0]
        pose.position.y = p[1]
        pose.position.z = p[2]
        pose.orientation.x = p[3]
        pose.orientation.y = p[4]
        pose.orientation.z = p[5]
        pose.orientation.w = p[6]
        pose_list.append(pose)
            
    msg.poses = pose_list
    #print(pose_list)
    
    rate = rospy.Rate(10)
    
    while not rospy.is_shutdown():
        h = std_msgs.msg.Header()
        h.stamp = rospy.Time.now()
        h.frame_id = "ceiling"
        msg.header = h
        wpose = group.get_current_pose()
        #print(wpose)
        wpose.header.stamp = rospy.Time.now()
        pub_current.publish(wpose)
        pub.publish(msg)
        rate.sleep()
        
        