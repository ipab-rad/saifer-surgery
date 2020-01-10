#!/usr/bin/env python

import roslib
roslib.load_manifest('ultrasound_imager')
import rospkg
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import moveit_commander
import numpy as np
import copy

from visual_irl.pairwise_reward  import pairwise_reward_model


class image_processor:

    def __init__(self):

	pkg_path = rospkg.RosPack().get_path('visual_irl')
	self.reward_model = pairwise_reward_model(fol_path=pkg_path+'/scripts/data/*',vae_path=pkg_path+'/scripts/visual_irl/logs/')
	self.reward_model.build_map_model(load=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/ultrasound_streamer/image_raw",Image,self.callback,queue_size=1)
	self.latent = None
	self.robot = moveit_commander.RobotCommander()
	group_name = "red_arm"
	self.group = moveit_commander.MoveGroupCommander(group_name)

	self.spin()

    def spin(self):
	self.go_to_start()
	x_pos_list = np.linspace(self.init_pose.pose.position.x-0.015,self.init_pose.pose.position.x+0.025,10)
	y_pos_list = np.linspace(self.init_pose.pose.position.y-0.015,self.init_pose.pose.position.y+0.025,10)
	z_pos_list = np.linspace(self.init_pose.pose.position.z,self.init_pose.pose.position.z+0.02,10)

	xx,yy,zz = np.meshgrid(x_pos_list,y_pos_list,z_pos_list)
	pos = np.vstack((xx.ravel(),yy.ravel(),zz.ravel())).T

        rate = rospy.Rate(0.5)
        while not rospy.is_shutdown():
    
	    p = pos[np.random.randint(pos.shape[0]),:]
    
            pose_goal = copy.deepcopy(self.init_pose.pose)
            pose_goal.position.x = p[0]
            pose_goal.position.y = p[1]
            pose_goal.position.z = p[2]
    
            waypoints = []
            waypoints.append(copy.deepcopy(pose_goal))

            (plan, fraction) = self.group.compute_cartesian_path(waypoints,0.0015,0.0)       
    
            self.group.execute(plan, wait=True)    
            self.group.stop()

	    if self.latent is not None:
   	        print('Current reward value: '+str(self.reward_model.gp.predict(self.latent)))
    
            rate.sleep()


    def go_to_start(self):
        start_pose = [1.870887041091919,-2.2503507773028772,1.8966856002807617,-1.222773853932516,1.6179306507110596,-0.34370404878725225]
	self.group.go(start_pose, wait=True)
	self.group.stop()

	self.init_pose = self.group.get_current_pose()

    def callback(self,data):
        try:
            image = self.bridge.imgmsg_to_cv2(data, "bgr8")

	    w = self.reward_model.vae_model.w
            h = self.reward_model.vae_model.h
            im = np.mean(cv2.resize(image[180:700,500:1020,:],(w,h)),axis=-1)
	    self.latent = self.reward_model.vae_model.encoder.predict(im.reshape(-1,w,h,1)/255.0)[0]	

        except CvBridgeError as e:
            print(e)
  
def main(args):
    moveit_commander.roscpp_initialize(args)
    rospy.init_node('image_processor', anonymous=True)
    ic = image_processor()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    main(sys.argv)
