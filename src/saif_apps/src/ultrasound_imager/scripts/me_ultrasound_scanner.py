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

from visual_irl.max_ent_reward import max_ent_reward_model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

class image_processor:

	def __init__(self,beta=15.0):
		
		pkg_path = rospkg.RosPack().get_path('visual_irl')
		self.reward_model = max_ent_reward_model(fol_path=pkg_path+'/scripts/data/*',vae_path=pkg_path+'/scripts/visual_irl/logs/')
		self.reward_model.build_map_model(load=True)
		self.bridge = CvBridge()
		self.image_sub = rospy.Subscriber("/ultrasound_streamer/image_raw",Image,self.callback,queue_size=1)
		self.latent = None
		self.robot = moveit_commander.RobotCommander()
		group_name = "red_arm"
		self.group = moveit_commander.MoveGroupCommander(group_name)
		self.beta = beta 
		self.logpath = pkg_path+'/results/max_ent/'
		self.spin()
		
	def randargmax(self,b):
		return np.argmax(np.random.random(b.shape) * (b==b.max()))

	def spin(self):
		self.go_to_start()
		x_pos_list = np.linspace(self.init_pose.pose.position.x-0.025,self.init_pose.pose.position.x+0.025,20)
		y_pos_list = np.linspace(self.init_pose.pose.position.y-0.025,self.init_pose.pose.position.y+0.025,20)
		z_pos_list = np.linspace(self.init_pose.pose.position.z,self.init_pose.pose.position.z+0.03,20)

		xx,yy,zz = np.meshgrid(x_pos_list,y_pos_list,z_pos_list)
		pos = np.vstack((xx.ravel(),yy.ravel(),zz.ravel())).T
		rate = rospy.Rate(0.5)
		pos_list = []
		reward_list = []
		gp = GaussianProcessRegressor(kernel=RBF(length_scale_bounds=[0.00001, 0.01]),alpha=2)

		next_bin = np.random.randint(pos.shape[0])
		while not rospy.is_shutdown():
	
			p = pos[next_bin,:] + 0.0005*np.random.randn(3)
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
				reward_mu,reward_sig = self.reward_model.gp.predict(self.latent,return_std=True)
				pos_list.append(p)
				reward_list.append(reward_mu)
				
				gp.fit(np.array(pos_list),np.array(reward_list))
				mu,sig = gp.predict(pos,return_std=True)
				
				aq = mu.ravel()+self.beta*sig.ravel()
				print(aq.shape)
				next_bin = self.randargmax(aq)
				self.save(pos_list,reward_list)
				print('Current reward value: '+str(reward_mu))
	
			rate.sleep()
			
	def save(self,poses,rewards):
		np.savetxt(self.logpath+'poses.txt',np.array(poses))
		np.savetxt(self.logpath+'rewards.txt',np.array(rewards))

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
