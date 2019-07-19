#!/usr/bin/env python

import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import WrenchStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
#from ur_kin_py.ur10_kin_py import forward, inverse
from ur_kin_py.kin import Kinematics

from trac_ik_python.trac_ik import IK
import tf

import numpy as np
import moveit_commander
import actionlib

class Compensation():
	def __init__(self,Fg = [90.06,-42.79,-52.23],F_thresh = 10.0, K = 0.015):
		rospy.init_node('ft_feedback_control')

		self.Fg = np.array(Fg)
		self.F_thresh = F_thresh
		self.K = K
		self.Tmax = 2.0

		self.js = None
		self.wind_up = 0
		self.IK = IK('ceiling','blue_robotiq_ft_frame_id',solve_type='Distance')

		self.robot = moveit_commander.RobotCommander()
		self.group = moveit_commander.MoveGroupCommander('blue_arm')
	
		self.listener = tf.TransformListener()
	
		rospy.Subscriber("/blue/joint_states",JointState,self.joint_callback)
		rospy.Subscriber("/robotiq_ft_wrench",WrenchStamped,self.ft_callback,queue_size=1)

		self.client = actionlib.SimpleActionClient('blue/follow_joint_trajectory',FollowJointTrajectoryAction)

		rospy.spin()

	def joint_callback(self,data):
		self.js = np.array(data.position)
		self.js_names = data.name

	def ft_callback(self,data):
		ft = np.array([data.wrench.force.x,data.wrench.force.y,data.wrench.force.z]) 
		self.wind_up += 1
		self.wind_up = min(self.wind_up,100)
		df = np.sum(np.abs(ft-self.Fg))
#		print('FT dev',df,self.wind_up)
		if (df > self.F_thresh) and (self.wind_up > 50):

			if self.js is not None:
				self.listener.waitForTransform(self.group.get_planning_frame(),"/blue_robotiq_ft_frame_id",data.header.stamp,rospy.Duration(4.0))				
				p = PointStamped()
				p.header = data.header
				p.point.x = self.K*(ft-self.Fg)[0]
				p.point.y = self.K*(ft-self.Fg)[1] 
				p.point.z = self.K*(ft-self.Fg)[2]				

				pose = self.group.get_current_pose().pose
				print('New force deviation:')
				print(ft,self.Fg)
				print(p.point.x,p.point.y,p.point.z)

				pnew = self.listener.transformPoint('ceiling',p)
			       
				print('Original pose:',pose.position.z,pose.position.x,pose.position.y)
				pose.position.z = pnew.point.z
				pose.position.y = pnew.point.y
				pose.position.x = pnew.point.x
				print('New pose',pose.position.z,pose.position.x,pose.position.y)

				sol = None
				retries = 0
				while not sol and retries < 20:

					sol = self.IK.get_ik(self.js,pose.position.x,pose.position.y,pose.position.z,pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w)
					retries +=1
				
				if retries < 20:
				
					print(sol)
					print(self.js)
					print(np.sum((self.js -np.array(sol))**2))
					if (np.sum((self.js -np.array(sol))**2) < 0.5):
						self.publish(sol)
			self.wind_up = 0
		else:
			self.Fg = 0.1*ft + 0.9*self.Fg

	def publish(self,q):
		if q is not None:

			goal = FollowJointTrajectoryGoal()

			jtp = JointTrajectoryPoint()
			jtp.positions = list(q)
			jtp.velocities = [0.0] * len(jtp.positions)
			jtp.time_from_start = rospy.Time(self.Tmax)

			goal.trajectory.points.append(jtp)
			goal.trajectory.header.stamp = rospy.Time.now()
			goal.trajectory.joint_names = self.js_names
			self.client.send_goal(goal)
			self.client.wait_for_result()

if __name__ == '__main__':
	ft_compensator = Compensation()
