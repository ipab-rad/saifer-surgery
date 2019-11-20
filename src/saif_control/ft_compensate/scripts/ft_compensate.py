#!/usr/bin/env python
import rospy

from geometry_msgs.msg import WrenchStamped
import moveit_commander
import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import time

class FT_COMP():

    def __init__(self):
        rospy.init_node('ft_compensator', anonymous=True)

        self.pub = rospy.Publisher("/red/robotiq_ft_wrench_compensated",WrenchStamped,queue_size=1)

        self.robot = moveit_commander.RobotCommander()
        group_name = rospy.get_param("ft_group_name","red_arm")
        self.group = moveit_commander.MoveGroupCommander(group_name)

        data_path = rospy.get_param("ft_compensate_path","./data/")
	
        F = np.genfromtxt(data_path+'F_'+group_name+'.txt')
        T = np.genfromtxt(data_path+'T_'+group_name+'.txt')
        theta = np.genfromtxt(data_path+'theta_'+group_name+'.txt')

        self.Freg_x = KNeighborsRegressor(n_neighbors=15).fit(theta,F[:,0])
        self.Freg_y = KNeighborsRegressor(n_neighbors=15).fit(theta,F[:,1])
        self.Freg_y = KNeighborsRegressor(n_neighbors=15).fit(theta,F[:,2])
#	self.Freg = DecisionTreeRegressor(max_depth=20).fit(theta,F)
        self.Treg_x = KNeighborsRegressor(n_neighbors=15).fit(theta,T[:,0])#DecisionTreeRegressor(max_depth=20).fit(theta,T)
        self.Treg_y = KNeighborsRegressor(n_neighbors=15).fit(theta,T[:,1])#DecisionTreeRegressor(max_depth=20).fit(theta,T)
        self.Treg_z = KNeighborsRegressor(n_neighbors=15).fit(theta,T[:,2])#DecisionTreeRegressor(max_depth=20).fit(theta,T)


    def run(self):
        rospy.Subscriber("/red/robotiq_ft_wrench", WrenchStamped, self.callback,queue_size=1)	
	rospy.spin()

    def callback(self,msg):

        pose = self.group.get_current_pose().pose
       	theta = R.from_quat(np.array((pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w)))
	theta = theta.as_euler('zxy').reshape(1.-1)
        self.Fx = self.Freg_x.predict(theta)
        self.Fy = self.Freg_y.predict(theta)
        self.Fz = self.Freg_z.predict(theta)
        self.Tx = self.Treg_x.predict(theta)
        self.Ty = self.Treg_y.predict(theta)
        self.Tz = self.Treg_z.predict(theta)

	msg_new = msg	
        msg_new.wrench.force.x = msg.wrench.force.x - self.Fx
       	msg_new.wrench.force.y = msg.wrench.force.y - self.Fy
        msg_new.wrench.force.z = msg.wrench.force.z - self.Fz

       	msg_new.wrench.torque.x = msg.wrench.torque.x - self.Tx
        msg_new.wrench.torque.y = msg.wrench.torque.y - self.Ty
       	msg_new.wrench.torque.z = msg.wrench.torque.z - self.Tz

	f_diff = np.sqrt(msg_new.wrench.force.x**2 + msg_new.wrench.force.y**2 + msg_new.wrench.force.z**2)
	print(f_diff)
        self.pub.publish(msg_new)


if __name__ == '__main__':
    ft = FT_COMP()
    ft.run()
