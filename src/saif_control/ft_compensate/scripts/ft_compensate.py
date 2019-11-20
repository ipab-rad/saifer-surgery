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

        self.Freg = KNeighborsRegressor(n_neighbors=15).fit(theta,F)
#	self.Freg = DecisionTreeRegressor(max_depth=20).fit(theta,F)
        self.Treg = KNeighborsRegressor(n_neighbors=15).fit(theta,T)#DecisionTreeRegressor(max_depth=20).fit(theta,T)




    def run(self):
        rospy.Subscriber("/red/robotiq_ft_wrench", WrenchStamped, self.callback,queue_size=1)	
	rospy.spin()

    def callback(self,msg):

        pose = self.group.get_current_pose().pose
       	theta = R.from_quat(np.array((pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w)))
	theta = theta.as_euler('zxy').reshape(1,-1)
        self.F = self.Freg.predict(theta)
	self.T = self.Treg.predict(theta)

	msg_new = msg	
        msg_new.wrench.force.x = msg.wrench.force.x - self.F[0,0]
       	msg_new.wrench.force.y = msg.wrench.force.y - self.F[0,1]
        msg_new.wrench.force.z = msg.wrench.force.z - self.F[0,2]

       	msg_new.wrench.torque.x = msg.wrench.torque.x - self.T[0,0]
        msg_new.wrench.torque.y = msg.wrench.torque.y - self.T[0,1]
       	msg_new.wrench.torque.z = msg.wrench.torque.z - self.T[0,2]

	f_diff = np.sqrt(msg_new.wrench.force.x**2 + msg_new.wrench.force.y**2 + msg_new.wrench.force.z**2)
	print(f_diff)
        self.pub.publish(msg_new)


if __name__ == '__main__':
    ft = FT_COMP()
    ft.run()
