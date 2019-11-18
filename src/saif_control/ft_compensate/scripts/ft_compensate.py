#!/usr/bin/env python
import rospy

from geometry_msgs.msg import WrenchStamped
import moveit_commander
import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.tree import DecisionTreeRegressor

class FT_COMP:

    def __init__(self):
        rospy.init_node('ft_compensator', anonymous=True)

        rospy.Subscriber("/red/robotiq_ft_wrench", WrenchStamped, self.callback)
        pub = rospy.Publisher("/red/robotiq_ft_wrench_compensated",WrenchStamped,queue_size=10)

        self.robot = moveit_commander.RobotCommander()
        group_name = rospy.get_param("ft_group_name","red_arm")
        self.group = moveit_commander.MoveGroupCommander(group_name)

        data_path = rospy.get_param("ft_compensate_path","./data/")

        F = np.genfromtxt(data_path+'F.txt')
        T = np.genfromtxt(data_path+'T.txt')
        theta = np.genfromtxt(data_path+'theta.txt')

        self.Freg = DecisionTreeRegressor(max_depth=15).fit(theta,F)
        self.Treg = DecisionTreeRegressor(max_depth=15).fit(theta,T)

        rospy.spin()

    def callback(self,msg):
        pose = self.group.get_current_pose().pose
        theta = R.from_quat(np.array((pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w)))
        F = self.Freg.predict(theta.as_euler('zyx'))
        T = self.Treg.predict(theta.as_euler('zyx'))

        msg_new = msg
        msg.Wrench.Force.x = msg.Wrench.Force.x - F[0]
        msg.Wrench.Force.x = msg.Wrench.Force.y - F[1]
        msg.Wrench.Force.x = msg.Wrench.Force.z - F[2]

        msg.Wrench.Force.x = msg.Wrench.Torque.x - T[0]
        msg.Wrench.Force.x = msg.Wrench.Torque.y - T[1]
        msg.Wrench.Force.x = msg.Wrench.Torque.z - T[2]

        pub.publish(msg)


if __name__ == '__main__':
    FT_COMP()
