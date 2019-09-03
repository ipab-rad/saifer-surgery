#!/usr/bin/env python
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState

import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from nav_msgs.msg import Path
import copy

from tf import TransformListener

class ContourFollower():

    def __init__(self,group):
        rospy.init_node('contour_follower',anonymous=True)
        rospy.Subscriber('contour_path',Path,self.path_cb)

        self.header = None
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander(group)

        self.listener = TransformListener()

        rospy.spin()

    def path_cb(self,msg):
        planning_frame = self.group.get_planning_frame()
        if self.listener.frameExists(msg.header.frame_id) and self.listener.frameExists(planning_frame):
            rospy.loginfo('Planning path through waypoints')
            waypoints = []
            for j in range(len(msg.poses)):
                pose = self.listener.transformPose(planning_frame,msg.poses[j]).pose
                waypoints.append(copy.deepcopy(pose))

            plan,fraction = self.group.compute_cartesian_path(waypoints,0.01,0.0)
            rospy.loginfo('Planning done')
            self.group.execute(plan,wait=True)
        else:
            rospy.logerr('No transform between '+planning_frame+' and '+msg.header.frame_id)

if __name__ == '__main__':
    pb = ContourFollower('red_arm') 

