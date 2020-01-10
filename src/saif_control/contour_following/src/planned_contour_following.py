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
	self.traj_pub = rospy.Publisher('/corrected_path',Path,queue_size=10)
        self.header = None
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander(group)

        self.listener = TransformListener()
	self.vertical_offset = rospy.get_param('countour_offset', 0.3)
	self.repositioning_offset = rospy.get_param('repositioning_offset', 0.3)

        rospy.spin()

    def path_cb(self,msg):
        planning_frame = self.group.get_planning_frame()
	self.listener.waitForTransform(msg.header.frame_id, planning_frame, rospy.Time.now(), rospy.Duration(1.0))
        
	if True:#self.listener.frameExists(msg.header.frame_id) and self.listener.frameExists('/camera_color/optical_frame'):
            rospy.loginfo('Planning path through waypoints')
	    pose = self.group.get_current_pose().pose
	    pose_start = copy.deepcopy(pose)
            pose.position = self.listener.transformPose(planning_frame,msg.poses[0]).pose.position
	    pose.position.z = pose.position.z - self.repositioning_offset
	    self.group.set_pose_target(pose)
	    plan = self.group.go(wait=True)
	    if not plan:
		return
	    self.group.stop()
	    print('Moved to start point')
            waypoints = [copy.deepcopy(self.group.get_current_pose().pose)]
	    path = Path()
	    path.header.frame_id = planning_frame
            p_pose = geometry_msgs.msg.PoseStamped()
	    for j in range(len(msg.poses)):
		print(waypoints[-1])
		pose = waypoints[-1]
                pose.position = self.listener.transformPose(planning_frame,msg.poses[j]).pose.position
		pose.position.z = pose.position.z - self.vertical_offset
                waypoints.append(copy.deepcopy(pose))
		p_pose.pose = pose
		path.poses.append(copy.deepcopy(p_pose))
	    self.traj_pub.publish(path)
            print(waypoints[-1])
            plan,fraction = self.group.compute_cartesian_path(waypoints[1:],0.2,0.0)
	    for j in range(len(plan.joint_trajectory.points)):
		plan.joint_trajectory.points[j].time_from_start = rospy.Duration(j)
            rospy.loginfo('Planning done')
            self.group.execute(plan,wait=True)

	    p_pose = path.poses[-1]
	    p_pose.pose.position.z = p_pose.pose.position.z
	    self.group.set_pose_target(p_pose.pose)
	    plan = self.group.go(wait=True)
	    if not plan:
		return
	    self.group.stop()
	    print('Moved above end')

	    self.group.set_pose_target(pose_start)
	    plan = self.group.go(wait=True)
	    if not plan:
		return
	    self.group.stop()
	    print('Moved to start point')
	    
        else:
            rospy.logerr('No transform between '+planning_frame+' and '+msg.header.frame_id)

if __name__ == '__main__':
    pb = ContourFollower('red_arm') 

