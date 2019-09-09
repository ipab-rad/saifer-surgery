#!/usr/bin/env python
import rospy
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Pose
import ros_numpy
import open3d as o3d

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from copy import copy
import math

class ContourFollower():

    def __init__ (self):

        rospy.init_node('contour_follower',anonymous=True)
        self.sample_density = rospy.get_param('sample_density',default=0.025)
	self.sat_solver_scale = rospy.get_param('sat_solver_scale',default=10000)
	self.sat_solver_timeout = rospy.get_param('sat_solver_timeout',default=10)
	self.min_points = rospy.get_param('min_points_selected',default=100)

        self.pc_sub = rospy.Subscriber('/rviz_selected_points',PointCloud2,self.pc_callback)
        self.traj_pub = rospy.Publisher('/contour_path',Path,queue_size=10)

        rospy.spin()

    def get_dist_dict(self,locations):
        distances = {}
        for from_counter, from_node in enumerate(locations):
            distances[from_counter] = {}
            for to_counter, to_node in enumerate(locations):
                if from_counter == to_counter:
                    distances[from_counter][to_counter] = 0
                else:
                    distances[from_counter][to_counter] = (int(self.sat_solver_scale*math.hypot((from_node[0] - to_node[0]), (from_node[1] - to_node[1]))))
        return distances

    def travelling_salesman(self,points):

        dist_m = self.get_dist_dict(points)
        manager = pywrapcp.RoutingIndexManager(points.shape[0],1,0)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return dist_m[from_node][to_node]

        routing = pywrapcp.RoutingModel(manager)
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.time_limit.seconds = self.sat_solver_timeout
        rospy.logdebug('Sat solving...')

        assignment = routing.SolveWithParameters(search_parameters)
        if assignment:
            plan = []
            index = routing.Start(0)
            while not routing.IsEnd(index):
                plan.append(copy(manager.IndexToNode(index)))

                index = assignment.Value(routing.NextVar(index))
            print(plan)
            return plan
        else:
            rospy.logerr('No solution to TSP.')
            return [] 

    def get_orientation(self):

	pose = Pose()
	pose.orientation.w = 1/np.sqrt(2)
	pose.orientation.z = 1/np.sqrt(2)
	pose.orientation.y = 0
	pose.orientation.x = 0
	
        return pose

    def pc_callback(self,points):

        if points.width < self.min_points:
            rospy.logwarn('Selection too small, select more points.')
            return
        else:
            header = points.header
            xyz = np.copy(ros_numpy.numpify(points))

            rospy.logdebug('Processing pointcloud...')
            pcd = o3d.geometry.PointCloud()
            xyz = np.array(xyz.tolist())
            pcd.points = o3d.utility.Vector3dVector(xyz)

            rospy.logdebug('Removing outliers...')
            cl, ind = o3d.geometry.radius_outlier_removal(pcd,nb_points=16,radius=0.08)
            pcd_clean = o3d.geometry.select_down_sample(cl, ind)

#            print('Estimating normals...')
#            o3d.geometry.estimate_normals(pcd_clean,search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
            #o3d.geometry.orient_normals_to_align_with_direction(pcd_clean,orientation_reference=[0.,0., -1.])
#	    o3d.geometry.orient_normals_towards_camera_location(pcd_clean)
            rospy.logdebug('Downsampling points...')
            downpcd = o3d.geometry.voxel_down_sample(pcd_clean, voxel_size=self.sample_density)
            points = np.asarray(downpcd.points)
            norms = np.asarray(downpcd.normals)
            
	    rospy.logdebug("Points to plan through: ",points.shape[0])
	   
            rospy.logdebug('Travelling salesman...')
            route = self.travelling_salesman(points.copy())

            rospy.logdebug('Publishing path')
            path = Path()
            path.header = header

            for r in route:
                pose = PoseStamped()
		pose.pose.orientation = self.get_orientation().orientation
                pose.header = header
                pose.pose.position.x = points[r,0]
                pose.pose.position.y = points[r,1]
                pose.pose.position.z = points[r,2]
		
                path.poses.append(pose)

            self.traj_pub.publish(path)           
#	    o3d.visualization.draw_geometries([downpcd]) 

if __name__ == '__main__':
    cf = ContourFollower()

