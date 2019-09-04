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

    def __init__ (self,sample_density=0.025):

        rospy.init_node('contour_follower',anonymous=True)
        self.sample_density = sample_density

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
                    distances[from_counter][to_counter] = (int(1000*math.hypot((from_node[0] - to_node[0]), (from_node[1] - to_node[1]))))
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
        search_parameters.time_limit.seconds = 10
        print('Solving...')

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
            print('No solution')
            return []
        #return self.two_opt(points,0.05)

    def orientation_from_vector3D(self,n):

	u = [1,0,0]
	norm = np.linalg.norm(n)
        v = -np.asarray(n)/norm 
	pose = Pose()
	if (np.array_equal(u, v)):
	    pose.orientation.w = 1
            pose.orientation.x = 0
            pose.orientation.y = 0
            pose.orientation.z = 0
        elif (np.array_equal(u, np.negative(v))):
            pose.orientation.w = 0
            pose.orientation.x = 0
            pose.orientation.y = 0
            pose.orientation.z = 1
        else:
            half = [u[0]+v[0], u[1]+v[1], u[2]+v[2]]
            pose.orientation.w = np.dot(u, half)
            temp = np.cross(u, half)
            pose.orientation.x = temp[0]
            pose.orientation.y = temp[1]
            pose.orientation.z = temp[2]
            norm = math.sqrt(pose.orientation.x*pose.orientation.x + pose.orientation.y*pose.orientation.y + 
            pose.orientation.z*pose.orientation.z + pose.orientation.w*pose.orientation.w)
        if norm == 0:
           norm = 1
           pose.orientation.x /= norm
           pose.orientation.y /= norm
           pose.orientation.z /= norm
           pose.orientation.w /= norm
        return pose

    def pc_callback(self,points):

        if points.width < 100:
            print ('Selection too small')
            return
        else:
            print (points.width)
            header = points.header
            xyz = np.copy(ros_numpy.numpify(points))

            print('Processing pointcloud...')
            pcd = o3d.geometry.PointCloud()
            xyz = np.array(xyz.tolist())
            pcd.points = o3d.utility.Vector3dVector(xyz)

            print('Removing outliers...')
            cl, ind = o3d.geometry.radius_outlier_removal(pcd,nb_points=16,radius=0.08)
            pcd_clean = o3d.geometry.select_down_sample(cl, ind)

            print('Estimating normals...')
            o3d.geometry.estimate_normals(pcd_clean,search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
            #o3d.geometry.orient_normals_to_align_with_direction(pcd_clean,orientation_reference=[0.,0., -1.])
	    o3d.geometry.orient_normals_towards_camera_location(pcd_clean)
            print('Downsampling points...')
            downpcd = o3d.geometry.voxel_down_sample(pcd_clean, voxel_size=self.sample_density)
            points = np.asarray(downpcd.points)
            norms = np.asarray(downpcd.normals)
            print("Points to plan through: ",points.shape[0])
            print(norms.shape)
	   
            print('Travelling salesman...')
            route = self.travelling_salesman(points.copy())

            print('Publishing path')
            path = Path()
            path.header = header

            for r in route:
                pose = PoseStamped()
		pose.pose = self.orientation_from_vector3D(norms[r,:])
                pose.header = header
                pose.pose.position.x = points[r,0] + norms[r,0]*0.25
                pose.pose.position.y = points[r,1] + norms[r,1]*0.25
                pose.pose.position.z = points[r,2] + norms[r,2]*0.25
		
                path.poses.append(pose)

            self.traj_pub.publish(path)
            print('Done')
#	    o3d.visualization.draw_geometries([downpcd]) 

if __name__ == '__main__':
    cf = ContourFollower()

