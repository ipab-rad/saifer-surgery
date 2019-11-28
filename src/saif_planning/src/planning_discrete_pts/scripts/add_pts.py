#!/usr/bin/env python

import rospy
import moveit_commander
from moveit_commander import conversions
from rospy.numpy_msg import numpy_msg
import numpy as np 
import os
from sensor_msgs.msg import JointState
from pr2_controllers_msgs.msg import JointTrajectoryControllerState
import argparse
import Queue
import copy

class PlanningGraph(object):

    def __init__(self, vertex_file, edge_file, robot):

        self.vertex_file = vertex_file
        self.edge_file = edge_file

        print(self.vertex_file)

        if os.path.isfile(vertex_file):
            nodes = list(np.load(vertex_file, allow_pickle=True))
        else:
            print('no such file')
            nodes = []
            
        if os.path.isfile(edge_file):
            connections = list(np.load(edge_file, allow_pickle=True))
        else:
            connections = [] 

        self.nodes = nodes
        self.connections = connections 
        self.current_node = None 
        self.robot = robot

    def getNodes(self):
        return self.nodes 

    def dist(self, node1, node2):
        return np.linalg.norm(node1 - node2)

    def getGraphDist(self, node1, node2):
        """number of transitions in shortest path between two nodes"""
        path = self.findShortestPath(node1, node2)
        return len(path) - 1

    # def getNodesWithinDist(self, position, dist):
    #     node, _ = self.findClosestNode(position)

    #     return [n for n in range(1, len(self.nodes)) if self.getGraphDist(node, n) <= dist]

    def getNodesWithinDist(self, position, dist=1):
        #print("node: " + str(position))
        #node, _ = self.findClosestNode(position)
        edges = [copy.copy(e) for e in self.connections if position in e]
        #print("all edges: " + str(self.connections))
        #print("edges with current pos: " + str(edges))	
        for e in edges:
            e.remove(position)
                
        children = [c.pop() for c in edges]
        #print("children: " + str(children))
        #children = [c for c in children if c not in visited]
        
        if dist == 0:
            return [position]
        #if dist == 1:
        #    return children

        return list(set(children + reduce(lambda x, y: x + y, [self.getNodesWithinDist(c, dist - 1) for c in children])))

    def findShortestPath(self, node_index1, node_index2):

        if node_index1 == node_index2:
            return [node_index1]

        path_found = False
        parent_dict = {}
        q = Queue.Queue()
        q.put(node_index1)
        visited = {node_index1}

        #print("searching for path between: " + str(node_index1) + " and " + str(node_index2))

        while not path_found:
            if q.empty():
                print("ERROR: no valid path")
                exit()
            current = q.get()
	    #print("current: " + str(current))
            edges = [copy.copy(e) for e in self.connections if current in e]
	

            for e in edges:
                e.remove(current)
                
	    #print("edges with current: " + str(edges))
            children = [c.pop() for c in edges]
            #print("children: " + str(children))
            children = [c for c in children if c not in visited]

            for c in children:
                parent_dict[c] = current
                q.put(c)
                visited.add(c)
	    #print("visited: " + str(visited))

            if node_index2 in children:
                path_found = True 

        point = node_index2
        waypoints = []

        while point != node_index1:

            waypoints.insert(0, point)
            point = parent_dict[point]

        return waypoints

    def findClosestNode(self, state):
        if len(self.nodes) == 0:
            print("no stored nodes")
            return None, None 
            
        dists = [self.dist(state, n) for n in self.nodes]
        return dists.index(min(dists)), min(dists)


    def state2index(self, state):
        return self.nodes.index(state)

    def index2state(self, index):
        return self.nodes[index]

    def storeNode(self, data):
        if self.robot == "ur10":
            position = np.array(data.position)[0:6]
        elif self.robot == "pr2":
            position = np.array(data.actual.positions)

     
        thresh = .05
        add_thresh = .2
        dist_list = [self.dist(node, position) for node in self.nodes]

        index, min_dist = self.findClosestNode(position)
#	print("index: {}, min dist: {}".format(index, min_dist))

        if index is None or min_dist > add_thresh:
	   
            self.nodes.append(position)
            data_index = len(self.nodes) - 1
	    #print("adding new node: " + str(data_index))
            print('adding node ' + str(data_index) + ' with dist: ' + str(min_dist) + " from " + str(index))

        elif min_dist < thresh:
            print("position: " + str(position))
            print("close to node {} with dist {}".format(index, min_dist))
            data_index = index

        else:
             print("no significant change, min dist: {}".format(min_dist))
             data_index = self.current_node


        # if len(self.nodes) == 0:
        #     self.nodes.append(position)
        #     data_index = len(self.nodes) - 1

        # else:
        #     min_dist = min(dist_list)
        #     if min_dist < thresh:
        #         data_index = dist_list.index(min_dist) 
        #     elif min_dist > add_thresh:
        #         print('adding node with dist: ' + str(self.dist(self.current_node, position)))
        #         self.nodes.append(position)
        #         data_index = len(self.nodes) - 1

        #     else:
        #         data_index = self.current_node

        if data_index and self.current_node:
            print("adding connection between {} and {}".format(data_index, self.current_node))
            self.addConnection(self.current_node, data_index)

        self.current_node = data_index 


    def addConnection(self, index1, index2):
        if {index1, index2} not in self.connections and index1 != index2:  
            self.connections.append({index1, index2})
        else:
            return 


    def buildGraph(self):

        rospy.init_node('set_pts', anonymous=True)

        if self.robot == "pr2":
            rospy.Subscriber("/l_arm_controller/state", numpy_msg(JointTrajectoryControllerState), self.storeNode)
        elif self.robot == "ur10":
            rospy.Subscriber("/joint_states", numpy_msg(JointState), self.storeNode)
        else:
            print("robot name not valid")
            exit() 

        rospy.spin()

        np.save(self.vertex_file, np.array(self.nodes))
        np.save(self.edge_file, np.array(self.connections))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--vfile", default="cup_graph_pts.npy", help="File path for saving vertices")
    parser.add_argument("--efile", default="cup_graph_edges.npy", help="File path for saving edges")
    parser.add_argument("--robot_name", default="ur10", help="Name of robot")
    args, unknown_args = parser.parse_known_args()

    gb = PlanningGraph(args.vfile, args.efile, args.robot_name)
    gb.buildGraph()


    
