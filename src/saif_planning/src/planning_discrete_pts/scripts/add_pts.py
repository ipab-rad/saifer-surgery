#!/usr/bin/env python

import rospy
import moveit_commander
from moveit_commander import conversions
from rospy.numpy_msg import numpy_msg
import numpy as np 
import os
from sensor_msgs.msg import JointState
import argparse
import Queue
import copy

class PlanningGraph(object):

    def __init__(self, vertex_file, edge_file):

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

    def getNodes(self):
        return self.nodes 

    def dist(self, node1, node2):
        return np.linalg.norm(node1 - node2)

    def getGraphDist(self, node1, node2):
        """number of transitions in shortest path between two nodes"""
        path = self.findShortestPath(node1, node2)
        return len(path) - 1

    # def getNodesWithinDist(self, position, dist):
    #     # TODO make faster
    #     node, _ = self.findClosestNode(position)

    #     return [n for n in range(1, len(self.nodes)) if self.getGraphDist(node, n) <= dist]

    def getNodesWithinDist(self, position, dist=1):
        #node, _ = self.findClosestNode(position)
        edges = [copy.copy(e) for e in self.connections if position in e]
	
        for e in edges:
            e.remove(position)
                
        children = [c.pop() for c in edges]
        #print("children: " + str(children))
        #children = [c for c in children if c not in visited]
        
        if dist == 1:
            return children

        return list(set(children + reduce(lambda x, y: x + y, [getNodesWithinDist(c, dist - 1) for c in children])))

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
            return None, None 
            
        dists = [self.dist(state, n) for n in self.nodes]
        return dists.index(min(dists)), min(dists)


    def state2index(self, state):
        return self.nodes.index(state)

    def index2state(self, index):
        return self.nodes[index]

    def storeNode(self, data):

        position = np.array(data.position)[0:6]
        thresh = .1
        add_thresh = .5
        dist_list = [self.dist(node, position) for node in self.nodes]

        index, min_dist = self.findClosestNode(position)

        if not index or min_dist > add_thresh:
            self.nodes.append(position)
            data_index = len(self.nodes) - 1
            #print('adding node with dist: ' + str(self.dist(self.current_node, position)))

        elif min_dist < thresh:
            data_index = index

        else:
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
            self.addConnection(self.current_node, data_index)

        self.current_node = data_index 


    def addConnection(self, index1, index2):
        if {index1, index2} not in self.connections and index1 != index2:  
            self.connections.append({index1, index2})
        else:
            return 


    def buildGraph(self):

        rospy.init_node('set_pts', anonymous=True)

        rospy.Subscriber("/joint_states", numpy_msg(JointState), self.storeNode)

        rospy.spin()

        np.save(self.vertex_file, np.array(self.nodes))
        np.save(self.edge_file, np.array(self.connections))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--vfile", default="test_graph_pts.npy", help="File path for saving vertices")
    parser.add_argument("--efile", default="test_graph_edges.npy", help="File path for saving edges")
    args, unknown_args = parser.parse_known_args()

    gb = PlanningGraph(args.vfile, args.efile)
    gb.buildGraph()


    
