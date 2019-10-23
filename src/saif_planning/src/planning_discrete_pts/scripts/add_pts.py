#!/usr/bin/env python

import rospy
import moveit_commander
from moveit_commander import conversions
from rospy.numpy_msg import numpy_msg
import numpy as np 
import os
from sensor_msgs.msg import JointState



class PlanningGraph(object):

    def __init__(self, vertex_file, edge_file):

        self.vertex_file = vertex_file
        self.edge_file = edge_file

        print(self.vertex_file)

        if os.path.isfile(vertex_file):
            nodes = list(np.load(vertex_file))
        else:
            print('no such file')
            nodes = []
            
        if os.path.isfile(edge_file):
            connections = list(np.load(edge_file))
        else:
            connections = [] 

        self.nodes = nodes
        self.connections = connections 
        self.current_node = None 

    def dist(self, node1, node2):
        return np.linalg.norm(node1 - node2)

    def getGraphDist(self, node1, node2):
        """number of transitions in shortest path between two nodes"""
        path = self.findShortestPath(node1, node2)
        return len(path) - 1

    def getNodesWithinDist(self, node, dist):
        return [n for n in range(self.nodes) if self.getGraphDist(node, n) <= dist]

    def findShortestPath(self, node_index1, node_index2):

        if node_index1 == node_index2:
            return [node_index1]

        path_found = False
        parent_dict = {}
        q = queue.Queue()
        q.put(node_index1)
        visited = {node_index1}

        while not path_found:
            current = q.get()
            edges = [e.remove(current).pop() for e in self.connections if current in e]
            children = [c for c in edges if c not in visited]

            for c in children:
                parent_dict[c] = current
                q.put(c)
                visited.add(c)

            if node_index2 is in children:
                path_found = True 

        point = node_index2
        waypoints = [point]

        while point != node_index1:
            point = parent_dict[point]
            waypoints.insert(0, point)

        return waypoints

    def findClosestNode(self, state):
        dists = [self.dist(state, n) for n in self.nodes]
        return dists.index(min(dists)), min(dists)


    def state2index(self, state):
        return self.nodes.index(state)

    def index2state(self, index):
        return self.nodes[index]

    def storeNode(self, data):

        position = np.array(data.position)
        thresh = .1
        add_thresh = .5
        dist_list = [self.dist(node, position) for node in self.nodes]

        if len(self.nodes) == 0:
            self.nodes.append(position)
            data_index = len(self.nodes) - 1

        else:
            min_dist = min(dist_list)
            if min_dist < thresh:
                data_index = dist_list.index(min_dist) 
            elif min_dist > add_thresh:
                self.nodes.append(position)
                data_index = len(self.nodes) - 1

            else:
                data_index = self.current_node
        if len(self.nodes) > 1:
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

    
    gb = PlanningGraph('test_graph_pts.npy', 'test_graph_edges.npy')
    gb.buildGraph()


    