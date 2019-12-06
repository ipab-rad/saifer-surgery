#!/usr/bin/env python
from add_pts import PlanningGraph
import rospy
import moveit_commander
import argparse

def planJointWaypoints(start, stop, graph, max_dist = .5):

    start_node, d1 = graph.findClosestNode(start)
    stop_node, d2 = graph.findClosestNode(stop)

    if d1 > .5 or d2 > .5:
        print("no safe solution guaranteed")
        return

    waypoints = graph.findShortestPath(start_node, stop_node)

    return waypoints

    
def planAndExecuteFromWaypoints(start, stop, graph, move_group, max_dist = .5):

    group = moveit_commander.MoveGroupCommander(move_group)

    start_node, d1 = graph.findClosestNode(start)
    stop_node, d2 = graph.findClosestNode(stop)

    if d1 > .5 or d2 > .5:
        print("no safe solution guaranteed")
        return

    waypoints = graph.findShortestPath(start_node, stop_node)
    #print("waypoints: " + str(waypoints))

    for w in waypoints:
	#print("moving to waypoint: " + str(w))
        w = graph.index2state(w)
        
        group.go(w, wait=True)
        group.stop()

    #(plan, fraction) = group.compute_cartesian_path(
    #                               waypoints,   
    #                               0.01,        
    #                               0.0)

    #group.execute(plan, wait=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--vfile", default="test_graph_pts.npy", help="File path for saving vertices")
    parser.add_argument("--efile", default="test_graph_edges.npy", help="File path for saving edges")
    parser.add_argument("--group_name", default="blue_arm", help="Name of moveit move group")
    parser.add_argument("--index", default=1, help="Index of node to move to")
    parser.add_argument("--robot_name", default="ur10", help="Name of robot")
    args, unknown_args = parser.parse_known_args()

    if args.robot_name == "ur10":
        group_name = "blue_arm"
    elif args.robot_name == "pr2":
        group_name = "left_arm"


    rospy.init_node('path_plan', anonymous=True)

    gb = PlanningGraph(args.vfile, args.efile, args.robot_name)

    group = moveit_commander.MoveGroupCommander(group_name)

    wpose = group.get_current_pose().pose
    print("wpose: " + str(wpose))
    print(wpose.position)
    joint_vals = group.get_current_joint_values()

    print("current joint vals" + str(joint_vals))
    nodes = gb.getNodes()

    print("nodes" + str(nodes))
    print("edges: " + str(gb.connections))
    cur_index, min_dist = gb.findClosestNode(joint_vals)
    current = gb.index2state(cur_index)
    print("min dist to graph: " + str(min_dist))    

    print("start at: " + str(current) + " index: " + str(cur_index))

    #for n in nodes[1:]:
        #print("attempting to reach: " + str(gb.state2index(n)))
    #    planAndExecuteFromWaypoints(current, n, gb, group_name, max_dist = .5)
    #    current = n
        #print("moved to node: " + str(gb.state2index(n)))


    planAndExecuteFromWaypoints(current, nodes[int(args.index)], gb, group_name, max_dist = .5)



    # target = group.get_joint_value_target() 

    # waypoints = planJointWaypoints(joint_vals, target, gb)

    # (plan, fraction) = group.compute_cartesian_path(
    #                                waypoints,   
    #                                0.01,        
    #                                0.0)

    # group.execute(plan, wait=True)


    rospy.spin()
