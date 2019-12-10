#!/usr/bin/env python
from add_pts import PlanningGraph
import rospy
import moveit_commander
import argparse
from scipy.spatial.transform import Rotation as Rot

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
	print("moving to waypoint: " + str(w))
        w = graph.index2state(w)
        
        group.go(w, wait=True)
        group.stop()

    #(plan, fraction) = group.compute_cartesian_path(
    #                               waypoints,   
    #                               0.01,        
    #                               0.0)

    #group.execute(plan, wait=True)


def recordData(group_name, gb):
    group = moveit_commander.MoveGroupCommander(group_name)
    joint_vals = group.get_current_joint_values()

    print("current joint vals" + str(joint_vals))
    nodes = gb.getNodes()

    print("nodes" + str(nodes))
    print("edges: " + str(gb.connections))
    cur_index, min_dist = gb.findClosestNode(joint_vals)
    current = gb.index2state(cur_index)

    planAndExecuteFromWaypoints(current, nodes[int(args.index)], gb, group_name, max_dist = .5)

    if self.robot == "pr2":
        im_sub = message_filters.Subscriber("/l_forearm_cam/image_color", Image, queue_size=1)
        joints_sub = message_filters.Subscriber("/l_arm_controller/state", JointTrajectoryControllerState, queue_size=1) 
    elif self.robot == "ur10":
        im_sub = message_filters.Subscriber("/camera/color/image_raw", Image, queue_size=1)
        joints_sub = message_filters.Subscriber("/blue/joint_states", JointState, queue_size=1)
    else:
        print("robot name not valid")
        exit() 

    synched_sub = message_filters.ApproximateTimeSynchronizer([im_sub, joints_sub], queue_size=1, slop=0.05)
    synched_sub.registerCallback(self.callback)
    rate = rospy.Rate(10) # 10hz

    done = False

    while not rospy.is_shutdown() and not done:
        current_node, _ = self.PG.findClosestNode(position)
        print("at node: " + str(current_node))

        self.next_view = self.PG.index2state((current_node + 1) % len(self.PG.getNodes()))

        pp.planAndExecuteFromWaypoints(position, self.next_view, self.PG, self.group_name, max_dist = .5)

def plotSimilarities(state_index, pose_file, feature_file, uncertainties):
    pf = np.load(pose_file)
    ff = np.load(feature_file)

    X = pf[:, 0]
    Y = pf[:, 1]
    Z = pf[:, 2]

    normals = []

    for i in range(np.size(X)):
        quat = [pf[i, 3], pf[i, 4], pf[i, 5], pf[i, 6]]
        rot_mat = Rot.from_quat(quat).as_dcm()
        normal = np.matmul(rot_mat, np.array(0, 0, 1))
        normals.append(normal)

    normals = np.array(normals)

    U = normals[:, 0]
    V = normals[:, 1]
    W = normals[:, 2]
     
    C = [np.dot(ff[state_index], fr)/(np.linalg.norm(ff[state_index]) * np.linalg.norm(fr))]

    from mpl_toolkits.mplot3d import Axes3D 
    fig = plt.figure() 
    ax = fig.add_subplot(111, projection='3d') 
    ax.quiver(X, Y, Z, U, V, W, C)

    plt.plot(X,Y,Z,’o’, markersize=uncertainties)


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
