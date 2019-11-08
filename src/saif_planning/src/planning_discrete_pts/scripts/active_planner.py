#!/usr/bin/env python
from cv_bridge import CvBridge, CvBridgeError
from sklearn.gaussian_process import GaussianProcessRegressor 
from keras.applications.inception_v3 import InceptionV3
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') # append back in order to import rospy
import rospy 
import message_filters
import argparse
from add_pts import PlanningGraph
from sensor_msgs.msg import Image
import path_plan as pp 
from sensor_msgs.msg import JointState
import numpy as np
from pr2_controllers_msgs.msg import JointTrajectoryControllerState

def kernel(dist):
    return np.exp(dist**2 / -2)

def acquisition(m, s, scale=.3):
    return m + scale * s 

class ActivePlanner(object):

    def __init__(self, target_img, vfile, efile, robot, target_name, search_dist=1):
        self.target_img = target_img
        self.training_pts = []
        self.training_labels = []
        self.PG = PlanningGraph(vfile, efile, robot)
        self.search_dist = search_dist
        self.rewards = []
        self.target_name = target_name
        self.views = 0

        self.next_view = None
        #self.image_topic = image_topic
        self.robot = robot

        if self.robot == "pr2":
            self.group_name = "left_arm"
        elif self.robot == "ur10":
            self.group_name = "blue_arm"
        else:
            print("robot name not valid")
            exit() 

        self.GP = GaussianProcessRegressor(kernel=None, alpha=0.001, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, normalize_y=True, copy_X_train=True, random_state=None)

    def setInitialPose(self):
        group = moveit_commander.MoveGroupCommander(self.group_name)

        wpose = group.get_current_pose().pose
        print("wpose: " + str(wpose))
        print(wpose.position)
        joint_vals = group.get_current_joint_values()

        print("current joint vals" + str(joint_vals))
        nodes = self.PG.getNodes()

        print("nodes" + str(nodes))
        print("edges: " + str(self.PG.connections))
        cur_index, min_dist = self.PG.findClosestNode(joint_vals)
        current = self.PG.index2state(cur_index)
        print("min dist to graph: " + str(min_dist))    

        print("start at: " + str(current) + " index: " + str(cur_index))

        index = random.randint(1, len(nodes) - 1)

        pp.planAndExecuteFromWaypoints(current, nodes[index], self.PG, self.group_name, max_dist = .5)

    def run(self, num_views=20):
        self.setInitialPose()
        rospy.init_node('active_planner', anonymous=False)

        if self.robot == "pr2":
            im_sub = message_filters.Subscriber("/l_forearm_cam/image_color", Image, queue_size=1)
            joints_sub = message_filters.Subscriber("/l_arm_controller/state", JointTrajectoryControllerState, queue_size=1) 
        elif self.robot == "ur10":
            im_sub = message_filters.Subscriber("/camera/color/image_raw", Image, queue_size=1)
            joints_sub = message_filters.Subscriber("/blue/joint_states", JointState, queue_size=1)
        else:
            print("robot name not valid")
            exit() 
        
        # message_filters.Subscriber("/kinect2/sd/image_depth_rect",Image)

        synched_sub = message_filters.ApproximateTimeSynchronizer([im_sub, joints_sub], queue_size=1, slop=0.05)
        synched_sub.registerCallback(self.callback)
        rospy.spin()

        if self.veiws >= num_views:
            break

    def chooseNextView(self, position):
        # get candidate set using graph, train gp

	print("current position: " + str(self.PG.findClosestNode(position)))
        # cand_pts = self.PG.getNodesWithinDist(self.PG.state2index(position), self.search_dist)
        # print("cand pts: " + str(cand_pts))
        # cand_pts = [self.PG.index2state(c) for c in list(cand_pts)]

        sampleTs = self.sampleTrajectories(self.PG.state2index(position))

        print("sampled trajectories: " + str(sampleTs))

        samplePreds = [[self.GP.predict(pts, return_std=True) for pts in traj] for traj in sampleTs]

        print("sample preds: " + str(samplePreds))

        scores = [sum([acquisition(*pred) for pred in preds]) for preds in samplePreds]

        # self.GP.fit(self.training_pts, self.training_labels)
        # print("training labels: " + str(self.training_labels))
        # means, stds = self.GP.predict(cand_pts, return_std=True)
        # print("means: " + str(means))

        #scores = [acquisition(m, s) for (m, s) in zip(means, stds)]
        print("scores: " + str(scores))
        best_index = np.argmax(np.array(scores))
        best_view = cand_pts[best_index]
        self.next_view = best_view
        print("best view: " + str(self.PG.findClosestNode(best_view)))
        pp.planAndExecuteFromWaypoints(position, best_view, self.PG, self.group_name, max_dist = .5)


    def sampleTrajectories(self, node, num_t=10, depth=5):
        

        children = self.PG.getNodesWithinDist(node, 1)
        to_expand = [children[random.randint(0, len(children) - 1)] for c in range(num_t)]
        trajectories = [[t] for t in to_expand]

        i = 1
        while i < depth:
            i += 1
            children = [self.PG.getNodesWithinDist(c, 1) for c in to_expand]
            to_expand = [c[random.randint(0, len(c) - 1)] for c in children]

            for k in range(num_t):
                trajectories[k].append(to_expand[k])

        return trajectories

    def callback(self, img, joint_state): # use eef
        cv_image = CvBridge().imgmsg_to_cv2(img, "bgr8")
#        cv2.imshow(cv_image)

        reward = self.imageCompare(self.toFeatureRepresentation(cv_image, (img.height, img.width, 3)))
        print("reward: " + str(reward))
        position = joint_state.position
        print("position: {}, index: {}".format(position, self.PG.findClosestNode(position)))
        self.training_pts.append(position)
        self.training_labels.append(reward)

        if not self.next_view or np.linalg.norm(np.array(position) - np.array(self.next_view)) < .1:
            self.rewards.append(reward)
            self.next_view = self.chooseNextView(position)
            self.views += 1

    def toFeatureRepresentation(self, img, img_shape=(480,640,3)):
        img = np.expand_dims(img, axis=0)

        model = InceptionV3(include_top=False, weights='imagenet', input_tensor=None, input_shape=(480,640,3), pooling='avg', classes=1000)

        return np.array(model.predict(img)).flatten()


    def imageCompare(self, img):
        target = self.toFeatureRepresentation(self.target_img)

        return np.dot(target, img)/(np.linalg.norm(target) * np.linalg.norm(img))

    def saveRewards(self, fname):
        with open(fname, "w+") as f:
            f.write(",".join(self.training_labels))

    def reset(self, saveTrajectory=False):
        self.training_pts = []
        self.training_labels = []
        self.next_view = None 
        self.views = 0
        self.GP = GaussianProcessRegressor(kernel=None, alpha=0.001, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, normalize_y=True, copy_X_train=True, random_state=None)
        self.saveRewards("rewards_{}.csv".format(self.target_name))
        if saveTrajectory == True:
            np.save("trajectory_{}.npy".format(self.target_name), self.training_pts)

    # def setNewTarget(self, target_file, target_name):
    #     self.reset()
    #     self.target_img = cv2.imread(target_file, target_name)
        
        


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--vfile", default="test_graph_pts.npy", help="File path for saving vertices")
    parser.add_argument("--efile", default="test_graph_edges.npy", help="File path for saving edges")
    parser.add_argument("--robot_name", default="ur10", help="Name of robot")
    args, unknown_args = parser.parse_known_args()

    targets = ['left0000.jpg']
    target_names = ['torso']

    for t, n in zip(targets, target_names):
        # send to initial position
        target_im = cv2.imread(t)
        ap = ActivePlanner(target_im, args.vfile, args.efile, args.robot_name, n)
        ap.setNewTarget(t, n)
        for i in range(10):
            ap.run()
            ap.reset()

    ap.saveRewards("rewards.csv")

    np.save("trajectory.npy", ap.training_pts)


