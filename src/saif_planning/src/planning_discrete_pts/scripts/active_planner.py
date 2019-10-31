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

def kernel(dist):
    return np.exp(dist**2 / -2)

def acquisition(m, s, scale=.3):
    return m + scale * s 

class ActivePlanner(object):

    def __init__(self, target_img, vfile, efile, group_name, search_dist=1):
        self.target_img = target_img
        self.training_pts = []
        self.training_labels = []
        self.PG = PlanningGraph(vfile, efile)
        self.search_dist = search_dist
        self.group_name = group_name




    def run(self):
   
        rospy.init_node('active_planner', anonymous=False)
        im_sub = message_filters.Subscriber("/camera/color/image_raw", Image, queue_size=1)
        joints_sub = message_filters.Subscriber("/blue/joint_states",JointState, queue_size=1)
        # message_filters.Subscriber("/kinect2/sd/image_depth_rect",Image)

        synched_sub = message_filters.ApproximateTimeSynchronizer([im_sub, joints_sub], queue_size=1, slop=0.05)
        synched_sub.registerCallback(self.callback)
        rospy.spin()

    def chooseNextView(self, position):
        # get candidate set using graph, train gp
	    print("current position: " + str(self.PG.findClosestNode(position)))
        cand_pts = self.PG.getNodesWithinDist(position, self.search_dist)
        print("cand pts: " + str(cand_pts))
        cand_pts = [self.PG.index2state(c) for c in list(cand_pts)]


        GP = GaussianProcessRegressor(kernel=None, alpha=0.001, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, normalize_y=True, copy_X_train=True, random_state=None)
        GP.fit(self.training_pts, self.training_labels)
        print("training labels: " + str(self.training_labels))
        means, stds = GP.predict(cand_pts, return_std=True)
        print("means: " + str(means))

        scores = [acquisition(m, s) for (m, s) in zip(means, stds)]
        print("scores: " + str(scores))
        best_index = np.argmax(np.array(scores))
        best_view = cand_pts[best_index]
        print("best view: " + str(self.PG.findClosestNode(best_view)))
        pp.planAndExecuteFromWaypoints(position, best_view, self.PG, self.group_name, max_dist = .5)



    def callback(self, img, joint_state): # use eef
        cv_image = CvBridge().imgmsg_to_cv2(img, "bgr8")
#        cv2.imshow(cv_image)

        reward = self.imageCompare(self.toFeatureRepresentation(cv_image, (img.height, img.width, 3)))
        print("reward: " + str(reward))
        position = joint_state.position
        print("position: {}, index: {}".format(position, self.PG.findClosestNode(position)))
        self.training_pts.append(position)
        self.training_labels.append(reward)

        next_view = self.chooseNextView(position)

    def toFeatureRepresentation(self, img, img_shape=(480,640,3)):
        img = np.expand_dims(img, axis=0)

        model = InceptionV3(include_top=False, weights='imagenet', input_tensor=None, input_shape=(480,640,3), pooling='avg', classes=1000)

        return np.array(model.predict(img)).flatten()


    def imageCompare(self, img):
        target = self.toFeatureRepresentation(self.target_img)

        return np.dot(target, img)/(np.linalg.norm(target) * np.linalg.norm(img))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--vfile", default="test_graph_pts.npy", help="File path for saving vertices")
    parser.add_argument("--efile", default="test_graph_edges.npy", help="File path for saving edges")
    parser.add_argument("--group_name", default="blue_arm", help="Name of moveit move group")
    args, unknown_args = parser.parse_known_args()



    target_im = cv2.imread('left0000.jpg')

    ap = ActivePlanner(target_im, args.vfile, args.efile, args.group_name)

    ap.run()


