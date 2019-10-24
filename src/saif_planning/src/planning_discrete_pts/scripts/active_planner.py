#!/usr/bin/env python
from cv_bridge import CvBridge, CvBridgeError
import sklearn
from keras.applications.inception_v3 import InceptionV3
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') # append back in order to import rospy
import rospy 

from add_pts import PlanningGraph
import path_plan as pp 

def kernel(dist):
    return np.exp(dist**2 / -2)

def acquisition(m, s):
    return m + .3 * s 

class ActivePlanner(object):

    def __init__(self, target_img, vfile, efile, group_name, search_dist=5):
        self.target_img = target_img
        self.training_pts = []
        self.training_labels = []
        self.PG = PlanningGraph(vfile, efile)
        self.search_dist = search_dist
        self.group_name = group_name

        im_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
        joints_sub = message_filters.Subscriber("joint_states",JointState)
        # message_filters.Subscriber("/kinect2/sd/image_depth_rect",Image)

        synched_sub = message_filters.ApproximateTimeSynchronizer([im_sub, joints_sub], queue_size=250, slop=0.05)
        synched_sub.registerCallback(callback)

    def chooseNextView(self, position):
        # get candidate set using graph, train gp
        cand_pts = self.PG.getNodesWithinDist(position, self.search_dist)

        GP = sklearn.gaussian_process.GaussianProcessRegressor(kernel=None, alpha=1e-10, optimizer=’fmin_l_bfgs_b’, n_restarts_optimizer=0, normalize_y=True, copy_X_train=True, random_state=None)
        GP.fit(self.training_pts, self.training_labels)
        means, stds = GP.predict(cand_pts, return_std=True)

        scores = [acquisition(m, s) for (m, s) in zip(means, stds)]
        best_index = np.argmax(np.array(scores))
        best_view = cand_pts[best_index]

        pp.planAndExecuteFromWaypoints(position, best_view, self.PG, max_dist = .5)



    def callback(self, img, joint_state): # use eef
        cv_image = self.bridge.imgmsg_to_cv2(img, "bgr8")

        reward = self.imageCompare(self.toFeatureRepresentation(cv_image, (img.height, img.width, 3)))

        position = joint_state.position
        self.training_pts.append(position)
        self.training_labels.append(reward)

        next_view = self.chooseNextView(position)

    def toFeatureRepresentation(self, img, img_shape=(480,640,3)):
        img = np.expand_dims(img, axis=0)

        model = keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_tensor=None, input_shape=img_shape, pooling='avg', classes=1000)

        return np.flatten(np.array(model.predict(img)))


    def imageCompare(self, img):
        target = self.toFeatureRepresentation(self.target_img)

        return np.dot(target, img)/(np.linalg.norm(target) * np.linalg.norm(img))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--vfile", default="test_graph_pts.npy", help="File path for saving vertices")
    parser.add_argument("--efile", default="test_graph_edges.npy", help="File path for saving edges")
    parser.add_argument("--group_name", default="ur10", help="Name of moveit move group")
    args, unknown_args = parser.parse_known_args()

    rospy.init_node('active_planner', anonymous=False)

    target_im = cv2.imread('left0000.jpg')

    ap = ActivePlanner(target_im, args.vfile, args.efile, args.group_name)

    rospy.spin()


