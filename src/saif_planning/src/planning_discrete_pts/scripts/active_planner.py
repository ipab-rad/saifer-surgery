#!/usr/bin/env python
from cv_bridge import CvBridge, CvBridgeError
from sklearn.gaussian_process import GaussianProcessRegressor 
from keras.applications.inception_v3 import InceptionV3
import tensorflow as tf
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
import moveit_commander
import random
import threading 
import matplotlib.pyplot as plt
import pylab as pl
from IPython import display
from keras.applications.inception_v3 import preprocess_input
from sklearn.gaussian_process.kernels import RBF
#from pr2_controllers_msgs.msg import JointTrajectoryControllerState

def kernel(dist):
    return np.exp(dist**2 / -2)

def acquisition(m, s, scale=.3):
    return m + scale * s 

class ActivePlanner(object):

    def __init__(self, target_img, vfile, efile, robot, target_name, search_dist=1, init_pose=None, visualize=False):
        self.target_img = target_img
        self.training_pts = []
        self.training_labels = []
        self.trajectory = []
        self.PG = PlanningGraph(vfile, efile, robot)
        self.search_dist = search_dist
        self.rewards = []
        self.target_name = target_name
        self.views = 0
        self.position = None
        self.all_imgs = []
        self.trial_imgs = []
        self.trial_num = 1
        self.done = False
	self.visualize = visualize

        self.next_view = None
        self.robot = robot

        if self.robot == "pr2":
            self.group_name = "left_arm"
        elif self.robot == "ur10":
            self.group_name = "blue_arm"
        else:
            print("robot name not valid")
            exit() 

        self.update = False

        self.GP = GaussianProcessRegressor(kernel=RBF(0.1), alpha=0.001, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, normalize_y=True, copy_X_train=True, random_state=None)
        self.model = InceptionV3(include_top=False, weights='imagenet', input_tensor=None, input_shape=(480,640,3), pooling='avg', classes=1000)
        rospy.init_node('active_planner', anonymous=False)
        self.setInitialPose(init_pose)

        self.graph = tf.get_default_graph()

    def setInitialPose(self, init_index=None):
        group = moveit_commander.MoveGroupCommander(self.group_name)

        wpose = group.get_current_pose().pose
        joint_vals = group.get_current_joint_values()

        nodes = self.PG.getNodes()

        cur_index, min_dist = self.PG.findClosestNode(joint_vals)
        current = self.PG.index2state(cur_index)

        #print("start at: " + str(current) + " index: " + str(cur_index))
        if init_index == None:
            index = random.randint(1, len(nodes) - 1)
        else:
            index = init_index
        print("initial index: " + str(index))
        pp.planAndExecuteFromWaypoints(current, nodes[index], self.PG, self.group_name, max_dist = .5)

        self.done = False

    def run(self, num_views=20, cycle=True):

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

        if rospy.is_shutdown():
            print("rospy shutdown")
        while not rospy.is_shutdown() and self.views < num_views:

            print("view: " + str(self.views))
            if self.update is True:
		if cycle == False:
               	    self.chooseNextView()
                else:
 		    self.cycleViews()

            rate.sleep()

    def cycleViews(self):
        position = self.position

        current_node, _ = self.PG.findClosestNode(position)
        print("at node: " + str(current_node))

        self.next_view = self.PG.index2state((current_node + 1) % len(self.PG.getNodes()))

        pp.planAndExecuteFromWaypoints(position, self.next_view, self.PG, self.group_name, max_dist = .5)

        self.views += 1
        self.update = False
        print("num views: " + str(self.views))

    def chooseNextView(self):
  
        print("current position: " + str(self.PG.findClosestNode(self.position)))

        points = self.training_pts
        labels = self.training_labels
        position = self.position
         
        try:
            self.GP.fit(points, labels)
        except ValueError:
            print("gp fit error")
            self.update = False
            return
        current_index, _ = self.PG.findClosestNode(position)
        

        # SAMPLE MPC
        #best_score, best_index = self.getMaxScore(current_index, depth=10)
        #best_view = self.PG.index2state(best_index)

        #self.trajectory.append(current_index)

        #print("gp preds: " + str(self.GP.predict(self.PG.getNodes())))
        #print("num nodes: " + str(len(self.PG.getNodes())))
        ##pl.plot(range(len(self.PG.getNodes())), self.GP.predict(self.PG.getNodes()))
        ##display.clear_output(wait=True)
        ##display.display(pl.gcf())
        ####

        # TRAJECTORY SAMPLING 
        sampleTs = self.sampleTrajectories(current_index)
        #print("sampled trajectories: " + str(sampleTs))

        #samplePreds = [[self.GP.predict(self.PG.index2state(pts).reshape(1, -1), return_std=True) for pts in traj] for traj in sampleTs]
        destinations = [self.PG.index2state(traj[-1]) for traj in sampleTs] 
        samplePreds = self.GP.predict(destinations, return_std=True)
        samplePreds = zip(samplePreds[0], samplePreds[1])
        #print("sample preds: " + str(samplePreds))

        scores = [acquisition(*pred) for pred in samplePreds]        

        #print("scores: " + str(scores))
        best_index = np.argmax(np.array(scores))
        best_view = self.PG.index2state(sampleTs[best_index][-1])

        #self.trajectory.append(sampleTs[best_index][-1])
        ####
        #### ALL WITHIN DIST
#         cand_pts = self.PG.getNodesWithinDist(current_index, 20)
#         preds = self.GP.predict([self.PG.index2state(n) for n in cand_pts], return_std=True)
#         scores = [acquisition(*pred) for pred in zip(preds[0], preds[1])] 
#         print("scores: " + str(scores))
#         best_index = cand_pts[np.argmax(np.array(scores))]
#         best_view = self.PG.index2state(best_index)
        ####
        
        if self.visualize == True:
            pl.clf()
            means, stds = self.GP.predict(self.PG.getNodes(), return_std=True)
	    x = range(len(self.PG.getNodes()))
            pl.plot(range(len(self.PG.getNodes())), [acquisition(*pred) for pred in zip(means, stds)] )
	    pl.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([means - 1.9600 * stds,
                        (means + 1.9600 * stds)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
            #pl.plot(range(len(self.PG.getNodes())), stds)
            
            display.clear_output(wait=True)
            display.display(pl.gcf())

        print("best view: " + str(self.PG.findClosestNode(best_view)))
        pp.planAndExecuteFromWaypoints(position, best_view, self.PG, self.group_name, max_dist = .5)
        self.views += 1
        #print("view: " + str(self.views))
	    
        self.update = False
        self.next_view = best_view

    def sampleTrajectories(self, node, num_t=10, depth=20):
        

        children = self.PG.getNodesWithinDist(node, 1)
        to_expand = [children[random.randint(0, len(children) - 1)] for c in range(num_t)]
        trajectories = [[t] for t in to_expand]

        i = 1
        while i < depth:
            i += 1
            children = [self.PG.getNodesWithinDist(c, 1) for c in to_expand]
            to_expand = [c[random.randint(0, len(c) - 1)] for c in children if len(c) > 0]

            for k in range(num_t):
                trajectories[k].append(to_expand[k])

        return trajectories

    def getMaxScore(self, node, depth=5, branch=10, fullFirstLayer=True):
        children = self.PG.getNodesWithinDist(node, 1) 
        #print("children of {}: {}".format(node, children))
        #to_expand = [children[random.randint(0, len(children) - 1)] for c in range(branch)]
        if fullFirstLayer == True:
            to_expand = [children[random.randint(0, len(children) - 1)] for c in range(branch)]
        else:
            to_expand = children

        preds = self.GP.predict([self.PG.index2state(t) for t in to_expand], return_std=True)
        scores = [acquisition(*pred) for pred in zip(preds[0], preds[1])] 
        
        if depth == 1:
            return max(scores), to_expand[np.argmax(np.array(scores))]
            
        scores = [scores[i] + self.getMaxScore(to_expand[i], depth - 1, max(1, int(branch/2)), False)[0] for i in range(0, len(scores))]
        
        return max(scores), to_expand[np.argmax(np.array(scores))]

    def callback(self, img, joint_state): # use eef
        print("entering callback")
        cv_image = CvBridge().imgmsg_to_cv2(img, "bgr8")
        
        position = joint_state.position
        try:
            reward = self.imageCompare(self.toFeatureRepresentation(cv_image, (img.height, img.width, 3)))

            print("reward: " + str(reward))
            self.training_pts.append(position)
            self.training_labels.append(reward)
       
        except ValueError:
            print("something isn't working right")
            reward = None
            print(self.toFeatureRepresentation(self.target_img, (img.height, img.width, 3)))
        #finally:


        if len(self.training_pts) > 1000:
            index = random.randint(0, len(self.training_pts) - 1)
            self.training_pts.pop(index)
            self.training_labels.pop(index)

        #print("training labels: {}".format(self.training_labels))
        self.position = position

        print("current {}, next {}".format(position, self.next_view))

        if reward is not None and (self.next_view is None or np.linalg.norm(np.array(position) - np.array(self.next_view)) < .1) and self.update is False and self.done is False:
            print("appending reward: {}, trial {}, view {}".format(reward, self.trial_num, self.views))
            self.rewards.append(reward)
            self.trajectory.append(self.PG.findClosestNode(position)[0])
            print("writing image: " + "data/{}_t{}_v{}.jpg".format(self.target_name, self.trial_num, self.views))
            cv2.imwrite("data/{}_t{}_v{}.jpg".format(self.target_name, self.trial_num, self.views), cv_image)
            self.update = True
            print("rewards: {}".format(self.rewards))
            print("trajectory: {}".format(self.trajectory))

    def toFeatureRepresentation(self, img, img_shape=(480,640,3)):
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        with self.graph.as_default():
            return np.array(self.model.predict(img)).flatten()


    def imageCompare(self, img):
        target = self.toFeatureRepresentation(self.target_img)
        return np.dot(target, img)/(np.linalg.norm(target) * np.linalg.norm(img))

    def saveRewards(self, dirname="data"):
        rewards = [str(tl) for tl in self.rewards]
        traj = [str(pt) for pt in self.trajectory]
        print("rewards: {}, array: {}".format(",".join(rewards), self.rewards))
        with open(dirname + "/" + self.target_name + "_rewards.csv", "ab") as f:
           f.write(",".join(rewards) + "\n")
        with open(dirname + "/" + self.target_name + "_trajectory.csv", "ab") as f:
           f.write(",".join(traj) + "\n")

    def reset(self, saveTrajectory=True):
        self.done = True
        self.training_pts = []
        self.training_labels = []
        
        self.next_view = None 
        self.views = 0
        self.GP = GaussianProcessRegressor(kernel=None, alpha=0.001, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, normalize_y=True, copy_X_train=True, random_state=None)
        self.saveRewards()

        self.rewards = []
        self.trajectory = []
        self.trial_num += 1

        self.setInitialPose()

        
        


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--vfile", default="test_graph_pts.npy", help="File path for saving vertices")
    parser.add_argument("--efile", default="test_graph_edges.npy", help="File path for saving edges")
    parser.add_argument("--robot_name", default="ur10", help="Name of robot")
    args, unknown_args = parser.parse_known_args()

    #targets = ['pink_ball.jpg'] 
    targets = ['liquid.jpg'] #, 
    #targets = ['cupcup.jpg']
    #target_names = ['pink_ball_'] #, 
    target_names = ['liquid'] #, 
    #target_names = ['cup_test']

    #num_views = 92
    num_trials = 10

    for t, n in zip(targets, target_names):
        print("t, n: {}, {}".format(t, n))
        # send to initial position
        target_im = cv2.imread(t)
        #print(np.shape(np.array(target_im)))
        #print(target_im)
        cv2.imshow('target', target_im)
        ap = ActivePlanner(target_im, args.vfile, args.efile, args.robot_name, n, init_pose=None)
        #num_views = len(ap.PG.getNodes()) - 1
        num_views = 15
        while ap.trial_num <= num_trials:
            print("trial: " + str(ap.trial_num))
            ap.run(num_views)
            ap.reset()

        
        #np.save(ap.target_name + "images_cycle_" + str(num_views), ap.all_imgs)


        #sub_thread.exit()
        #ap.run()
        #for i in range(10):
        #    ap.run()
        #    if ap.views >= 20:
        #        print("resetting")
        #        ap.reset()

        # ap.saveRewards("rewards_{}.csv".format(n))

        # np.save("trajectory_{}.npy".format(n), ap.training_pts)


