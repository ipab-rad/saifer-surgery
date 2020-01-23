#!/usr/bin/env python
from cv_bridge import CvBridge, CvBridgeError
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') # append back in order to import rospy
import rospy 
import message_filters
from sensor_msgs.msg import Image
import std_msgs
from sensor_msgs.msg import JointState
import numpy as np

class data_collector:

    def __init__(self, num_steps):
        self.image = None
        self.joint_state = None 
        self.step = 0
        self.STEPS = num_steps
        self.joint_states = []

    #def collect_data():

        im_sub = message_filters.Subscriber("/camera/color/image_raw", Image, queue_size=1)
        joints_sub = message_filters.Subscriber("/joint_states", JointState, queue_size=1)


        synched_sub = message_filters.ApproximateTimeSynchronizer([im_sub, joints_sub], queue_size=1, slop=0.05)
        synched_sub.registerCallback(callback)

        

        #while not rospy.is_shutdown():



    def callback(self, img, joint_state): # use eef
            print("entering callback")
            cv_image = CvBridge().imgmsg_to_cv2(img, "bgr8")

            state = joint_state.position[0:6]

            cv2.imwrite("image_data/{}.jpg".format(self.step), cv_image)

            self.joint_states.append(state)

            self.step += 1




if __name__ == "__main__":

    dc = data_collector()

    rospy.init_node('data_collector', anonymous=True)

    try:
       rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down module")
        np.save(np.array("joint_data.npy", dc.joint_states))

    # rate = rospy.Rate(10) # 10hz

    # if rospy.is_shutdown():
    #     print("rospy shutdown")
    
    # 



    # TODO  COLLECT 10 SEQUENCES FOR EACH OF 10 DIFFERENT TARGET OBJECTS

    # FOR EACH SEQUENCE, RECORD COSINE SIM TO TARGET 

    # TO TRAIN, INPUT 2 IMGS WITH SIMILAR SCORES AND 1 WITH A SIGNIFICANTLY DIFFERENT SCORE (> THAN SOME MIN DIST AWAY)

    #LET LOSS BE C * |I1 - I2| - |I1 - I3|