HOW TO CALIBRATE A KINECT2 WITH ROS 
+++++++++++++++++++++++++++++++++++++++++++++

##If you haven't already, start the kinect2_bridge with a low number of frames per second (to make it easy on your CPU): 
 - ./K1.sh    ========  rosrun kinect2_bridge kinect2_bridge _fps_limit:=2

##create a directory for your calibration data files, for example: 
 - mkdir ~/kinect_cal_data; cd ~/kinect_cal_data

##Record images for the color camera: 
 - rosrun kinect2_calibration kinect2_calibration kinect_1 chess6x9x0.0247 record color

##calibrate the intrinsics: 
 - rosrun kinect2_calibration kinect2_calibration chess5x7x0.03 calibrate color

##Record images for the ir camera: 
 - rosrun kinect2_calibration kinect2_calibration chess5x7x0.03 record ir

##Calibrate the intrinsics of the ir camera: 
 - rosrun kinect2_calibration kinect2_calibration chess5x7x0.03 calibrate ir

##Record images on both cameras synchronized: 
 - rosrun kinect2_calibration kinect2_calibration chess5x7x0.03 record sync

##Calibrate the extrinsics: 
 - rosrun kinect2_calibration kinect2_calibration chess5x7x0.03 calibrate sync

##Calibrate the depth measurements: 
 - rosrun kinect2_calibration kinect2_calibration chess5x7x0.03 calibrate depth

##Find out the serial number of your kinect2 by looking at the first lines printed out by the kinect2_bridge. The line looks like this: device serial: 012526541941

##Create the calibration results directory in kinect2_bridge/data/$serial: 
 - roscd kinect2_bridge/data; mkdir 012526541941

##Copy the following files from your calibration directory (~/kinect_cal_data) into the directory you just created: 
 - calib_color.yaml calib_depth.yaml calib_ir.yaml calib_pose.yaml

##Restart the kinect2_bridge and be amazed at the better data.
