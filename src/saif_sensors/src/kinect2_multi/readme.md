# Multiple Depth Sensor drivers and calibration

This package allows to calibrate 4 static Kinect + 1 arm-mounted Realsense RGBD sensors in a global coordinate frame and run them simultaneously.

## Calibration

1. Place aprilgrid board on the tabletop, centered, aligned
2. Run `src/mk_calib_tags.m` in Matlab
3. Exports `calib/data/tags/kinect2_multi_reg_tf_link.launch`

## Lauch

1. Use `launch/kinect_multi4.launch`
2. Reduce fps framerate param if load is too high (default 15 fps)

