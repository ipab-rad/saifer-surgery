# Multiple Depth Sensor drivers and calibration

This package allows to calibrate 4 static Kinect + 1 arm-mounted Realsense RGBD sensors in a global coordinate frame and run them simultaneously.

## Launch

1. Use `launch/kinect_multi4.launch`
2. Reduce fps framerate param if load is too high (default 15 fps)

## Calibration

1. Place aprilgrid board on the tabletop, centered, aligned
   - Generate april-grid with [Kalibr package](https://github.com/ethz-asl/kalibr/wiki/calibration-targets)
3. Run `src/mk_calib_tags.m` in Matlab
4. Exports `calib/data/tags/kinect2_multi_reg_tf_link.launch`
