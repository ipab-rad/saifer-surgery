# Ultrasound_epiphan

This package contains a launch script to start publishing images received using the [epiphan](https://www.epiphan.com/support/dvi2usb-3-0-software-documentation/?utm_source=Support&utm_medium=Web&utm_campaign=DVI2USB3.0) DVI2USB 3.0 framegrabber.

`roslaunch ultrasound_epiphan us.launch` 

will start the [usb_cam](http://wiki.ros.org/usb_cam) node and publish data on the topic 

`/ultrasound_streamer/image_raw` 

at 30 fps. This launch file also opens an image viewer to show the stream.
