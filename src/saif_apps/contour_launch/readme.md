## Contour following launch file

Launches the dual arm robots with rviz point selection, contour tracing and following code. 

To use, execute:

`roslaunch contour_launch contour.launch`.

* Initialise robot to suitable start pose and orientation
* Select surface to follow using Selected Points Publisher

<img src="https://github.com/ipab-rad/saifer-surgery/blob/master/src/saif_ui/contour_launch/ims/s1.png" width="400" /><img src="https://github.com/ipab-rad/saifer-surgery/blob/master/src/saif_ui/contour_launch/ims/s2.png" width="400" />

* Nav path will be displayed, and robot will follow this

<img src="https://github.com/ipab-rad/saifer-surgery/blob/master/src/saif_ui/contour_launch/ims/s3.png" width="400" /><img src="https://github.com/ipab-rad/saifer-surgery/blob/master/src/saif_ui/contour_launch/ims/s4.png" width="400" />


<p align="center">
<img src="https://github.com/ipab-rad/saifer-surgery/blob/master/src/saif_ui/contour_launch/ims/surface.gif" width="200" />
</p>

This node relies on the [contour_tracing](https://github.com/ipab-rad/contour_tracer) node for perception and the [contour_following](../../saif_control/contour_following) node for planning and control.
