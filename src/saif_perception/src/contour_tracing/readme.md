## Contour tracing

This node listens for a pointcloud selection (published using the [selected points publisher](../../../saif_ui/publish_selected_patch) plugin in rviz), subsamples these points and then solves a travelling salesman problem to produce a path moving through these. This path can be used by the [contour_following](../../../saif_control/contour_following) node for contour following.

### Parameters

* `sample_density` - resample the points on the selected surface at this density.
* `sat_solver_scale` - scaling parameter for the ortools sat solver.
* `sat_solver_timeout` - maximum time for ortools travelling salesman problem to run for.
* `min_points` - minimum number of points required in selection.

### Topics

* `/rviz_selected_points` - `sensor_msgs/PointCloud2` input message (typically from selected patch publisher)
* `/contour_path` - `nav_msgs/Path` message published by node
