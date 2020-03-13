## Contour following

Follows a set of points provided by a suitable node (eg. the [contour tracing](../../saif_perception/src/contour_tracing) node). The MoveIt! Cartesian planner is used to move between points and send commands to the robot. Only positions are controlled, orientations are set to remain the same (initial pose) throughout.

### Parameters

* `contour_offset` - vertical offset above contour point to account for end effector and tool.
* `repositioning_offset` - vertical offset above contour point to move to before and after contour following. Generally best for this to be about twice the vertical offset.

### Topics

* `/corrected_path` - `nav_msgs/Path` offset path to be followed, published for display purposes.
* `contour_path` - `nav_msgs/Path` path on surface (typically from contour tracing node).
