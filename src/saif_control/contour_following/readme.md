## Contour following

Follows a set of points provided by the [contour tracing](../../saif_perception/src/contour_tracing) node. The MoveIt! Cartesian planner is used to move between points. Only positions are controlled, orientations are set to remain the same (initial pose) throughout.

### Parameters

* `contour_offset` - vertical offset above contour point to account for end effector and tool.
* `repositioning_offset` - vertical offset above contour point to move to before and after contour following. Generally best for this to be about twice the vertical offset.
