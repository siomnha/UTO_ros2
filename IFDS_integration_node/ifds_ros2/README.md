# ifds_ros2

ROS 2 package for running the IFDS local path planner with PX4 x500 in Gazebo Harmonic.

The node ports the MATLAB IFDS velocity modulation from `new_dynamic/src/create_shape.m` and
`new_dynamic/src/calc_ubar.m` to Python.  It assumes **known obstacles** supplied by YAML,
which can later be generated from Gazebo world obstacle parameters.

## Interfaces

- Inputs:
  - GNSS condition (`gnss_denied:=false`): `nav_msgs/Odometry` on `/x500/gnss/odometry`.
  - GNSS-denied condition (`gnss_denied:=true`): RGL LiDAR → converter node → FAST-LIO2 `nav_msgs/Odometry` on `/Odometry`.
  - Goal: `geometry_msgs/PoseStamped` on `/ifds/goal`.
  - Optional obstacle updates: YAML text in `std_msgs/String` on `/ifds/obstacles`.
- Outputs:
  - `nav_msgs/Path` on `/ifds/local_path` for visualization/debugging.
  - `geometry_msgs/PoseStamped` on `/mavros/setpoint_position/local` for a low-level PX4 offboard/MAVLink bridge.
  - `std_msgs/String` on `/ifds/status`.

This baseline follows the AIAA 2024-2091 receding full-path structure:
current UAV odometry position → complete IFDS replanning to the unchanged
global goal → immediate replacement of the tracked path → high-rate
carrot/CCA tracking.  Set `dynamic_obstacles:=true` to refresh dynamic obstacle
centers once per ROS re-plan while keeping geometry frozen inside each IFDS
plan call, and set `optimizer_mode:=2` only when comparing against the optional
local optimiser variant.

Path following uses a fixed-distance carrot rather than a waypoint index.  The
30 Hz setpoint loop is independent from the slower IFDS re-planning loop and
does not run reactive IFDS or emergency-escape logic; if a complete plan fails,
it keeps publishing a hold setpoint until the next planning cycle.

## Run

```bash
colcon build --packages-select ifds_ros2
source install/setup.bash
ros2 launch ifds_ros2 ifds_planner.launch.py gnss_denied:=false
# or, for GNSS-denied FAST-LIO2 odometry from the RGL LiDAR converter flow:
ros2 launch ifds_ros2 ifds_planner.launch.py gnss_denied:=true dynamic_obstacles:=true optimizer_mode:=2
```

Publish a goal:

```bash
ros2 topic pub --once /ifds/goal geometry_msgs/msg/PoseStamped \
  '{header: {frame_id: map}, pose: {position: {x: 50.0, y: 0.0, z: 10.0}, orientation: {w: 1.0}}}'
```

See `docs/RUN_IFDS.md` for the full simulator data flow and run procedure.

## Oscillating known-obstacle world

`worlds/my_rgl_world_osci.sdf` matches
`config/known_obstacles_osci.yaml`: obstacles 1 and 4 are static, while obstacles 2 and 3 are Gazebo
models moving laterally along Y.  Build `ifds_gz_plugins` together
with `ifds_ros2`; the run guide contains the exact plugin-path and launch
commands.
