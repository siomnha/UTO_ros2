# Running IFDS in Gazebo Harmonic with PX4 x500

This package is the local path-planning block in the current simulation data flow:

```text
Gazebo Harmonic + PX4 x500
  ├─ GNSS available: x500 GNSS/local odometry ─────────────┐
  └─ GNSS denied: RGL LiDAR → RGL-to-FAST-LIO2 converter → FAST-LIO2 odometry
                                                            ↓
                                      ifds_ros2 IFDS local planner
                                                            ↓
                         local path + position setpoint / hold setpoint
                                                            ↓
                              PX4 offboard / MAVLink low-level control
```

The planner assumes a **known-obstacles** environment.  Gazebo can generate the
obstacle YAML passed to `obstacles:=...`, or you can start with
`config/known_obstacles.yaml`.  Static mode keeps obstacle centers fixed; dynamic
mode refreshes each obstacle motion model once at the start of every ROS re-plan.
Within one candidate IFDS path, obstacle geometry is frozen to match the MATLAB
`IFDS()` behavior; the UAV does not predict future obstacle trajectories.

## 1. Build

From the ROS 2 workspace containing this repository:

```bash
colcon build --packages-select ifds_ros2
source install/setup.bash
```

## 2. Start the simulator and upstream odometry

Start Gazebo Harmonic with the PX4 x500 model and the offboard/MAVLink bridge
that consumes `/mavros/setpoint_position/local` or your remapped setpoint topic.

For GNSS-denied tests, also start the RGL LiDAR pipeline before launching IFDS:

```text
RGL LiDAR point cloud → converter node → FAST-LIO2 → nav_msgs/Odometry
```

By default, IFDS expects FAST-LIO2 odometry on `/Odometry`.  Remap or override
`fast_lio_odom_topic` if your FAST-LIO2 launch publishes a different topic.

### Oscillating known-obstacle RGL world

This scenario is intentionally limited to the oscillation test.  Its YAML is
the source of truth and contains four obstacles:

1. `obstacle_1`: fixed ellipsoid at `[20, 0, 4]`.
2. `obstacle_2`: radius-3 sphere with `y(t) = -5 + 4 sin(0.25 t)`.
3. `obstacle_3`: ellipsoid with axes `[3, 2, 4]` and
   `y(t) = 5 + 5 sin(0.20 t + pi/2)`.
4. `obstacle_4`: fixed ellipsoid at `[65, 0, 6]`.

Unlike the previous actor implementation, obstacles 2 and 3 are ordinary
Gazebo models with both visual and collision geometry.  The
`ifds_gz_plugins` system moves their model poses directly, so RGL and Gazebo
physics observe the same obstacles.  IFDS still samples the current pose once
per outer re-plan and does not predict motion inside a candidate path.

Build both packages and expose the installed Gazebo plugin:

```bash
colcon build --packages-select ifds_gz_plugins ifds_ros2
source install/setup.bash
export GZ_SIM_SYSTEM_PLUGIN_PATH="$(ros2 pkg prefix ifds_gz_plugins)/lib:${GZ_SIM_SYSTEM_PLUGIN_PATH}"
```

Start the world:

```bash
gz sim -r "$(ros2 pkg prefix ifds_ros2 --share)/worlds/my_rgl_world_osci.sdf"
```

Spawn the existing PX4 x500 + RGL model using the normal simulation launch,
then start IFDS with the matching YAML:

```bash
ros2 launch ifds_ros2 ifds_planner.launch.py \
  dynamic_obstacles:=true \
  obstacles:="$(ros2 pkg prefix ifds_ros2 --share)/config/known_obstacles_osci.yaml"
```

The default `use_sim_time: true` is required so the Gazebo oscillator plugins
and IFDS motion models use the same simulation clock.

## 3. Launch IFDS with GNSS odometry

Use this mode for the normal GNSS condition.  `gnss_denied:=false` selects
`gnss_odom_topic`.

```bash
ros2 launch ifds_ros2 ifds_planner.launch.py \
  gnss_denied:=false \
  obstacles:=/path/to/known_obstacles.yaml
```

Default GNSS odometry topic: `/x500/gnss/odometry`.

## 4. Launch IFDS with GNSS denied / FAST-LIO2 odometry

Use this mode after the RGL LiDAR converter and FAST-LIO2 are publishing
odometry.  `gnss_denied:=true` selects `fast_lio_odom_topic`.

```bash
ros2 launch ifds_ros2 ifds_planner.launch.py \
  gnss_denied:=true \
  obstacles:=/path/to/known_obstacles.yaml
```

Default FAST-LIO2 odometry topic: `/Odometry`.

## 5. Send a goal

```bash
ros2 topic pub --once /ifds/goal geometry_msgs/msg/PoseStamped \
  '{header: {frame_id: map}, pose: {position: {x: 50.0, y: 0.0, z: 10.0}, orientation: {w: 1.0}}}'
```

## 6. Static vs dynamic obstacles

`dynamic_obstacles:=false` keeps all YAML obstacle centers fixed.
`dynamic_obstacles:=true` enables deterministic simulation motion for entries marked
`dynamic: true`.  The motion is sampled once per re-plan and then frozen for
the candidate path, so the UAV does not use future obstacle trajectory knowledge.
The built-in motion model is a simple circular/sinusoidal motion in the y
direction, with optional z radius:

```yaml
header:
  frame_id: map
obstacles:
  - name: moving_sphere
    center: [35.0, -5.0, 6.0]
    axes: [3.0, 3.0, 3.0]
    exponents: [1.0, 1.0, 1.0]
    safety_margin: 1.0
    dynamic: true
    motion:
      type: circular_y
      radius: 4.0
      radius_z: 0.0
      angular_speed: 0.25
      phase: 0.0
```

Gazebo or another simulator bridge can also update the obstacle list at runtime
by publishing the same YAML schema as `std_msgs/String` on `/ifds/obstacles`.
The YAML may include an optional top-level `header` with `frame_id` and stamp
metadata; the planner logs the frame and uses the configured planner frame for
published paths/setpoints.

## 7. Optimiser mode

`optimizer_mode:=0` keeps the configured `rho0` and `sigma0` fixed.
`optimizer_mode:=2` enables a lightweight local optimiser inspired by the MATLAB
`path_opt2` / `norm_ubar` logic: every `local_optimizer_period_steps`, the node
searches bounded nearby `rho0`/`sigma0` candidates and selects the pair that
minimises `||M(rho0, sigma0)u||^2`.  This avoids a SciPy/fmincon dependency and
keeps the planner suitable for online ROS 2 Gazebo use.

Example:

```bash
ros2 launch ifds_ros2 ifds_planner.launch.py \
  gnss_denied:=true \
  dynamic_obstacles:=true \
  optimizer_mode:=2 \
  obstacles:=/path/to/gazebo_obstacles.yaml \
  params:=/path/to/ifds_params.yaml
```

Set these in `ifds_params.yaml`:

```yaml
dynamic_obstacles: true
optimizer_mode: 2
local_optimizer_period_steps: 5
```

## 8. Observe outputs

- `/ifds/local_path` (`nav_msgs/Path`): planned local IFDS path for RViz/debugging.
- `/mavros/setpoint_position/local` (`geometry_msgs/PoseStamped`): high-rate carrot/CCA
  setpoint for the PX4 offboard/MAVLink low-level controller.
- `/ifds/status` (`std_msgs/String`): `PLAN_OK_REPLACED`, `PLAN_FAILED_HOLDING`,
  or `GOAL_REACHED` state.

This paper-faithful baseline follows the AIAA 2024-2091 receding full-path
loop: current UAV odometry position -> complete IFDS replanning to the unchanged
global goal -> immediate path replacement -> high-rate carrot/CCA tracking.
Every successful complete IFDS path replaces the previously tracked path, so
large left/right or vertical reroutes are accepted instead of being rejected by
path-consistency logic.

If a complete IFDS plan fails, the node clears the active path and publishes a
hold setpoint at the normal setpoint timer rate until the next planning cycle.
No old-path reuse, reactive setpoint escape, future obstacle prediction, or
emergency override is run in the setpoint callback.  Dynamic obstacle centers
are sampled at the current simulation time once per outer re-plan and then stay
frozen within that single `IFDSPlanner.plan()` call, matching the receding
full-path replanning behavior used in the paper.

### Stable carrot/CCA path following

Planning and path following use separate timers.  `planning_rate_hz` controls
the slower IFDS re-plan, while `setpoint_rate_hz` publishes the carrot setpoint
at a controller-friendly rate.  They run in separate callback groups on a
two-thread executor, so a longer IFDS calculation does not pause the offboard
setpoint stream.  The carrot is interpolated at `lookahead_distance` metres
along the latest replaced path, independent of waypoint spacing.

`max_setpoint_speed` limits how quickly the commanded carrot can move and
`yaw_rate_limit` limits yaw change through shortest-angle interpolation.  Useful
starting values for PX4 x500 are:

```yaml
planning_rate_hz: 2.0
setpoint_rate_hz: 30.0
lookahead_distance: 2.5
max_setpoint_speed: 3.0
yaw_rate_limit: 1.0
```

## 9. Common parameter overrides

```bash
ros2 launch ifds_ros2 ifds_planner.launch.py \
  gnss_denied:=true \
  dynamic_obstacles:=true \
  optimizer_mode:=2 \
  obstacles:=/path/to/gazebo_obstacles.yaml \
  params:=/path/to/ifds_params.yaml
```

Edit `ifds_params.yaml` to change:

- `gnss_odom_topic`
- `fast_lio_odom_topic`
- `setpoint_topic`
- `rho0`, `sigma0`, `cruise_speed`, `dt`, `delta_g`
- `dynamic_obstacles`
- `optimizer_mode`, `local_optimizer_period_steps`
- `planning_rate_hz`, `setpoint_rate_hz`, `lookahead_distance`
- `max_setpoint_speed`, `yaw_rate_limit`
- `hold_on_failure`
