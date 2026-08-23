# UTO ROS 2: FAST-LIO2 → IFDS → UTO → PX4

This repository provides one control chain:

```text
FAST-LIO2 belief ────────────────────────────────┐
mission goal → IFDS (path only) → local Path ───→ UTO → px4_offboard_bridge → PX4
```

`ifds_planner` is **only** a path provider. It publishes no MAVROS or PX4 flight
setpoint. `uto_planner` is the only local trajectory optimizer, and
`px4_offboard_bridge` is the only PX4 setpoint publisher.

> **Safety boundary:** UTO currently has no obstacle constraints. IFDS path
> tracking and the dense path-tube admission gate are not collision checks or a
> collision guarantee. Dynamic-obstacle delay prediction and commit-time obstacle
> revalidation are not implemented in this phase.

## IFDS–UTO integration

The runtime `ros2_packages/uto_ros2/uto_ros2/ifds_core.py` is a byte-for-byte migration of the original
modulation-based IFDS core from `IFDS_integration_node/ifds_ros2`. It retains
superellipsoid gamma/normal/tangent modulation, planar corridor walls, `rho0`,
`sigma0`, `delta_g`, `alpha_deg`, shape following, wall gains, dynamic and
ping-pong motion, normal/relative velocity modes, and optimizer modes 0/2. It is
not the former sphere/detour substitute.

The nested original ROS package is retained as a non-runtime reference and has a
`COLCON_IGNORE`; its historical carrot/setpoint wrapper cannot be discovered by
colcon. The three discoverable runtime packages are standard siblings under
`ros2_packages/`: `uto_ros2`, `ifds_gz_plugins`, and `rgl_livox_converter`.
Gazebo plugins and the RGL Livox converter are simulation support packages, not
additional flight-command owners.

### Topic ownership

| Topic | Type | Publisher | Consumer / meaning |
|---|---|---|---|
| `/ifds/goal` | `geometry_msgs/PoseStamped` | mission operator | raw goal for IFDS |
| `/ifds/mission_goal` | `PoseStamped` | IFDS (reliable, transient-local) | validated global goal for UTO |
| `/ifds/local_path` | `nav_msgs/Path` | IFDS | local path references only |
| `/ifds/path_status` | `std_msgs/String` | IFDS | strict JSON path validity/generations |
| `/Odometry` | `nav_msgs/Odometry` | FAST-LIO2 | default UTO 6-D belief |
| `/uto/trajectory` | JSON `String` | UTO | admitted physical trajectory |
| `/fmu/in/*` | `px4_msgs` | PX4 bridge only | offboard heartbeat/commands/setpoints |

Send a goal:

```bash
ros2 topic pub --once /ifds/goal geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: map}, pose: {position: {x: 15.0, y: 0.0, z: 1.5}, orientation: {w: 1.0}}}"
```

IFDS transforms goal and odometry coordinates with TF2 when their frame differs
from `map`; empty frames, missing TF, or failed transforms invalidate the path.
Obstacle YAML uses `header.frame_id`. A differently framed online obstacle update
is rejected (geometry is never relabelled).

### Path/status ordering and generations

Path and valid status travel on different DDS topics, so UTO caches both by the
exact `path_stamp_ns` and activates them atomically in either arrival order.
Unmatched items expire after `path_pair_timeout` (0.20 s) and produce an
`IFDS_PATH_PAIR_TIMEOUT` warning containing the expired stamp and remaining cache
counts. An incomplete refresh
does not invalidate a still-fresh active pair. An explicit invalid status fails
closed immediately: pending candidates are discarded; execution states request
`SAFE_HOLD`.

IFDS owns `goal_generation`, `obstacle_generation`, and semantic
`path_generation`. Timestamp-only refreshes and normal progress along an old path
do not increment path generation. Remaining old and new polylines are arc-length
resampled (`path_resample_spacing=0.10 m`); generation changes when maximum or RMS
geometry error exceeds `path_geometry_change_threshold=0.05 m`, or goal/obstacle
context changes. UTO uses the paired status generation; its geometry hash is only
a fallback helper.

### Status and terminal semantics

| Status/command | Meaning |
|---|---|
| `NEW_GOAL_PENDING` | Invalidates the previous path, candidate, and request generation while IFDS replans. Active execution changes to `HOLD_CURRENT`. |
| `valid=true, terminal=false` | Normal Path status; requires stamp pairing with a `nav_msgs/Path`. |
| `valid=false, terminal=false` | Genuine planning failure; fails closed and may produce `SAFE_HOLD`. |
| `ALREADY_AT_MISSION_GOAL`, `terminal=true` | No zero-length Path is required; UTO clears old planning work and performs fresh-belief goal dwell from hold. |
| `HOLD_CURRENT` | Non-fault hold at the last actually issued setpoint during a mission transition. |
| `SAFE_HOLD` | Fault/safety fallback; distinct from normal terminal hold. |

IFDS publishes `NEW_GOAL_PENDING` for every newly accepted `/ifds/goal` before
planning. Mission-goal and status topics may arrive in either order: neither can
reuse an old Path because the pending status increments UTO's request generation.

### Exact goal and zero-length mission

IFDS uses `target_threshold=0.05 m`. After the original integrator reaches that
threshold, the wrapper samples the last segment and applies the original wall and
superellipsoid gamma safety functions before appending the exact mission goal.
If start equals goal, IFDS publishes `ALREADY_AT_MISSION_GOAL` and no false
zero-length Path; UTO may complete goal dwell from hold using fresh belief,
velocity, and mission goal.

## Modes, localization, and worlds

| Mode | IFDS params | obstacle YAML | world |
|---|---|---|---|
| `global` | `ifds_global.yaml`, static | `corridor_static_4_obstacles.yaml` | `my_rgl_corridor_static_4.sdf` |
| `online` | `ifds_online.yaml`, dynamic | `corridor_dynamic_4_obstacles.yaml` | `my_rgl_corridor_dynamic_4.sdf` |

Original `known_obstacles*.yaml` files are also installed. They support `wall` and
`superellipsoid` entries with `axes`, `exponents`, `dynamic`, and motion
`start/end/velocity`. SDF ellipsoid `radii` are semi-axes and therefore correspond
directly to YAML `axes` (not full box dimensions). In simulation,
`validate_world_consistency=true` checks obstacle names in both directions,
centers, radii, dynamic classification, walls, and plugin motion parameters;
mismatch prevents a valid path. For a wall with axis `i`, SDF center `c_i`, size
`d_i`, and YAML `inside_sign`, its inner boundary is
`c_i + inside_sign*d_i/2`. The validator also rejects non-static walls, bad box
coverage, duplicate/extra/missing obstacle names, relative poses, and rotated
obstacle models rather than silently comparing coordinates in the wrong frame.
Use `allow_empty_obstacles:=true` only for an explicit obstacle-free test.

| `gnss_denied` | IFDS mean subscription |
|---|---|
| `false` | `/x500/gnss/odometry` |
| `true` | `/Odometry` (FAST-LIO2) |

The IFDS mean source is independent of UTO belief. `uto_belief_topic` defaults to
`/Odometry` and must carry finite, nonzero full position–attitude covariance. A
GNSS-only position message with zero attitude covariance is not a valid UTO
belief; use a GNSS/IMU fusion odometry topic and override the argument.

```bash
ros2 launch uto_ros2 uto_ifds_gazebo.launch.py mode:=global gnss_denied:=false \
  uto_belief_topic:=/fusion/odometry
ros2 launch uto_ros2 uto_ifds_gazebo.launch.py mode:=online gnss_denied:=true
```

All three nodes share simulation time and `map`; no fixed launch sleep is used.
During the approximately 0.9 s cold build/solve (about 0.4 s subsequent solve in
the supplied reference measurements), the PX4 bridge continues its independent
40 Hz takeoff/terminal hold stream. These are reference figures, not measurements
from every machine.

## Build and verification

```bash
python3 -m compileall .
python3 -m pytest -q
source /opt/ros/jazzy/setup.bash
colcon list
rosdep install --from-paths ros2_packages -yi
colcon build --symlink-install \
  --packages-select ifds_gz_plugins rgl_livox_converter uto_ros2
colcon test \
  --packages-select ifds_gz_plugins rgl_livox_converter uto_ros2
colcon test-result --verbose
```

After sourcing `install/setup.bash`, verify plugin installation and automatic
Gazebo discovery:

```bash
ros2 pkg prefix ifds_gz_plugins
find "$(ros2 pkg prefix ifds_gz_plugins)" -name 'libifds_obstacle_path.so'
find "$(ros2 pkg prefix ifds_gz_plugins)" -name 'libifds_obstacle_oscillator.so'
echo "$GZ_SIM_SYSTEM_PLUGIN_PATH"
```

The environment hook prepends the package's `lib` directory; a manual export is
only a troubleshooting fallback. Three distinct validation levels exist:

1. ordinary pure tests parse and compare YAML/SDF contracts;
2. build verification checks both plugin binaries and the environment hook;
3. `UTO_RUN_GAZEBO_TESTS=1 python3 -m pytest -m gazebo` launches the dynamic
   world and observes `dynamic_obstacle_3` motion. Missing Gazebo causes an
   explicit skip, not a pass.

The required simulation matrix is: global + GNSS supported, global + GNSS
denied, online + GNSS supported, and online + GNSS denied. `gnss_denied` only
selects the IFDS mean-position source; `uto_belief_topic` independently selects
UTO's mean/covariance source. GNSS-supported operation therefore does not imply
that raw GNSS odometry satisfies UTO's 6-D covariance contract.

```bash
ros2 launch uto_ros2 uto_ifds_gazebo.launch.py mode:=global gnss_denied:=false uto_belief_topic:=/fusion/odometry
ros2 launch uto_ros2 uto_ifds_gazebo.launch.py mode:=global gnss_denied:=true
ros2 launch uto_ros2 uto_ifds_gazebo.launch.py mode:=online gnss_denied:=false uto_belief_topic:=/fusion/odometry
ros2 launch uto_ros2 uto_ifds_gazebo.launch.py mode:=online gnss_denied:=true
```

Pure tests cover the original-core regression oracle, no-obstacle/wall/
superellipsoid/dynamic/velocity/optimizer behavior, original YAML, SDF contracts,
semantic generations, and cross-topic pairing. ROS, CasADi/IPOPT, PX4, and Gazebo
tests require those dependencies and a sourced workspace; skipped or unavailable
tests must not be interpreted as passing SITL validation.

Before real flight, validate FAST-LIO covariance block ordering and twist validity,
TF/time synchronization, the selected world/YAML pairing, PX4 message fields and
ACK behavior, offboard failsafes, and all four global/online × GNSS-supported/
GNSS-denied SITL combinations.
