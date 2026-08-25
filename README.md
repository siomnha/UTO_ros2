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
| `global` | `ifds_global.yaml`, one-shot static | `simple_obstacles.yaml` | `my_rgl_simple.sdf` |
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

## Global one-shot simple environment

`mode:=global` now defaults to `my_rgl_simple.sdf` and
`simple_obstacles.yaml`: the mission starts near `[0, 0, 1.5]`, uses the original
static IFDS around `static_obstacle_1` near `[5.32173, -0.210448, 1.0]`, and is
intended for a goal near `[10, 0, 1.5]`. The configured six references at 2 m
spacing cover the full 10 m path and clamp at the exact IFDS/mission endpoint.

Global startup is deliberately preflight-only:

```text
PX4 READY at takeoff hold
→ stable FAST-LIO belief + valid mission goal + valid static IFDS path
→ freeze current hold belief (no delay propagation or delay process noise)
→ build NLP → IPOPT → non-dense LGR feasibility gates
→ latest-hold-belief continuity check
→ set commit time to now + 0.10 s
→ publish and execute the complete trajectory
→ no in-flight IPOPT replanning
→ goal dwell and terminal hold
```

The bridge remains an independent 40 Hz hold publisher while build/solve/gate are
running. A failed preflight solve can retry at most three times, no faster than
0.5 s. After commit, ordinary odometry updates, trajectory remaining time, and
the one-shot path timestamp do not cause replanning. A new mission resets the
one-shot state; explicit invalid status, stale belief/velocity, or PX4 failsafe
still causes hold/safety handling. Online mode retains delay compensation and
latest-wins replanning.

Example sequence after building and sourcing the workspace:

```bash
# Terminal 1: simple Gazebo world (then start PX4 SITL with your normal model command)
gz sim -r "$(ros2 pkg prefix uto_ros2)/share/uto_ros2/worlds/my_rgl_simple.sdf"

# Terminal 2: start RGL/Livox conversion and FAST-LIO2 using their deployment configs
# Verify /Odometry publishes a nonzero 6-D position-attitude covariance.

# Terminal 3: IFDS + one-shot UTO + PX4 bridge
ros2 launch uto_ros2 uto_ifds_gazebo.launch.py mode:=global gnss_denied:=true

# Wait for /uto/px4_status mode READY and stable-belief diagnostics, then:
ros2 topic pub --once /ifds/goal geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: map}, pose: {position: {x: 10.0, y: 0.0, z: 1.5}, orientation: {w: 1.0}}}"

ros2 topic echo /uto/diagnostics
```

The trajectory topic is published only after solver status, the enabled
feasibility checks, and post-solve continuity all succeed. During flight, diagnostics report
`global_trajectory_committed=true` and `global_replan_blocked=true`.

### Experimental global static-world dense-rollout switch

The supplied simple-world global profile sets
`enable_dense_rollout_gate: false`. This skips only the independent RK4 dense
rollout checks (dense mean/sigma path tube, velocity/attitude/control bounds,
and rollout-to-LGR endpoint errors). It does **not** bypass admission: solver
status, shapes/finite values, time/horizon, request generation, LGR dynamics
residual, extracted LGR state/control/path/terminal checks, covariance and
post-solve hold continuity, IFDS validity, commit policy, and PX4 failsafe
handling remain active.

Skipped metrics are emitted as JSON `null`, with
`dense_rollout_gate_enabled=false` and `dense_rollout_gate_skipped=true`; `0.0`
is never used to imply that an unexecuted check had zero error. This is an
explicit execution-chain experiment for the current Gazebo simple static world.
It does not demonstrate independent real-dynamics or PX4 closed-loop validity
and is unsuitable for dynamic obstacles, narrow environments, or real flight.
Online mode explicitly keeps the gate enabled. Re-enable it with:

```yaml
enable_dense_rollout_gate: true
```

The gate is constructed from startup parameters. Changing
`enable_dense_rollout_gate` requires restarting `uto_planner`; this package does
not claim that `ros2 param set` rebuilds the gate at runtime.

Pure tests cover the original-core regression oracle, no-obstacle/wall/
superellipsoid/dynamic/velocity/optimizer behavior, original YAML, SDF contracts,
semantic generations, and cross-topic pairing. ROS, CasADi/IPOPT, PX4, and Gazebo
tests require those dependencies and a sourced workspace; skipped or unavailable
tests must not be interpreted as passing SITL validation.

Before real flight, validate FAST-LIO covariance block ordering and twist validity,
TF/time synchronization, the selected world/YAML pairing, PX4 message fields and
ACK behavior, offboard failsafes, and all four global/online × GNSS-supported/
GNSS-denied SITL combinations.

## Deterministic TO versus UTO validation

`planner_mode` is the only planner-formulation selector and accepts `uto` or
`deterministic`. UTO retains seven simplex trajectories for the six-dimensional
position/attitude uncertainty and adds terminal position-covariance trace to the
shared objective. Deterministic TO builds a genuinely separate one-state-trajectory
graph: it has no sigma trajectories, sigma dynamics constraints, or covariance
objective. Both formulations share the 9-D physical dynamics, LGR grid, references,
scaling, bounds, IPOPT options, and the non-covariance weights. Weight order remains:

```yaml
weights: [path, terminal_position, covariance,
          terminal_velocity, control_effort, control_smoothness]
```

DTO ignores only the third entry. Both publish `uto_trajectory/v1`, so the PX4
bridge has one execution path. Run either formulation with:

```bash
ros2 launch uto_ros2 uto_ifds_gazebo.launch.py \
  mode:=global planner_mode:=deterministic gnss_denied:=true \
  uto_belief_topic:=/Odometry
# replace deterministic with uto for UTO
```

PX4 message versions are configured independently. The SITL profile subscribes
to `VehicleStatus` on `/fmu/out/vehicle_status_v4` and
`VehicleLocalPosition` on `/fmu/out/vehicle_local_position_v1`; changing one
parameter does not rewrite the other. `px4_status_timeout` defaults to 2 s.
After READY, a status dropout latches takeoff completion, aborts the old timed
trajectory, and holds the last valid reference instead of returning to the
takeoff point. Bridge status reports dropout counters, abort state, and hold source.

### Passive metrics and paired seeds

`uto_validation_metrics` subscribes to parameterized Gazebo ground truth,
FAST-LIO belief, trajectory, diagnostics, PX4 status, mission goal, and reliable
transient-local `/validation/trial_context`. **Only Gazebo ground truth** is used
for terminal and tracking error; `/Odometry` is planner belief, not truth.
Each finish/abort atomically updates:

* `validation_runs.csv` — one row per trial, including failures;
* `validation_summary.json` — bias, RMSE, sample covariance (`N-1`), trace,
  p95, success rate, timing, tracking, and attitude-rate statistics;
* `validation_matrices.npz` — `terminal_positions_DTO/UTO` (`N×3`, metres),
  `terminal_errors_DTO/UTO` (`N`, metres), `terminal_mean_DTO/UTO` (`3`, metres),
  `terminal_covariance_DTO/UTO` (`3×3`, m²), and
  `paired_error_difference` (`N`, metres, UTO minus DTO).

With fewer than two successful trials, covariance is JSON `null` and the NPZ
matrix is NaN with a warning. Rebuild outputs from an interrupted CSV with:

```bash
ros2 run uto_ros2 summarize_validation \
  --input ~/uto_validation_results/validation_runs.csv
```

Generate the default ten paired samples with:

```bash
ros2 launch uto_ros2 validation_experiment.launch.py sample_count:=10 base_seed:=1
```

The supplied runner is simulation-only, defaults to dry-run, records the actual
six-dimensional error vector, and orders each seed as DTO then UTO. Set
`sample_count` to 30, 50, or 100 to extend the same experiment. Full restart is
the required reset strategy because planner, IFDS, PX4, and FAST-LIO retain
state. This repository does **not** contain a verified Gazebo Harmonic
set-entity-state interface: automatic runs therefore abort with
`INITIAL_STATE_INJECTION_UNCONFIRMED` unless an external, confirmed injector is
provided. Processes created by the runner use their own process group and are
stopped with SIGINT followed by scoped SIGTERM; it never uses `pkill`/`killall`.
Neither the metrics node nor dry-run runner publishes flight setpoints.
The runner readiness contract is event-driven—advancing `/clock`, PX4 connection
and hold readiness, stable belief, valid IFDS status, committed trajectory,
execution completion, and `GOAL_REACHED` each have a distinct timeout reason;
fixed sleeps are not treated as readiness evidence.
