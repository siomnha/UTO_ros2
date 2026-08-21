# UTO ROS 2 — IFDS → Unscented Trajectory Optimisation → PX4

This package implements a fixed-size, seven-sigma Unscented Trajectory Optimisation (UTO) local planner for ROS 2, with an IFDS `Path` input and a PX4 offboard output. The MATLAB program and paper in this repository remain the numerical references.

> **Safety limitation:** the current UTO NLP has no obstacle constraints and cannot independently guarantee collision safety. Dynamic avoidance depends on IFDS producing fresh collision-free paths, UTO tracking those paths, runtime validity checks, and PX4 safety fallback.

## Architecture

```text
FAST-LIO2 /Odometry ─> belief adapter (6x6 position/SO(3) tangent covariance) ─┐
PX4 velocity ─────────────────────────────────────────────────────────────────┤
IFDS path-only /ifds/local_path ─> arc-length projection/lookahead ──────────┤
active trajectory ─> sigma-level delay predictor ────────────────────────────┤
                                                                             v
      state machine -> latest-wins worker -> fixed CasADi/IPOPT UTO -> gate/buffer
            |                                                           |
            └──────────── SAFE_HOLD <──────── diagnostics ───────────────┘
                                                                        |
                        40 Hz independent PX4 bridge <───────────────────┘
                          ENU/map -> NED -> PX4 /fmu/in/*
```

The planner callback thread only updates snapshots and submits requests. One background worker owns IPOPT; a newer request replaces the pending request, never starts a concurrent solve, and makes the old result stale. The PX4 timer is in a separate node/process, so cold graph construction or IPOPT cannot interrupt heartbeat/setpoints.

## Mathematical model

The physical state is `x=[p(3),v(3),roll,pitch,yaw]`, control is `[thrust acceleration, roll command, pitch command, yaw rate]`, and initial uncertainty is `z=[p(3),delta_theta(3)]`. Seven equal-weight regular-simplex points exactly reconstruct the supplied 6-D covariance. Attitude points use `R_i=R_mean Exp(delta_theta_i)`, not Euclidean Euler perturbations.

The dynamics reproduce the MATLAB model: thrust is rotated into the world frame, gravity and linear drag are applied, roll/pitch follow first-order command dynamics (`tau=0.35 s`), and yaw integrates yaw rate. State/control scaling is explicit. One graph contains 2 regions × 5 nodes, seven state trajectories, one shared control, velocity/attitude/control bounds, mean path tracking at fixed references, terminal mean position/velocity and position-covariance costs, effort and smoothness. Initial sigma states, references, horizon, mode, weights and velocity bounds are CasADi parameters. `regions`, `lgr_nodes_per_region`, `sigma_count`, and reference count are **startup-only** because they set graph dimensions.

The Python graph uses fixed-node direct transcription. The repository's LGR MATLAB reference documents the original differentiation and quadrature operators; further fidelity validation against MATLAB is required before real flight.

## Nodes and ROS interfaces

| Node | Subscribes | Publishes | Responsibility |
|---|---|---|---|
| `uto_planner` | `/Odometry` (`nav_msgs/Odometry`), `/ifds/local_path` (`nav_msgs/Path`) | `/uto/trajectory`, `/uto/diagnostics` (`std_msgs/String`, JSON) | belief/path snapshots, fixed NLP, async latest-wins replanning |
| `px4_offboard_bridge` | `/uto/trajectory`, `/fmu/out/vehicle_status_v1` | `/fmu/in/offboard_control_mode`, `/fmu/in/trajectory_setpoint`, `/fmu/in/vehicle_command` | uninterrupted hold/offboard stream and ENU→NED boundary |

JSON trajectory is intentionally minimal pending selection of a site-standard trajectory message. Important parameters are documented in the three YAML files: frames/topics, timeouts, covariance thresholds, horizon/lookahead/tube, P90 window/margins, process noise, scales, bounds/weights, IPOPT tolerances, rates, takeoff and fallback.

The intended state progression is `WAIT_PX4 → TAKEOFF → HOLD → WAIT_BELIEF_STABLE → WAIT_IFDS_INITIAL_PATH → BUILDING_NLP → FIRST_SOLVE → TRAJECTORY_READY → EXECUTING ↔ REPLANNING`, with `GOAL_REACHED`, `SAFE_HOLD`, and `FAULT` exits. No trajectory may execute before stable belief, fresh path, successful first solve, and feasibility admission.

## FAST-LIO2 belief contract

`Odometry.pose.covariance` must be mapped deliberately into the full `[position, attitude-tangent]` 6×6 covariance, retaining cross terms. The adapter symmetrises it, rejects non-finite/all-zero/meaningfully indefinite matrices, floors eigenvalues, and inflates covariance. Stability additionally requires fresh messages, available TF, valid velocity, trace thresholds, and N consecutive samples with bounded mean/covariance changes.

**Verify in your FAST-LIO2 build** the covariance block ordering and publish timing, and whether `/Odometry.twist` is actually populated. Configure velocity as PX4 `VehicleOdometry`, a separate velocity topic, or explicitly patched odometry twist. Missing/stale velocity is a hold condition; it must never silently become zero. This repository does not patch FAST-LIO2.

## IFDS connection

Run IFDS in **path-only** mode. IFDS is the path provider, UTO is the only local trajectory generator, and this PX4 bridge is the only low-level setpoint publisher. Disable IFDS/MAVROS direct setpoint publication. Both systems must use the same planning frame (normally ENU `map`) or provide an explicit TF2 transform.

Recommended QoS is reliable/volatile depth 5–10 for path, 2–5 Hz for dynamic paths and 0.5–1 Hz for static paths. Every dynamic update must change `header.stamp`; UTO combines timestamp and path content hash into a generation ID. Lookahead begins at the delay-predicted commit position projected onto the polyline, then samples a fixed K by arc length with endpoint padding.

```bash
ros2 run your_ifds_pkg ifds_node --ros-args -p path_only:=true -r local_path:=/ifds/local_path
ros2 launch uto_ros2 uto_ifds_gazebo.launch.py mode:=online \
  --ros-args -r /Odometry:=/fast_lio/Odometry
```

Global mode uses a longer horizon and 0.75 Hz requests. Online mode uses a shorter horizon and 1.5 Hz latest-wins requests; intermediate IFDS updates may be coalesced. Dynamic avoidance is “IFDS changes path + UTO tracks path”, not collision avoidance inside UTO. `/ifds/obstacles` is deliberately not decoded because no message type/schema was specified; a future typed adapter may add post-checking only.

## Delay, buffering, feasibility and diagnostics

Commit delay is `clamp(P90(solve window)+validation/interface margin)`, falling back to 0.5 s while samples are sparse. Seven sigma states are propagated with active controls and RK4, process noise is added, and lookahead is selected from the predicted mean. Candidate time zero is its commit instant. A candidate finishing after `commit-guard`, carrying stale generations, or disagreeing with actual commit belief must be discarded.

The runtime gate is designed to check solver status, finite values, dynamics residual, bounds, terminal/path error, monotonic horizon coverage, generation freshness, start continuity, and mean/all-sigma path tube. An invalid or exhausted trajectory causes hold, not extrapolation. Diagnostics separate cold build, parameter update, solve, extraction, gate and end-to-end time plus iterations, deadline misses, stale discards and setpoint jitter. The current implementation publishes build/solve/generation; remaining gate/timing counters are integration work and must be completed before flight qualification.

## Frames and PX4

FAST-LIO/IFDS use ENU/map. The bridge alone converts vectors as `[E,N,U]→[N,E,-U]` and yaw as `pi/2-yaw` for PX4 NED/FRD conventions. ROS time honors `use_sim_time`. The code targets PX4 ROS 2 topics `/fmu/in/offboard_control_mode`, `/fmu/in/trajectory_setpoint`, and `/fmu/out/vehicle_status_v1`; confirm these and every `px4_msgs` field against the exact PX4/px4_msgs checkout used to build the workspace.

Simulation YAML permits automatic arm/takeoff. Real vehicles must use a separate override with `auto_arm_takeoff: false` (the conservative default in code), tested failsafe, geofence, RC takeover and an independent kill path.

## Build and launch (Gazebo Harmonic + PX4 SITL)

```bash
source /opt/ros/jazzy/setup.bash
python3 -m pip install casadi numpy pytest
cd ~/ros2_ws/src && git clone https://github.com/siomnha/UTO_ros2.git
cd ~/ros2_ws && rosdep install --from-paths src -yi
colcon build --symlink-install --packages-select uto_ros2
source install/setup.bash
ros2 launch uto_ros2 uto_ifds_gazebo.launch.py mode:=online
```

Complete launch order:
1. Gazebo Harmonic and PX4 SITL; 2. sensors and FAST-LIO2; 3. IFDS path-only; 4. UTO planner/bridge; 5. IFDS goal/mission enable; 6. PX4 takeoff and hold; 7. wait for stable belief; 8. wait for initial path; 9. build NLP once; 10. first solve; 11. execute; 12. online replan.

```bash
ros2 topic hz /fmu/in/trajectory_setpoint
ros2 topic hz /ifds/local_path
ros2 topic echo /uto/diagnostics
ros2 topic echo /fmu/out/vehicle_status_v1
ros2 topic echo /Odometry --once
```

## Tests and troubleshooting

```bash
python3 -m pytest -q test/test_core.py
python3 -m pytest -q test/test_solver_optional.py  # skipped without CasADi/IPOPT
colcon test --packages-select uto_ros2 && colcon test-result --verbose
```

* Never leaves hold: check fresh timestamps, TF/frame agreement, nonzero covariance, covariance traces, and the configured velocity source.
* IPOPT fails: inspect normalization, horizon, velocity bounds, path discontinuities and initial covariance; do not loosen gates blindly.
* PX4 rejects offboard: verify bridge rate >2 Hz (configured 40 Hz), DDS agent, exact px4_msgs branch/topic suffix, nav/arming/failsafe state, and timesync.
* Cutting corners: increase path tracking weight/density and reduce horizon; remember this is not an obstacle-constrained optimizer.
* Late candidates: reduce request rate/horizon only after examining P90 solve time; latest-wins intentionally drops intermediate paths.

## Current qualification status

The pure-Python math, path, delay, buffer, state and concurrency units are testable without ROS/Gazebo. CasADi, ROS 2, PX4 and Gazebo integration tests require those dependencies. Before vehicle use, add the site-specific velocity/TF adapters, complete the full feasibility gate and command/arming state handling, validate transcription numerically against MATLAB, run SITL fault injection, and review all safety behavior.
