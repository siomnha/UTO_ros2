# UTO ROS 2：FAST-LIO2 + IFDS + LGR UTO + PX4 Offboard

本仓库实现一个 `ament_python` ROS 2 包：FAST-LIO2 提供 position/attitude belief，IFDS 只提供无碰路径，固定尺寸 CasADi/IPOPT UTO 生成动态可执行轨迹，独立 PX4 bridge 以 40 Hz 执行轨迹或安全 hold。MATLAB 文件和论文保留为数值参考。

> **安全边界：当前 UTO NLP 没有 obstacle constraints，不能单独保证碰撞安全。动态避障依赖 IFDS 及时发布无碰路径、UTO path/tube tracking、实时 admission gate 和 PX4 fallback。Path-tube 检查不等价于 collision checking。**

## 实际闭环

```text
FAST-LIO /Odometry (pose, quaternion, full 6x6 covariance)
       + configured fresh velocity
       -> covariance sanitize -> SO(3) simplex 7 sigma states -> stable detector
IFDS /ifds/local_path -> frame/freshness/hash -> Polyline -> commit-position lookahead
active trajectory control -> RK4 sigma prediction -> P90 commit belief
       -> one latest-wins worker -> reusable 2x5 LGR CasADi/IPOPT graph
       -> feasibility gate -> candidate buffer -> atomic commit
       -> /uto/trajectory (validated physical JSON v1)
       -> independent PX4 bridge -> interpolation -> ENU/NED -> PX4 setpoints
```

Planner 和 bridge 是不同进程；IPOPT build/solve 不在 ROS callback 或 PX4 setpoint timer 中运行。旧 solve 若因 pending request 变 stale，只计数并丢弃，不触发 hold。只有数据失效、solve/gate 失败且没有安全 active tail、PX4 failsafe 或 trajectory 结束才进入安全 hold。

## 数学模型与真正的 LGR transcription

状态为 `[px,py,pz,vx,vy,vz,roll,pitch,yaw]`，控制为 `[thrust_acceleration,roll_command,pitch_command,yaw_rate]`。初始随机变量是 position 和 SO(3) tangent attitude 共 6 维。七个等权 regular-simplex 点重建 full covariance；姿态点严格使用 `R_i=R_mean Exp(delta_theta_i)`。

`uto_ros2/lgr.py` 移植 MATLAB 的 Jacobi eigenvalue LGR nodes、barycentric differentiation matrix、endpoint interpolation 和 moment-matching quadrature。`uto_nlp.py` 使用 `X[sigma][region]` 二维 CasADi variables，默认 2 regions × 5 LGR nodes：

* 每条 sigma trajectory 都满足 `X Dᵀ = duration/2 f_normalized`；
* region state continuity 和 endpoint-interpolated shared-control continuity；
* shared physical control、velocity、roll/pitch 和 control bounds；
* 多个 LGR node 上的 mean IFDS path tracking；
* terminal mean-position tolerance、mean-velocity bounds、position covariance trace；
* terminal velocity、control effort 和 LGR control-rate smoothness cost；
* extraction 后 state/control 全部恢复 SI physical units，并输出每个样本的 UT covariance。

初始 sigma states、K 个 references、horizon、mode、weights、terminal bounds/tolerance是 parameters。`build()` 幂等；同一个 worker 是唯一 solver owner。`regions`、`lgr_nodes_per_region`、`sigma_count`、`lookahead_count`、state/control dimensions/scales 是 **startup-only**，改变它们必须重启节点，不会在线 rebuild。

## ROS interfaces

| Node | Input | Output |
|---|---|---|
| `uto_planner` | `/Odometry` `nav_msgs/Odometry`; `/ifds/local_path` `nav_msgs/Path`; `/uto/px4_status` JSON; optional velocity | `/uto/trajectory`; `/uto/diagnostics` |
| `px4_offboard_bridge` | `/uto/trajectory`; `/fmu/out/vehicle_status_v1`; `/fmu/out/vehicle_local_position_v1` | `/fmu/in/offboard_control_mode`; `/fmu/in/trajectory_setpoint`; `/fmu/in/vehicle_command`; `/uto/px4_status` |

### Physical trajectory schema

ROS distributions/PX4 workspaces do not consistently install `trajectory_msgs`, so this package currently uses a validated, versioned `std_msgs/String` JSON schema rather than an un-timed ad-hoc list:

```json
{"schema":"uto_trajectory/v1","generation":1,"path_generation":"hash","frame_id":"map","commit_time_ns":0,"times":[0.0],"states_physical":[[0,0,1,0,0,0,0,0,0]],"controls_physical":[[9.81,0,0,0]],"mean_covariances":[[[0]]]}
```

Real arrays have shapes `times=[N]`, `states_physical=[N,9]`, `controls_physical=[N or N-1,4]`, covariance `[N,9,9]`. Parser validates schema, dimensions, finite values and monotonic time. Bridge interpolates by ROS time, switches only at `commit_time_ns`, and returns to hold after the last sample—never extrapolates an expired path.

## FAST-LIO belief and velocity

ROS pose covariance ordering `[x,y,z,rotation_x,rotation_y,rotation_z]` is mapped intact into tangent `[position,delta_theta]`, including position-attitude cross blocks. It is checked for finite values, symmetrised, rejected if all-zero or meaningfully indefinite, eigenvalue-floored and inflated. Quaternion is converted to `R_mean`, then seven SO(3) sigma states are generated.

`velocity_source` supports:

* `patched_odometry_twist`: read `/Odometry.twist` explicitly;
* `px4_vehicle_odometry`: subscribe `/fmu/out/vehicle_odometry`, convert PX4 NED velocity to ENU and enforce `velocity_timeout`;
* `separate_velocity_topic`: subscribe a planning-frame `geometry_msgs/TwistStamped`.

Missing/non-finite/stale configured velocity is rejected and diagnosed; it is never silently replaced by zero. Check the exact FAST-LIO covariance write order/timestamp and whether its twist is populated. This repository does not modify FAST-LIO2.

Belief becomes stable only after `stable_samples` consecutive fresh, frame-correct, finite, nonzero PSD samples below position/attitude trace thresholds, with mean and covariance deltas below configured limits. NLP build cannot start before PX4 hold-ready, stable belief and a valid initial path.

## IFDS contract

IFDS must run **path-only**. Disable all IFDS/MAVROS/PX4 direct setpoint outputs. IFDS is the path provider, UTO is the only local trajectory generator, PX4 bridge is the only low-level publisher. IFDS and UTO must use the same planning frame; this release rejects a mismatched frame rather than silently using it (transform the IFDS path upstream if needed).

Every dynamic update must update `header.stamp`; generation is timestamp + content hash. Planner checks receive freshness, projects the delay-predicted commit position onto the polyline and arc-length samples exactly K references with endpoint padding. Suggested publication: static 0.5–1 Hz, dynamic 2–5 Hz, reliable volatile depth 5–10.

## State machine, delay and modes

Real state sequence is:

`WAIT_PX4 → TAKEOFF → HOLD → WAIT_BELIEF_STABLE → WAIT_IFDS_INITIAL_PATH → BUILDING_NLP → FIRST_SOLVE → TRAJECTORY_READY → EXECUTING ↔ REPLANNING`, plus `GOAL_REACHED`, `SAFE_HOLD`, `FAULT`.

Bridge prestreams setpoints, optionally sends PX4 custom-main-mode OFFBOARD and arm commands, climbs/holds at `hold_altitude`, and reports hold-ready. Simulation sets `auto_arm_takeoff=true`; code/default for a real vehicle is false.

For replanning, commit delay is `clamp(P90(solve)+validation_time+commit_margin,min,max)`; sparse history uses `initial_delay`, while cold graph build uses `cold_start_delay`. Seven sigma states propagate from belief timestamp through active physical controls to commit with RK4, and `Q_delay*dt` is added to covariance. A result arriving after `commit-commit_guard`, with stale request/path/belief generation, failing continuity or gate checks is discarded. Active trajectory continues until its safe end.

* `mode:=global`: 6 s horizon, 0.75 Hz check; unchanged path does not re-solve until the active tail is short.
* `mode:=online`: 3 s horizon, 1.5 Hz latest-wins; intermediate dynamic path updates coalesce naturally.

Both use the same fixed graph structure; only parameters differ.

## Parameters

All parameters in `config/uto_global.yaml`, `config/uto_online.yaml`, and `config/gazebo_harmonic_px4.yaml` are declared and consumed. Groups include topics/frames/source, mode/horizon/lookahead/rate, freshness and stability, covariance floor/inflation, process noise, LGR startup dimensions, scales/bounds/weights/IPOPT settings, terminal tolerances, tube limits, P90 window/delay/guard, bridge rate/hold/auto-arm/status timeout.

## Diagnostics

`/uto/diagnostics` JSON reports mission state/reason, cold build time (once), parameter update, IPOPT solve/P90, extraction, predicted commit delay, deadline misses, stale discards, build count and active remaining time. `/uto/px4_status` reports connection, hold-ready, failsafe, arm/nav states, output mode, maximum setpoint jitter and active remaining time.

Cold start is `build + first update + first solve + extraction + gate`; steady online excludes build. Build time is never added again.

## Build and Gazebo Harmonic + PX4 SITL

Confirm the installed PX4 and `px4_msgs` are matching checkouts. This code uses current-style `/fmu/in/*`, `/fmu/out/vehicle_status_v1`, `/fmu/out/vehicle_local_position_v1`, and `/fmu/out/vehicle_odometry`; inspect `ros2 topic list/type` and generated message fields in your workspace before launching.

```bash
source /opt/ros/jazzy/setup.bash
cd ~/ros2_ws
rosdep install --from-paths src -yi
python3 -m pip install casadi
colcon build --symlink-install --packages-select uto_ros2
source install/setup.bash
ros2 launch uto_ros2 uto_ifds_gazebo.launch.py mode:=online
# or: mode:=global
```

Launch order: Gazebo/PX4 SITL → Micro XRCE-DDS agent → sensors/FAST-LIO2 → IFDS path-only → UTO launch → IFDS goal. Then observe takeoff/hold, stable belief, first build/solve/commit and execution.

```bash
ros2 topic hz /fmu/in/trajectory_setpoint
ros2 topic echo /uto/px4_status
ros2 topic echo /uto/diagnostics
ros2 topic hz /ifds/local_path
ros2 topic echo /Odometry --once
```

## Tests

```bash
python3 -m pytest -q test/test_core.py test/test_mock_pipeline.py
python3 -m pytest -q test/test_solver_optional.py
colcon build --packages-select uto_ros2
colcon test --packages-select uto_ros2 && colcon test-result --verbose
```

Tests cover simplex/full cross covariance, SO(3), LGR polynomial differentiation/quadrature, MATLAB hover dynamics, frames, belief stability, trajectory schema/interpolation, delay/P90, state sequence, latest-wins shutdown, gate/late/stale rejection and mock first/replan/expiry. The optional solver test builds and solves a small CasADi/IPOPT graph twice, asserts one build, distinct sigma states, physical output and PSD terminal covariance.

## Qualification and real-flight checklist

The repository must first pass the above tests and an actual Gazebo Harmonic/PX4 SITL fault-injection run in the target workspace. Before real flight: keep auto-arm false; verify exact PX4 topics/constants/fields and timesync; add/validate site TF if frames differ; validate Python LGR results against the MATLAB mission; test estimator/path/velocity/PX4 loss, late solve and DDS interruption; configure geofence, RC override and kill switch; independently review admission thresholds and vehicle bounds. Absence of UTO obstacle constraints remains a fundamental limitation.
