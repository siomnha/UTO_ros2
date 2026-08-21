# UTO ROS 2：FAST-LIO2 → IFDS → LGR UTO → PX4

本包把 FAST-LIO2 position/attitude belief 和 IFDS path 转为延迟补偿的 2×5 LGR、7-sigma Unscented Trajectory Optimization，并由独立 PX4 bridge 执行。

> **当前 UTO 没有 obstacle constraints。IFDS path tracking 和 path-tube admission gate 不等价于 collision guarantee。避障仍依赖 IFDS 持续提供无碰路径和 PX4 安全机制。**

## 目录和职责

```text
uto_ros2/
├── math_utils.py              # SO(3), quaternion, ENU/NED, velocity yaw alignment
├── dynamics.py                # physical quadrotor dynamics and RK4
├── lgr.py                     # nodes, D, quadrature, interpolation, dense checks
├── belief_adapter.py          # covariance sanitation, SO(3) sigma/UT, stable detector
├── ifds_path_adapter.py       # Polyline projection/lookahead/hash
├── uto_nlp.py                 # one reusable fixed CasADi/IPOPT LGR graph
├── trajectory.py              # physical JSON trajectory, interpolation, buffer, hold policy
├── planner_runtime.py         # state/request/P90/delay/gate/worker/command sequencer
├── uto_planner_node.py        # ROS inputs → request → solve → gate → commit → publish
└── px4_offboard_bridge_node.py# PX4 status/ACK → one setpoint/timer
```

旧的 `async_worker.py`、`state_machine.py`、`planner_core.py`、`delay_compensator.py`、`feasibility_gate.py` 和 `trajectory_buffer.py` 已合并并删除；运行时只有两个有明确边界的核心模块。

## Planner 闭环

```text
Odometry + configured velocity
  → full 6×6 position/SO(3)-tangent covariance sanitize
  → seven R_mean Exp(delta_theta) simplex states
  → consecutive stable detector
IFDS Path → frame/freshness/hash → fixed arc-length references
active control history + belief timestamp
  → RK4 to commit time → SO(3) UT → P + Q_delay Δt
  → regenerate seven SO(3) sigma states
  → one background reusable LGR solve
  → real LGR residual + feasibility gate
  → local pending candidate
  → at commit compare latest actual belief
  → publish to PX4 bridge only after continuity acceptance
```

`uto_planner_node.py` 的 timer 只取得 snapshot、推进 mission state、执行到期 commit、按需创建 request 和发布 diagnostics。ROS callbacks 不调用 IPOPT。`LatestWinsWorker` 是唯一 solver owner。

### First solve 和 online replanning

状态为：

`WAIT_PX4 → TAKEOFF → HOLD → WAIT_BELIEF_STABLE → WAIT_IFDS_INITIAL_PATH → BUILDING_NLP → FIRST_SOLVE → TRAJECTORY_READY → EXECUTING ↔ REPLANNING`，并有 `GOAL_REACHED`、`SAFE_HOLD`、`FAULT`。

`first_request_submitted` 和 `solve_in_progress` 防止 0.9 s cold build/solve 期间重复 first request。新 belief/path 仍更新 latest snapshot，但 first result 完成前不提交普通 replanning。正常 online solve 使用 latest-wins；stale result只丢弃，不导致 hold。

### Delay 和 process noise

控制查询严格使用：

```python
control_at(belief_stamp + relative_time)
```

若 clock 非 finite 或 `commit_time < belief_stamp`，request 被拒绝。传播后的 position/attitude tangent covariance显式加入 `Q_delay × Δt`，再围绕 SO(3) mean重新生成七个 sigma states，所以 process noise真正进入 NLP initial parameters。姿态 mean/covariance用 SO(3) Log/Exp，不对 Euler angles 做线性平均。

### Commit-time continuity

Solve/gate 成功后 trajectory 只进入 planner local candidate buffer，不立即发布。到 commit time 读取最新 belief，检查 position、velocity 和 SO(3) geodesic attitude error。全部低于 YAML tolerance 才原子 commit并发布；否则丢弃并立即允许重规划。有 active safe tail 时继续执行，否则进入 `SAFE_HOLD`。

`/uto/resume` 使用 `std_srvs/Trigger`。只有 PX4 connected、无 failsafe、hold ready、belief重新稳定、velocity fresh、path fresh时才允许从 `SAFE_HOLD` 恢复。`FAULT` 不能通过普通 resume 恢复。

## LGR UTO

默认固定结构：2 regions、5 LGR nodes/region、7 sigma trajectories、10 references。每个 sigma/region 使用 `X Dᵀ = Δt/2 Sx⁻¹ f(SxX,SuU)`；region state连续，所有 sigma共享 controls，control endpoint插值连续。

目标包含 mean path tracking、terminal mean position、terminal position covariance trace、terminal velocity、effort和smoothness。约束包含 velocity、roll/pitch、terminal mean position/velocity及control bounds。

Control bounds不再只检查五个nodes。每region使用 LGR nodes、endpoint、相邻midpoints及默认31个dense points的并集：`U_check = U_nodes × interpolation_matrix`。NLP extraction用实际 normalized state blocks、physical control blocks、D和dynamics重新计算 `max_lgr_dynamics_residual`；gate不再接受默认零 residual。

`regions`、`lgr_nodes_per_region`、`sigma_count`、`lookahead_count`、`control_check_points_per_region`、state/control dimensions和scales均为 **startup-only**。在线只更新 initial sigma、references、horizon、mode、weights和terminal bounds；`build_count`应始终为1。

## FAST-LIO2、velocity和IFDS contract

`/Odometry` 使用 position、quaternion、timestamp、frame和完整ROS pose covariance `[xyz, rotation-about-xyz]`，保留cross blocks。Covariance会finite/symmetry/all-zero/PSD/eigen-floor/inflation检查。

Velocity source：

* `patched_odometry_twist`；
* `px4_vehicle_odometry`；
* `separate_velocity_topic` (`TwistStamped`)。

PX4 velocity先NED→ENU，再根据 `velocity_frame_alignment_mode` 使用 `identity` 或 `yaw_offset`。未知alignment mode会拒绝velocity/belief并保持hold。本包没有声称已实现TF velocity alignment。

IFDS必须为 **path-only**：IFDS只发布 `/ifds/local_path`，UTO是唯一local trajectory generator，PX4 bridge是唯一setpoint publisher。禁止IFDS/MAVROS同时直接控制PX4。Path frame必须已与planning frame一致；当前不在planner内做TF转换。Generation由header timestamp和content hash组成，lookahead从predicted commit position开始并固定尺寸padding。

## Trajectory和PX4 bridge

`/uto/trajectory` 是验证过的 `uto_trajectory/v1` JSON：包含generation、path generation、frame、commit nanoseconds、physical `[N,9]` states、physical controls、times和covariances。Bridge只接收已经通过commit-time实际belief检查的可执行trajectory。

Bridge每个非failsafe timer严格执行：heartbeat → command sequencer →选择一个takeoff/trajectory/terminal/emergency hold setpoint →发布**一个** `TrajectorySetpoint` →status。PX4 failsafe时不发布规划器setpoint覆盖PX4自身failsafe。

Hold策略：

* 起飞前使用takeoff hold；
* 正常trajectory结束hold trajectory最终position/yaw；
* 暂时不能继续执行时hold最后安全setpoint；
* 不外推过期trajectory；
* 绝不自动飞回 `[0,0,hold_altitude]`。

Command sequencer状态为 `WAIT_CONNECTION → PRESTREAM → REQUEST_OFFBOARD → REQUEST_ARM → TAKEOFF_HOLD → READY`，另有 `FAULT`。它订阅参数化的 `VehicleCommandAck` topic，按retry interval/ACK timeout/max retries受控重发；拒绝或超限进入FAULT。`auto_arm_takeoff=false`时不发送arm/offboard command，只等待外部完成。

所有PX4 topics均参数化，因为不同PX4/px4_msgs checkout的`_v1` suffix可能不同。必须用目标workspace的`ros2 topic list/type`核对。

## ROS interfaces

| Interface | 默认值 |
|---|---|
| FAST-LIO belief | `/Odometry` (`nav_msgs/Odometry`) |
| IFDS path | `/ifds/local_path` (`nav_msgs/Path`) |
| Executable trajectory | `/uto/trajectory` (`std_msgs/String`) |
| Planner diagnostics | `/uto/diagnostics` |
| PX4/bridge state | `/uto/px4_status` |
| Resume service | `/uto/resume` (`std_srvs/Trigger`) |
| PX4 heartbeat/setpoint/command | `/fmu/in/offboard_control_mode`, `/trajectory_setpoint`, `/vehicle_command` |
| PX4 status/local/ACK | YAML参数，默认`vehicle_status_v1`, `vehicle_local_position_v1`, `vehicle_command_ack` |

Diagnostics包括cold build、parameter update、solve/P90、extraction、gate time/status/reasons、真实LGR residual、commit delay、deadline/stale counters、build count、first-request/solve flags和active remaining。Bridge status包括command state/retries/fault、arming/offboard/failsafe、jitter、setpoint count和remaining time。

## Global / online

```bash
ros2 launch uto_ros2 uto_ifds_gazebo.launch.py mode:=global
ros2 launch uto_ros2 uto_ifds_gazebo.launch.py mode:=online
```

Launch会实际选择对应YAML。Global使用6 s horizon/0.75 Hz并跳过未变化path上的不必要solve；online使用3 s/1.5 Hz并合并pending updates。两个mode复用同一固定graph结构。

## Build、启动和检查

```bash
source /opt/ros/jazzy/setup.bash
cd ~/ros2_ws
rosdep install --from-paths src -yi
python3 -m pip install casadi
colcon build --symlink-install --packages-select uto_ros2
source install/setup.bash
ros2 launch uto_ros2 uto_ifds_gazebo.launch.py mode:=online
```

启动顺序：Gazebo Harmonic/PX4 SITL → Micro XRCE-DDS → sensors/FAST-LIO2 → IFDS path-only → UTO → IFDS goal。

```bash
ros2 topic hz /fmu/in/trajectory_setpoint
ros2 topic echo /uto/px4_status
ros2 topic echo /uto/diagnostics
ros2 service call /uto/resume std_srvs/srv/Trigger '{}'
```

## Tests和验证状态

```bash
python3 -m pytest -q test/test_core.py test/test_mock_pipeline.py
python3 -m pytest -q test/test_solver_optional.py
python3 -m pytest -q test/test_ros_integration_optional.py
colcon test --packages-select uto_ros2 && colcon test-result --verbose
```

Pure tests覆盖SO(3) UT、delay absolute control time、process-noise resampling、commit mismatch、dense control overshoot、real residual admission、terminal hold、ACK retry、resume、velocity yaw alignment、first request单提交和YAML duplicate keys。`test_mock_pipeline.py`是ROS-independent的first solve/candidate/commit/replan模型测试。`test_ros_integration_optional.py`当前只是目标ROS/PX4/CasADi workspace的节点import/constructibility smoke入口，**不是Gazebo测试，也不声称已经验证topic频率闭环**；目标CI仍应扩展fake publishers和launch fixture。

## Real-flight前

必须在目标PX4/px4_msgs版本完成colcon build、小规模IPOPT solve、ROS fake-topic integration和Gazebo SITL fault injection；核对ACK result/constants/topics、timesync和frame alignment；定量对照MATLAB LGR mission；测试belief/path/velocity/DDS/PX4 loss和late solve；保留`auto_arm_takeoff=false`；配置geofence、RC takeover、kill switch并独立审查vehicle bounds及gate thresholds。
