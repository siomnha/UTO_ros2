# UTO ROS 2：FAST-LIO2 → IFDS → LGR UTO → PX4

本仓库只有一套运行主链路：`uto_planner_node.py` 使用 `planner_runtime.py` 和 `trajectory.py`，PX4 bridge复用相同runtime/trajectory定义。MATLAB文件和论文保留为数值参考。

> **当前UTO没有obstacle constraints。IFDS path tracking和path-tube validation不等价于collision guarantee。** IFDS必须负责持续生成无碰path，PX4 bridge是唯一低层setpoint publisher。

## 真实目录

```text
uto_ros2/
├── math_utils.py                 # SO(3), quaternion, ENU/NED
├── dynamics.py                   # physical dynamics + RK4
├── lgr.py                        # LGR nodes/D/quadrature/interpolation
├── belief_adapter.py             # FAST-LIO covariance, sigma, SO(3) stability
├── ifds_path_adapter.py          # polyline/project/lookahead/hash
├── uto_nlp.py                    # fixed reusable CasADi/IPOPT LGR graph
├── trajectory.py                 # sole trajectory/schema/interpolation/buffer implementation
├── planner_runtime.py            # sole state/request/delay/gate/worker runtime
├── uto_planner_node.py           # ROS planner node
└── px4_offboard_bridge_node.py   # ROS PX4 bridge node
```

旧的`async_worker.py`、`delay_compensator.py`、`feasibility_gate.py`、`state_machine.py`、`trajectory_buffer.py`和`planner_core.py`已删除且无残余import。系统只有两个ROS nodes；“10个LGR collocation nodes”是2 regions × 5 nodes，“7 sigma trajectories”是同一NLP中的不确定轨迹，不是ROS nodes。

## Planner双timer和线程模型

* **Planning timer**：使用`replan_rate`（global 0.75 Hz，online 1.5 Hz），只决定是否replan、生成delay-compensated request和提交latest-wins worker。
* **Commit/safety timer**：默认`commit_check_rate=50 Hz`，消费worker completion queue、推进mission state、执行freshness/goal检查、gate admission、candidate due/continuity/late检查、atomic commit和轻量diagnostics。它不调用NLP build/solve。

Worker线程是唯一CasADi/IPOPT owner，只set parameters、solve、extract并把immutable `CompletionEvent`放入线程安全queue。所有state、candidate、publisher更新都在ROS timer线程完成。First solve使用不可替换提交，避免cold solve期间重复first request；普通solve仍为latest-wins，stale completion只丢弃。

Candidate允许的commit lateness由`allowed_commit_lateness`限制；超过阈值会拒绝，避免从trajectory中间开始。通过gate的candidate只留在planner本地，到commit时用最新belief检查position、velocity和SO(3) geodesic attitude error，成功后才发布给bridge。Global mode的active path generation也只在成功commit后更新。

## Belief、delay和时间域

FAST-LIO `/Odometry`读取position、quaternion、ROS timestamp和完整6×6 position/attitude-tangent covariance，保留cross terms并执行finite、all-zero、PSD、eigen-floor和inflation检查。Stable detector使用position Euclidean变化、`Log(R_previousᵀR_current)` attitude变化及covariance变化，所以`179°→-179°`不会被误判为358°跳变。Conversion、frame、quaternion、covariance或velocity失败会立即清除stable状态。

Delay从belief timestamp开始，以相同绝对时钟查询active controls并用RK4传播七条sigma trajectories。SO(3) UT后只向6D position/attitude tangent covariance加入`Q_delay Δt`；重新注入该spread时保留动力学传播产生的每条sigma velocity deviation，并保持mean velocity不变。`commit_time < belief_stamp`或clock非finite会拒绝request。

PX4 `VehicleOdometry.timestamp_sample`存在时优先使用，否则使用`timestamp`。`px4_velocity_time_mode=offset`显式估计PX4→ROS offset并监测offset跳变；`ros`模式要求timestamp已在ROS `/clock`域。超过`source_clock_tolerance`的变化使velocity invalid。Diagnostics发布belief/path/velocity age和velocity clock offset。IFDS path的header stamp必须非零，且每次有效dynamic update必须更新ROS timestamp。

Velocity先NED→ENU，再按`velocity_frame_alignment_mode=identity|yaw_offset`映射到planning map；本包未声称支持TF velocity alignment。

## LGR UTO和dense gate

默认固定graph为2 regions × 5 LGR nodes、7 shared-control sigma trajectories、10 references。NLP包含normalized LGR dynamics、region continuity、dense polynomial control bounds、velocity/attitude bounds、path tracking、terminal mean/covariance/velocity、effort和smoothness。Startup-only参数包括regions、nodes、sigma、references、control checks和scales；在线只`set_value`，`build_count`保持1。

Admission先检查真实LGR residual和collocation output，再独立执行每region默认15点RK4 rollout：用优化后的LGR control polynomial做barycentric interpolation，从七条commit sigma initial states传播，检查finite、dense mean/all-sigma path tube、velocity、roll/pitch、control bounds以及rollout/LGR endpoint consistency。Diagnostics发布所有dense maxima和endpoint error。Dense path tube仍不是collision checking。

## GOAL_REACHED和hold

IFDS polyline终点是当前goal。Active trajectory结束后，最新belief必须连续`goal_dwell_time`满足position和velocity tolerance才进入`GOAL_REACHED`；否则trajectory过期进入`SAFE_HOLD`，不能误报goal。Goal reached后停止普通replanning并保持bridge terminal hold；开始新mission采用保守规则：重启planner/mission或显式重新启动流程，不因新path自动离开goal。

Trajectory JSON (`uto_trajectory/v1`)验证time/state/control/covariance shape、finite和monotonic。Yaw插值先unwrap再wrap，避免`+π/-π`走长路径。正常结束hold最终trajectory position/yaw；emergency hold保持最后实际安全setpoint，不回起飞点，不外推过期trajectory。

## PX4 bridge

Bridge独立40 Hz运行，solver不会阻塞。每个非failsafe周期发布一个heartbeat和恰好一个`TrajectorySetpoint`。默认`offboard_control_level=position`，对应：

```text
position=true, velocity=false, acceleration=false,
attitude=false, body_rate=false
```

也支持`velocity`或`acceleration`互斥层级。即使position是控制层级，TrajectorySetpoint仍填写velocity/acceleration feed-forward。ENU yaw rate转NED使用`yawspeed_ned=-yaw_rate_enu`，不再固定NaN。不同PX4版本对OffboardControlMode优先级、topic suffix和message fields可能不同，必须在目标`px4_msgs` checkout核对。

Bridge订阅参数化`VehicleCommandAck`，以`WAIT_CONNECTION→PRESTREAM→REQUEST_OFFBOARD→REQUEST_ARM→TAKEOFF_HOLD→READY`顺序运行，带retry interval、ACK timeout和max retries；拒绝或超限进入FAULT。`auto_arm_takeoff=false`不主动发arm/offboard。PX4 failsafe时planner不覆盖PX4自己的failsafe setpoint。

## IFDS contract和ROS interfaces

IFDS必须为path-only，关闭IFDS/MAVROS direct setpoint。IFDS和UTO必须使用同一planning frame；当前frame mismatch会拒绝，不在planner内静默TF转换。

| Interface | 默认值 |
|---|---|
| FAST-LIO belief | `/Odometry` |
| IFDS path | `/ifds/local_path` |
| Executable trajectory | `/uto/trajectory` |
| Emergency execution command | `/uto/execution_command` |
| Diagnostics / PX4 state | `/uto/diagnostics`, `/uto/px4_status` |
| Resume | `/uto/resume` (`std_srvs/Trigger`) |
| PX4 topics | 全部在`gazebo_harmonic_px4.yaml`参数化 |

`SAFE_HOLD`只有PX4 connected/no-failsafe/hold-ready、belief重新稳定、velocity/path fresh时才能通过resume；FAULT不能普通resume。

## Build和运行

```bash
source /opt/ros/jazzy/setup.bash
cd ~/ros2_ws
rosdep install --from-paths src -yi
python3 -m pip install casadi
colcon build --symlink-install --packages-select uto_ros2
source install/setup.bash
ros2 launch uto_ros2 uto_ifds_gazebo.launch.py mode:=online
```

启动顺序：Gazebo Harmonic/PX4 SITL → Micro XRCE-DDS → sensors/FAST-LIO2 → IFDS path-only → UTO → IFDS goal。确保ROS `/clock`、FAST-LIO/IFDS header stamps和PX4 timestamp offset稳定。

## Tests和当前验证状态

```bash
python3 -m pytest -q test/test_core.py test/test_mock_pipeline.py
python3 -m pytest -q test/test_solver_optional.py
python3 -m pytest -q test/test_ros_integration_optional.py
colcon test --packages-select uto_ros2 && colcon test-result --verbose
```

Pure tests覆盖SO(3)、yaw wrap、delay velocity dispersion/process noise、dense rollout、dual-rate commit helpers、completion queue、goal dwell、global commit generation、trajectory interpolation、PX4 control levels和旧模块检查。Solver test包含小型1×2×7和默认2×5×7的build/two-solve/reuse/residual/covariance/timing检查。

ROS optional test在依赖齐全时实际初始化rclpy并构造planner/bridge，检查独立planning/commit timers和默认PX4层级；它仍不是完整fake-topic状态序列或Gazebo验证。若目标CI需要完整消息闭环，应继续加入匹配该PX4版本的fake publishers/ACK fixture。当前README不声称Gazebo、PX4 SITL或未执行tests已经通过。

## Real-flight前

必须在目标ROS/PX4 workspace通过pure tests、默认CasADi solve、colcon build/test和完整fake-topic integration，再运行Gazebo/PX4 SITL fault injection并记录40 Hz setpoint、cold/first/steady timing。核对PX4 ACK result、topic、timesync、frame alignment和OffboardControlMode优先级；定量对照MATLAB；测试belief/path/velocity/DDS/PX4 loss及late candidate；保持real vehicle `auto_arm_takeoff=false`，配置geofence、RC takeover、kill switch并独立审查bounds、dense gate和commit thresholds。
