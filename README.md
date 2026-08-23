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
├── ifds_core.py                  # path-only IFDS geometry/status primitives
├── ifds_planner_node.py          # IFDS path/status/mission-goal ROS node
├── uto_nlp.py                    # fixed reusable CasADi/IPOPT LGR graph
├── trajectory.py                 # sole trajectory/schema/interpolation/buffer implementation
├── planner_runtime.py            # sole state/request/delay/gate/worker runtime
├── uto_planner_node.py           # ROS planner node
└── px4_offboard_bridge_node.py   # ROS PX4 bridge node
```

旧的`async_worker.py`、`delay_compensator.py`、`feasibility_gate.py`、`state_machine.py`、`trajectory_buffer.py`和`planner_core.py`已删除且无残余import。系统有三个在线ROS nodes（IFDS、UTO planner、PX4 bridge）；“10个LGR collocation nodes”是2 regions × 5 nodes，“7 sigma trajectories”是同一NLP中的不确定轨迹，不是ROS nodes。

## Planner双timer和线程模型

* **Planning timer**：使用`replan_rate`（global 0.75 Hz，online 1.5 Hz），只决定是否replan、生成delay-compensated request和提交latest-wins worker。
* **Commit/safety timer**：默认`commit_check_rate=50 Hz`，消费worker completion queue、推进mission state、执行freshness/goal检查、candidate due/continuity/late检查、atomic commit和轻量diagnostics。它不调用NLP build/solve或dense gate。

Worker线程是唯一CasADi/IPOPT owner，负责set parameters、solve、extract和与ROS无关的dense gate，再把immutable `CompletionEvent`放入线程安全queue。所有state、candidate、publisher更新都在ROS timer线程完成。First solve使用不可替换提交，避免cold solve期间重复first request；普通solve仍为latest-wins，stale completion只丢弃。当前节点明确以`SingleThreadedExecutor`为支持基线。

Candidate允许的commit lateness由`allowed_commit_lateness`限制；超过阈值会拒绝，避免从trajectory中间开始。每次高频tick先drain completion、取得一致snapshot，再优先处理due candidate；仅在没有成功commit时才判定旧active是否过期。一个tick只会选择`commit / continue / safe hold / goal reached`之一，避免同周期先hold再发trajectory。通过gate的candidate只留在planner本地，到commit时用最新belief检查position、velocity和SO(3) geodesic attitude error，成功后才发布给bridge。Global mode的active path generation也只在成功commit后更新。

决策优先级为：`COMMITTED → COMMIT`、`GOAL_REACHED → GOAL_REACHED`、fresh `WAITING → CONTINUE`、stale `WAITING → SAFE_HOLD`、合法且terminal-data-fresh的goal dwell → `CONTINUE`，最后才按active tail和planning freshness选择continue或hold。尤其first candidate尚未due且active为空时继续PX4 takeoff/terminal hold，不会提前进入`SAFE_HOLD`；late/rejected candidate且无active tail仍会安全hold。

## Belief、delay和时间域

FAST-LIO `/Odometry`读取position、quaternion、ROS timestamp和完整6×6 position/attitude-tangent covariance，保留cross terms并执行finite、all-zero、PSD、eigen-floor和inflation检查。Stable detector使用position Euclidean变化、`Log(R_previousᵀR_current)` attitude变化及covariance变化，所以`179°→-179°`不会被误判为358°跳变。Conversion、frame、quaternion、covariance或velocity失败会立即清除stable状态。

Delay从belief timestamp开始，以相同绝对时钟查询active controls并用RK4传播七条sigma trajectories。传播后在`[position, velocity, Log(R_mean^T R_i)]`的9维联合tangent space计算均值、covariance和原始simplex vertex identity。`Q=0`时原始七条trajectory完全不重采样，完整tangent covariance保持不变。七个等权centered sigma最多只有rank 6；`Q>0`后九维target一般高于rank 6，因此代码明确执行最佳rank-6 PSD近似，而不声称精确恢复target。Diagnostics量化relative Frobenius covariance error、position–velocity/attitude–velocity cross-block error、target/reconstructed rank及discarded eigenvalue energy；cross terms不会被静默清零。`commit_time < belief_stamp`或clock非finite会拒绝request。

Delay历史记录的是实际admission latency，而不只是IPOPT时间：request enqueue → worker queue → build/parameter prepare → IPOPT → extraction → dense gate → completion queue consumption；成功commit后另记request-to-commit。`first_cold`显式使用`cold_start_delay=1.2 s`；first solve后、有效样本少于`delay_p90_min_samples=5`时为`initial_history`并使用`initial_delay=0.5 s`；达到门槛后才进入`steady_p90`。History最多`delay_p90_window=20`个样本并先按`latency_clip_min/max`裁剪。`validation_time`是额外固定安全overhead，不是dense gate实测时间（dense gate已包含在admission sample内）；之后再加commit/interface scheduling margins并限制到minimum/maximum delay。Diagnostics使用`admission_latency_p90`，旧`solve_p90`仅作deprecated兼容。

PX4 `VehicleOdometry.timestamp_sample`存在时优先使用，否则使用`timestamp`。`px4_velocity_time_mode=offset`显式估计PX4→ROS offset并监测offset跳变；`ros`模式要求timestamp已在ROS `/clock`域。超过`source_clock_tolerance`的变化使velocity invalid。Diagnostics发布belief/path/velocity age和velocity clock offset。IFDS path的header stamp必须非零，且每次有效dynamic update必须更新ROS timestamp。

Velocity先NED→ENU，再按`velocity_frame_alignment_mode=identity|yaw_offset`映射到planning map；本包未声称支持TF velocity alignment。

## LGR UTO和dense gate

默认固定graph为2 regions × 5 LGR nodes、7 shared-control sigma trajectories、10 references。NLP包含normalized LGR dynamics、region continuity、dense polynomial control bounds、velocity/attitude bounds、path tracking、terminal mean/covariance/velocity、effort和smoothness。Startup-only参数包括regions、nodes、sigma、references、control checks和scales；在线只`set_value`，`build_count`保持1。

Admission先检查真实LGR residual和collocation output，再独立执行每region默认15点RK4 rollout：用优化后的LGR control polynomial做barycentric interpolation，从七条commit sigma initial states传播，检查finite、dense mean/all-sigma path tube、velocity、roll/pitch、control bounds以及rollout/LGR endpoint consistency。Endpoint一致性拆成position Euclidean error（m）、velocity Euclidean error（m/s）和`Log(R_LGR^T R_rollout)` SO(3) geodesic error（rad），对应三个独立参数与diagnostic，避免Euler `+pi/-pi`误报。Dense path tube仍不是collision checking。

## 显式mission goal、GOAL_REACHED和hold

`/ifds/local_path`只提供滚动lookahead，末点绝不是全局任务终点。IFDS须用`geometry_msgs/PoseStamped`在`/ifds/mission_goal`为每个任务发布一次显式goal；timestamp必须非零、frame必须等于planning frame、position/orientation必须finite。默认`mission_goal_timeout=0`表示收到后持久有效；正值才启用freshness timeout。内容generation变化会清除旧dwell，重复发布相同goal或滚动local path不会重置它。

Active/terminal trajectory结束后，最新belief必须连续`goal_dwell_time`满足显式goal的position、velocity，以及可选的wrapped-yaw tolerance，才进入`GOAL_REACHED`。Terminal goal freshness只要求fresh belief、fresh velocity和有效persistent mission goal，不要求rolling local path继续更新；belief/velocity stale会立即清空dwell，PX4 failsafe优先。Goal reached后停止普通replanning并保持bridge terminal hold。

`goal_yaw_enabled=false`时goal可为position-only，零/未设置quaternion被接受，generation只取position且orientation变化不会重置dwell。启用yaw后quaternion必须finite、nonzero、可归一化，generation包含wrapped yaw。新goal到达`GOAL_REACHED`只设置restart-pending；`/uto/resume`还必须验证PX4 connected/hold-ready/no-failsafe、goal valid、belief present且stable、velocity和local path fresh。成功后清除旧candidate/dwell、保留旧active terminal hold，并从`WAIT_BELIEF_STABLE`重新走startup readiness，直到新trajectory commit才替换hold reference。

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
| IFDS explicit mission goal | `/ifds/mission_goal` (`geometry_msgs/PoseStamped`) |
| Executable trajectory | `/uto/trajectory` |
| Emergency execution command | `/uto/execution_command` |
| Diagnostics / PX4 state | `/uto/diagnostics`, `/uto/px4_status` |
| Resume | `/uto/resume` (`std_srvs/Trigger`) |
| PX4 topics | 全部在`gazebo_harmonic_px4.yaml`参数化 |

IFDS契约是：(1) 持续发布有新ROS timestamp的local path；(2) 每个任务或全局goal变化时发布mission goal；(3) 绝不直接向PX4/MAVROS发布setpoint。`SAFE_HOLD`只有PX4 connected/no-failsafe/hold-ready、belief重新稳定、velocity/path fresh时才能通过resume；FAULT不能普通resume。

## IFDS–UTO integration（第一阶段）

唯一在线控制链为：

```text
FAST-LIO2 /Odometry ───────────────────────────────→ uto_planner
/ifds/goal → ifds_planner → /ifds/mission_goal ───→ uto_planner
                         └→ /ifds/local_path ───────→ uto_planner
                         └→ /ifds/path_status ──────→ uto_planner
uto_planner → /uto/trajectory → px4_offboard_bridge → /fmu/in/* → PX4
```

`ifds_planner`是严格的path-only节点：它只消费position mean、任务goal和障碍物，发布path、结构化status和检查后的mission goal；代码中不创建MAVROS/PX4 publisher、setpoint timer、carrot chasing或hold command。FAST-LIO的完整belief只进入UTO，IFDS不消费其covariance。`px4_offboard_bridge`是唯一飞控setpoint发布者。仓库若同时包含`IFDS_integration_node/ifds_gz_plugins`、`rgl_livox_converter`和`world_sdf`，这些只是Gazebo/传感器辅助包，不属于在线控制链，也不能发布飞控setpoint。

### Topics与安全契约

| Owner | Topic | Type/语义 |
|---|---|---|
| mission client → IFDS | `/ifds/goal` | `geometry_msgs/PoseStamped`原始任务输入 |
| IFDS → UTO | `/ifds/mission_goal` | reliable + transient-local检查后目标 |
| IFDS → UTO | `/ifds/local_path` | `nav_msgs/Path`，精确goal为末点 |
| IFDS → UTO | `/ifds/path_status` | `std_msgs/String`，严格JSON schema |
| UTO → bridge | `/uto/trajectory` | 唯一可执行physical trajectory |

Goal和odometry若不在`planning_frame=map`，IFDS必须先取得TF并实际变换坐标；empty frame、TF timeout或转换失败会发布`valid=false`，绝不只替换`frame_id`。静态障碍物YAML必须声明`frame_id: map`；在线Marker障碍物frame不一致时fail closed。`target_threshold=0.05 m`仅控制IFDS积分终止，发布前还会用现有segment-clearance判据验证并追加精确mission goal，最终到达仍由UTO的position/velocity/dwell/freshness判断。

每条Path紧接一个status；`path_stamp_ns`必须精确等于Path header stamp，`valid_until`默认0.8 s。UTO只有在status valid、stamp匹配且未过期时使用Path。首次执行前invalid会清除path/candidate并留在hold等待；`TRAJECTORY_READY/EXECUTING/REPLANNING`收到invalid、无法解析的JSON或stamp mismatch会立即丢弃candidate并进入`SAFE_HOLD`，不会等待普通`path_timeout`。Path generation只hash按`path_generation_resolution=0.05 m`量化后的几何，不含timestamp，因此相同路径的高频重发不会让约0.9 s first solve永久stale；freshness仍使用最新Path timestamp/status。

发送任务示例：

```bash
ros2 topic pub --once /ifds/goal geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: map}, pose: {position: {x: 5.0, y: 0.0, z: 1.5}, orientation: {w: 1.0}}}"
ros2 topic echo /ifds/path_status
```

启动模式与障碍物选择：

```bash
ros2 launch uto_ros2 uto_ifds_gazebo.launch.py mode:=global \
  ifds_obstacles:=/absolute/path/static_obstacles.yaml dynamic_obstacles:=false
ros2 launch uto_ros2 uto_ifds_gazebo.launch.py mode:=online \
  ifds_obstacles:=/absolute/path/static_seed.yaml dynamic_obstacles:=true
```

三个节点统一使用sim time和`map` frame，不使用固定sleep。PX4先takeoff/hold，UTO按`WAIT_PX4 → TAKEOFF → HOLD → WAIT_BELIEF_STABLE → WAIT_IFDS_INITIAL_PATH → BUILDING_NLP → FIRST_SOLVE → EXECUTING`推进；约0.9 s cold build/solve和约0.4 s online solve期间bridge仍独立发布hold/setpoint。当前第一阶段没有动态障碍物未来位置预测、commit-time path compatibility检查或UTO obstacle constraints；path-tube gate仍不等价于碰撞保证。

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

Pure tests覆盖SO(3)、yaw wrap、delay velocity dispersion/process noise、dense rollout、dual-rate commit helpers、completion queue、goal dwell、semantic path generation、IFDS exact-goal path/status/TF fail-closed contract、trajectory interpolation、PX4 control levels和旧模块检查。Solver test包含小型1×2×7和默认2×5×7的build/two-solve/reuse/residual/covariance/timing检查。

ROS optional test在依赖齐全时实际初始化rclpy并构造planner/bridge，检查独立planning/commit timers和默认PX4层级；它仍不是完整fake-topic状态序列或Gazebo验证。若目标CI需要完整消息闭环，应继续加入匹配该PX4版本的fake publishers/ACK fixture。当前README不声称Gazebo、PX4 SITL或未执行tests已经通过。

本次提交所处容器没有NumPy、CasADi、ROS 2、`px4_msgs`、Gazebo或`colcon`：Python AST/bytecode和YAML重复键检查已执行；core/solver pytest在collection阶段因缺少NumPy而未运行，ROS optional test按设计skip，SITL未运行。必须在依赖完整的目标workspace重新执行上述全部命令，不能把这些环境阻塞视为功能通过。

## Real-flight前

必须在目标ROS/PX4 workspace通过pure tests、默认CasADi solve、colcon build/test和完整fake-topic integration，再运行Gazebo/PX4 SITL fault injection并记录40 Hz setpoint、cold/first/steady timing。核对PX4 ACK result、topic、timesync、frame alignment和OffboardControlMode优先级；定量对照MATLAB；测试belief/path/velocity/DDS/PX4 loss及late candidate；保持real vehicle `auto_arm_takeoff=false`，配置geofence、RC takeover、kill switch并独立审查bounds、dense gate和commit thresholds。
