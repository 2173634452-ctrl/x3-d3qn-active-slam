# 数据和网络接口

1. Gazebo 发布 RGB、深度、CameraInfo。`runtime/rgbd_node.cpp` 将图像交给 ORB-SLAM3，发布估计位姿、跟踪状态等；`prepare_runtime.py` 生成带 `/slam/map_epoch` 的封装，字段为当前 map ID 和 big-change 索引。
2. `base_navigation.py` / `exploration_bridge.py` 将相机位姿转换成车体位姿，同步深度并投影到自由/障碍/未知栅格。地图版本变化时取消旧目标。同一地图内，运行节点通过 `/slam/map_anchors` 发布带地图版本、关键帧 ID 和采集时间的位姿快照。栅格按保留的深度观测相对于附近关键帧的位姿重建。旧动作跨越修正时，其转移不进入训练数据。该方法以关键帧局部刚性变换近似非关键帧的修正，最多保留 3,000 帧已积分的深度观测，缺少存活锚点的观测不参与重放。
3. `frontiers.py` 找自由区与未知区边界，构造安全接近点、检查连通性；`episode_worker.py` 调用 Nav2 规划检查可达性。保留最多10个候选；不足时填零并用动作掩码屏蔽。
4. 状态为 `float32[36]`：归一化车体 xy 共2维、四元数4维、10个候选的车体相对 xy 共20维、候选局部未知面积收益代理10维；掩码 `int8[10]`。它不包含论文完整 D-optimality 指标。
5. 网络输入 `[B,36]`，共享层 `36→128 ReLU→64 ReLU`；V 分支 `64→32 ReLU→1`，A 分支 `64→32 ReLU→10`；输出 `Q=V+A-mean(A,dim=-1)`，共17,515参数。非法动作在选最大值前屏蔽。
6. 动作为候选索引0–9，转成 `NavigateToPose` 目标，经 NavFn / Regulated Pure Pursuit 输出速度。不是网络直接输出轮速。
7. 执行动作后记录下一状态、导航结果、模拟时长和新增自由面积。奖励为新增自由面积减 `0.01 × 模拟秒数`，再减失败罚项。通信、传感器或地图故障不应伪装成正常有效转移，具体以 `valid_transition` 为准。
8. JSONL 每行包含 `state, action_mask, action, reward, next_state, next_action_mask, terminated, truncated, duration`。`collect_long.py` 只保留有效转移；采集时网络更新次数为0。
9. 离线训练采用 online 网络选下一动作、target 网络估值，折扣 `gamma ** (duration / 10)`；terminal 或无有效下一动作时不 bootstrap。使用 Smooth L1、梯度裁剪和定期 target 同步。真正时间截断与 MDP 终止分开保存。
10. 保存检查点后新进程加载网络做 greedy 推理，核对固定输入 Q 值，重新运行仿真。短程通过只证明加载和闭环可执行，不证明策略最优。

每个回合重新启动 Gazebo / ORB / Nav2，Gym 在决策边界暂停仿真，通过原子写入的文件协议交换请求和响应。新增统一入口为并行采集器设置互不重叠的 ROS domain 范围；Gazebo partition 同时使用回合 UUID。
