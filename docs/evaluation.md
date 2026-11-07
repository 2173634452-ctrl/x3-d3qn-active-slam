# 训练监控、对照评估与模型选择

## 指标各自回答什么问题

| 指标 | 用途 | 不能据此推断 |
|---|---|---|
| Smooth L1 TD loss / 绝对 TD 误差 | 网络是否在拟合自举目标 | 探索能力提升或收敛 |
| Q 均值 / 最大绝对 Q / V、A 梯度 | 检查数值漂移、价值过估计、分支未学习 | Q 值是真实回报 |
| 独立仿真的回合回报 | 同一协议下比较策略 | 跨奖励定义或跨预算直接比较 |
| 新增局部自由面积 | 探索产出的地图代理量 | 全屋真实覆盖率 |
| 导航成功率、异常和提前中断 | 闭环可靠性 | ORB / Nav2 的贡献全来自 DRL |

D3QN 使用 epsilon-greedy，不包含 PPO 式的策略熵正则项。固定数据训练时不执行行为策略，因此不提供训练 epsilon / entropy 曲线。数据采集策略中的 epsilon 为0.35，不随离线梯度更新变化。历史未记录的 Q / TD 指标不补造。

## 初步对照协议

`evaluate_suite.py` 比较随机合法动作、纯启发式、随机初始化网络、训练后模型。四组共享世界、默认 X3 初始位置、回合种子列表、目标次数与模拟时间上限；每回合完整重启。网络初始化固定 seed42；每组默认3回合，种子11/22/33。Gazebo 与 ORB-SLAM 不保证严格确定性，种子不等于确定的不同场景。

初步默认最多6次动作，模拟时间180秒。时间限制在动作完成时检查，存在单次动作造成的超时；记录实际模拟时间和超出秒数。它是短程、按动作边界计量的基准，**不是严格相同模拟秒数的全屋覆盖实验**。若因地图版本变更或传感器/跟踪故障结束，保留此前有效回报和面积并统计中断，不丢弃整回合。

导航成功率分母为所有尝试目标。发生无完整时长记录的中断时，不计算该回合面积/时间效率；不使用短暂高效率掩盖故障。组内显示均值与样本标准差，配对差值只在协议相同时计算；3回合不足以声称显著改进。

受资源影响时，使用 `--workers 1` 串行评估。四组并行只是快速排查；如果跟踪频繁中断，应在串行或低并发下重测，避免把资源竞争归因于策略。

## 命令

```bash
# 独立比较现有模型；不更新参数、不录像
bash scripts/container.sh python3 learning/evaluate_suite.py \
  --checkpoint /results/run_001/offline/d3qn_offline.pt \
  --output /results/comparison_001 --workers 1 --seeds 11 22 33 --decisions 6

# 固定数据训练并定期做独立仿真验证；保存初始模型与每2500次更新的模型
bash scripts/container.sh python3 learning/train_offline.py \
  --dataset /results/run_001/transitions.jsonl --output /results/training_002 \
  --updates 10000 --checkpoint-every 2500 --eval-seeds 11 22 33 --eval-decisions 6

# 单独生成或持续更新监控页；历史训练报告可省略
bash scripts/container.sh python3 learning/build_dashboard.py \
  --root /results/training_002 --output /results/training_002/monitor --watch-seconds 5

# 在宿主机只对本机开放页面；网络访问请使用 SSH 隧道
python3 -m http.server 17864 --bind 127.0.0.1 --directory /absolute/path/to/results/training_002/monitor
```

默认 `scripts/pipeline.py` 已启用：每2500次更新后3回合验证，训练完成后4组串行对照，运行时每5秒更新 `monitor/index.html` 和 `monitor-data.json`，结束后关闭指标生成进程。HTTP 服务不会自动创建；也可直接打开生成的 HTML 文件查看快照，或在页面导入 JSON/JSONL。

启用仿真评估后，**整个训练流程耗时包含验证，明显长于纯梯度更新**。`optimization_seconds` 单独记录更新耗时。纯训练脚本不传 `--eval-seeds` 时保持无仿真的离线训练，但仍保存检查点与逐步指标。

## 最佳模型和收敛

训练脚本仅从完整、至少3回合、没有异常和提前中断的检查点中，按验证平均回报选择 `d3qn_best_validation.pt`，同时写 `best_checkpoint.json`；没有满足条件的检查点就明确不选。最终模型 `d3qn_offline.pt` 始终单独保留。验证集上选出的最佳模型不等于独立测试集成绩。

收敛目前不自动判定。较完整的结论还需要：更多初始位置/地图、独立保留的测试场景与种子、多次训练种子、足够长的探索回合，以及检查点成绩与不确定性在多个评估点趋稳。可以预先约定“连续3次验证相对改善不足5%”作为早停候选，但不能把它直接命名为收敛。固定数据 D3QN 的分布外价值估计风险也不会因 loss 平稳而消失。

## 输出

- `metrics.jsonl`：每次更新落盘的训练指标。
- `checkpoints/update_XXXXXXXX.pt`：保留的模型快照，初始检查点反映实际初始权重（续训时为载入模型）。
- `evaluations/update_XXXXXXXX/benchmark.json`：各检查点独立验证。
- `evaluation/{random,heuristic,untrained,trained}/benchmark.json`：策略对照原始记录。
- `comparison.json`：均值、标准差、配对差值、异常统计，始终不宣称收敛。
- `monitor/index.html`：离线也能打开的交互曲线页面；通过 HTTP 查看时每5秒读取同目录指标快照。
