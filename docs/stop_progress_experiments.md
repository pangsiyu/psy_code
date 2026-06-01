# Stop Progression Head Experiments

本文档用于把 Stop Progression Head 做成一个快速、可复现的实验闭环。默认不加任何新参数时，JanusVLN 仍按原来的 action text generation 方式评测。

## 数据准备

给训练标注补 `stop_progress`：

```bash
python tools/add_stop_progress_to_jsonl.py \
  data/train_r2r_rxr.json \
  data/train_r2r_rxr_stop_progress.jsonl \
  --mode step_ratio
```

可选 label 模式：

- `step_ratio`: `step_id / max(traj_len - 1, 1)`
- `binary_stop`: assistant action 是 `STOP` 时为 1，否则为 0
- `distance_progress`: `1 - distance_to_goal / start_distance_to_goal`，clip 到 `[0, 1]`

带 `stop_progress` 训练时必须保持 `--data_flatten False`。packed sequence 下样本边界会被打包，`stop_progress` target 无法可靠对齐到单个样本。

## 快速评测

本项目当前固定子集放在：

- `data/eval_subsets/val_unseen_smoke20.json`: 20 条 smoke test。
- `data/eval_subsets/val_unseen_dev100.json`: 100 条 dev 消融。
- `data/eval_subsets/val_unseen_mid300.json`: 300 条中等规模复验。

不要用“临时跑前 N 条 episode”的结果和这些固定子集混在一起比较。`scripts/stop_progress/eval_smoke_baseline.sh` 和 `scripts/stop_progress/eval_smoke_stop_gate.sh` 默认使用 smoke20；`eval_threshold_sweep.sh` 默认使用 dev100。

当前资源约定：

- Stop Progression Head 小闭环脚本默认使用 `CUDA_VISIBLE_DEVICES=0,1`，并通过 `scripts/stop_progress/gpu_guard.sh` 禁止使用物理 3、7 号卡。
- 如需换卡，可以显式设置 `CUDA_VISIBLE_DEVICES=2,4`、`CUDA_VISIBLE_DEVICES=4,5` 等不包含 3/7 的组合；同一组消融必须记录实际资源设定。
- 当前主路线不再硬跑完整 JanusVLN backward，而是先提取 hidden cache，再离线训练 stop head。

20 条 smoke test：

```bash
bash scripts/stop_progress/eval_smoke_baseline.sh
bash scripts/stop_progress/eval_smoke_stop_gate.sh
```

Stop Head 训练/推理闭环 smoke：

```bash
bash scripts/stop_progress/train_stop_head_smoke_gpu3.sh
bash scripts/stop_progress/eval_stop_gate_smoke_gpu3.sh
```

固定 100 条 dev subset：

```bash
EPISODE_SUBSET_JSON=data/eval_subsets/val_unseen_dev100.json \
MAX_EVAL_EPISODES=100 SAVE_STEP_LOGS=1 \
bash scripts/stop_progress/eval_smoke_stop_gate.sh
```

如果已经有固定子集：

```bash
EPISODE_SUBSET_JSON=data/dev_100_subset.json SAVE_STEP_LOGS=1 \
bash scripts/stop_progress/eval_smoke_stop_gate.sh
```

`data/dev_100_subset.json` 格式：

```json
[
  {"scene_id": "2azQ1b91cZZ", "episode_id": "123"}
]
```

## 主消融 A0-A4

- A0: 原始 `JanusVLN_Base`，不开 stop head，不开 stop gate。
- A1: 训练 stop head，但 evaluation 不开 gate，确认新增 head 不改变 action generation 主路径。
- A2: 训练 stop head，evaluation 开 `--use_stop_progress_gate`，默认 threshold `0.85`。
- A3: 在固定 100 episodes dev subset 上做 threshold sweep，选择 SR/SPL 和 stop metrics 最稳的阈值。
- A4: 只对 A3 选出的最佳配置跑完整 `val_unseen`。

## 训练范围消融 B1-B4

- B1: `--trainable_scope stop_head_only`
- B2: `--trainable_scope stop_head_merger`
- B3: `--trainable_scope stop_head_lora`
- B4: `--trainable_scope stop_head_merger_lora`

推荐先跑 B1，因为它只训练 `stop_progress_head`，最快也最能判断 stop label 本身是否有效。

## Label 消融 C0-C3

- C0: `step_ratio`，旧方案，只作为对照。
- C1: `distance_progress`，距离进度回归。
- C2: `stop_ready`，主方案，`distance_to_goal <= stop_success_radius` 时为 1。
- C3: `stop_ready + hard negatives`，强化靠近但不该停或模型早停的负例。

当前优先执行 C2。`data/train_r2r_only.json` 本身没有 `distance_to_goal`，需要用 `tools/build_r2r_distance_stop_labels.py` 结合 `data/datasets/r2r/train/train.json.gz` 和 `data/datasets/r2r/train/train_gt.json.gz` 生成 `data/train_r2r_only_stop_ready_geo.jsonl`。

## 指标

`result.json` 的 episode 行和最终 summary 会包含：

- `early_stop_count`
- `correct_stop_count`
- `missed_stop_count`
- `total_stop_count`
- `gate_trigger_count`
- `gate_correct_count`
- `invalid_action_count`
- `avg_stop_distance`

开启 `--save_step_logs` 后，`step_logs.jsonl` 每步保存：

- `raw_action`
- `parsed_action`
- `final_action`
- `stop_progress`
- `distance_to_goal`
- stop/gate/invalid 相关布尔标记

## 推荐执行顺序

1. 跑 20 条 smoke test，确认模型能加载、step log 正常、非法 action 不再默认 STOP。
2. 固定 100 episodes dev subset，所有 A/B/C 消融都在同一子集上比较。
3. 在 100 episodes 上做 threshold sweep。
4. 只把最佳方案跑完整 `val_unseen`，避免每个中间版本都消耗 6-8 天。
