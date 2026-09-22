# SD3 + ControlNet + RA DPO 使用说明

本流程始终冻结 SD3 Transformer 主干，并通过 YAML 独立控制是否使用 DPO 更新 ControlNet、RA，或同时更新二者。DPO 训练开始时加载的 ControlNet 和 RA 权重构成参考策略（reference policy）。

## 数据划分

严格实验应使用训练集或独立验证集的 GT 构建离线偏好数据，最终测试集只用于评估。不要使用测试集 GT 构造偏好对后，再在同一测试集上宣称无偏性能提升；这种结果属于测试集适配。

当前配置优先读取经过筛选的 source manifest：

```text
/root/autodl-tmp/datasets2/manifests/dpo_source_selection.json
```

只有将 `selection_manifest` 设为 `null` 时，才会回退到按天气目录扫描。

## 1. 生成离线候选组

配置文件为 `config/dpo_sd3.yaml`：

```yaml
candidate_generation:
  dataset_rain: "/root/autodl-tmp/datasets2/train/rain"
  dataset_snow: "/root/autodl-tmp/datasets2/train/snow"
  dataset_haze: "/root/autodl-tmp/datasets2/train/haze"
  selection_manifest: "/root/autodl-tmp/datasets2/manifests/dpo_source_selection.json"
  splits: ["train"]

  # 从 curated manifest 中按固定 seed 抽样。
  sample_mode: "random"
  rain_max_samples: 2000
  snow_max_samples: 2000
  haze_max_samples: 2000

  num_candidates_per_image: 6
  candidate_guidance_scales: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
  seed: 20240805

  weather_psnr_gap_thresholds:
    rain: 0.0
    snow: 0.0
    haze: 0.0

  # 每种天气最多保存多少个通过 gap 筛选的候选组。
  max_saved_groups_per_weather: 2000
```

`rain_max_samples`、`snow_max_samples`、`haze_max_samples` 控制参与推理检查的源图片数量；`max_saved_groups_per_weather` 控制最终保存的有效组数量，两者含义不同。

运行：

```bash
python -m scripts.generate_dpo_candidates --config config/dpo_sd3.yaml
```

生成器按源图逐张处理：

1. 读取一张 LQ 和对应 GT。
2. 根据基础 `seed` 为 6 个候选派生不同随机种子。
3. 为 6 个候选分别使用配置的 guidance scale。
4. 顺序生成 `candidate_00` 至 `candidate_05`，控制单次推理显存。
5. 分别计算 6 个 candidate 与同一 GT 的 PSNR、SSIM、LPIPS。
6. 计算 `PSNR gap = max(PSNR) - min(PSNR)`。
7. gap 未达到天气阈值时不保存图片；达到阈值时立即保存完整候选组。
8. 某天气保存数量达到 `max_saved_groups_per_weather` 后，跳过该天气剩余图片。

候选噪声直接在内存中生成，不创建 Noise Bank 文件。相同数据、配置和 seed 下可复现。

生成时只显示一个 `tqdm` 进度条，后缀包括：

```text
weather=rain
psnr=28.13
ssim=0.8381
lpips=0.0670
gap=0.274/0.200
kept=126
status=saved | rejected | limit
```

候选按 guidance scale 顺序执行，pipeline 单次 batch size 为 1。候选间不再计算 pairwise LPIPS。

输出结构：

```text
dpo_candidates/
|-- candidates/
|   |-- rain/
|   |   `-- image_000000_<stem>/
|   |       |-- lq.png
|   |       |-- gt.png
|   |       |-- metrics.txt
|   |       |-- candidate_00.png
|   |       `-- candidate_05.png
|   |-- snow/
|   `-- haze/
|-- per_candidate_metrics.csv
|-- selected_samples.json
|-- rejected_samples.json
|-- sample_summary.csv
|-- dataset_per_noise.csv
|-- dataset_summary.csv
`-- summary.json
```

每个候选组的 `metrics.txt` 保存 6 个 candidate-to-GT 的 PSNR、SSIM、LPIPS，以及实际 PSNR gap 和对应天气阈值。

已有候选图不需要重新生成。安装 `pyiqa` 后，对 PNG 离线补算美学指标：

```bash
python -m scripts.rescore_dpo_candidates \
  --input_csv /root/autodl-tmp/sd3/experiment/dpo_candidates/per_candidate_metrics.csv \
  --output_csv /root/autodl-tmp/sd3/experiment/dpo_candidates/per_candidate_metrics_aesthetic.csv
```

脚本计算 MUSIQ、CLIP-IQA+、NIMA 和 DISTS，并按天气使用候选全集固定的
`(score - median) / IQR` 生成 `musiq_z`、`clipiqa_z` 和 `nima_z`。
归一化统计写入同目录的 `*_normalization.json`。

## 2. 构建偏好对

运行：

```bash
python -m scripts.filter_dpo_pairs --config config/dpo_sd3.yaml
```

偏好筛选配置示例：

```yaml
preference_filter:
  candidate_metrics_path: "/root/autodl-tmp/sd3/experiment/dpo_candidates/per_candidate_metrics_aesthetic.csv"
  output_dir: "/root/autodl-tmp/sd3/experiment/dpo_preferences"
  pair_strategy: "all_pairs"
  min_psnr_gap: -0.15
  weather_specific_thresholds:
    rain: -0.15
    snow: -0.15
    haze: -0.15
  min_reward_gap: 0.05
  fidelity_constraints:
    max_dists_pair_degradation: 0.01
    baseline_mode: "group_median"
    baseline_psnr_tolerance: 0.50
    baseline_dists_tolerance: 0.02
  max_samples_per_pair: 3
  max_pairs_per_weather: null
  shuffle: true
  random_seed: 42
```

支持三种配对策略：

- `best_vs_worst`：每组只使用 aesthetic reward 最高和最低的候选。
- `best_vs_all`：reward 最高候选分别与其他候选配对。
- `all_pairs`：组内所有候选两两组合，reward 较高者为 chosen。

所有策略都应用 PSNR、DISTS 和候选组中位数保真约束，然后按 aesthetic reward gap
从大到小最多保留 `max_samples_per_pair` 对。

输出：

```text
dpo_preferences/
|-- preference_pairs.jsonl
|-- preference_pairs.csv
`-- preference_summary.json
```

每行偏好数据包含 LQ、GT、chosen、rejected、天气、候选索引、PSNR、reward 和 gap 等字段。

## 3. 离线奖励

奖励配置：

```yaml
reward:
  weights:
    musiq_z: 0.55
    clipiqa_z: 0.35
    nima_z: 0.10
  directions:
    musiq_z: 1.0
    clipiqa_z: 1.0
    nima_z: 1.0
```

`weights` 决定每个指标对总奖励的权重，值为 0 或未配置表示不参与奖励。

`directions` 将不同指标统一成“奖励越大越好”。三个 z-score 都是越大越好。

奖励计算公式为：

```text
reward = 0.55 * musiq_z + 0.35 * clipiqa_z + 0.10 * nima_z
```

PSNR 和 DISTS 不进入 aesthetic reward，而作为保真门槛。chosen 最多允许损失
0.15 dB PSNR；DISTS pair 退化最多 0.01。每个源图现有候选的 PSNR/DISTS
中位数作为 SFT baseline，chosen 还必须满足绝对保真下限。

## 4. DPO 训练

运行：

```bash
accelerate launch scripts/train_dpo_sd3.py --config config/dpo_sd3.yaml
```

每个 preference pair 的 chosen 和 rejected latent 使用相同 timestep 和噪声。模型分别计算二者的 flow-matching MSE，并使用负 MSE 作为扩散 log-probability surrogate：

```text
policy_logratio = MSE_policy(rejected) - MSE_policy(chosen)
reference_logratio = MSE_ref(rejected) - MSE_ref(chosen)
loss = -log sigmoid(beta * (policy_logratio - reference_logratio))
```

参考分支使用训练开始时的 ControlNet/RA 初始参数快照，并共享冻结的 SD3 主干，不需要再加载一套完整 SD3。ControlNet reference 快照使用 BF16 以降低显存占用。

支持普通单卡和 DDP，不支持当前快速路径下的 FSDP 和 DeepSpeed。

训练模式：

```yaml
training:
  train_controlnet: true
  train_ra_fusion: true
  controlnet_learning_rate: 5.0e-8
  ra_fusion_learning_rate: 1.0e-7
```

- 仅训练 RA：`train_controlnet: false`、`train_ra_fusion: true`。
- 仅训练 ControlNet：`train_controlnet: true`、`train_ra_fusion: false`。
- 联合训练：二者都为 `true`。
- 二者都为 `false` 时程序报错。

推荐初始超参数：

| 参数 | 默认值 | 建议范围 |
|---|---:|---:|
| `beta` | 0.1 | 0.05–0.5 |
| ControlNet 学习率 | 5e-8 | 2e-8–2e-7 |
| RA 学习率 | 1e-7 | 5e-8–5e-7 |
| batch size | 4 | 1–4 |
| 梯度累积 | 2 | 2–16 |
| 训练步数 | 5000 | 1000–10000 |
| `sft_weight` | 0 | 0–0.1 |
| `gt_flow_weight` | 0.1 | 0.03–0.3 |
| `gt_x0_l1_weight` | 0.01 | 0–0.1 |

第一轮建议保持：

```yaml
sft_weight: 0.0
gt_flow_weight: 0.1
gt_x0_l1_weight: 0.01
weight_by_psnr_gap: false
```

`sft_weight` 只拟合 chosen candidate，并不直接监督 GT。`gt_flow_weight` 使用
GT latent 的 SD3 velocity target，`gt_x0_l1_weight` 则约束由 velocity 还原的
predicted-x0。两个 GT 分支只更新 policy，reference policy 仍保持训练开始时的
不可变快照。GT 与对应 preference pair 复用相同的 timestep、noise、LQ 和 prompt。

如果直接按 PSNR gap 加权，gap 较大的 haze 可能主导多天气训练。

训练输出：

```text
SD3_ControlNet_RA_DPO/
|-- checkpoint-250/
|   |-- controlnet/
|   `-- ra_fusion/
|-- controlnet/
|-- ra_fusion/
|   |-- config.json
|   `-- ra_fusion.safetensors
|-- dpo_config.yaml
`-- training_summary.json
```

### Checkpoint 验证与保留数量

```yaml
training:
  checkpointing_steps: 500
  checkpoints_total_limit: 3

  run_checkpoint_validation: true
  validation_eval_config: "./config/eval_sd3.yaml"
  validation_num_samples_per_weather: 10
  validation_num_inference_steps: 20
  validation_guidance_scale: 1.0
  validation_strength: 1.0
  validation_seed: 42
  validation_lpips_net: "alex"
```

每次保存 `checkpoint-N` 后，程序使用 `validation_eval_config` 中的 rain/snow/haze 测试路径，每种天气按确定性顺序取前 `validation_num_samples_per_weather` 张，计算 candidate-to-GT 的 PSNR、SSIM、LPIPS。验证使用固定 seed，并复用训练进程中当前的 ControlNet、RA、SD3 和 VAE，不重复加载完整模型。

每个 checkpoint 新增：

```text
checkpoint-N/
|-- controlnet/
|-- ra_fusion/
|-- validation_metrics.json
`-- validation_per_image.csv
```

`validation_metrics.json` 包含每天气和整体平均指标；启用 TensorBoard/W&B 时还会上报：

```text
validation/overall_psnr
validation/overall_ssim
validation/overall_lpips
validation/rain_psnr
validation/snow_psnr
validation/haze_psnr
...
```

`checkpoints_total_limit: 3` 表示每次完成保存和验证后，只保留 step 最大的 3 个 `checkpoint-*` 目录，旧 checkpoint 会被递归删除。设为 `0` 表示不限制数量。最终输出目录下的 `controlnet/`、`ra_fusion/` 不计入该限制。

保存的模型组件由训练开关决定；此外 checkpoint 还包含 optimizer、scheduler、RNG 和训练游标状态，因此既可用于评估，也支持完整断点续训。

### 断点续训

当前 checkpoint 同时保存：

- 当前 policy ControlNet/RA 权重。
- AdamW optimizer 状态。
- LR scheduler 状态。
- mixed-precision scaler 状态（如适用）。
- Python、NumPy、CPU/CUDA RNG 状态。
- `global_step`、epoch 和下一 dataloader batch 游标。

配置方式：

```yaml
training:
  # 不续训，从原始 Baseline + RA 开始。
  resume_from_checkpoint: null
```

自动选择 `output_dir` 中最新的可续训 checkpoint：

```yaml
training:
  resume_from_checkpoint: "latest"
```

或者显式指定：

```yaml
training:
  resume_from_checkpoint: "/root/autodl-tmp/sd3/experiment/SD3_ControlNet_RA_DPO/checkpoint-2500"
```

续训启动命令不变：

```bash
accelerate launch scripts/train_dpo_sd3.py --config config/dpo_sd3.yaml
```

resume 会先从 `model` 中的原始 Baseline 权重建立不可变 reference 快照，再把 checkpoint 权重恢复到 policy，因此 reference policy 不会被错误替换为续训权重。

训练 dataloader 使用固定 seed 的 seedable sampler；checkpoint 记录 epoch 和下一 batch 位置。恢复时会重建同一 epoch 的样本顺序并跳过已经消费的 batch。Checkpoint 验证前后会保存并恢复 Python、NumPy、CPU 和 CUDA RNG，验证不会改变后续训练随机序列。

允许提高 `max_train_steps` 延长训练。为保证 optimizer、scheduler 和数据游标正确，以下配置必须与 checkpoint 一致：

- `train_controlnet`、`train_ra_fusion`
- `train_batch_size`、`gradient_accumulation_steps`
- `preference_manifest`
- `beta`
- ControlNet/RA 学习率
- LR scheduler 类型和 warmup steps
- GPU/进程数量、seed、数据集长度和 preference manifest 内容哈希
- resolution、`sft_weight`、`gt_flow_weight`、`gt_x0_l1_weight`、`weight_by_psnr_gap`

旧版仅含模型权重、没有 `dpo_resume.json` 和 Accelerate 状态文件的 checkpoint 不能完整续训；程序会明确报错，不会静默重置 optimizer。

## 5. 推理与评估

运行：

```bash
python -m scripts.evaluate_dpo --config config/dpo_sd3.yaml
```

评估脚本会自动选择权重：

- 训练过的组件从 `<training.output_dir>` 加载。
- 未训练的组件继续使用 `model` 中的原始路径。
- 如果最终权重不存在，会查找最新且包含所有已启用组件的 `checkpoint-N`。

也可以显式指定：

```bash
python -m scripts.evaluate_dpo --config config/dpo_sd3.yaml \
  --controlnet_model_path /path/to/checkpoint-750/controlnet \
  --ra_fusion_path /path/to/checkpoint-750/ra_fusion
```

评估输出：

```text
<timestamp>_eval/
|-- metrics.txt
|-- metrics.json
|-- per_image_metrics.csv
|-- rain_test/per_image_metrics.txt
|-- snow_test/per_image_metrics.txt
`-- haze_test/per_image_metrics.txt
```

对比 Baseline + RA 和 DPO 模型时，应保持推理步数、strength、CFG、RA scale、测试样本、seed 和指标配置一致。
