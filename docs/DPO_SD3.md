# SD3 + ControlNet + RA：DPO 主流程

本文是 DPO 操作的统一入口，按当前 **M=32 候选、静态/动态奖励、简版导出、单卡训练** 整理。
命令均在服务器项目根目录 `/root/sd3_controlnet` 执行。

已有候选后的必需流程：

```text
per_candidate_metrics.csv
    → IQA 评分：per_candidate_metrics_aesthetic.csv
    → 偏好筛选：自动准备共用 FR/NR CSV，再生成 preference_pairs.jsonl
    → 单卡 DPO 训练
    → 评估
```

候选已成功导出时，从第 3 节开始；已有美学评分时，从第 4 节开始。
均值、方差等 CSV 分析是可选步骤，不是训练前置条件。

## 1. 配置、路径与当前参数

### 配置选择

| 配置 | 用途 |
|---|---|
| `config/dpo_sd3_reward_static.yaml` | 独立完整的 M32 静态配置，包含评分、筛选、训练与评估全部参数 |
| `config/dpo_sd3_reward_dynamic.yaml` | 独立完整的 M32 基础保真贡献、二次惩罚及双优优先动态配置 |
| `config/dpo_sd3.yaml` | 保留的历史配置，供旧实验使用；不是上述两份配置的父配置 |

静态和动态各只保留一份配置，均不使用 `base_config`，也不相互继承。
修改某种模式时，直接编辑该模式的文件；另一模式不会自动改变。
两份配置共用 M32 候选评分，分别保存偏好、训练和评估结果。做公平对照时，请同步共同超参数。
静态 PSNR 下界消融也直接修改 `dpo_sd3_reward_static.yaml`，不再使用额外叠加文件。

### 本批数据的固定路径

| 内容 | 路径 |
|---|---|
| 原中断任务 | `/root/autodl-tmp/sd3/experiment/dpo_candidates_m32_selected2000_3gpu` |
| 简版导出候选集 | `/root/autodl-tmp/sd3/experiment/dpo_candidates_m32_complete_groups` |
| 静态偏好目录 | `/root/autodl-tmp/sd3/experiment/dpo_preferences_static_m32_partial` |
| 动态偏好目录 | `/root/autodl-tmp/sd3/experiment/dpo_preferences_dynamic_m32_quadratic` |
| 静态训练目录 | `/root/autodl-tmp/sd3/experiment/SD3_ControlNet_RA_DPO_Static_M32_Partial` |
| 动态训练目录 | `/root/autodl-tmp/sd3/experiment/SD3_ControlNet_RA_DPO_Dynamic_M32_Quadratic` |

更换候选集时，同步修改 `candidate_generation.output_dir`、
`preference_filter.source_metrics_path`、`preference_filter.candidate_metrics_path`；
更换偏好或训练目录时，同步修改 `preference_filter.output_dir`、
`training.preference_manifest`、`training.output_dir` 和 `evaluation.output_dir`。
单独覆盖脚本输出路径不会自动修改 YAML。

### 当前生效的训练设置

以下是两份独立 M32 配置直接定义的值，以实际 YAML 为准：

| 参数 | 当前值 |
|---|---:|
| `training.beta` | 5000 |
| `training.controlnet_learning_rate` | 5e-8 |
| `training.ra_fusion_learning_rate` | 1e-7 |
| `training.max_train_steps` | 10000 |
| `training.train_batch_size` | 8 |
| `training.gradient_accumulation_steps` | 1 |
| `training.mixed_precision` | `bf16` |
| `training.checkpointing_steps` | 1000 |
| `training.checkpoints_total_limit` | 2 |
| `training.logging_steps` | 10 |
| `training.train_controlnet` / `training.train_ra_fusion` | `true` / `true` |
| `training.use_ema` / `training.validation_use_ema` | `true` / `true` |
| `training.resume_from_checkpoint` | `null` |
| `tail_risk.enabled` | `false` |
| `preference_filter.pairs_per_group` | 12 |

模型初始路径位于 `model` 配置，当前为 `SD3ControlNet_RA_Joint/checkpoint-5000` 及其
`ra_fusion`。它们必须与生成候选时记录的参考模型一致；候选来源会在训练前检查。
SD3 主干始终冻结，当前配置联合训练 ControlNet 和 RA。

## 2. 中断任务：简单检查并导出完整组

确认原生成进程已停止后执行。目标目录必须是新目录或空目录，且不能与源目录重叠。

```bash
python -m scripts.export_candidates_simple \
  --source /root/autodl-tmp/sd3/experiment/dpo_candidates_m32_selected2000_3gpu \
  --dest /root/autodl-tmp/sd3/experiment/dpo_candidates_m32_complete_groups \
  --expected-candidates 32
```

脚本逐组处理并流式写 CSV，仅做基础检查：

- 组缓存能读取，原状态为 `retained`。
- 候选编号完整覆盖 `0～31`，源图编号一致，不重复导出同一源图组。
- 候选路径属于对应分片，且同组图片位于同一目录。
- 32 张候选及缓存 `lq.png`、`gt.png` 存在且非空。

合格组复制到新目录，不补生成缺失图片，不修改源文件。编号、随机种子和指标保留，
仅改写 CSV 中的候选图片路径。脚本不解码图片、不计算哈希、不验证种子推导或模型指纹，
也不重新应用全局天气组数上限，因此不能识别非空但损坏的图片。
该步骤不需要 GPU，但复制图片仍需时间和额外磁盘空间。

输出：

```text
dpo_candidates_m32_complete_groups/
  candidates/<weather>/image_<global_index>_<stem>/
  per_candidate_metrics.csv
  skipped_groups.csv
  summary.json
  EXPORT_COMPLETE.json
```

`skipped_groups.csv` 只记录有缓存但被跳过的组，从未写入缓存的组不在此表中。
`summary.json` 记录实际导出数量和模型来源信息；`EXPORT_COMPLETE.json` 表示简版导出成功，
不是严格完整性认证，也不表示原生成计划全部完成。
CSV 中的原始 LQ/GT 路径仍用于后续来源检查，请保持原数据集可访问。
失败重试需使用新的空目标目录，并同步后续配置路径。

## 3. 计算美学与感知评分

原始 CSV 只有 PSNR、SSIM、LPIPS，必须先补算无参指标，再按当前奖励构建偏好对：

```bash
python -m scripts.rescore_dpo_candidates \
  --config config/dpo_sd3_reward_static.yaml \
  --input_csv /root/autodl-tmp/sd3/experiment/dpo_candidates_m32_complete_groups/per_candidate_metrics.csv \
  --output_csv /root/autodl-tmp/sd3/experiment/dpo_candidates_m32_complete_groups/per_candidate_metrics_aesthetic.csv \
  --device cuda:0 \
  --batch_size 4
```

此步骤加载 IQA 模型，不加载 SD3 生成模型，也不重新生成候选。建议使用 GPU；
改成 `--device cpu` 也能运行，但评分更慢。需要 `pyiqa` 及其模型权重；
CLIP 后端缺少 `pkg_resources` 时，可运行 `python -m pip install "setuptools>=68,<81"`。

| 新增指标 | 类型 | 方向 |
|---|---|---|
| MUSIQ | 无参考质量 | 越高越好 |
| CLIP-IQA | 无参考质量 | 越高越好 |
| NIMA | 无参考美学 | 越高越好 |
| DISTS | 全参考感知距离 | 越低越好 |

实际模型变体以评分器日志和输出元数据为准。评分 CSV 旁会生成对应的
`*_normalization.json`，记录按天气拟合的归一化参数和指标模型来源。
静态/动态共用这一份评分及统计量，不需要分别运行 IQA。
未评分的原始 CSV 不能使用 `--normalize_only`，已有评分文件则无需重新计算。

## 4. 构建静态与动态偏好对

需要比较两种模式时，按顺序各运行一次；只训练一种模式时，只执行对应命令：

```bash
python -m scripts.filter_dpo_pairs \
  --config config/dpo_sd3_reward_static.yaml

python -m scripts.filter_dpo_pairs \
  --config config/dpo_sd3_reward_dynamic.yaml
```

首次筛选时，若 `per_candidate_metrics_fr_nr.csv` 不存在，会从已有美学 CSV 自动执行
数值归一化并写入该共用文件，不运行 IQA 模型。第二种模式复用同一文件。
显式指定 `--candidate_metrics_path` 时，不会自动替换或生成该路径。

每个偏好目录输出 `preference_pairs.jsonl`、`preference_pairs.csv`、
`preference_summary.json` 和相关诊断。训练读取 `preference_pairs.jsonl`。
开始训练前，查看汇总中的保留组数、偏好对数、移除原因及动态分支比例；
合格对不足时不会放宽门槛或重复凑数，没有合格对时会输出空清单。

### 奖励与偏好规则

统一使用按天气拟合的中位数/IQR 标准化；尺度不足时依次回退为标准差、1，
标准化分数截断到 [-3,3]。缺失或非有限的必需指标不会用零填充。
已有归一化参数需保留，不能按 GPU、奖励模式或任意子集分别重新拟合。

```text
Q = 0.55*MUSIQ_z + 0.35*CLIP-IQA_z + 0.10*NIMA_z
F = (PSNR_z + SSIM_z - LPIPS_z) / 3
R_static = 0.5*F + 0.5*Q
```

FR 权重由 `reward.fr_weights` 指定，非负且归一化后总和为 1；上式为当前等权配置。
LPIPS 标准化分数保留原始方向，只在 F 中取负一次。

| 模式 | 偏好方向与差值门槛 |
|---|---|
| 静态 `static` | 按 R 决定方向；平分或绝对差值小于 `margin_static=0.05` 时跳过 |
| 动态 `dynamic` | 先跳过 Q/F 均不明显的组合，再优先确定双优方向；其他组合按 R_dynamic 与奖励差门槛判断 |

等于阈值时接受。静态公式保持不变；当前动态模式已替换旧的逐对 Q/F 阈值切换，
在同一源图内每张候选都有统一分数，排序满足传递性。
每组 32 个候选有 `32*31/2=496` 个无序组合，应用奖励和保真约束后，
用固定随机种子无放回均匀抽样，最多保留 `pairs_per_group=12` 对。
此数量可配置，不是程序硬编码上限。

默认情况下，两种模式都必须满足以下硬保真约束：

```text
chosen_PSNR - rejected_PSNR >= -0.15 dB
chosen_DISTS - rejected_DISTS <= 0.01
chosen_PSNR >= group_median_PSNR - 0.50 dB
chosen_DISTS <= group_median_DISTS + 0.02
```

#### 动态奖励：基础保真贡献、二次惩罚与双优优先

Q、F 和共用的天气归一化保持不变。对同一源图的全部候选先计算：

```text
B = median(F_1, F_2, ..., F_32)
d_i = max(0, B - F_i)
R_dynamic_i = Q_i + eta_fidelity * F_i - lambda_fidelity * d_i^2
```

- `eta_fidelity > 0`：无论是否达到 B，F 都按正系数计入基础贡献；在同一 B、Q 不变时，F 下降必使奖励下降。
  冲突时，美学增益仍可能抵消保真代价，由下述规则决定取舍。
- F 低于 B 时，额外按缺口平方扣分；缺口翻倍时，在相同 λ 下二次惩罚变为四倍。
- B 是同图候选的相对典型保真水平，不是绝对合格线，也不是原始 PSNR 中位数。
  原始 PSNR/DISTS 的硬约束仍在决定偏好方向后应用，不能由动态奖励绕过。

直接编辑唯一的动态配置：

```yaml
reward:
  mode: dynamic
  eta_fidelity: 0.5
  lambda_fidelity: 1.0
  margin_dynamic: 0.05
  min_gap_q: 0.01
  min_gap_fr: 0.01
```

`eta_fidelity`、`min_gap_q`、`min_gap_fr` 必须为正数，`lambda_fidelity` 不小于零。
λ 为 0 时，奖励仍为 `Q + eta_fidelity*F`，不是纯 Q。`margin_dynamic` 可为非负数。
以上数值仅为试验起点，不代表已验证最优；Q/F 差值阈值作用于归一化后的分量，不是原始 PSNR 的 dB 阈值。
这些参数只影响离线偏好构建，不直接作为训练损失或 GT 辅助项系数。
旧动态参数 `tau_q`、`margin_fr` 不再有效，配置中保留它们会明确报错。

B 在组内去重后、硬保真/文件过滤前，由 F 有效的候选统一计算；完整数据应为 32 张。
候选缺少 Q 时不参与配对，但其有效 F 仍可参与基准。F、身份或索引无效的记录及重复候选不重复计入。
不要在只剩下合格偏好对后重算 B，也不能只拿当前 A/B 两张图的 F 中位数作为 B。
`baseline_f_candidate_count` 会记录实际参与基准的数量，便于发现不足 32 张的组。

对枚举到的原始 A/B，定义 `delta_Q=Q_A-Q_B`、`delta_F=F_A-F_B`、`delta_R=R_A-R_B`，依次处理：

1. 若 `abs(delta_Q) < min_gap_q` 且 `abs(delta_F) < min_gap_fr`，跳过，原因为 `small_differences`。
2. 若 `delta_Q >= min_gap_q` 且 `delta_F >= min_gap_fr`，A 为 chosen，原因为 `both_better`。
3. 若 Q/F 差值都达到对称的负阈值，B 为 chosen，原因为 `both_better`。
4. 其他情况按 R_dynamic 定向；平分或 `abs(delta_R) < margin_dynamic` 时跳过，
   否则原因为 `reward_tradeoff`。差值等于门槛时接受。

双优分支绕过 `margin_dynamic`，但所有接受分支都必须再通过原 PSNR、DISTS 和组中位数硬约束。
旧的无参差值门槛不会再次删除已接受的有参优势样本。
最后按固定种子 42 无放回均匀抽取每组最多 12 对，不改变配对预算，也不强制总是用最高分候选配对。
偏好记录中的 `chosen_reward`/`rejected_reward` 都是统一的 R_dynamic，
另存 B、基础保真贡献、缺口、惩罚及奖励参数；`reward_branch` 为 `dynamic`，
`reason` 为 `both_better` 或 `reward_tradeoff`。
记录中的 `delta_Q`、`delta_F`、`delta_R` 始终是原始 A−B，不是 chosen−rejected；
对应 `candidate_a_noise_index`、`candidate_b_noise_index` 会一并保存。
每组 B 和基准候选数也保存在 `evaluated_pairs_per_group` 中。

筛选还会输出 `dynamic_candidate_rewards.csv`，按组内动态分数降序排列，
保留原列，并增加 `eta_fidelity` 和 `base_fidelity_contribution`（η×F），
同时记录 `rank_in_group`、Q、F、B、缺口、二次惩罚和最终动态奖励。
若重新评分脚本使用 dynamic 配置，也会在评分 CSV 添加 `reward_dynamic` 等组基准字段；
已有共用评分 CSV 不需要重写，偏好构建器会按当前配置重新计算奖励。

`dynamic_pair_decisions.csv.gz` 流式记录所有已枚举候选对的原始索引、B、三类差值和决策/拒绝原因。
`status` 区分 `decision_rejected`、`hard_rejected`、`not_selected` 和 `kept`；
双优定向后未通过硬约束的对仍记录其初始 `decision_reason=both_better` 和实际硬拒绝原因。
通过规则和硬约束却未被最终抽样/数量预算保留时，状态为 `not_selected`。
M32 完整组可能合计约 300 万条决策；使用压缩 CSV 和临时磁盘流式处理，不把这些记录集中存入内存。

`preference_summary.json` 中的 `dynamic_selection` 汇总：

- `accepted_before_hard`、`eligible_after_hard`、`kept`：双优/奖励权衡分支在硬筛前、硬筛后及最终保留的对数。
- `rejected`：小差异、奖励差不足、奖励平分、硬保真失败及未被最终抽样保留的数量。
- `hard_fidelity_by_reason`、`hard_fidelity_by_decision`：硬约束拒绝的细分原因与最初接受分支。
- `chosen_minus_rejected`：最终偏好对的原始 PSNR、原始 LPIPS、Q 的平均差及负差比例，按偏好对等权。
  LPIPS 的负差代表改善，不应误读为退化；另有 `worse_fraction` 按各指标方向报告变差比例。

这是本项目的实验性取舍方案，不保证所有指标同时提高。
新动态奖励与偏好契约版本为 3，旧 Q/F 切换及上一版纯二次惩罚的数据都不能只改标签继续使用。
训练前会核验新契约和每条记录的公式、参数、方向、原因及三类差值，不能静默复用旧公式数据。
更新后只需重新运行动态筛选，不重新生成候选或计算 IQA，也不自动生成静态结果：

```bash
python -m scripts.filter_dpo_pairs --config config/dpo_sd3_reward_dynamic.yaml
```

沿用现有配置内的输出路径管理方式，不另建训练入口。需保留上一轮结果时，先在同一动态 YAML 中
同步修改偏好输出目录、训练清单、训练输出和评估输出路径，再筛选；后续训练应从同一 SFT 初始化新实验，
不要直接续训另一奖励版本的检查点。筛选本身不启动训练。

#### 静态消融：是否启用 PSNR 对间差值下界

直接修改唯一的静态配置 `config/dpo_sd3_reward_static.yaml` 中的布尔项：

```yaml
preference_filter:
  enable_psnr_pair_constraint: true
```

- `true`（默认）：应用 `chosen_PSNR - rejected_PSNR >= min_psnr_gap`，天气单独设置优先；当前通常为 `-0.15 dB`。
- `false`：只跳过上述下界，包括 `weather_specific_thresholds` 对该下界的覆盖。
  PSNR 仍参与 F，组中位数 PSNR 下限、DISTS 约束，以及可选的 `max_psnr_gap` 都继续生效。
- 此项只控制 `reward.mode: static`。动态及旧模式忽略该开关，保留原 PSNR 对间检查。
  必须填写 YAML 布尔值 `true`/`false`，不能写字符串 `"false"` 或数字。

进行消融时，不新建或叠加 YAML；在同一静态配置中修改以下字段，保留其他参数：

```yaml
preference_filter:
  enable_psnr_pair_constraint: false
  output_dir: "/root/autodl-tmp/sd3/experiment/dpo_preferences_static_m32_no_psnr_pair"
training:
  preference_manifest: "/root/autodl-tmp/sd3/experiment/dpo_preferences_static_m32_no_psnr_pair/preference_pairs.jsonl"
  output_dir: "/root/autodl-tmp/sd3/experiment/SD3_ControlNet_RA_DPO_Static_M32_NoPSNRPair"
  resume_from_checkpoint: null
evaluation:
  output_dir: "/root/autodl-tmp/sd3/experiment/eval_sd3_dpo_static_m32_no_psnr_pair"
```

这样可以保留原启用约束的对照数据。候选评分和每组 12 对的设置仍共用原值。
命令始终使用同一个静态文件名：

```bash
python -m scripts.filter_dpo_pairs \
  --config config/dpo_sd3_reward_static.yaml

CUDA_VISIBLE_DEVICES=0 python -m scripts.train_dpo_sd3 \
  --config config/dpo_sd3_reward_static.yaml
```

无需重新生成候选或重跑 IQA，但切换开关后必须重新筛选偏好，不能直接复用另一开关状态的数据或检查点。
默认启用/未配置时保持已有偏好契约兼容；关闭时会写入不同契约，并在训练及续训前检查。
筛选终端会显示 `PSNR minimum=disabled`，偏好汇总和健康检查结果会报告
`psnr_pair_constraint_enabled: false`。对照和消融应使用同版本代码及相同的其他参数，
并记录最终保留组数、偏好对数及各天气分布，因为关闭门槛后合格候选对集合可能改变。
每次训练输出中的 `dpo_config.yaml` 会记录当次配置；将开关改回 `true` 时，也需同步恢复或指定对应对照实验路径。

新模式不使用旧版的覆盖优先选择、候选出现次数上限、`max_samples_per_pair` 或
`min_reward_gap`。不要通过这些旧参数调整当前每组对数，使用 `pairs_per_group`。
构建器按当前配置重算 Q/F，不直接信任旧奖励列；历史 `reward` 可另存为 `original_reward`。
静态对保存 R_static，动态对保存 R_dynamic，训练损失本身保持同一 DPO 形式。

更改模式、FR 权重或门槛后，应重建偏好对，无需重新运行 IQA。
偏好汇总绑定奖励、筛选参数、候选 CSV、归一化和清单的签名/哈希，训练及续训会验证契约。
旧版未指定 `reward.mode` 的数据仍受支持，但不能只改标签冒充当前静态/动态偏好。
改变候选总体或归一化时，使用一套新的对应输出，避免复用旧共用 CSV。

## 5. 单卡训练

静态模式：

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_dpo_sd3 \
  --config config/dpo_sd3_reward_static.yaml
```

动态模式：

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_dpo_sd3 \
  --config config/dpo_sd3_reward_dynamic.yaml
```

每次运行一个实验。训练脚本内部使用 Accelerate，直接通过 Python 启动即为普通单卡流程，
混合精度使用 YAML 中的 `bf16`。当前有效批大小为 `8*1*1=8`
（批大小 × 梯度累积 × 进程数）。若需有效批大小 24，可把梯度累积设为 3；
两种模式的对照应使用一致的训练参数，命令本身不会自动修改这些设置。

### 损失与可训练模块

参考策略为训练开始时从 `model` 加载的 ControlNet/RA 不可变快照，共享冻结的 SD3 主干。
参考参数保留与策略参数一致的精度，不再单独把 ControlNet 参考权重压成 BF16；
两次前向的计算精度统一由自动混合精度控制。旧实现中的不对称量化会在更新前产生预测差异，
大 beta 可将其放大成明显的 DPO 间隔误差。保留 FP32 参考快照比 BF16 每参数多约 2 字节显存。
每对偏好候选使用相同时间步和噪声；流匹配误差按潜变量元素取平均：

```text
policy_logratio = MSE_policy(rejected) - MSE_policy(chosen)
reference_logratio = MSE_ref(rejected) - MSE_ref(chosen)
dpo_loss = -log sigmoid(beta * (policy_logratio - reference_logratio))
```

当前实现没有额外的 0.5 系数。beta 必须结合该误差定义理解，不沿用旧文档中的过期推荐范围。
`train_controlnet` 和 `train_ra_fusion` 可分别控制训练模块，不能同时为 `false`。
当前 `sft_weight=0`、`gt_flow_weight=0.1`、`gt_x0_l1_weight=0.01`、
`weight_by_psnr_gap=false`。两个 GT 辅助分支只更新当前策略，保持独立反向传播，
不因奖励模式改变而改写目标。

### 指标持续退化时的检查

静态奖励是加权和，不保证每个指标都改善。当前总奖励中 MUSIQ 标准化分数的系数为 0.275，
CLIP-IQA 为 0.175，NIMA 为 0.05，PSNR/SSIM/负 LPIPS 各为 1/6。
MUSIQ 增益可以抵消其他指标下降；离线 PSNR/DISTS 门槛只约束已选候选，不直接约束训练后的模型输出。
DPO 也是相对误差目标：偏好候选误差和非偏好候选误差都上升时，相对间隔仍可能改善。
不能仅凭 DPO loss 降低或 MUSIQ 上升判断恢复质量变好，也不能直接用两个损失的数值大小比较梯度强度。

为区分训练退化与验证/加载问题，可在新实验的配置中启用零步 SFT 基线，并缩短诊断间隔：

```yaml
training:
  output_dir: "/root/autodl-tmp/sd3/experiment/SD3_ControlNet_RA_DPO_Static_M32_Diagnostic"
  resume_from_checkpoint: null
  validation_before_training: true
  checkpointing_steps: 200
```

此开关默认关闭；启用后，仅新训练会在第一次优化器更新前，用同一验证流程运行 step=0 原始权重，
结果写入 `initial_validation/` 和验证历史，续训不会重复执行。
零步结果已经很差时，先检查 SFT 权重、LQ/GT 配对、预处理、文本条件及推理设置；
零步正常而后续退化时，再检查实际学习率、beta、GT 辅助约束及偏好取向。
原始权重与 EMA 也应在相同图片和推理设置下对照，EMA 不是保真约束。
训练输出根目录的 `dpo_config.yaml` 保存本次生效配置，比事后修改的源 YAML 更适合排查。

### 检查点、EMA 与断点续训

每 1000 步保存检查点并执行验证，默认只保留最新 2 个 `checkpoint-*`。
检查点包含当前模型、EMA、优化器、调度器、随机数状态及数据游标，供完整续训使用。
训练正常结束后，根目录才写入最终 `controlnet/`、`ra_fusion/` 导出；
启用 EMA 时，最终组件使用 EMA 权重，不是每个检查点都更新根目录导出。

新实验使用 `resume_from_checkpoint: null`。同一实验断点续训时，在实际启动的配置中设置：

```yaml
training:
  resume_from_checkpoint: latest
```

然后使用原单卡训练命令。也可将该值设为具体检查点路径。
恢复时先从原模型建立参考快照，再把检查点恢复到当前策略，不会用续训权重替换参考策略。
允许提高 `max_train_steps` 延长训练；奖励/偏好契约、模型初始化、beta、学习率、调度器、
批大小、梯度累积、进程数、种子、数据及辅助损失等须满足检查点兼容性要求。
三卡检查点不能直接改成单卡后忽略进程数检查继续训练。
旧版仅有模型权重而没有续训状态的检查点不能用于完整断点续训。
当前续训元数据版本为 5，训练 ControlNet 时额外绑定 `reference_controlnet_precision`。
缺少新精度契约的旧 ControlNet 训练检查点不会被直接续训，以免静默改变参考策略；
使用修正后的代码应从原 SFT 权重、新输出目录开始。旧检查点仍可用于独立推理评估。

## 6. 验证、日志与 TensorBoard

当前每种天气验证 10 张，使用固定种子 42、30 步推理、CFG 1.0。
验证路径由 `training.validation_eval_config` 指向的配置决定。
每次验证记录 PSNR、SSIM、LPIPS、MUSIQ、CLIP-IQA、NIMA；三项无参评分默认使用 CPU。
评分在当前验证输出上计算，不复制离线标签，也不重新拟合小验证集的归一化参数。
不可用指标保留为空并记录有效数量，不伪造零值；验证不改变后续训练随机序列。

```text
<training.output_dir>/
  training.log
  training_metrics.jsonl
  validation_metrics.jsonl
  tensorboard/sd3_ra_dpo/events.out.tfevents.*
  checkpoint-N/
    validation_metrics.json
    validation_per_image.csv
    validation_weather_metrics.csv
```

训练指标按 `logging_steps=10` 的窗口汇总；验证历史记录步数、原始/EMA 权重、时间戳、
种子、整体/各天气均值和有效数量。根目录日志与 TensorBoard 事件不会随旧检查点删除，
续训时追加写入。`training.log` 是指标日志，不捕获所有依赖库的标准输出。
主要标签包括 `loss`、`implicit_accuracy`、`lr/controlnet` 和 `validation/overall_<metric>`。
默认在首次检查点验证结束后出现验证曲线；启用 `validation_before_training` 时，还会包含 step=0 基线。

在服务器 Jupyter 中查看静态训练：

```python
%load_ext tensorboard
%tensorboard --logdir /root/autodl-tmp/sd3/experiment/SD3_ControlNet_RA_DPO_Static_M32_Partial/tensorboard
```

动态实验使用相应动态训练目录。缺少 TensorBoard 时运行 `python -m pip install tensorboard`。
运行中的 Python 进程不会自动加载新代码；要启用新日志，应在保持原训练参数的前提下从有效检查点续训。

## 7. 训练后评估

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate_dpo \
  --config config/dpo_sd3_reward_static.yaml

CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate_dpo \
  --config config/dpo_sd3_reward_dynamic.yaml
```

评估优先使用训练根目录的最终组件；尚未最终导出时，回退到最近的完整检查点，
启用 EMA 时使用其 `ema/` 组件。未训练的组件继续使用基础模型路径。
可通过 `--controlnet_model_path`、`--ra_fusion_path` 显式指定权重。
结果写入对应 `evaluation.output_dir` 下的时间戳目录，包括整体指标及逐图 CSV。
对比 SFT 与 DPO 时，保持样本、种子、推理步数、强度、CFG 和缩放系数一致。
离线偏好应来自训练集或独立验证集，最终测试集用于独立评估。

## 8. 可选：候选差异与方差分析

只分析已保存 CSV，不需要 GPU，不检查图片完整性，也不重新评分：

```bash
python -m scripts.analyze_candidate_csv \
  --input-csv /root/autodl-tmp/sd3/experiment/dpo_candidates_m32_complete_groups/per_candidate_metrics_aesthetic.csv \
  --output-dir /root/autodl-tmp/sd3/experiment/dpo_candidates_m32_complete_groups/nr_fr_analysis \
  --expected-candidates 32 \
  --top-k 20
```

尚未计算美学指标时，输入改为 `per_candidate_metrics.csv`，使用另一个新分析目录，
例如 `fr_analysis`。分析目录必须不存在或为空。
输出 `report.md`、`summary.json`、`group_metrics.csv` 和 `high_variability_groups.csv`。

- **组内差异**：同一源图不同候选的样本方差（N−1）、标准差、极差及分位数。
- **组间差异**：各源图平均分之间的方差和差距，另按天气与子数据集汇总。
- 统计已有 PSNR、SSIM、LPIPS、DISTS、MUSIQ、CLIP-IQA、NIMA 等字段，不凭空推算缺失指标。
- 缺失/重复编号、重复种子及无效数值会单独统计；缺失指标不填零。
- 最佳相对均值的增益是单指标观察到的择优空间，不保证不同指标的最佳值属于同一候选。

不要把不同场景的组间方差解释为随机种子多样性，也不要仅凭极差判断能保留多少偏好对。

### M12 扩至 M32 后的数据健康检查

增加候选数和每组配对上限本身不是错误。M12 有 66 个无序组合，M32 有 496 个；
每组上限从 8 调至 12 只增加 50%，并不会把全部组合都用于训练。
当前静态/动态模式使用 `pairs_per_group`，不是旧的 `max_samples_per_pair`。

只检查候选记录，不读取图片：

```bash
python -m scripts.check_candidate_health \
  --candidate-csv /root/autodl-tmp/sd3/experiment/dpo_candidates_m32_complete_groups/per_candidate_metrics.csv \
  --output-dir /root/autodl-tmp/sd3/experiment/dpo_candidates_m32_complete_groups/health_csv \
  --expected-candidates 32 \
  --expected-cfg 1.0 \
  --base-seed 20240805
```

`--base-seed` 必须与实际生成时的基础种子一致；不知道原种子时省略该选项。
脚本检查编号、种子、CFG、重复路径/已记录图片哈希、源图身份、指标有限性和源路径划分。
若有归一化附加文件，还会核对原始指标与标准化分数是否一致。
程序按行读取，并使用临时磁盘索引，不加载模型或把全部候选行放入内存。

已有偏好对时，使用实际筛选所用的 FR/NR CSV，核对偏好方向、保存的奖励、配置契约及保真门槛：

```bash
python -m scripts.check_candidate_health \
  --candidate-csv /root/autodl-tmp/sd3/experiment/dpo_candidates_m32_complete_groups/per_candidate_metrics_fr_nr.csv \
  --pairs /root/autodl-tmp/sd3/experiment/dpo_preferences_static_m32_partial/preference_pairs.jsonl \
  --config config/dpo_sd3_reward_static.yaml \
  --output-dir /root/autodl-tmp/sd3/experiment/dpo_candidates_m32_complete_groups/health_static_pairs \
  --expected-candidates 32 \
  --check-files \
  --sample-groups 12
```

`--check-files` 仅查询文件存在且非空；`--sample-groups 12` 只解码抽中的 12 组，
包含高 PSNR 极差组和固定种子的随机组，不全量解码约 19 万张图片。
抽样检查 RGB、组内尺寸、实际像素哈希；有同目录缓存 `gt.png` 时，还重算 PSNR，
与记录相差超过 0.15 dB 的样本列为复查项。实际完成的抽样和 PSNR 检查数量会写入报告。

报告输出 `summary.json`、`report.md`、`groups.csv` 和 `issues.csv`；输出目录必须是新目录或空目录。
重点查看：

- `issue_counts`：缺编号、重复种子、路径/GT 错配、标准化不一致、偏好反向或保真门槛违规等。
- `pairs.configured_pairs_per_group`：实际生效的每组对数。当前配置为 12；若预期 12 却报告 9，核对实际启动的配置和旧偏好契约。
- `pairs.metric_deltas`：偏好候选相对非偏好候选的指标差及变差比例。
- `pairs.musiq_up_but_clipiqa_and_nima_down`：MUSIQ 上升而另外两项无参指标下降的偏好对数量。
- `pairs.max_candidate_appearances`：候选是否在偏好清单中反复出现。
- 对新动态偏好，检查器还会从全组重建 B，核对 λ、缺口、惩罚和 R_dynamic，不信任手工填写的基准。
- `prefix_vs_added`：同组内前 12 张与后 20 张的平均指标差异，逐组配对后汇总。
  PSNR 等高分为优的指标若新增部分系统性更低，或 LPIPS 系统性更高，应优先检查新增候选。
  `groups.csv` 还记录前 12 张与全组的 PSNR 中位数，便于判断组中位数保真基线是否被拉低。

若还保留旧 M12 CSV，可在命令中增加 `--baseline-csv /path/to/m12/per_candidate_metrics.csv`。
程序按天气、子数据集和原 LQ 路径匹配旧候选，比较其种子、CFG、像素哈希及已有原始指标，
不按可能重排的全局编号强行匹配，也不比较可能合法重拟合的标准化分数。
默认前 12 张仅是候选位置分组；只有与旧 CSV 的身份、种子和哈希核对后，才能认为它们保留了旧 M12。
真正“仅追加”时，旧候选应保持不变；重新生成、改变 CFG/评分模型会产生差异，不能直接认定为损坏。
没有匹配记录时，该比较不能给出结论。

错误项会使退出码为 2；高极差、旧候选变化等警告返回 0，供人工判断。
一致性检查通过不等于语义配对或偏好质量已得到保证。尤其是高极差组，应实际查看 LQ、GT 和最差候选。

## 9. 按需：重新生成候选、恢复与严格审计

### 从头生成新的 M32 候选集

已有候选时不需要重复本节。两份独立配置的候选设置均为每图 32 个独立种子、全部 CFG=1.0，
基础种子 20240805、每种天气最多选 2000 张、30 步推理。

源图优先读取 `selection_manifest`；设为 `null` 才扫描天气目录。
当前指向 `dpo_source_selection_2000.json`，按 `sample_mode: head` 使用预选清单。
新任务先确认清单存在、每种天气数量限制与清单一致；不在工作进程中二次抽样。

```bash
python -m scripts.generate_dpo_candidates \
  --config config/dpo_sd3_reward_static.yaml \
  --gpu_ids 0 1 2 \
  --output_dir /root/autodl-tmp/sd3/experiment/dpo_candidates_m32_new_3gpu \
  --dry_run

python -m scripts.generate_dpo_candidates \
  --config config/dpo_sd3_reward_static.yaml \
  --gpu_ids 0 1 2 \
  --output_dir /root/autodl-tmp/sd3/experiment/dpo_candidates_m32_new_3gpu
```

这是按源图分片的独立进程推理，不是 DDP；每张 GPU 需容纳一套完整推理模型。
第 i 个工作进程负责 `global_index % K == i` 的源图，同一源图的全部候选留在同一卡。
种子由 `base_seed:candidate_index:global_index` 派生，工作进程不重新编号。
单卡生成时使用 `--gpu_ids 0`；不要用 `accelerate launch --multi_gpu` 启动候选生成。
新生成目录与本批数据目录不同，后续评分/筛选配置需同步指向实际生成结果。

### 原任务恢复或只合并

要补齐中断任务，用原配置、原分片数量和原输出目录运行生成命令，并加 `--resume`。
有效缓存会复用，只有缺失/无效组补生成；不想补生成时，使用第 2 节的简版导出。
`--monitor` 可只读查看已有任务进度。进度 100% 后仍可能在校验和合并，
正常全量合并需等待根目录 `COMPLETE.json`。

若全部分片已完成，只是根目录尚未合并：

```bash
python -m scripts.generate_dpo_candidates \
  --config config/dpo_sd3_reward_static.yaml \
  --output_dir /root/autodl-tmp/sd3/experiment/dpo_candidates_m32_new_3gpu \
  --merge_only
```

此命令不推理，但严格要求全部分片完成。生成器的同级锁文件用于避免重复写入；
异常终止后，确认记录的进程已经停止再处理遗留锁。
原冻结计划绑定生成实现的哈希，恢复旧任务时不能随意改写其依赖实现。

### 正常合并数据的严格审计

只适用于已有完整合并元数据的目录，不适用于第 2 节的简版导出：

```bash
python -m scripts.summarize_dpo_candidates \
  --candidate-dir /root/autodl-tmp/sd3/experiment/dpo_candidates_m32_new_3gpu \
  --expected-candidates 32 \
  --check-files
```

报告默认写入 `generation_analysis/`，严重问题返回退出码 2，警告或正常结果返回 0。
`--check-files` 检查文件状态；`--verify-checksums` 会执行更耗时的像素校验。
正常合并可能使用硬链接，不要直接修改合并图片；简版导出则使用独立复制。
简版数据的完成标记是 `EXPORT_COMPLETE.json`，不能冒充严格审计要求的 `COMPLETE.json`。

## 10. 独立实验与验证入口

以下是按需运行的独立实验，不是主流程的额外必需步骤：

- [尾部风险加权](TAIL_RISK_DPO.md)：在既有偏好上做加权对照。当前主配置关闭此功能；新动态虽有组内统一标量，但现有尾部工具没有提供冻结组基准上下文，仍不支持该路径。
- [同图在线/离线 IQA 诊断](OFFLINE_IQA_ANALYSIS.md)：对比 GT、SFT、DPO，并检查候选中的保真合格择优空间。
- [M=8/M=12 随机性实验](SEED_RANDOMNESS.md)：独立的嵌套样本量和输出多样性分析，不改变主候选集的 M=32 设置。

相关回归测试：

```bash
python -m unittest discover -s tests -p 'test_export_candidates_simple.py'
python -m unittest discover -s tests -p 'test_analyze_candidate_csv.py'
python -m unittest discover -s tests -p 'test_candidate_health.py'
python -m unittest discover -s tests -p 'test_reward_modes.py'
python -m unittest discover -s tests -p 'test_dynamic_fidelity_reward.py'
python -m unittest discover -s tests -p 'test_reward_modes_smoke.py'
python -m unittest discover -s tests -p 'test_preference_provenance.py'
```

测试使用小型夹具、模拟评分或模拟推理，不代替服务器上的真实 IQA、显存和训练验证。
