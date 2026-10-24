# 尾部风险加权 DPO

本文是独立的加权对照实验。候选评分、静态/动态筛选、单卡训练、日志和续训的统一说明见
[DPO 主流程](DPO_SD3.md)。当前主流程默认关闭尾部加权，无需执行本文命令。

奖励模式兼容性：静态模式使用统一标量奖励，受本算法支持。
当前动态偏好使用 `Q + eta_fidelity*F - lambda_fidelity*max(0,B-F)^2` 并有双优优先规则，
其统一分数依赖历史同源候选组的冻结 B。
本文现有逐行标量导出和尾部工具未提供这一上下文，仍明确拒绝动态模式；这不表示新动态奖励不存在标量。
动态模式保持 `tail_risk.enabled: false`，可查看偏好构建器输出的 `dynamic_candidate_rewards.csv`。
共用评分与偏好对构建说明见[DPO 主流程的奖励规则](DPO_SD3.md#奖励与偏好规则)。

这是实验性的偏好对加权方案，不保证降低随机种子方差。
尾部指各天气完整训练候选总体中的低奖励输出，不表示某个输入的异常种子或生成失败。
它无法添加已有偏好对中不存在的尾部样本。

## 作用范围

原偏好对、偏好/非偏好候选路径、顺序、天气比例、奖励公式、归一化、SFT 初始化、
可训练模块、beta、学习率、训练步数及辅助损失系数均保持不变。
只有 DPO 项应用偏好对权重，不重新生成候选或计算 IQA。

对天气 w，使用完整候选台账中按唯一身份去重的候选：

```text
T_w = Q20(candidate reward)
IQR_w = Q75(candidate reward) - Q25(candidate reward)
tail_deficit = clip(max(0, T_w - rejected_reward) / (IQR_w + eps), 0, 2)
raw_pair_weight = 1 + lambda_tail * tail_deficit
pair_weight = raw_pair_weight / mean(raw_pair_weight for all final pairs of w)
```

在每种实际存在的天气内，离线对完整偏好对集合计算归一化，使 `pair_weight` 的平均值为 1。
普通偏好对的权重通常小于 1；尾部偏好对相对普通偏好对权重更大，
但偏离阈值较浅的尾部对在归一化后仍可能小于 1。默认原始权重范围为 [1,3]，
归一化后的权重不再具有相同范围。

## 配置

现有 `config/dpo_sd3.yaml` 包含默认关闭的设置：

```yaml
tail_risk:
  enabled: false
  candidate_reward_file: null
  tail_quantile: 0.20
  lambda_tail: 1.0
  tail_deficit_max: 2.0
  eps: 1.0e-8
```

两份可直接运行的对照配置通过 `base_config` 继承已有设置，路径相对于引用它的 YAML 解析。
各顶层配置段按键覆盖，不进行通用的递归合并：

- `config/dpo_sd3_tail_risk_standard.yaml`：强制将有效权重设为 1。
- `config/dpo_sd3_tail_risk.yaml`：启用离线偏好对权重。

这两份独立配置继承基础 `dpo_sd3.yaml` 的旧版无参奖励与路径，不是 M32 静态/动态主配置。
若要对当前静态奖励执行尾部实验，需显式配置一致的奖励契约、候选评分、偏好文件与新输出目录，
不能把旧版加权文件直接当作当前静态偏好使用。

两者读取完全相同的加权 JSONL，使用相同随机种子、采样器、预处理、初始化和训练参数，
仅有效偏好对权重及结果目录不同。运行前，基础 `model` 路径与缩放系数必须和原偏好汇总中的
`candidate_policy` 一致。这些示例不会改变所选检查点。
请保持 `require_candidate_provenance: true`。

## 离线命令

已有美学 CSV 不包含最终标量奖励。首先使用已保存的标准化分数和不变的 `build_reward` 公式，
将现有奖励写入新文件。这是显式导出，不运行 IQA 推理、不重新归一化，也不定义新奖励。
偏好与非偏好候选分数都会与原偏好对中的分数核对；所有导出的候选奖励还绑定完整的原始指标，
包括未出现在偏好对中的候选。不匹配时命令会停止。
除非原完整候选 CSV 已包含显式奖励，否则权重生成器要求提供通过验证的导出元数据。

```bash
python -m scripts.export_dpo_candidate_rewards \
  --config config/dpo_sd3_tail_risk.yaml

python -m scripts.weight_dpo_tail_risk \
  --config config/dpo_sd3_tail_risk.yaml
```

默认导出到 `tail_risk.candidate_reward_file`。权重生成从 `preference_filter.output_dir`
读取原始未加权 JSONL，不读取新的训练清单。
也可按需通过 `--input_csv`、`--output_csv`、`--input_pairs` 和 `--output_dir` 指定输入输出路径。

脚本拒绝覆盖已有输出。运行另一组实验时，请选择新路径，并同步更新两份对照配置。
程序要求 `splits: [train]` 和训练用 `selection_manifest`，检查候选的 LQ/GT/天气归属，
并将完整候选身份集合与原指标 CSV 和偏好汇总核对。不允许回退为只使用非偏好候选的分布。
身份和奖励均相同的重复候选只计一次；重复项冲突、奖励缺失、IQR 接近零或关键字段缺失都会报错。
将偏好对移至新目录时，需要使用当前筛选器生成的绝对图片路径。
相对路径会被拒绝，不会静默改写原偏好对字段。

新增输出：

```text
dpo_candidates/
  per_candidate_rewards.csv
  per_candidate_rewards.csv.metadata.json
dpo_preferences_tail_risk/
  preference_pairs.jsonl
  preference_summary.json
  tail_risk_statistics.json
```

每条新偏好对保留全部原字段，仅增加 `tail_threshold`、`tail_deficit`、`raw_pair_weight`、
`pair_weight` 和 `is_tail_pair`。统计包括阈值、IQR、去重候选/偏好对数量、尾部对比例、
原始权重均值、归一化权重分布、输入输出哈希及配置。复制的偏好汇总保留模型来源信息。

```bash
python -c "import json; p=json.load(open('/root/autodl-tmp/sd3/experiment/dpo_preferences_tail_risk/tail_risk_statistics.json')); print(json.dumps(p['per_weather'], indent=2))"
```

只有在服务器上实际执行这些命令后，才能获得对应服务器数据的统计结果。

## 匹配对照训练

使用新输出目录，不替换已有结果。进行此对照时，不要重新选源图、生成候选、计算 IQA 或筛选偏好对。

```bash
# 标准 DPO：使用同一偏好对文件，全部有效权重设为 1
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_dpo_sd3 \
  --config config/dpo_sd3_tail_risk_standard.yaml

# 尾部风险加权 DPO
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_dpo_sd3 \
  --config config/dpo_sd3_tail_risk.yaml
```

当前主流程的静态/动态配置保持尾部加权关闭，单卡启动命令见[DPO 主流程](DPO_SD3.md)。

旧偏好对默认权重为 1。关闭模式会完全忽略已存储权重。
启用模式但使用旧版偏好对时，会提示权重全部为 1，即没有重新加权。
启用加权训练时，加载模型前会验证离线配置、文件哈希、数量及各天气平均权重。
本实验要求显式关闭已有的 PSNR 差值加权，同时启用两种加权方式会报错。

## 损失、日志与断点续训

```python
per_pair_loss, stats = diffusion_dpo_loss(..., reduction="none")
weighted_dpo_loss = (pair_weight * per_pair_loss).mean()
loss = weighted_dpo_loss + sft_weight * chosen_mse.mean()
# GT 流匹配和 GT x0 目标保持原有的独立反向传播。
```

不除以权重总和，也不手动除以梯度累积步数。梯度累积仍由 Accelerate 处理。
权重不应用于 SFT、GT 流匹配、GT x0 或其他辅助损失。

`training_metrics.jsonl` 和配置的跟踪器会记录 `dpo_loss_unweighted`、`dpo_loss_weighted`、
`pair_weight_mean`、`pair_weight_min`、`pair_weight_max`、`tail_pair_fraction`，
以及已有的间隔/准确率和辅助损失统计。日志以 `training.logging_steps` 次优化器更新为一个窗口
（默认 10），按样本数汇总全部微批次和进程。保存检查点和最后一次更新时也会刷新窗口，
因此批大小为 1 时，通常不会只记录一对样本。
这些是观测窗口内的权重，不是在批次内重新归一化后的权重，每个窗口的均值不一定等于 1。
两组对照训练均保留尾部标签供日志使用；没有标签的旧偏好对报告尾部比例为 0。

断点续训时，在对应配置中设置 `training.resume_from_checkpoint: latest`。
`dpo_resume.json` 保存并校验实际生效的尾部配置、候选奖励和统计文件哈希，
以及已有的偏好文件哈希、采样器/优化器设置和随机数生成器（RNG）状态。
切换加权方式或修改离线输入后，不能静默续训同一实验。
旧检查点在关闭尾部加权且满足原兼容性检查时，仍支持断点续训。
