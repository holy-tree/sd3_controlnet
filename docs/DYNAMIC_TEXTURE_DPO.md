# Dynamic 专用：GT 纹理加权 DPO + 全图 GT MSE

本实验使用 `config/dpo_sd3_reward_dynamic_texture.yaml`，奖励模式为
`dynamic_texture`，数据 schema 为 `sd3_dynamic_texture_v1`。
原 static 和旧 quadratic dynamic 的奖励、选对公式以及默认训练路径继续保留。
**static 与本模式是完整方法对比，不是只修改奖励公式的消融。**

## 1. 实际训练目标

```text
L_total = L_DPO_texture + lambda_fidelity * L_GT

error_map = mean_channels((prediction - target)^2)
W_Q = 1 + texture_strength * resize_to_prediction_grid(M)
weighted_error = sum(W_Q * error_map) / sum(W_Q)

L_GT = mean((RGB_prediction_01 - RGB_GT_01)^2)
```

chosen/rejected、当前模型/冻结参考模型的四项 Flow Matching 误差都使用同一份
GT 权重图。后面的参考差比较、符号、beta、pair reduction 沿用现有 DPO。
当前 DPO 采样时间步分布以及原有单位 pair 权重不变，不额外引入时间步加权。
`texture_strength=0` 使用与原 `per_sample_mse` 完全相同的浮点归约顺序。
按 `sum(W_Q)` 归一化，不因 mask 面积变化而任意放大 loss。

`lambda_fidelity` 为固定正数，没有动态 lambda。GT MSE 对所有像素/通道等权，
不使用 mask、`1+(1-M)`、mask 外区域或额外平坦区加权。
这是两个目标的联合训练；共享参数会相互影响，不能称为严格空间解耦。
空间加权是区域加权的 DPO **代理目标**，不声称与原始概率推导完全等价，
也不保证只改变 mask 内的最终输出像素。

## 2. GT 辅助分支：复用已有前向，替换其监督

现有 trainer 已有独立 GT Flow Matching 前向，旧监督为 velocity MSE + latent x0 L1。
新模式复用这一段 ControlNet + RA 前向，关闭旧 `gt_flow_weight`、
`gt_x0_l1_weight` 及 `sft_weight`，避免重复叠加辅助目标。

```text
clean_GT_latent = (VAE.encode(GT).mode - shift_factor) * scaling_factor
noisy_GT = (1 - sigma) * clean_GT_latent + sigma * noise
FM_target = noise - clean_GT_latent
estimated_clean = noisy_GT - sigma * model_prediction
RGB_prediction = VAE.decode(estimated_clean / scaling_factor + shift_factor)
```

sigma/timestep 成对取自原 SD3 scheduler 的真实训练表，不构造另一套 scheduler。
GT 默认只采样 `sigma ∈ [0.05, 0.70]`，排除极高噪声以及接近零的步。
x0 估计不除以 `1-sigma`，避免高噪声时的奇异放大。
VAE 冻结且要求 `upcast_vae: true`，但 **decode 不能放进 no_grad**；
梯度通过解码输入传回当前 ControlNet 与 RA。RGB 从原 `[-1,1]` 线性变到 `[0,1]`
后求 MSE；不 clamp 训练预测，以免饱和区梯度被截断。非有限结果直接报错。

GT 使用独立的、包含所有源图的 LQ/GT 数据流，不拿 chosen/rejected latent
冒充 GT 重建。没有可靠纹理图块、没有合格偏好对的源图依然参与 GT 监督。
每个 epoch 中 GT 源索引独立循环，默认每步计算 GT 分支，覆盖全部源图。
全部源图都无偏好时，仅使用零 DPO 权重的张量占位，不导出伪偏好，仍可训练 GT 目标。

`gt.interval: 1` 时每步都是上述联合目标；设为 K>1 则仅在每 K 个 optimizer step
计算 GT 项，跳过步只优化 DPO，**不会自动调整或放大 lambda**。
因此改变间隔也改变监督频率，主实验建议保持 1。

一步重建 MSE 只是最终完整多步推理质量的训练代理；不运行每 batch 30 步采样，
不能保证一步 MSE 降低就会改善最终 PSNR/视觉质量。

## 3. GT mask 与图块

```text
GT RGB [0,1]
→ 灰度
→ 归一化 Sobel x/y 梯度幅值
→ window×window 局部平均
→ 每张 GT 的稳健分位数归一化
→ smoothstep 软阈值
→ 局部平均平滑，得到 M ∈ [0,1]
```

全过程 float32、no_grad。Sobel 核除以 8，统计窗口用平均而非累加。
用 `gradient_floor` 抑制平坦图的数值噪声；平坦图返回零图。
当稳健区间塌缩但仍有明显正梯度能量时，使用零下锚点，避免将均匀重复纹理误当平坦图。
M 只表示局部边缘/纹理变化丰富程度，不能识别“可以牺牲保真的区域”。

评分与训练共用候选生成时的 Resize(shorter side) + CenterCrop + RGB 预处理。
优先使用候选组目录中保存的 `gt.png`/`lq.png`，并记录内容哈希。
偏好组绑定一份 GT、mask 哈希及图块坐标，所有候选共用该坐标。
默认不随机裁剪/翻转；可选水平翻转会同时翻转 chosen/rejected/GT/LQ/M。
mask 通过 area resize 对齐预测 latent 空间网格，不跨通道生成不同 mask。

从 GT mask 按图块均值排序、y/x 稳定打破平分，并根据覆盖门槛和 IoU 抑制重叠。
选取完整正方形图块，**不把非纹理位置涂黑**。没有可靠图块时不构造局部偏好，
但仍写入 `gt_sources.jsonl`。

## 4. 局部 IQA 与选对

模型名按配置明确指定，不静默换用其他变体。当前默认：

| 指标 | pyiqa 名称 | 输入处理 |
|---|---|---|
| MUSIQ | `musiq-spaq` | 完整 RGB 图块；backend 处理 multiscale 224/384 及归一化 |
| CLIP-IQA | `clipiqa+` | 整个图块缩放为 224；backend 执行 CLIP 归一化 |
| NIMA | `nima` | 检查 backbone 的输入尺寸为 224 或 299，整块缩放后交给 backend |

真实 IQA 的图块大小要求至少 224。NIMA 收到已适配的正方形图块，
backend 的 center crop 不再额外丢掉图块边缘。模型版本、输入预处理策略写入统计文件。
实现按 pyiqa 的 MUSIQ/CLIPIQA/NIMA 架构输入约定检查，不在 wrapper 外重复做其归一化。
模型后续升级时，以运行时可用的变体/输入尺寸检查及已记录版本为准。

同一候选的多个图块先分别评分，再对每个指标求图块均值。
只对可靠局部评分总体拟合天气内 median/IQR → std → 1 的冻结统计，截断 z 到 ±3：

```text
Q_texture = 0.55*MUSIQ_texture_z + 0.35*CLIP-IQA_texture_z + 0.10*NIMA_texture_z
```

该统计仅包含三项局部指标，不读取/复用整图归一化参数。
局部 Q 高的一方为 chosen，差值必须达到独立的 `reward.min_texture_q_gap`。
不使用旧 quadratic 奖励、旧 `margin_dynamic` 或旧 Q/F 双优分支。
随后应用原始 PSNR 对间下界/可选上界、DISTS 退化限制以及组中位数底线，
最终固定种子、无放回均匀随机选最多 12 对，不按奖励差取前 12。

## 5. 配置与运行

完整独立配置：`config/dpo_sd3_reward_dynamic_texture.yaml`。
以下值仅是实验起点，不代表已验证最优：

```yaml
reward:
  mode: dynamic_texture
  min_texture_q_gap: 0.05
spatial_dynamic:
  version: 1
  lambda_fidelity: 0.1
  texture_strength: 2.0
  mask:
    window: 7
    smooth_window: 3
    lower_quantile: 0.10
    upper_quantile: 0.99
    threshold: 0.35
    transition: 0.15
    gradient_floor: 0.0001
  patches:
    size: 224
    stride: 112
    max_patches: 4
    min_mean: 0.08
    min_coverage: 0.05
    max_iou: 0.5
  gt:
    interval: 1
    sigma_min: 0.05
    sigma_max: 0.70
```

在项目根目录执行；下列命令不会自动串联启动正式训练：

```bash
# 1. 只读取已有候选图片，新增局部 IQA 评分；可以显式改为 --device cuda:0。
python -m scripts.prepare_texture_dpo score --config config/dpo_sd3_reward_dynamic_texture.yaml --device cpu

# 原评分进程已退出，且上次仅留下不完整输出时：归档失败目录，从头重试。
# 显式覆盖无效的 OpenMP 环境变量；此例将 IQA 放在 GPU 上。
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m scripts.prepare_texture_dpo score --config config/dpo_sd3_reward_dynamic_texture.yaml --device cuda:0 --restart-incomplete

# 2. 在新的偏好目录中筛选 Q_texture 偏好。
python -m scripts.prepare_texture_dpo filter --config config/dpo_sd3_reward_dynamic_texture.yaml
# 等价入口：python -m scripts.filter_dpo_pairs --config config/dpo_sd3_reward_dynamic_texture.yaml

# 3. 用户确认配置和准备结果后，才手动启动训练。
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_dpo_sd3 --config config/dpo_sd3_reward_dynamic_texture.yaml

# CPU 数值、数据对齐、版本和实际 trainer 微批次路径梯度检查，无需模型下载。
python -m unittest discover -s tests -p 'test_spatial_dynamic.py' -v

# 保存合成 GT/candidate/M/图块坐标可视化；输出目录须为新目录。
python -m scripts.check_texture_dpo --output experiment/texture_dpo_check_v1
```

评分/偏好默认要求新目录，现有输入和旧结果不会被覆盖。
评分最后发布 `texture_scores.json` 作为完成标记；失败留下的目录不能被误当成完整评分。
`score --restart-incomplete` 仅在确认旧进程已退出后使用：将可识别的失败目录重命名为
`原目录.incomplete-UTC时间戳`，保留所有失败产物，然后从头评分；不是续算已评分部分。
带完成标记的目录和含无关文件的目录不会被该选项覆盖/归档。首次发布版本产生的
无 attempt marker、但含合法 texture CSV 表头的失败目录也支持此方式重试。
输入 CSV 要求同源图记录连续（当前完整组导出满足），否则明确报错。
评分启动时流式统计候选总数，随后显示按候选图片更新的 `Texture IQA` 进度条，
包括百分比、处理速度、预计剩余时间，以及已完成组数、当前天气和图块数。
局部归一化写出阶段另显示 `Normalize local IQA` 进度条；计数不将整张 CSV 载入内存。

### 候选校验和含义

生成器的 `output_checksum_sha256` 是量化 RGB **像素**的 SHA256：原始生成器使用
CHW 通道顺序，另有历史写入器使用 HWC 顺序。局部评分与现有健康检查一样兼容这两种
像素布局，并保留输入字段原值，不能拿它与 PNG 文件字节哈希比较。

新增 `candidate_file_sha256` 专门保存完整 PNG 文件字节 SHA256；偏好记录中的
`chosen_checksum`/`rejected_checksum` 使用此文件哈希，供训练时检查内容变化。
已完成的早期 texture 产物若尚无新列，筛选器只兼容其旧字段中确实匹配文件字节的
旧约定；不会重写其文件或元数据。真正的像素不匹配仍然报错，不关闭校验。

主要新增产物：

- `per_candidate_texture_raw.csv`、`per_candidate_texture.csv`。
- `texture_normalization.json`：独立局部统计、backend 和预处理记录。
- `gt_sources.jsonl`：所有源图，包括无纹理/无偏好者；GT/LQ/mask/图块绑定。
- `texture_scores.json`：准备合同、完整文件哈希、无可靠纹理的组数。
- 新偏好目录的 `preference_pairs.jsonl`、`preference_summary.json`。
- 评分前若干组的 `visualizations/group_*/{gt,candidate,texture_mask,patch_locations}.png`。

训练 checkpoint 使用 resume metadata v6，额外绑定固定 lambda、texture_strength、mask、
GT 调度等完整目标设置。偏好合同只绑定局部评分/筛选，故最小消融可以复用同一偏好文件。
旧 static/旧 dynamic checkpoint 与新模式相互拒绝混用，旧 checkpoint 仍按原模式恢复。

## 6. 最小消融与日志

复制完整配置到另一个文件，只修改：

1. `spatial_dynamic.texture_strength: 0.0`。
2. `training.output_dir` 及 `evaluation.output_dir` 为不同的新目录。

局部评分目录、同一偏好清单、独立 GT manifest、固定 lambda、GT 分支设置均不变。
不重算局部 IQA，不重新筛选偏好。这才是在新方法内部检验空间加权本身的最小消融。

训练日志单独记录 `loss_dpo_texture`、`loss_gt_rgb_mse`、`loss_gt_weighted`、
`gt_aux_active`、`texture_mask_mean`、`texture_mask_coverage`、`texture_pairs_in_batch`。
启动日志另记录总偏好数、独立 GT 源数及筛选拒绝原因；准备摘要保留 Q 差不足、
无纹理、PSNR/DISTS/组中位数拒绝及未抽中等数量。
GT interval>1 时，未计算步的 GT loss 记为 0，并由 `gt_aux_active` 区分。

## 7. 成本与验证边界

- 局部评分成本约为候选数 × 所选图块数 × 3 个 IQA 模型，只运行一次离线评分；
  不再运行 diffusion 生成候选。CPU 能运行但可能很慢，IQA 权重也需要额外内存。
- DPO 本体前向次数不变，只增加 mask 和空间归约。
- 读取训练图片时检查候选内容哈希，GT/LQ 在每 worker 的两项源缓存缺失时检查哈希和 mask；
  这增加少量读盘与 CPU 预处理成本。
- 原项目 GT 辅助前向继续使用；新模式增加独立 GT-LQ 的 VAE encode 和可微 VAE decode。
  RGB decode 反传可能显著增加显存/耗时，默认 microbatch=2、梯度累计=4，等效 batch=8；
  仍需按实际显存调整，不自动修改 lambda。
- 本地 CPU 测试使用合成图片、fake IQA 和小型 ControlNet/RA/VAE 桩，执行实际 trainer 的
  微批次代码，检查两次 backward 都能给两个可训练分支提供有限非零梯度，参考快照无梯度。
  这不等于已验证真实 SD3 的完整训练质量或实际 IQA 分数。
- 合成可视化中的 edge 可能没有达到完整 224 图块的覆盖门槛；其 GT 监督仍保留。
- 最终性能必须通过原有完整多步推理评估；推理接口和模型结构不变，不需要 GT/mask。
