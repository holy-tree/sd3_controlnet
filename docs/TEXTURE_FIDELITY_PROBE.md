# 下一步：固定 GT 系数的单变量短程探针

适用于已经得到真实 `dynamic_texture` 梯度诊断，且发现加权 GT 信号远小于 DPO、
部分累计窗口存在反向分量的情况。此阶段只检查固定权重能否有效影响联合方向，
不直接宣布某个 lambda 合理或能保证最终 PSNR。

入口：`scripts/plan_texture_fidelity_check.py`，仅依赖标准库和 PyYAML，**CPU 运行**。
不读取当前默认训练 YAML、不加载模型、不生成候选、不重算 IQA、不筛选偏好，
也不启动训练。以真实诊断 JSON 内嵌的历史训练快照和 checkpoint 有效设置为来源。

## 1. 从已有诊断做代数标定

```text
GT_dot_total(lambda) = dot(g_GT,g_DPO) + lambda * ||g_GT||²

lambda_SGD_boundary = max(0, -dot(g_GT,g_DPO) / ||g_GT||²)
```

边界仅描述**所测参数点、所测样本和一致步长 SGD 的当前 GT loss 一阶趋势**。
不包含 Adam 预条件、各组学习率、裁剪、曲率，也不能从 checkpoint-37000
直接外推到初始 SFT 或后续训练全过程。边界不是等范数目标、不是最优超参数。

给定示例报告中，雾/雪累计窗口边界约为 97/919，向上取一位有效数字得到 1000。
因此只把 1000 作为一个固定探针；不是把它直接设为正式长训默认值。
脚本以实际 CSV 的完整精度重新计算，而不是按 Markdown 四舍五入数字计算。

```bash
BASE=/root/autodl-tmp/sd3/experiment
python -m scripts.plan_texture_fidelity_check --diagnostic-dir "$BASE/diagnostic_texture_37000_full_v2" --candidate-lambda 1000 --output "$BASE/texture_fidelity_calibration_only_v1"
```

产物 `calibration_report.md` / `calibration.json` 包括：

- 每个累计窗口及 ControlNet/RA/全参数的原始梯度和候选系数的代数影响。
- 原/候选 GT-to-DPO 比例、候选 `g_GT·g_total`、SGD 边界。
- Q/F 共同缺失、GT 独有缺失、DPO 独有缺失的名字预览，以及完整缺失事件数。

先核对名字与实际 forward：共同缺失不自动代表正常；GT 独有缺失也可能有条件分支原因，
需要结合结构排查。没有预览列时会明确标记不可用，不把空预览当作没有缺失。
无梯度、零梯度或非有限累计数据不能据此生成权重校准结论。

## 2. 准备两组 200-step 配置（不会启动训练）

```bash
BASE=/root/autodl-tmp/sd3/experiment
python -m scripts.plan_texture_fidelity_check --diagnostic-dir "$BASE/diagnostic_texture_37000_full_v2" --candidate-lambda 1000 --write-configs --steps 200 --output "$BASE/texture_fidelity_probe_plan_v1" --run-root "$BASE/Texture_FidelityProbe_200_v1"
```

生成：

```text
texture_fidelity_probe_plan_v1/
  calibration_report.md
  calibration.json
  baseline.yaml       # 原实验固定 lambda（本例 0.1）
  fidelity.yaml       # 显式指定的固定 lambda（本例 1000）
```

两组均从同一原始 SFT ControlNet/RA 初始化，**不是改 lambda 后 resume 37000**。
同一偏好清单、全 GT 流、纹理强度、beta、学习率、sigma、累计、seed；
相同 max_train_steps=200、checkpointing_steps=200。200 是稳定性/训练信号探针预算，
不代表足以评估方法最终表现，并要求超过原 warmup。

两组之间只允许 lambda 与输出路径不同，工具会检查此条件。
checkpoint 记录的有效 mask/GT 默认设置会显式展开，避免被当前默认值重新解释。
原训练配置、输入诊断、偏好、归一化统计和任何原输出都不会被覆盖。

### 独立验证集

项目默认 `config/eval_sd3.yaml` 的 roots/splits 指向 **test**，不能用于本轮选系数。
未传独立验证配置时，工具同时设置：

```yaml
training:
  run_checkpoint_validation: false
  validation_before_training: false
  validation_eval_config: null
```

因此默认两组只是训练诊断探针。不能靠训练样本上的 PSNR 选择最终超参数或宣称泛化提升。
若已有真正独立的验证数据，可在生成命令加入 `--validation-config /实际路径/validation.yaml`。
该文件必须明确使用 `splits: [val]` 或 `[validation]`，且不能指向 test roots；
目录须符合项目配对数据布局，工具不会虚构或划分数据，也不会改写验证配置。
此时两组都会启用训练前基线与 checkpoint 验证。

## 3. 手动依次运行，不并行抢占同一单卡

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
BASE=/root/autodl-tmp/sd3/experiment

CUDA_VISIBLE_DEVICES=0 python -m scripts.train_dpo_sd3 --config "$BASE/texture_fidelity_probe_plan_v1/baseline.yaml"
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_dpo_sd3 --config "$BASE/texture_fidelity_probe_plan_v1/fidelity.yaml"
```

两条命令各自最多 200 optimizer steps，到期保存 `checkpoint-200`。
不要传额外的 `--max_train_steps` 去覆盖短程预算。lambda 仍固定，训练时没有自动重标定。

## 4. 使用相同种子再测两组

```bash
BASE=/root/autodl-tmp/sd3/experiment
RUNS="$BASE/Texture_FidelityProbe_200_v1"

python -m scripts.diagnose_dynamic_texture --config "$RUNS/baseline/dpo_config.yaml" --checkpoint "$RUNS/baseline/checkpoint-200" --batches 3 --seed 42 --full-inference --device cuda:0 --output "$BASE/diagnostic_fidelity_probe_baseline_200_v1"
python -m scripts.diagnose_dynamic_texture --config "$RUNS/fidelity/dpo_config.yaml" --checkpoint "$RUNS/fidelity/checkpoint-200" --batches 3 --seed 42 --full-inference --device cuda:0 --output "$BASE/diagnostic_fidelity_probe_candidate_200_v1"
```

重点比较：原始/加权 GT 梯度、累计联合方向、原始 GT MSE，以及相同训练样本的一步/完整推理差值。
同时看 DPO loss 和偏好差距，不能只看乘上大 lambda 后的加权 loss 标量。
两组实际训练后的梯度与 37000 点的代数预测可能不同，这是正常的参数点变化。

- 固定权重改变后仍存在反向联合方向：下一步才研究冲突处理，而不是一次再改 beta/LR/纹理强度。
- GT MSE 改善但完整推理持续变差：提高代理失配的排查优先级。
- 训练信号已能约束保真：再以独立验证集评估能否在 PSNR 预算内保留美学收益。

脚本的代数与配置检查，不等价于已经在真实模型上验证 lambda=1000 有效。

```bash
python -m unittest discover -s tests -p 'test_texture_fidelity_plan.py' -v
```
