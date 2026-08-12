# RA 消融实验配置

所有训练变体均继承 `config/train.yaml`，冻结同一个 ControlNet，只训练 RA，并从头初始化
RA 参数。模型保存时会将消融结构配置与权重一起保存，评估程序会自动加载对应结构。
每次训练还会在输出目录中生成 `effective_config.yaml`，记录合并后的 CLI/YAML 配置。

## 核心实验

| 配置文件 | 实验变体 |
| --- | --- |
| `full.yaml` | M + C + Q + T + Local Token Adapter 完整模型 |
| `wo_main.yaml` | 移除主干特征 `main_states` |
| `wo_control.yaml` | 移除 RA 内部的 ControlNet 特征分支，保留原始 ControlNet 残差注入 |
| `wo_condition.yaml` | 移除 LQ token 和递归 condition state 路径 |
| `wo_temb.yaml` | 移除 RA 内部的 timestep + pooled text 联合嵌入 |
| `adapter_identity.yaml` | 只保留四路融合，不使用 Local Token Adapter |
| `adapter_channel.yaml` | 只使用通道 Adapter，不使用局部空间卷积 |

## 训练方法

运行一次完整模型训练：

```bash
accelerate launch train_controlnet_sd3.py --config config/ablations/full.yaml
```

建议每个变体分别使用 `42`、`3407` 和 `2026` 三个训练种子。覆盖 `seed` 时必须同时覆盖
`output_dir`，避免不同实验相互覆盖：

```bash
accelerate launch train_controlnet_sd3.py --config config/ablations/full.yaml \
  --seed 3407 --output_dir /root/autodl-tmp/sd3/experiment/ra_ablation/full_seed3407
```

## 评估方法

使用固定噪声协议评估完整测试集。将 `ra_fusion_path` 和 `output_dir` 指向待评估的实验，
并确保所有模型复用完全相同的 Noise Bank：

```bash
python -m utils.randomness_check --config config/ablations/eval.yaml \
  --ra_fusion_path /root/autodl-tmp/sd3/experiment/ra_ablation/full_seed42/ra_fusion \
  --output_dir /root/autodl-tmp/sd3/experiment/ra_ablation/eval/full_seed42 \
  --max_samples_per_weather 0
```

评估 `eval_baseline.yaml` 时使用 `--no-use_ra_fusion`，得到不含 RA 的 SD3+ControlNet
Baseline。禁止针对不同模型分别选择随机种子、推理步数或 ControlNet scale，否则会破坏
消融实验的公平性。
