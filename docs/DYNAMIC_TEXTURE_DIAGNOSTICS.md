# dynamic_texture 真实模型诊断（只诊断，不训练）

入口：`scripts/diagnose_dynamic_texture.py`。本工具不构造 optimizer，不调用 step，
不裁剪梯度，不改奖励、选对、GT 目标或训练配置；现有 trainer 文件保持不变。
真实运行只调用项目实际 SD3/ControlNet/RA/VAE 加载函数，不提供小模型或 CPU 模型替代。

## 输入必须来自实际训练

- `--config`：训练输出目录的 **`dpo_config.yaml`**，不是当前 `config/` 下的默认 YAML。
  校验 trainer 保存的 `training.reward/preference_filter/spatial_dynamic` 嵌入块。
- `--checkpoint`：含 `dpo_resume.json`、`controlnet/`、`ra_fusion/` 的原始训练 checkpoint。
  校验 v6 目标签名、偏好数据哈希、实际参数设置及 dataset 长度。
- 原始 SFT ControlNet/RA 路径从保存的 model 块读取，用来重建训练时的固定参考，
  不应将它们替换成当前 DPO 权重。
- 真实梯度和可选完整推理都检查 **raw** 权重，不把 EMA 当作训练时参数。
  如果原评测使用 EMA，不能将本次 raw 对照直接当成对该 EMA 评测的复现。

## 直接运行

命令在服务器项目根目录执行。下面以 37000 步实验为例，输出均使用新的独立目录。

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# 只做梯度/DPO诊断，不进行完整采样。
python -m scripts.diagnose_dynamic_texture --config /root/autodl-tmp/sd3/experiment/SD3_ControlNet_RA_DPO_Dynamic_Texture_M32_V1/dpo_config.yaml --checkpoint /root/autodl-tmp/sd3/experiment/SD3_ControlNet_RA_DPO_Dynamic_Texture_M32_V1/checkpoint-37000 --batches 3 --device cuda:0 --output /root/autodl-tmp/sd3/experiment/diagnostic_texture_37000_grad_v1

# 可选：同次诊断后增加同源图的 SFT/raw 当前权重一步重建与完整采样对照。
python -m scripts.diagnose_dynamic_texture --config /root/autodl-tmp/sd3/experiment/SD3_ControlNet_RA_DPO_Dynamic_Texture_M32_V1/dpo_config.yaml --checkpoint /root/autodl-tmp/sd3/experiment/SD3_ControlNet_RA_DPO_Dynamic_Texture_M32_V1/checkpoint-37000 --batches 3 --full-inference --inference-samples-per-weather 1 --device cuda:0 --output /root/autodl-tmp/sd3/experiment/diagnostic_texture_37000_full_v1

# 在完整推理命令中再加 --local-iqa，才会计算两个输出的局部美学分数。
# 同一个冻结局部统计用于两者；若 IQA backend 版本已改变，仅给出同 backend 原始分数，不重新拟合。

# 只检查源码、环境和可选快照，不加载真实模型。
python -m scripts.diagnose_dynamic_texture --inspect-only --output experiment/texture_diagnostic_code_audit_v1
```

`--batches` 是**完整梯度累计窗口数**。例如真实 microbatch=2、累计=4，
`--batches 3` 测量 12 个微批次、24 对样本，不是只测 3 个微批次。
仅单 GPU 顺序运行，原多卡的 world_size 和全局有效 batch 会报告，但不冒称是多卡
all-reduce 梯度。每个窗口按一个 `(DPO天气, GT天气)` 层抽样，优先 rain/rain、snow/snow、
haze/haze；完全保留 dataset 的原 pair/独立 GT 索引映射，不换 GT 来凑同一天气。
如果没有某个天气或同天气组合，会明确标记。分层抽样不是精确恢复训练时下一批数据。

## 梯度测量口径

```text
g_Q = grad(L_DPO_texture)
g_F = grad(L_GT)
g_F_weighted = lambda_fidelity * g_F
g_total = g_Q + g_F_weighted

cosine = dot(g_Q,g_F) / (norm(g_Q)*norm(g_F))
dot(g_F,g_total) = dot(g_Q,g_F) + lambda_fidelity*norm(g_F)^2
```

- 复用实际 `texture_weighted_mse`、`training_dpo_loss` 和 `decoded_gt_mse`。
  chosen/rejected 和 policy/reference 的四项误差共用 GT mask，β 和 pair 权重来自快照。
- BF16/no 下没有 AMP scaling。FP16 读取 checkpoint 的 `scaler.pt`，没有则必须显式
  提供 `--loss-scale` 并记录，不猜测当时 scaler；每次反传后除去 scaling 和累计因子。
- 每个微批次统计其自身 batch mean 的梯度；累计统计是同窗口微批梯度的平均。
  参数集合依据真实可训练 ControlNet 和 RA 参数，以对象身份去重。模块集合可以重叠，
  `all` 不重复计数；注册表和重叠记录在 JSON。
- 两项 backward 之间把 Q 梯度流式转存磁盘，再测原始 GT 梯度；不在 RAM 保存多份
  巨型梯度。临时磁盘需求约为 **12 × 可训练参数数目（字节）**，启动时检查剩余空间。
  最多同时使用现有 backward 所需的一套 GPU 参数梯度，额外 CPU 数值块约 MB 量级。
  诊断完成后删除自己的工作向量；`--keep-gradient-buffers` 可显式保留。
- None 梯度按向量贡献为零处理，但单独报告缺失事件数；全缺失、全零、非有限都
  不称为正常。零分母的余弦/比例为 N/A，不填零。统计用 float64 点积避免范数溢出。
- 默认真实测量每对样本对 DPO batch mean 的参数梯度贡献范数，需要额外 backward，
  可用 `--no-per-sample-gradients` 关闭；关闭后不能声称已经查明参数梯度是否由少数样本主导。
  范数份额/有效贡献数量不等于向量和中的可加份额，存在抵消时尤其不能混淆。
- GT interval>1 时额外探测本来会跳过的原始 GT 梯度以便比较，并明确标为 probe-only。
  `gt_dot_total` 对应题设固定 λ 公式，`scheduled_gt_dot_total` 才对应当步的实际开关。
- 未调用 optimizer，没有 Adam 预条件；负内积只提供当前 loss 的普通小步 SGD 一阶趋势。
  不设置任意梯度比例阈值去判断 λ 是否合理。

## DPO 拉差诊断

逐对记录四个 FM 误差、policy/reference margin、βlogit、sigmoid、DPO loss、
稳定计算的 logit 导数及 chosen/rejected 相对参考的误差变化。
当两个 FM 误差都升高、但 rejected 升得更多时，偏好 margin 仍可改善。
这里的“DPO改善”是相对固定参考的中性 log(2)，不是缺少历史记录时虚构的训练时间曲线。
记录 sigmoid 恰好舍入为 0/1 的数量，并与导数/真实逐对梯度区分；
不能仅凭 β=5000 或 sigmoid 舍入成1就断言梯度过大/完全消失。

## 一步与完整推理的配对控制

- 来源仅为上述诊断采样涉及的训练 GT source manifest；不加载 `eval_sd3.yaml`
  的测试集。训练样本检查只能诊断机制，不能当验证集调参或泛化结论。
- SFT 与当前 checkpoint 共用相同 GT latent、噪声、实际 sigma/timestep、LQ、prompt 和 seed。
  sigma 取配置允许的 scheduler 表中低/中/高三个位置；记录 BF16/FP16 实际取值与原 scheduler 值。
- 完整推理使用原 `run_with_initial_noise` 路径以及保存的 validation 推理参数。
  对 SFT 用 functional substitution 换入原始 SFT parameters/buffers，退出后自动恢复
  当前模型绑定，不写入其参数。DPO 梯度诊断则仍使用原 trainer 的 parameter-only reference 语义。
- 两种方法都计算 PSNR/SSIM/LPIPS，保存 GT/LQ/M、三种 sigma 的一步图、完整推理图及误差图。
  一步 loss 在 clamp 前计算，保存图像时才 clamp；每对误差图使用共同的显示缩放因子。
- 按每源图三个 sigma 的平均一步 GT MSE 与完整 PSNR 分类 C/D，同时保存单 sigma 结果，
  避免把对 sigma 敏感的结论隐藏在均值中。小样本相关性不足时为 N/A，不做因果判断。
- `--local-iqa` 放在释放 SD3 后执行，避免三个 IQA 模型和大模型同时占用显存。
  两者同坐标、同 backend，并使用唯一的原实验冻结统计；backend 已变时只比较原始分数。

## 产物与证据等级

完成真实诊断后，可按[固定 GT 权重短程探针](TEXTURE_FIDELITY_PROBE.md)处理：
先 CPU 复查缺失梯度和固定系数的代数影响，再显式生成一组不覆盖原实验的单变量配置。

```text
output/
  diagnostics.csv              # row_type 区分 pair_sample / micro_gradient /
                               # accum_gradient / one_step_compare / inference_compare / local_iqa
  diagnostic_state.json        # 配置、源码哈希、环境、参数注册表及全部分位数
  diagnostic_report.md         # 先主要发现，再必要数据及优先级
  visualizations/              # 仅 --full-inference 时生成真实对齐图像
```

报告明确区分真实运行、代码检查和未验证推测。运行中异常保留已收集的真实数据并标记 partial，
不会写入或修改训练/候选/偏好目录。运行前后核对可训练参数、参考快照和共享 buffer 的摘要。
诊断后不变只能证明本次没有参数更新，不能替代过去 37000 步的 reference 历史日志。

下一步依据证据：优先排查无梯度/非有限/参考变化；出现 C 则先复查 GT 代理；
累计梯度出现可复现负内积则优先分析冲突；只有相对 GT 信号弱、冲突证据少且代理方向一致时，
才考虑单独验证更强的固定 GT 权重。本工具不自动修改任何训练设置，也不安排参数扫描。

数值工具检查命令（**只检查统计公式和磁盘记账，不代表真实模型已通过**）：

```bash
python -m unittest discover -s tests -p 'test_texture_diagnostics_math.py' -v
```
