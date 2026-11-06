# 多 lambda 单卡等预算对照

入口：`python -m scripts.texture_lambda_sweep`。
默认比较 `lambda_fidelity=1000,2000,5000,10000`，每组从相同 SFT 初始化训练 **200个 optimizer steps**，
再在同一份固定验证集、同样 seed 和推理设置下，比较各模型与 SFT 的差距。

这里的 lambda 是 `L_DPO_texture + lambda_fidelity * L_GT` 中的训练权重。
不同 lambda 必须有各自训练过的 checkpoint，不能只在推理时改配置。
候选局部 IQA、偏好对和全 GT 流复用已有数据，不重新生成19万候选或重算局部评分。

## 1. 固定验证集（已有 plan 则直接复用）

在服务器项目根目录执行：

```bash
cd /root/sd3_controlnet
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export BASE=/root/autodl-tmp/sd3/experiment
export PROBE="$BASE/Texture_FidelityProbe_200_v1"
export VAL="$BASE/texture_final_validation_v1"
export SWEEP="$BASE/texture_lambda_sweep_200_v1"
export RUNS="$BASE/Texture_LambdaSweep_200_v1"
```

若 `$VAL/validation_plan.json` 已存在，**跳过下面这条 prepare**；不能因某个 lambda 的表现重新抽图。
若尚未准备验证集，执行：

```bash
python -m scripts.validate_texture_release prepare --probe-root "$PROBE" --rain-root /root/autodl-tmp/datasets2/train/rain --snow-root /root/autodl-tmp/datasets2/train/snow --haze-root /root/autodl-tmp/datasets2/train/haze --samples-per-weather 30 --seeds 42,2024 --data-role dpo_holdout --psnr-tolerance 0.0 --output "$VAL"
```

这一步只固定数据、hash和预算，不加载 SD3，不执行原三模型验证。
会排除偏好对应的原训练路径、全 GT 流以及规范预处理像素重复内容。
`dpo_holdout` **只保证本次 DPO 未见，不代表 SFT 未见**；不能把此前15张训练诊断图当验证集。
若有真实独立 val，将三个 root 换成该 val 路径并使用 `--data-role external_validation`。
不够合格图像就报错，不回退到 test。更多排除规则见[验证说明](TEXTURE_FINAL_VALIDATION.md)。

## 2. 只生成等预算配置，并显式复用已完成的1000组

```bash
python -m scripts.texture_lambda_sweep prepare --validation-plan "$VAL/validation_plan.json" --lambdas 1000,2000,5000,10000 --steps 200 --reuse "1000=$PROBE/fidelity" --run-root "$RUNS" --output "$SWEEP"
python -m scripts.texture_lambda_sweep space --plan "$SWEEP/sweep_plan.json"
```

**此时没有开始训练。** 工具读取真实训练输出的 `dpo_config.yaml` 与 checkpoint 元数据，
物化当时的有效空间设置，生成完整独立配置。组间只改变 lambda 和输出路径，保留：

- 初始 SFT ControlNet/RA 路径，固定 reference 的定义；`resume_from_checkpoint=null`。
- 偏好与全 GT manifest、beta、学习率、optimizer、scheduler、warmup、纹理强度、sigma。
- microbatch、梯度累积、训练 seed、混合精度和 EMA 训练设置。
- 相同总步数、最后一步 checkpoint；自动训练验证关闭，避免调用默认 test 配置。

`--reuse` 是显式声明，**不是拿某个1000组的最终导出目录冒充本次模型**。
工具要求存在该运行的 `checkpoint-200`、有效历史配置和训练元数据，并核对 SFT、seed、目标、
数据、optimizer 设置及预算一致。必须从 SFT 起训且训练自动验证关闭；条件不一致就报错。
评估始终读取 checkpoint 内的 **raw** 权重，不使用可能已导出为 EMA 的运行根目录权重。
可重复指定 `--reuse LAMBDA=RUN_DIRECTORY`；如要四组全部重训，省略 `--reuse`。

对于这组已有的1000-200实验，复用后只需新增训练 **2000、5000、10000**，共600个 optimizer steps。
更改 `--steps` 时，不能复用预算不同的旧200步模型；先另起新的计划/运行目录。

### 先看磁盘估算

`storage_estimate.json` 包含：

- `pending_training_runs`：还需训练的组数。
- `per_run_checkpoint_plus_export_bytes`：按真实 checkpoint 估计的每组 checkpoint、optimizer、EMA和最终导出总空间。
- `estimated_validation_bytes`：各方法逐图 LQ/GT/预测图的保守存储估算。
- `estimated_additional_peak_bytes`、`free_bytes`、`enough_space_estimate`。

保留所有组的结果，因此训练空间会累计；已有训练曾因磁盘不足报错，开训阶段会再次检查剩余空间。
不足时将 `RUNS` 改为真正空间充足的磁盘目录，并用新 `SWEEP` 路径重新 prepare。
若报告目录和训练目录位于不同磁盘，还应分别 `df -h "$SWEEP"` 和 `df -h "$RUNS"`；
估算为保守地在训练盘预留验证空间，不能替代另一个磁盘的实际配额检查。
默认 `--storage-mode full` 保留完整训练结果；显式 `weights-only` 模式见下文。

### 磁盘不足：显式仅保留评估权重

如果三组完整输出需要87.31GiB、可用54.76GiB，可以采用 `--storage-mode weights-only`。
该模式仍以原有设置完成每组训练（包括 EMA 更新和完整 checkpoint 保存），成功结束后再精简该新组：

- **保留**：`checkpoint-200/controlnet`、`checkpoint-200/ra_fusion` 的 raw 权重及配置、
  `dpo_resume.json`、`dpo_config.yaml`、训练日志/summary，以及精简记录 `weights_only.json`。
- **删除本轮新组生成的**：checkpoint 的 optimizer/scheduler/RNG/scaler/EMA state、
  checkpoint 内重复的 EMA 导出、运行根目录重复的 policy/EMA 导出。
- **复用的原1000组不精简**；未知文件不删除。只清理开训前已有本计划 ownership 记录、
  且完成指定步数的本轮新组，删除前校验保留权重的hash，支持中断后重试。

这样按“前面已完成组的 raw 权重 + 当前一组完整训练/导出峰值 + 验证空间”估算，
而不是累计三组完整训练状态。每组仍需先有空间写出完整 checkpoint，不能把峰值只按 raw 权重算。
**这些新组可按同一协议评估，但不再支持完整恢复 optimizer 续训。**
如选中的lambda需要完整200步状态续训，应从SFT用已固定的配置重跑该组，保留完整状态后继续；
不能把仅存权重目录当完整 resume，亦不能改 reference 初始化路径。

更新 `dpo/texture_lambda_sweep.py` 和 `scripts/texture_lambda_sweep.py` 后，原计划的 code hash 会变化，
需要**新建计划目录**，不要手改旧JSON的hash。沿用原固定验证集即可，示例：

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export BASE=/root/autodl-tmp/sd3/experiment
export PROBE="$BASE/Texture_FidelityProbe_200_v1"
export VAL="$BASE/texture_final_validation_v1"
export SWEEP="$BASE/texture_lambda_sweep_200_weights_v1"
export RUNS="$BASE/Texture_LambdaSweep_200_weights_v1"

python -m scripts.texture_lambda_sweep prepare --validation-plan "$VAL/validation_plan.json" --lambdas 1000,2000,5000,10000 --steps 200 --reuse "1000=$PROBE/fidelity" --storage-mode weights-only --run-root "$RUNS" --output "$SWEEP"
python -m scripts.texture_lambda_sweep space --plan "$SWEEP/sweep_plan.json"
```

查看新估算的 `storage_mode`、`raw_policy_bytes`、`estimated_training_peak_bytes` 和 `enough_space_estimate`。
确认空间足够，再使用下方相同的 `train` 和 `run` 命令。
原来因存储预检查失败的计划尚未开训，不需要重建验证集、重算IQA或删除原1000探针。

## 3. 显式开始单卡串行训练

```bash
python -m scripts.texture_lambda_sweep train --plan "$SWEEP/sweep_plan.json" --gpu 0
```

逐组启动项目真实 `scripts.train_dpo_sd3`，每组结束后验证 checkpoint，再进入下一组。
`--gpu 0` 为该子进程的物理 GPU 选择，训练时统一线程数为4。不要从多进程 launcher 内启动此命令。
任一组失败就停止，不继续烧后续组的 GPU 时间。
重新执行时会校验并跳过已完成、条件匹配的组；未完成的目录需先保留/重命名再重试，
不会静默覆盖或将半程状态当成新的 SFT 起点。

每个新组的文件位于 `$RUNS/lambda_2000` 等目录；复用组仍从原 `$PROBE/fidelity` 读取。
训练后不会自动启动评估或继续到更长步数。

## 4. 同一套 SFT 和固定验证协议，一条命令完成比较

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.texture_lambda_sweep run --plan "$SWEEP/sweep_plan.json" --device cuda:0
```

- 复用项目真实 checkpoint-validation 和初始噪声生成路径；所有方法采用匹配精度、raw权重。
- 按组顺序加载模型，SFT仅推理一次；核对 reference、共享冻结权重/buffers、源图与初始噪声一致。
- 默认90张源图 × 2seed ×（SFT + 4个lambda）= **900次完整推理**，不是改几个数就能离线算出的指标。
  每天气10张、2seed时是300次；样本量要在首次prepare时确定，不看完分数再挑图。
- SD3释放后，统一评分整图 **PSNR、SSIM、LPIPS、DISTS、MUSIQ、CLIP-IQA、NIMA、TOPIQ-NR**。
  其中前三项来自原验证回调，其余复用原评估器的真实指标模型。
- 比较原始整图分数；不对不同lambda各自拟合归一化，也不将局部`Q_texture`当成整图美学。
- 指标缺失记录为缺失，不填零；模型/seed已完成输出可核对复用，未完成子目录拒绝覆盖。

若完整推理已完成、只有 IQA 权重下载或评分失败，修复后使用：

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.texture_lambda_sweep score --plan "$SWEEP/sweep_plan.json" --device cuda:0
```

这条只重新评分已保存图片，不重新加载 SD3 推理。重建报告使用：

```bash
python -m scripts.texture_lambda_sweep report --plan "$SWEEP/sweep_plan.json"
```

## 5. 看这些结果

```text
$SWEEP/
  sweep_plan.json                  # lambda、预算、复用声明、配置/输入/code哈希
  lambda_1000.yaml ...              # 完整独立配置
  storage_estimate.json
  evaluation/
    lambda_report.md               # 总体/各天气相对SFT差值与PSNR CI
    lambda_comparison.csv          # 全部指标的绝对均值、SFT均值、差值、CI、有效源图数
    paired_summary.csv             # 另含中位数、p05/p95、下降比例
    paired_source_deltas.csv       # 逐源图变化，带天气和原子数据集
    per_seed_summary.csv           # 每个seed的方向复查
    per_image_metrics.csv          # 原始逐图逐seed指标及输出路径
    psnr_musiq_tradeoff.svg         # 横轴PSNR差、纵轴整图MUSIQ差；SFT位于(0,0)
    lambda_summary.json            # 观测的均值Pareto前沿与完整汇总
    inference/<method>/seed_*/     # 真实模型输出
```

先读 `$SWEEP/evaluation/lambda_report.md`。
例如 `PSNR delta=-0.05` 表示比 SFT 低0.05dB，`MUSIQ delta=+1` 表示比 SFT 高1分。
SSIM和美学越大越好，LPIPS/DISTS越小越好，所有差值仍统一写成“候选减SFT”。

CI先在每个源图内平均seed，再以源图为单位bootstrap，总体按天气分层，
不会把两个seed当两张独立图片。区间是逐比较区间，未作多重比较校正；
扫描后选出的最佳数值仍需后续冻结方案验证，不能把选参集表现当最终泛化结果。

Pareto前沿只表示本轮预算下的 PSNR/MUSIQ 均值非支配点，包含 SFT，缺失/不完整点不参与。
脚本不会自动选最大lambda、自动续训或声称“GT与DPO已无冲突”。
判断时看各天气/每seed方向、逐图退化和美学收益，200步对照只能比较这个预算，不能保证长训排序不变。

## 本地检查范围

```bash
python -m unittest discover -s tests -p 'test_texture_lambda_sweep.py' -v
```

测试覆盖完整配置单变量、历史有效设置、复用的seed/预算/目标隔离、文件改变检测、
磁盘不足不启动/失败即停止、weights-only的完整单组峰值/保留边界/中断恢复、
SFT仅分配一次、源级配对统计、缺失数据与均值Pareto方向。
这些是配置/文件处理/代数测试，不加载模型，不替代服务器真实SD3训练或验证。
