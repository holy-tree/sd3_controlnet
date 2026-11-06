# 一轮完成验证，然后固定方案继续训练

入口：`scripts/validate_texture_release.py`。本轮不再重复梯度测量、一步 sigma 机制诊断，
不重算 19 万候选 IQA，不修改奖励、选对、训练 loss 或 optimizer 更新方式。
先固定验证样本/预算，再比较真实 **SFT、lambda0.1-200、lambda1000-200**。

如果要比较多个固定 GT 权重（例如1000、2000、5000、10000），使用[多 lambda 等预算脚本](TEXTURE_LAMBDA_SWEEP.md)。
可直接沿用本页 `prepare` 生成的固定验证集，无需先执行本页的三模型推理或续训。

## 希望补齐的证据

1. **DPO 未见源图**上，lambda1000 的保真改善是否仍成立；不能只看之前 15 张训练图。
2. 美学收益是否体现在**整图**而不只是被奖励的纹理图块。
3. PSNR、SSIM、LPIPS、DISTS 是否方向一致；全图 MUSIQ/CLIP-IQA/NIMA 与非奖励 TOPIQ-NR 是否一致。
4. 雾天与各子数据集是否仍有明显退化，平均值是否掩盖少数坏图。
5. 两个固定 seed 下方向是否稳定。
6. 后续固定 lambda 的训练是否继续漂移：先监控到总步数2000，再决定长训，不一次扫多个参数。

不保证验证一定通过；严格“不下降”的预算默认就是 **0 dB**，不会看完结果后悄悄放宽。

## 数据角色必须说清

仓库默认 `config/eval_sd3.yaml` 指向 test，不能直接拿来选择 lambda。
当前没有已登记的独立 val 配置，因此默认提供 `dpo_holdout` 方案：

- 从既有 train 目录中抽样，不读 test。
- 排除所有已进入 DPO **全 GT manifest** 的源图，不仅排除有偏好者。
- 同时从原候选输入 CSV 排除原 GT/LQ 路径，并按匹配训练预处理的 RGB 像素哈希排除重复内容。
- 验证样本间 GT/LQ 内容也去重，按子数据集轮流取样，不按模型评分选图。
- 固定种子抽每天气30张，总90张，只生成 symlink，不移动/改写原图片。

**这只能证明 DPO 阶段留出，不代表 SFT 没见过。** 如果有全流程独立验证数据，
把三个 root 换成真实独立 val 根目录，并指定 `--data-role external_validation`。
仍做 DPO 重合排除，但 SFT 是否见过需要由数据来源确认；工具不能自动证明。
路径/精确像素去重也不能证明不存在重压缩或近重复内容。

## 0. 先确认存储，避免训练后又因保存失败

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export BASE=/root/autodl-tmp/sd3/experiment
export PROBE="$BASE/Texture_FidelityProbe_200_v1"
export VAL="$BASE/texture_final_validation_v1"
export TRAIN_OUT="$BASE/Texture_L1000_Continue_v1"

df -h "$BASE"
python -m scripts.validate_texture_release space --probe-root "$PROBE" --training-output "$TRAIN_OUT" --total-steps 2000
```

验证主要占预测图像和报告空间，建议留1–2GB，不申请13GB梯度缓存。
**训练保存不是这个量级**：根据真实 checkpoint-200 的大小估算 optimizer、EMA、
最终导出及“先保存新 checkpoint 再 prune”峰值，可能需要数十GB。
若空间不足，先扩容或把 `TRAIN_OUT` 改为你实际空间充足的磁盘新目录。
不要删除后面要 resume 的 fidelity/checkpoint-200 状态文件，也不要改 SFT 初始化路径。
工具不自动删除任何训练结果。估算不能代替真实磁盘配额，开训前仍需看实际空闲。

## 1. CPU 固定验证集与预算

以下根目录来自项目原训练数据配置，脚本会核对是否存在；无合格数据就报错，
不回退到 test。输入同源 GT/LQ 必须配对且宽高比一致。

```bash
python -m scripts.validate_texture_release prepare --probe-root "$PROBE" --rain-root /root/autodl-tmp/datasets2/train/rain --snow-root /root/autodl-tmp/datasets2/train/snow --haze-root /root/autodl-tmp/datasets2/train/haze --samples-per-weather 30 --seeds 42,2024 --data-role dpo_holdout --psnr-tolerance 0.0 --output "$VAL"
```

准备阶段会 hash 已用 GT/LQ 和读取 checkpoint 文件校验和，有少量 CPU/I/O 成本，不加载SD3。
`validation_plan.json` 固化样本、推理设置、原训练快照、模型文件哈希、seed、判据。
`validation.yaml` 供终检与后续训练 callback 共用。保留整个验证目录。
Linux 默认 symlink；需要独立副本或 Windows 无 symlink 权限时可显式 `--materialization copy`，
这只复制验证图片到新目录，不改变原训练目录。

## 2. 单卡完整验证，一条命令

```bash
python -m scripts.validate_texture_release run --plan "$VAL/validation_plan.json" --device cuda:0
```

实际执行：

- 复用真实 `RealEngine` 加载训练期精度的 ControlNet/RA、冻结 SD3/VAE。
- 复用 `trainer.run_checkpoint_validation` 与 `run_with_initial_noise`，不另造推理流程。
- SFT 使用 initial SFT 的真实参数/buffers，functional substitution 后恢复当前权重。
- 90张 × 2seed × 3模型 = **540次完整推理**；GT/LQ、prompt、scheduler、步数、CFG、strength、
  初始噪声和权重精度匹配。验证输出标注 **raw**，不把 EMA 与 raw 混着比较。
- SFT/base 两个方法共用一次模型加载；candidate 另一次。两套模型顺序加载，参数前后hash核对。
- SD3 释放后逐个运行整图 DISTS、MUSIQ、CLIP-IQA、NIMA、TOPIQ-NR，复用项目现有 metrics wrapper。
  三模型在同一评分流程使用原始分数，不分别拟合归一化；不与旧的局部Q分数混淆。
- 单图batch=1，IQA逐模型清理缓存，避免小容器RAM累计多份IQA权重。
- 不计算小样本 FID/Oracle，不进行反传、不分配梯度向量、不执行训练更新。

完成的 model/seed 子运行可校验复用。若某个子运行目录未完成，先保留/重命名该报错目录后重试；
不覆盖未识别的中间产物。如果只是评分模型缺失导致失败，修复权重后只运行：

```bash
python -m scripts.validate_texture_release score --plan "$VAL/validation_plan.json" --device cuda:0
```

它会复用已经保存的真实输出，不再跑 SD3。

## 3. 看一个报告，不再来回追加小诊断

```bash
cat "$VAL/validation_report.md"
```

主要产物：

```text
validation_report.md            # 主要结论、各天气指标、最差图、后续建议
validation_decision.json        # 目标状态与受控继续状态，分开记录
per_image_metrics.csv           # 所有真实输出的原始逐图指标
paired_summary.csv              # source内平均seed之后的配对汇总/CI
per_seed_summary.csv            # 每seed复查
paired_source_deltas.csv        # 逐源图变化、子数据集身份
review_panels/*.png             # 每天气最差3图的 LQ/GT/SFT/base/candidate 与误差对照
```

### 两种结论不能混为一谈

**最终目标状态**：

- `PASS`：总体/每天气平均PSNR的95%源图配对CI不越预设预算；整图MUSIQ有正增益证据，
  辅助均值没有退化或缺失/seed警报。
- `NOT_MET`：已经观察到预算或主指标不满足。
- `INCONCLUSIVE`：方向可能满足，但CI跨界、辅助指标/seed有待核对。
- `INCOMPLETE`：有缺失数据，不能宣称通过。

默认 `--psnr-tolerance 0.0` 表示严格不降；只有事先明确接受容差时才在 prepare 阶段设置。
不要看完结果后改预算。上述保护的是平均指标，不保证每张图都不下降。
CI先在每源图内平均两个seed，再按源图bootstrap；总体保持各天气配额，不能把180个seed观测当180张独立图。

**支持受控继续**：如果最终仍略低SFT，但候选相对lambda0.1的PSNR改善在留出集上得到支持，
各天气保真方向不差于base，且仍有相对SFT的整图美学增益、辅助指标/seed没有对应阻断，
可支持固定方案继续到2000步观察收敛。**这不是最终“不下降”已通过**，不会允许直接据此延到20000。

人工看一次最差图，重点关注雾天色偏/低频结构、雪天错纹理、雨天过锐。
不需要重新检查此前已经验证的所有GT梯度机制；若本轮方向不支持，不开始长训，不一次扫多个参数。

## 4. 生成受控续训配置，明确验证方向后再开训

```bash
python -m scripts.validate_texture_release continue-config --plan "$VAL/validation_plan.json" --training-output "$TRAIN_OUT" --total-steps 2000
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_dpo_sd3 --config "$VAL/continue_2000.yaml"
```

第一条仅准备配置，不自动执行第二条。证据不支持或磁盘不足时不会写出“已验证可继续”的配置。
保留 lambda=1000、beta、学习率、纹理强度、GT目标、sigma、偏好/GT流、batch和optimizer设置。
原 SFT 初始化路径保留，`resume_from_checkpoint` 指向原 fidelity/checkpoint-200；
续训输出放新目录，原200步试验保留。总步数2000意味着再训练1800步。

每500步用同一验证集、seed42检查 **raw** PSNR/SSIM/LPIPS。为节省训练内存，
callback 中暂不加载美学模型；美学在阶段结束后离线复查。EMA仍按原训练方式更新和保存，
仅改变验证使用的权重视图，不改变 optimizer 或 loss。
由于终检使用同一个 trainer validation/noise 协议，callback可与 seed42 的终检SFT基线比较。

出现以下情况不要机械跑满后再决定：多个验证点保真持续走坏、GT/总loss非有限、图像明显出现色偏或虚构纹理。
本工具不自动调整lambda或训练更新，不根据验证结果悄悄改参数。

## 5. 2000步结束，固定同一批图复查一次

```bash
export VAL2="$BASE/texture_final_validation_2000_v1"
python -m scripts.validate_texture_release recheck --plan "$VAL/validation_plan.json" --candidate-run "$TRAIN_OUT" --checkpoint "$TRAIN_OUT/checkpoint-2000" --output "$VAL2"
python -m scripts.validate_texture_release run --plan "$VAL2/validation_plan.json" --device cuda:0
```

不重新抽样或改预算，复用已校验的SFT/base输出；只对新的candidate做完整推理，再统一评分。
此时 candidate2000 与 base200 预算不同，只用于看继续训练轨迹；不能写成纯lambda等预算消融。
若目标稳定达标，再准备更长预算；若还未达标，仅支持中程的判据不能冒充长训通过。
最终冻结模型后才做正式test报告，不再用test挑lambda。

## 本地检查范围

```bash
python -m unittest discover -s tests -p 'test_texture_release_validation.py' -v
```

这些只检查选样、排重、几何、统计、续训配置和数据隔离，不用小模型替代真实SD3。
真实三模型验证需要服务器的快照/模型/数据与实际依赖，本地缺失时不产生“真实验证通过”结论。
