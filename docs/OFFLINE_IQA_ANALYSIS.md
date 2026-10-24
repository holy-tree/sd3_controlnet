# 同图配对的在线与离线 IQA 分析

本文用于独立的 GT/SFT/DPO 同图诊断，不是常规候选评分步骤。
候选生成、无参指标评分、偏好筛选和单卡训练的统一命令见[DPO 主流程](DPO_SD3.md)。

`scripts/analyze_offline_iqa.py` 分析 GT、SFT、DPO 之间的 IQA 分数差异，并检查已有候选集中是否存在满足保真约束且评分更高的结果。启用 `--online` 时，脚本先在历史候选清单中的同一批源图上运行选定的 SFT/DPO 检查点；不启用时，只读取已保存的模型输出。脚本不会重新生成历史候选，也不会启动训练。

## 在线验证

```bash
python -m scripts.analyze_offline_iqa \
  --online \
  --candidate-csv /path/to/per_candidate_rewards.csv \
  --normalization-json /path/to/per_candidate_metrics_aesthetic_normalization.json \
  --eval-config config/eval_sd3.yaml \
  --sft-checkpoint /path/to/sft/checkpoint-N \
  --dpo-checkpoint /path/to/dpo/checkpoint-M \
  --sft-weights raw \
  --dpo-weights ema \
  --seeds 42 43 44 \
  --reference-seed 42 \
  --num-samples-per-weather 20 \
  --sample-seed 2026 \
  --max-images 6 \
  --batch-size 4 \
  --output-dir /path/to/online_iqa_sft_vs_dpo \
  --device cuda
```

移除 `--max-images 6` 后，每种天气分析 20 个源图组；各天气数据充足时共 60 组。按天气固定的样本排序允许从小规模试运行扩展到更大规模，并复用已生成结果。`--num-samples-per-weather 0` 表示选择完整的有效历史候选总体，开销可能很大。基于小规模诊断作出广泛结论前，应先增加样本量。

默认的三个随机种子包含用于保真参照的固定种子 42。SFT 和 DPO 使用相同的源 LQ、提示词、显式生成的 CPU 初始噪声、预处理、CFG、步数和强度。历史候选的逐图提示词及生成设置优先于评估默认值；`--inference-steps`、`--guidance-scale`、`--strength`、`--ra-fusion-scale` 和 `--controlnet-scale` 用于显式覆盖，并会记录到配置中。`--inference-dtype` 默认为 `bf16`。每次仅推理一张图的一个种子，且只驻留一个模型；IQA 评分前会释放两套推理流水线。

必须提供具体的检查点路径。DPO 默认使用 EMA，不会静默回退到原始权重。如果没有 EMA 导出，应显式使用 `--dpo-weights raw`，并在实验中说明。不要用训练输出根目录代替选定的检查点。

在线模式写入 `online_validation/source_manifest.csv`、各模型的 `manifest.csv`、PNG 和来源附加文件、生成跳过/错误记录，以及 `run_metadata.json`。源文件内容、模型权重、实现代码、运行库版本、提示词和推理策略均记录指纹。再次运行时会直接复用有效输出，无需加载扩散模型；缺失或损坏的输出会逐个修复。策略变化时，程序会拒绝复用，并要求使用新输出目录或 `--no-resume`。IQA 使用独立的增量缓存。原候选、源图片和清单均不会被修改。

这些结果属于候选总体诊断，通常基于训练源图，不代表独立留出集上的泛化证据。历史候选可能由不同于当前对比模型的 SFT 检查点生成，报告会注明检查点来源。从 K 个历史候选中择优的结果，相比模型的三个种子平均值具有选择优势；未超过该择优结果不意味着模型没有学到东西。报告分别展示 DPO 相对 SFT 的直接增益、保真约束通过情况，以及候选是否超过模型表现。

## 离线输入

- `--candidate-csv`：已有候选清单，图片相对路径以 CSV 所在目录为基准解析。
- `--sft-input`、`--dpo-input`：明确指定的 CSV/JSONL 清单，或包含 `validation_per_image.csv`、`per_image_metrics.csv`、按种子划分的运行目录，或 `*_pred.png`/`*_gt.png`/`*_lq.png` 三联图片的目录。
- `--output-dir`：独立的分析输出与缓存目录。
- `--normalization-json`：可选的候选归一化文件。未指定时，若存在 `<candidate_csv_stem>_normalization.json`，则自动使用它。

关联始终使用“天气 + 子数据集 + 源图 ID”，不会仅凭文件名在全局匹配。关联键对三个字段统一进行大小写折叠，从子数据集名开头移除一个 `<weather>_`，并且只移除源图 ID 中可识别的已保存三联图数字前缀/后缀。输出记录保留原展示字段。清单中的身份字段可使用 `image_id`、`source_id`、`pair_id` 或 `name`，路径字段可使用 `prediction_path`、`candidate_path` 或 `path`。发现记录和跳过记录 CSV 会保存匹配方式、置信度、重复项、缺失文件及内容冲突。

## 离线运行

```bash
python -m scripts.analyze_offline_iqa \
  --candidate-csv /path/to/per_candidate_metrics_aesthetic.csv \
  --sft-input /path/to/sft_saved_eval \
  --dpo-input /path/to/dpo_saved_eval \
  --sft-default-seed 42 \
  --dpo-default-seed 42 \
  --output-dir /path/to/offline_iqa_analysis \
  --device cuda
```

`--sft-default-seed` 和 `--dpo-default-seed` 默认为 `None`。它们只补充产物中缺失的种子，并记录 `seed_method=cli_default`；未显式指定时，未知种子仍保持未知。这适用于没有记录固定验证种子的检查点 `validation_per_image.csv` 文件。

常用参数包括 `--resolution 512`、`--batch-size`、`--reference-seed 42`、`--max-images`、`--max-visualizations`、`--crop-size` 和 `--no-resume`。保真约束默认采用严格 PSNR/DISTS 阈值 `0/0` 和容忍阈值 `0.15/0.01`，可通过四个 `--*-threshold` 参数覆盖。

预处理沿用评估器策略：使用双线性插值将短边缩放到指定分辨率，中心裁剪，转为 RGB，并在 `[0,1]` 范围内评分。各指标独立、按需初始化。依赖包、权重或模型缺失时，对应单元格留空，并在 `report.md`/`COMPLETE.json` 中记录，不会用零替代。

## 结果解读

- 实验 A 仅使用无参考 IQA：MUSIQ-SPAQ、CLIP-IQA+、NIMA、TOPIQ-NR-SPAQ、TOPIQ-IAA 和 NIQE。只统计 GT/SFT/DPO 精确共有的图片，先在每张图内对种子取平均，再报告按指标方向定义的配对增益和图像级汇总。
- GT 是**恢复目标的评分参照**，不是分数上界。
- 实验 B 还使用 PSNR、SSIM、LPIPS-Alex 和 DISTS。候选集择优结果要求奖励为有限值；严格/容忍约束下的择优结果还要求 PSNR 和 DISTS 为有限值。每个择优结果都对应一条真实候选记录，不可用的可选指标保持为空。严格/容忍约束使用 SFT 的固定种子 42，不会回退到其他种子。
- 共用候选归一化对候选、SFT 和 DPO 应用同一套按天气计算的中位数/IQR 及 `clip_z`。缺少该归一化时，历史候选奖励仅用于候选选择，模型间奖励比较留空。
- 当前动态奖励需要历史同源组的 F 中位数 B。本分析器未提供该冻结上下文，因此拒绝动态归一化附加文件，
  不会静默改用 Q 或给每种模型重新计算 B；候选动态排序可查看偏好构建器的 `dynamic_candidate_rewards.csv`。
- 候选 PSNR、SSIM、LPIPS 为有限值时，直接使用已保存数值。只有归一化 JSON 明确记录了对应指标的预期模型时，才信任候选的 PyIQA 分数；未声明模型来源的 TOPIQ/NIQE 会重新计算。
- 从 K 个候选中择优包含利用已知评分选择最优项的优势。非配对集合的均值差异不能作为模型学到改进的证据。
- 诊断标记用于识别需要人工查看的频率、颜色或细节不匹配，不推断语义层面的原因。
- 严格/容忍约束下的“无合格候选比例”只统计存在有效固定种子 SFT 保真参照的组。缺少该参照时，比例保持为空，而不是报告为 100% 失败。

运行会输出：模型逐种子评分、两个实验的配对表与汇总表、候选指标/组汇总/单指标择优结果、有效/跳过记录、复现配置、包含总体表格和审慎诊断结论的 Markdown 报告、增量校验和缓存、可视化 PNG，以及 `COMPLETE.json`。`run_config.json` 和 `COMPLETE.json` 还包含候选/归一化文件校验和及模型清单发现记录。
