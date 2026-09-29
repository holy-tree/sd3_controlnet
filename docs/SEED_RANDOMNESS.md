# M=8 / M=12 逐图 Seed 随机性测试

入口 `scripts/evaluate_seed_randomness.py` 复用项目已有的：

- `build_dataset_for_eval()`：天气、子数据集和 LQ/GT 映射。
- `select_evaluation_records()`：按天气固定随机抽样。
- `setup_pipeline()`：ControlNet、RA Fusion、LoRA 和 VAE 加载。
- `run_with_initial_noise()`：strength、steps、CFG、ControlNet scale、prompt 和预处理一致的固定噪声推理。
- `utils.metrics`：PSNR、SSIM 和 LPIPS 实现。

脚本只生成每图 12 个候选。M=8 读取配置 Seed 列表的前 8 个结果，M=12
读取全部 12 个结果，不会执行第二套推理。
每张源图都会分别将 CPU noise generator 重置为配置中的 Seed，因此 Seed 含义不受
图片顺序、batch size 或断点恢复位置影响；相同 Seed 在不同图片上使用相同初始噪声。

## 配置

默认配置为 `config/seed_randomness.yaml`：

```yaml
eval_config: "./config/eval_sd3.yaml"
num_samples_per_weather: 20
max_candidates: 12
compare_candidate_counts: [8, 12]
sample_seed: 2026
seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]
batch_size: 4
num_workers: 4
device: "cuda"
dtype: "bf16"
omp_num_threads: 1
output_dir: "/root/autodl-tmp/sd3/experiment/seed_randomness_m8_m12"
lpips_net: "alex"
metric_models:
  dists: "dists"
  musiq: "musiq-spaq"
  clipiqa: "clipiqa+"
  nima: "nima"
stability_relative_error_threshold: 0.20
```

`batch_size` 是同一源图一次推理或度量的候选数量。`num_workers` 只负责预读取
LQ/GT；启用 worker 时使用 `spawn`，避免 CUDA 初始化后 fork。

## 运行

```bash
uv run python -m scripts.evaluate_seed_randomness \
  --config config/seed_randomness.yaml
```

覆盖输出目录：

```bash
uv run python -m scripts.evaluate_seed_randomness \
  --config config/seed_randomness.yaml \
  --output_dir /root/autodl-tmp/sd3/experiment/seed_randomness_m8_m12
```

## 统计逻辑

质量指标方向：PSNR、SSIM、MUSIQ、CLIP-IQA+、NIMA 越高越好；LPIPS、DISTS
越低越好。

对每张图、每个 M 和每个质量指标：

```text
mean            = M 个 Seed 结果的平均值
sample_variance = 与 mean 的平方差之和 / (M - 1)
sample_std      = sample_variance 的平方根
Worst@M         = ↑指标的最小值，或 ↓指标的最大值
```

同时输出 min、max、median、p10、p90 和 range。总体统计严格先计算逐图统计，
再在 overall、weather 和 subdataset 内求平均：

```text
Mean         = 逐图 mean 的平均值
MeanStd      = 逐图 sample_std 的平均值
MeanVariance = 逐图 sample_variance 的平均值
Worst@M      = 逐图 Worst@M 的平均值
```

输出多样性不使用 GT。它在同一输入的 M 个输出之间计算：平均 pairwise LPIPS、
平均 pairwise DISTS、平均 pairwise L1，以及 M 个输出逐像素标准差的平均值。

M=8 稳定性分析精确枚举全部 `C(12,8)=495` 个子集。CSV 同时保存配置前 8 个
Seed 的方差误差、495 个子集的平均绝对/相对误差、方差 95% 区间和遗漏 M=12
最差候选的概率。报告中的结论阈值来自
`stability_relative_error_threshold`；M 的差异仅表示估计样本量变化，不表示模型
分布发生变化。

## 输出

```text
<output_dir>/
|-- sample_manifest.csv
|-- per_seed_metrics.csv
|-- per_image_randomness.csv
|-- randomness_summary.csv
|-- m8_vs_m12_bootstrap.csv
|-- stability_conclusions.json
|-- report.md
|-- run_manifest.json
|-- COMPLETE.json
|-- cache/<weather>/<image-key>/seed_<seed>.{png,json}
`-- diversity/<image-key>.json
```

- `sample_manifest.csv`：固定的 60 张图及其天气、子数据集、LQ/GT 路径。
- `per_seed_metrics.csv`：每张图每个 Seed 的七项 GT/NR 质量指标。
- `per_image_randomness.csv`：逐图 M=8/M=12 质量统计和输出多样性。
- `randomness_summary.csv`：overall/weather/subdataset 的逐图优先汇总。
- `m8_vs_m12_bootstrap.csv`：精确 8-of-12 子集稳定性分析。
- `report.md`：中文结论、总体表格及每项指标的高方差异常图片。

## 断点续跑

相同命令默认启用 resume。每个候选 PNG 和 JSON sidecar 都会校验：运行策略指纹、
image_id、Seed、prompt、尺寸和 SHA-256。有效缓存不会重新生成；缺失、损坏或指纹
不一致的候选会单独补算。质量指标和 diversity 也分别缓存并绑定候选 checksum。

模型、checkpoint、输入内容、prompt、strength、steps、CFG、scale、dtype、Seed 或
采样列表改变时，旧缓存不会被静默复用。策略不一致的 resume 会要求使用新目录或：

```bash
uv run python -m scripts.evaluate_seed_randomness \
  --config config/seed_randomness.yaml \
  --no-resume
```

PyIQA 模型在推理阶段释放后只初始化一次。任何必需权重或依赖缺失都会写入
`metric_initialization_error.txt` 并终止，不会用 NaN 静默完成汇总。
