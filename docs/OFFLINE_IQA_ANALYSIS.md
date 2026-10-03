# Pure-offline IQA analysis

`scripts/analyze_offline_iqa.py` compares saved GT, SFT, DPO, and candidate images. It does not import or construct a diffusion pipeline and never runs training or generation.

## Inputs

- `--candidate-csv`: existing candidate manifest. Relative image paths are resolved from the CSV directory.
- `--sft-input`, `--dpo-input`: an explicit CSV/JSONL manifest or a directory containing `validation_per_image.csv`, `per_image_metrics.csv`, seed-run directories, or saved `*_pred.png`/`*_gt.png`/`*_lq.png` triplets.
- `--output-dir`: independent analysis output/cache directory.
- `--normalization-json`: optional candidate normalization file. If omitted, `<candidate_csv_stem>_normalization.json` is used when present.

Joins always use `weather + subdataset + source ID`. A basename is never joined globally. Join keys case-fold all three fields, remove one leading `<weather>_` from the subdataset, and remove only a recognized saved-triplet numeric prefix/suffix from the source ID. Display fields remain in output records. Manifests may use `image_id`, `source_id`, `pair_id`, or `name` and `prediction_path`, `candidate_path`, or `path`. Discovery and skipped-record CSVs record the match method, confidence, duplicates, missing files, and content conflicts.

## Run

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

`--sft-default-seed` and `--dpo-default-seed` default to `None`. They fill only missing artifact seeds and record `seed_method=cli_default`; without these explicit options an unknown seed remains unknown. This is useful for checkpoint `validation_per_image.csv` files that omit their fixed validation seed.

Useful controls include `--resolution 512`, `--batch-size`, `--reference-seed 42`, `--max-images`, `--max-visualizations`, `--crop-size`, and `--no-resume`. Fidelity defaults are strict PSNR/DISTS `0/0` and tolerant PSNR/DISTS `0.15/0.01`; override them with the four `--*-threshold` options.

The preprocessing policy is the evaluator policy: resize the short side to the resolution with bilinear interpolation, center crop, convert to RGB, and score in `[0,1]`. Metrics are initialized independently and lazily. Missing packages, weights, or models leave blank cells and are reported in `report.md`/`COMPLETE.json`; they are never replaced by zero.

## Interpretation

- Experiment A uses NR IQA only: MUSIQ-SPAQ, CLIP-IQA+, NIMA, TOPIQ-NR-SPAQ, TOPIQ-IAA, and NIQE. It includes only exact common GT/SFT/DPO images, averages seeds within each image, and reports directional paired gains and image-level summaries.
- GT is **恢复目标的评分参照**, not an upper bound.
- Experiment B also uses PSNR, SSIM, LPIPS-Alex, and DISTS. Pool winners require a finite reward; strict/tolerant winners additionally require finite PSNR and DISTS. Every winner remains one actual candidate row, while unavailable optional metrics stay blank. Strict/tolerant qualification uses exact SFT seed 42 and never falls back to another seed.
- Shared candidate normalization applies the same per-weather median/IQR and `clip_z` to candidates, SFT, and DPO. Without it, historical candidate reward is selection-only and model reward comparisons stay blank.
- Candidate PSNR, SSIM, and LPIPS values are trusted when finite. Candidate PyIQA values are trusted only when the normalization JSON records the exact expected model for that metric; undeclared TOPIQ/NIQE values are recomputed.
- Best-of-K candidate results include an oracle selection advantage. Unpaired set mean differences are not evidence of learning.
- Diagnostic flags identify frequency/color/detail mismatches requiring visual review; they do not infer semantic causes.

The run writes per-seed model scores, paired and summary tables for both experiments, candidate metrics/group summaries/single-metric winners, valid/skipped records, reproducibility config, a Markdown report with Overall tables and cautious diagnostic states, progressive checksum cache, visualization PNGs, and `COMPLETE.json`. `run_config.json` and `COMPLETE.json` include candidate/normalization checksums and model manifest discovery inventories.
