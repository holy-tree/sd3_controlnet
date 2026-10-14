# Recover Complete Groups From Interrupted M32 Generation

This workflow exports verified completed source groups without generating missing
images. It reads the original frozen plan and group journals; worker CSVs and
worker `COMPLETE.json` files are not required. It does not load restoration or IQA
models, change worker files, or restart workers.

## Export

Run from the project root after confirming the original generator has stopped.
Keep the original source dataset, generation plan and `_shards` directories.
The destination must be new or empty and must not overlap the original directory.

```bash
python -m scripts.generate_dpo_candidates \
  --config config/dpo_sd3.yaml \
  --output_dir /root/autodl-tmp/sd3/experiment/dpo_candidates_m32_selected2000_3gpu \
  --export_complete_groups \
  --export_dir /root/autodl-tmp/sd3/experiment/dpo_candidates_m32_complete_groups
```

The frozen plan, not current generation settings, determines the expected groups,
candidate slots, seeds, CFG, source hashes and model provenance. Original global
indices are preserved, including gaps. Missing/damaged groups are excluded as a
whole, never completed with generated replacements. Original threshold rejections
and global weather-cap rejections remain separate from exclusions.

The exporter validates cached PNGs/metrics and copies retained candidates and
cached LQ/GT images into an independent snapshot. It does not use hardlinks;
budget additional disk space and time for validation/copying. CSV source LQ/GT
references remain the original paths for downstream identity/provenance, so keep
the raw source dataset available as well. Export progress shows verification and
copy stages. A stale lock on the original generation directory produces a warning,
not a requirement to delete it: only the new destination is locked.

If all 5,977 observed cached groups are valid and retained, the snapshot has
191,264 candidate rows; this is an expectation, not a guaranteed export count.
Read the printed counts and `snapshot_manifest.json` for actual results.

Additional outputs:

- `excluded_samples.json`: missing/invalid groups and exclusion reasons.
- `snapshot_manifest.json`: original plan/journal hashes, population counts,
  retained/excluded indices and copied-file hashes.
- `summary.json` and `COMPLETE.json`: `partial_export: true`,
  `dataset_scope: "verified_completed_groups"`, `generation_finished: false`.

`COMPLETE.json` is published last and certifies the exported subset, not completion
of the original job. A failed export may leave incomplete destination artifacts,
but no completion marker; use a different new destination for a retry. The copied
full plan is provenance, not a resumable subset plan. Do not run `--resume` against
the snapshot. `--merge_only` is different: it requires all shards to be complete.

## Audit

```bash
python -m scripts.summarize_dpo_candidates \
  --candidate-dir /root/autodl-tmp/sd3/experiment/dpo_candidates_m32_complete_groups \
  --expected-candidates 32 \
  --check-files
```

A certified partial export is labeled as a verified subset of an unfinished
generation job. This is a warning, not an incomplete retained-group error. Missing
slots, inconsistent manifests, hashes or completion metadata remain errors.
Reports are written separately under `generation_analysis/`.

## Score Once

The raw generation CSV does not contain MUSIQ/CLIP-IQA/NIMA/DISTS. This step runs
the existing IQA scorer on the exported images, but does not generate candidates:

```bash
python -m scripts.rescore_dpo_candidates \
  --config config/dpo_sd3_reward_static_m32_partial.yaml \
  --input_csv /root/autodl-tmp/sd3/experiment/dpo_candidates_m32_complete_groups/per_candidate_metrics.csv \
  --output_csv /root/autodl-tmp/sd3/experiment/dpo_candidates_m32_complete_groups/per_candidate_metrics_aesthetic.csv \
  --device cuda:0 \
  --batch_size 8
```

Do not use `--normalize_only` on unscored generation data. The scorer needs its
normal runtime dependencies and IQA weights. It refuses existing output unless
explicitly asked to overwrite. Both experiments use this one score population;
do not fit normalization independently per shard or mode.

## Filter Both Modes

Run these sequentially. The first filter prepares the shared
`per_candidate_metrics_fr_nr.csv` from saved numeric scores if absent; no further
IQA inference is run by filtering. Both modes reuse its weather normalization.

```bash
python -m scripts.filter_dpo_pairs \
  --config config/dpo_sd3_reward_static_m32_partial.yaml

python -m scripts.filter_dpo_pairs \
  --config config/dpo_sd3_reward_dynamic_m32_partial.yaml
```

Preferences are written to separate `dpo_preferences_static_m32_partial` and
`dpo_preferences_dynamic_m32_partial` directories. The common fidelity gates and
9-pair-per-group limit are unchanged. Groups with too few qualifying pairs are
not padded by weakening gates.

## Train Separately

These are three-GPU DDP commands; each GPU holds a complete training replica.
Run one experiment at a time, not both on the same GPUs concurrently:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch \
  --multi_gpu --num_processes 3 --mixed_precision bf16 \
  scripts/train_dpo_sd3.py \
  --config config/dpo_sd3_reward_static_m32_partial.yaml

CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch \
  --multi_gpu --num_processes 3 --mixed_precision bf16 \
  scripts/train_dpo_sd3.py \
  --config config/dpo_sd3_reward_dynamic_m32_partial.yaml
```

Both configs use beta 5000, disabled Tail-risk, equal FR weights, independent new
training/evaluation output directories and `resume_from_checkpoint: null`.
They start from the configured SFT reference rather than resuming older DPO runs.
Training settings otherwise inherit the base YAML: currently ControlNet LR
`5e-8`, RA LR `1e-7`, 10,000 steps and batch size 8 per process (effective batch
24 with three processes and accumulation 1). The previously discussed table with
LRs `1e-6`/`2e-6` and 20,000 steps did not change those YAML values. Check desired
training hyperparameters and checkpoint paths before launching either experiment.

## Verification

Local tests use tiny PNGs, fake inference processes and synthetic numeric IQA
scores. They cover missing/damaged groups, preserved indices/seeds, no worker
completion markers, frozen policy reconstruction, global caps, independent copies,
source preservation, interrupted publication, certified-subset audits and both
reward-mode preference contracts. They do not execute the real server export,
IQA models or GPU training.
