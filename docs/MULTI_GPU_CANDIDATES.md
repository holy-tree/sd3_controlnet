# Source-Sharded Multi-GPU Candidate Generation

Each GPU owns disjoint source images, not different seeds for the same image.
All candidates of a source image stay in one worker. The existing inference,
preprocessing, metrics, CSV format, and downstream preference workflow are reused.
This is independent-process inference, not DDP or model parallelism.

## Commands

Preview without creating files, loading model weights, or running inference:

```bash
python -m scripts.generate_dpo_candidates \
  --config config/dpo_sd3.yaml \
  --gpu_ids 0 1 2 \
  --output_dir /root/autodl-tmp/sd3/experiment/dpo_candidates_3gpu \
  --dry_run
```

Generate on three GPUs:

```bash
python -m scripts.generate_dpo_candidates \
  --config config/dpo_sd3.yaml \
  --gpu_ids 0 1 2 \
  --output_dir /root/autodl-tmp/sd3/experiment/dpo_candidates_3gpu
```

Use a fresh output directory. Coordinated generation refuses an existing legacy
candidate directory rather than deleting it. Each GPU must fit one complete
pipeline; GPUs do not split a model's memory. Explicit GPU IDs/UUIDs become each
child's `CUDA_VISIBLE_DEVICES` value, overriding inherited visibility masks.

Recover an interrupted job using the same settings and shard count:

```bash
python -m scripts.generate_dpo_candidates \
  --config config/dpo_sd3.yaml \
  --gpu_ids 0 1 2 \
  --output_dir /root/autodl-tmp/sd3/experiment/dpo_candidates_3gpu \
  --resume
```

Physical GPU assignments may change when shard count stays the same. Valid
completed shards are verified and skipped. In incomplete shards, valid group
journals are reused and only damaged/missing groups are regenerated. Model,
source-image, inference-setting, or implementation changes reject stale reuse;
use a new directory instead. A duplicate-coordinator lock prevents simultaneous
jobs from writing one output. After abnormal termination, inspect the PID in the
sibling `.dpo_candidates_3gpu.generation.lock` file and confirm the job stopped
before removing that stale lock.

If all workers finished but the coordinator stopped before final merge, recover
the merged files without starting inference:

```bash
python -m scripts.generate_dpo_candidates \
  --config config/dpo_sd3.yaml \
  --output_dir /root/autodl-tmp/sd3/experiment/dpo_candidates_3gpu \
  --merge_only
```

This reads the existing frozen plan, checks all shard completion records and
images, then publishes the ordinary root CSV/summary/COMPLETE files. No GPU IDs
or model inference are needed. Missing/corrupt shards are rejected rather than
silently merged; use `--resume` with original settings when actual recovery is
needed. Do not merge concurrently with a live generator. A lock error requires
checking that generator's status, not blindly removing the lock. The check may
take time for a large M32 pool because image integrity is verified.

Do not use `accelerate launch --multi_gpu` for generation. DDP environment
launches are explicitly rejected. Without `--gpu_ids`, the old single-process
entry point remains available, including its historical candidate cleanup
behavior; `--resume` belongs to the coordinated mode.

To keep only verified complete groups from an interrupted job without generating
missing groups, use `--export_complete_groups --export_dir <new_directory>`.
This reads group journals even when worker CSVs/completion markers are absent,
copies an independent subset, and never modifies/resumes the original job.
The subset is explicitly marked as partial; its completion marker does not mean
the original generation finished. See
[PARTIAL_CANDIDATE_RECOVERY.md](PARTIAL_CANDIDATE_RECOVERY.md) for export, audit,
shared scoring, static/dynamic filtering and separate training commands.

## Selection And Seeds

The plan first applies the existing global source selection and sampling limits,
then assigns `global_index` once. Worker i owns `global_index % K == i`. For K=3:

```text
GPU 0: source indices 0, 3, 6, ...
GPU 1: source indices 1, 4, 7, ...
GPU 2: source indices 2, 5, 8, ...
```

With 6,000 selected sources, this is 2,000 source groups per GPU. With 12
candidates each, it is 24,000 candidate slots per GPU. Uneven populations and
empty shards are supported without duplicating sources.

Candidate noise remains derived from `base_seed:candidate_index:global_index`.
Workers never reindex or sample their source subset again. Weather prompts are
computed globally and checked against the frozen plan, so random prompt fallback
cannot depend on which weather a worker happens to contain.

Weather retained-group caps apply once during the global merge, not once per
GPU. The merger keeps the earliest qualifying groups by global index. Workers
may generate extra qualifying/rejected groups before the cap is applied; this
is deliberate to preserve the globally selected retained population. Seed
identity is consistent between one-shard and three-shard execution; exact pixel
equality additionally depends on compatible GPU hardware, precision, runtime,
and deterministic kernels.

## Outputs And Logs

```text
<output_dir>/
  generation_plan.json
  shard-0000.log
  shard-0001.log
  shard-0002.log
  _shards/shard-0000/...
  _shards/shard-0001/...
  _shards/shard-0002/...
  candidates/<weather>/image_<global_index>_<stem>/...
  per_candidate_metrics.csv
  selected_samples.json
  rejected_samples.json
  sample_summary.csv
  dataset_per_noise.csv
  dataset_summary.csv
  summary.json
  COMPLETE.json
```

Monitor a worker on the server:

```bash
tail -f /root/autodl-tmp/sd3/experiment/dpo_candidates_3gpu/shard-0000.log
```

The coordinator now displays one overall source-group progress bar with each
GPU's completed/assigned counts and a combined rate/ETA. A group is counted
after its atomic completion journal is published, including groups rejected by
the quality threshold. It is not an individual-candidate or training-step bar.
For 32 candidates per group, the bar can stay at zero during model loading and
the first complete 32-candidate group. At 100%, reports may still be checked and
merged; wait for the root `COMPLETE.json` and final success message.

To observe a job already launched with an older coordinator, open another
terminal and run this read-only command using the same output directory:

```bash
python -m scripts.generate_dpo_candidates \
  --config config/dpo_sd3.yaml \
  --gpu_ids 0 1 2 \
  --output_dir /root/autodl-tmp/sd3/experiment/dpo_candidates_3gpu \
  --monitor
```

Monitoring starts no processes, writes no generation artifacts, and does not
affect workers when stopped with Ctrl+C. Existing journals are checked once
against their PNG caches; subsequent updates inspect new atomic journals rather
than repeatedly reading all images. Final integrity is still enforced by the
merger. Omit GPU IDs to label bars by shard number. Current inference workers and
frozen plan formats are unchanged by this display feature.

Each worker has separate reports and group journals. Before merge, all assigned
groups, candidate indices/seeds, PNG dimensions/checksums, metric records,
model provenance and frozen settings are validated. Final paths are rewritten
to the standard merged `candidates/` tree. Global statistics are recomputed from
all retained rows, never averaged from per-worker means. `COMPLETE.json` is
published last; a failed shard never yields a completed merged dataset.

Merged PNGs use hardlinks on compatible filesystems, with copy fallback. Do not
edit generated images in place: hardlinked worker/final files share storage.
Keep `_shards` and the plan while recovery is needed. Existing source images and
old candidate directories are not modified by coordinated generation. Do not
run downstream consumers against a directory while it is being merged.

## Downstream Workflow

After merge, the read-only `scripts/summarize_dpo_candidates.py` audits the M32
pool and summarizes saved metrics without scoring or inference. See
[CANDIDATE_GENERATION_AUDIT.md](CANDIDATE_GENERATION_AUDIT.md) for the command.

After successful merge, use the existing rescoring and filtering commands on the
merged CSV. NR/FR normalization must be fit once on the complete merged candidate
population, not separately per GPU. Set subsequent config paths to the new tree:

```yaml
candidate_generation:
  output_dir: /root/autodl-tmp/sd3/experiment/dpo_candidates_3gpu
preference_filter:
  source_metrics_path: /root/autodl-tmp/sd3/experiment/dpo_candidates_3gpu/per_candidate_metrics_aesthetic.csv
  candidate_metrics_path: /root/autodl-tmp/sd3/experiment/dpo_candidates_3gpu/per_candidate_metrics_fr_nr.csv
```

Scoring and preference selection are not automatically launched by generation.
Output directory overrides do not update existing training/preference configs;
keep those paths consistent when choosing a new candidate population.

## Verification

Tests use tiny images and fake child/inference processes, not real GPU execution.
They cover disjoint union, global index/noise/prompt consistency, global caps,
uneven/empty shards, cached recovery, policy/hash mismatches, duplicate/missing
records, interrupted publication, atomic completion, source preservation,
GPU-pinned launch environments, child failure cleanup and dry-run behavior.
