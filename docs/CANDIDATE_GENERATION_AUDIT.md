# Candidate Generation Audit

The generator already writes summary.json, sample_summary.csv and
dataset_summary.csv. `scripts/summarize_dpo_candidates.py` reads those metadata
and the saved candidate CSV to audit one merged pool. It never generates images,
loads a restoration model, starts training, or computes IQA. It has no ML imports.

## Run For M32

```bash
python -m scripts.summarize_dpo_candidates \
  --candidate-dir /root/autodl-tmp/sd3/experiment/dpo_candidates_m32_selected2000_3gpu \
  --expected-candidates 32 \
  --check-files
```

Use the merged root, not `_shards/shard-0000`. The terminal shows counts, metric
means/variability, weather totals, missing-data notes, and output paths. The file
check only stats existing files; it does not decode 192,000 images. Full pixel
verification is optional and much slower:

```bash
python -m scripts.summarize_dpo_candidates \
  --candidate-dir /root/autodl-tmp/sd3/experiment/dpo_candidates_m32_selected2000_3gpu \
  --expected-candidates 32 \
  --verify-checksums
```

The original pool is read-only. Reports are written to `generation_analysis/`:

```text
summary.json
report.md
group_metrics.csv
high_variability_groups.csv
issues.csv
run_config.json
```

`--output-dir` selects another independent report directory. Root metadata,
images, manifests and shard files cannot be overwritten by audit output.
Critical problems produce exit code 2 after writing the report; warning-only
or clean results return 0. Legacy absence of a completion marker is reported as
unknown, while a missing merged marker for a sharded run is a problem.

## What It Reports

- Raw versus deduplicated candidate rows, retained source groups, expected slots,
  complete/incomplete groups, per-weather/subdataset totals and count distribution.
- Duplicate paths/slots, conflicting source identities, invalid values, missing
  files when requested, and seed/CFG mismatches against saved settings.
- Selected/rejected group membership, rejection reasons, retention ratios,
  global caps, merged completion, and metadata/CSV hash consistency.
- Saved PSNR/SSIM/LPIPS and any available additional IQA scores, without filling
  missing values with zero or launching a scorer.
- Per-source mean, sample std/variance (N-1), population std, range and metric-
  specific best/worst candidate indices/paths; then image-weighted aggregate
  means and variability percentiles. Across-scene variance is not seed variance.
- High-variability groups to inspect in `high_variability_groups.csv`.

If no groups were rejected and the selected manifest contains 6,000 sources,
the expected retained pool is 6,000 groups / 192,000 candidates, with 2,000
groups / 64,000 candidates per weather. Actual counts may be lower when quality
thresholds reject groups; the audit explains the retained/rejected counts.

Best@32 is an observed selection advantage, not a theoretical quality limit.
Per-metric maxima can belong to different actual candidates; do not combine
them into an imaginary best image. Fixed CFG and unique recorded seeds are
needed to interpret within-group variation as seed sensitivity.

## Existing Aesthetic Scores

Generation normally only saves PSNR/SSIM/LPIPS. MUSIQ, CLIP-IQA, NIMA and DISTS
are obtained by the separate existing rescoring step. The audit flags their
absence and does not compute them. To summarize a CSV already rescored:

```bash
python -m scripts.summarize_dpo_candidates \
  --candidate-dir /path/to/merged_pool \
  --metrics-csv /path/to/merged_pool/per_candidate_metrics_aesthetic.csv \
  --expected-candidates 32 \
  --check-files
```

Input and report configuration/hashes are saved for reproducibility. Small
unit tests cover complete M32 pools, missing/duplicate records, metric direction,
group variance, uneven group sizes, metadata/hash errors and unchanged inputs.
