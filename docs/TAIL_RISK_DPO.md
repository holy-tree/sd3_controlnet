# Tail-Risk Weighted DPO

This is an experimental pair-weighting scheme, not a guarantee of reduced seed
variance. The tail is low reward within each weather's complete TRAIN candidate
population, not an abnormal seed for a particular input or a generation failure.
It cannot add tail examples absent from the existing preference pairs.

## Scope

The original pairs, chosen/rejected paths, order, weather proportions, reward
formula, normalization, SFT initialization, trainable modules, beta, learning
rates, training steps, and auxiliary loss coefficients are unchanged.
Only the DPO term receives pair weights. Candidate images and IQA are not rerun.

For weather w, use unique candidate identities from the full candidate ledger:

```text
T_w = Q20(candidate reward)
IQR_w = Q75(candidate reward) - Q25(candidate reward)
tail_deficit = clip(max(0, T_w - rejected_reward) / (IQR_w + eps), 0, 2)
raw_pair_weight = 1 + lambda_tail * tail_deficit
pair_weight = raw_pair_weight / mean(raw_pair_weight for all final pairs of w)
```

The mean of pair_weight is 1 separately for each represented weather, computed
offline over the full pair set. Ordinary pairs typically have weight below 1;
tail pairs have larger weights relative to ordinary pairs, but a shallow tail
pair can still be below 1 after normalization. The default raw weight range is
[1, 3]; normalized weights do not have that same range.

## Configuration

The existing `config/dpo_sd3.yaml` contains disabled defaults:

```yaml
tail_risk:
  enabled: false
  candidate_reward_file: null
  tail_quantile: 0.20
  lambda_tail: 1.0
  tail_deficit_max: 2.0
  eps: 1.0e-8
```

Two runnable comparison configs inherit all existing settings via `base_config`,
resolved relative to the referring YAML. Each top-level section is overridden
by key (not a general-purpose recursive merge):

- `config/dpo_sd3_tail_risk_standard.yaml`: weights forced to 1.
- `config/dpo_sd3_tail_risk.yaml`: offline pair weights enabled.

Both read the exact same weighted JSONL, use the same seed, sampler, preprocessing,
initialization and training parameters. They differ only in effective pair weights
and separate result directories. Before running, the base `model` paths and scales
must match `candidate_policy` in the original preference summary. These examples
do not change your checkpoint. Keep `require_candidate_provenance: true`.

## Offline Commands

The existing aesthetic CSV does not contain the final scalar reward. First
materialize the existing reward into a NEW file, using saved z-scores and the
unchanged `build_reward` formula. This is an explicit export, not IQA inference,
new normalization, or a new reward definition. Both chosen and rejected scores
are checked against the original pair scores; all exported candidate rewards are
also bound to the full original saved metrics, including candidates absent from
pairs. Mismatches stop the command. The weight generator requires validated export
metadata, unless the original full candidate CSV already contains explicit rewards.

```bash
python -m scripts.export_dpo_candidate_rewards \
  --config config/dpo_sd3_tail_risk.yaml

python -m scripts.weight_dpo_tail_risk \
  --config config/dpo_sd3_tail_risk.yaml
```

Export defaults to `tail_risk.candidate_reward_file`. Weight generation reads
the original unweighted JSONL from `preference_filter.output_dir`, not the new
training manifest. Input/output paths can also be provided with `--input_csv`,
`--output_csv`, `--input_pairs`, and `--output_dir` as appropriate.

The scripts refuse to overwrite existing outputs. If running another experiment,
choose new paths and update both matched configs. They require `splits: [train]`
and the training `selection_manifest`, check candidate LQ/GT/weather membership,
and check the full candidate identity set against the original metrics CSV and
preference summary. No fallback to rejected-only distributions is allowed.
Duplicate candidate rows with identical identity/reward are counted once;
conflicting duplicates, missing reward, near-zero IQR, and missing key fields fail.
Moving pairs to a new directory requires absolute image paths, as produced by
the current filter. Relative paths are rejected rather than silently rewriting
the original pair fields.

New outputs:

```text
dpo_candidates/
  per_candidate_rewards.csv
  per_candidate_rewards.csv.metadata.json
dpo_preferences_tail_risk/
  preference_pairs.jsonl
  preference_summary.json
  tail_risk_statistics.json
```

Each new pair preserves all original fields and adds only `tail_threshold`,
`tail_deficit`, `raw_pair_weight`, `pair_weight`, and `is_tail_pair`.
Statistics include thresholds, IQR, unique candidate/pair counts, tail-pair
fractions, raw-weight means, normalized weight distributions, input/output hashes,
and configuration. The copied preference summary preserves model provenance.

```bash
python -c "import json; p=json.load(open('/root/autodl-tmp/sd3/experiment/dpo_preferences_tail_risk/tail_risk_statistics.json')); print(json.dumps(p['per_weather'], indent=2))"
```

No statistics for server data are claimed until these commands are run there.

## Matched Training

Use fresh output directories; do not replace existing results. Do not rerun source
selection, candidate generation, IQA, or preference filtering for this comparison.

```bash
# Standard DPO, same pair file with all effective weights equal to 1
accelerate launch scripts/train_dpo_sd3.py \
  --config config/dpo_sd3_tail_risk_standard.yaml

# Tail-risk DPO
accelerate launch scripts/train_dpo_sd3.py \
  --config config/dpo_sd3_tail_risk.yaml
```

The original command still runs with tail risk disabled:

```bash
accelerate launch scripts/train_dpo_sd3.py --config config/dpo_sd3.yaml
```

Old pairs default to weight 1. Disabled mode ignores stored weights entirely.
Enabled mode with legacy pairs reports that weights are all 1 (no reweighting).
Enabled weighted training validates the offline config, file hashes, counts,
and per-weather mean weights before loading models. Existing PSNR-gap weighting
must explicitly be false for this experiment; simultaneous enabling is an error.

## Loss, Logs And Resume

```python
per_pair_loss, stats = diffusion_dpo_loss(..., reduction="none")
weighted_dpo_loss = (pair_weight * per_pair_loss).mean()
loss = weighted_dpo_loss + sft_weight * chosen_mse.mean()
# GT flow and GT x0 objectives retain their existing independent backward pass.
```

There is no division by the weight sum or manual accumulation-step division.
Accelerate handles gradient accumulation as before. Weighting is not applied to
SFT, GT flow, GT x0 or other auxiliary losses.

`training_metrics.jsonl` and configured trackers receive `dpo_loss_unweighted`,
`dpo_loss_weighted`, `pair_weight_mean`, `pair_weight_min`, `pair_weight_max`, and
`tail_pair_fraction`, plus existing margin/accuracy and auxiliary-loss statistics.
Logs aggregate by sample count over all microbatches and processes in a window
of `training.logging_steps` optimizer updates (default 10). Windows flush at
checkpoints and the final update too, so batch size 1 does not normally log only
one pair. These are observed-window weights, not batch-renormalized weights;
their mean need not equal 1 in every window. Tail labels are retained for logging
in both comparison runs; legacy pairs without labels report tail fraction 0.

For resume, set `training.resume_from_checkpoint: latest` in the relevant config.
`dpo_resume.json` saves/checks effective tail configuration, candidate reward and
statistics hashes, along with the existing preference-file hash, sampler/optimizer
settings and RNG state. Switching weighting or changing the offline inputs cannot
silently resume the same experiment. Old checkpoints remain resumable with tail
weighting disabled, subject to the existing compatibility checks.
