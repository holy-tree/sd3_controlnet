# Static / Dynamic Preference Rewards

Both modes use the existing candidate CSV, pair builder, DPO JSONL format, and training loss. No candidate generation or training is required to prepare preferences.

## Configuration

Use `config/dpo_sd3_reward_static.yaml` or `config/dpo_sd3_reward_dynamic.yaml`. They inherit the same model, candidate scores, and fidelity settings. Separate output directories protect historical experiments; there is only one implementation.

```yaml
reward:
  mode: static                  # static | dynamic
  fr_weights:
    psnr: 1.0
    ssim: 1.0
    lpips: 1.0
  tau_q: 0.05
  margin_static: 0.05
  margin_fr: 0.05

preference_filter:
  pairs_per_group: 9
  random_seed: 42
```

FR weights are nonnegative relative weights, normalized to sum one. Q is fixed at `0.55*MUSIQ_z + 0.35*CLIP-IQA_z + 0.10*NIMA_z`. F is `w_psnr*PSNR_z + w_ssim*SSIM_z - w_lpips*LPIPS_z`. LPIPS z-scores retain raw metric direction and are negated exactly once in F. Equal weights yield `(PSNR_z + SSIM_z - LPIPS_z)/3`.

The original `dpo_sd3.yaml` intentionally has no `reward.mode`, preserving already shipped NR-only rewards and checkpoint semantics. Explicit modes always select the new contract and require rebuilding preferences. Do not relabel old pairs as static or dynamic.

## Prepare Shared Scores Once

If the existing aesthetic CSV contains MUSIQ, CLIP-IQA, NIMA, PSNR, SSIM, LPIPS, and DISTS, no model needs to run:

```bash
python -m scripts.rescore_dpo_candidates \
  --config config/dpo_sd3_reward_static.yaml \
  --input_csv /root/autodl-tmp/sd3/experiment/dpo_candidates/per_candidate_metrics_aesthetic.csv \
  --output_csv /root/autodl-tmp/sd3/experiment/dpo_candidates/per_candidate_metrics_fr_nr.csv \
  --normalize_only
```

The command finds the input CSV's `_normalization.json`, preserves existing NR parameters, and fits missing FR parameters on the attested full population. It writes one CSV/normalization sidecar shared by both modes. Without historical parameters, it fits complete six-metric rows per weather using median/IQR, standard-deviation, then 1 fallback, clipped to [-3,3]. Missing/nonfinite required values stay blank and are recorded, never replaced by zero.

`--normalize_only` never loads images, IQA models, or diffusion models. Without it the existing rescoring path computes IQA on saved candidate images, without regenerating candidates. Frozen statistics may be supplied with `--normalization_json`; subsets must not refit population parameters.

The new CSV adds six z-score columns and `reward_q`, `reward_f`, `reward_static`, `reward_mode`. Historical `reward` is retained as `original_reward`. The builder recomputes Q/F from z-scores using current config, never stale precomputed reward columns. Changing mode, FR weights, or thresholds only requires rebuilding pairs, not recalculating IQA or normalization. Existing outputs are not overwritten unless `--overwrite` is explicit.

## Build Current Preferences

```bash
python -m scripts.filter_dpo_pairs --config config/dpo_sd3_reward_static.yaml
```

Switch through configuration:

```bash
python -m scripts.filter_dpo_pairs --config config/dpo_sd3_reward_dynamic.yaml
```

The existing `--candidate_metrics_path` and `--output_dir` overrides remain supported; keep training paths consistent with overrides.

The same builder enumerates each unordered combination once (12 candidates -> 66 comparisons):

1. Static: orient by `R=0.5*F+0.5*Q`; skip ties and gaps strictly smaller than `margin_static`.
2. Dynamic: if `abs(Q_A-Q_B) >= tau_q`, orient only by Q; otherwise orient only by F, skipping ties and gaps strictly smaller than `margin_fr`. Equality with a threshold is accepted. Zero `tau_q` selects Q for every comparison, with ties skipped.
3. Apply identical fidelity constraints to the final orientation, including Q-branch pairs.
4. Uniformly sample up to `pairs_per_group` eligible pairs without replacement using a fixed seed. Default 9 is not a hard cap: configured 15 can retain 15. Insufficient pairs are never repeated or padded.

Dynamic preferences can be nontransitive, so no global dynamic ranking is used. Old `pair_strategy`, coverage selection, appearance caps, and `min_reward_gap` are ignored by new modes and effective replacements are recorded. F-branch pairs cannot be removed by the old Q-gap gate.

Inherited fidelity defaults remain:

```text
chosen_PSNR - rejected_PSNR >= -0.15 dB
chosen_DISTS - rejected_DISTS <= 0.01
chosen_PSNR >= group median PSNR - 0.50 dB
chosen_DISTS <= group median DISTS + 0.02
```

These are pair constraints, not the online analyzer's SFT-seed42 constraints. The builder prints actual conditions, mode, kept groups/pairs, removal reasons and branch fractions. Details remain in `preference_summary.json` and missing candidate reasons in `skipped_candidates.csv`. No surviving pairs produces an empty manifest and diagnostic summary, not invented pairs.

## Metadata And Reuse

The existing JSONL/CSV layout is retained. New pairs add mode/branch, Q/F components, absolute Q gap and thresholds. `chosen_reward`/`rejected_reward` contain R for static and the active Q or F score for dynamic. DPO still consumes the oriented images and pair weights.

The summary binds reward parameters, sampling/group limit, fidelity settings, candidate CSV, normalization parameters and manifest with signatures/hashes. The builder always reconstructs pairs under current config. Sidecars with saved output CSV checksums are checked before publication. Training validates the contract before loading models, independently of candidate-model provenance or tail weighting. Resume also binds reward/data provenance. Genuine old mode-absent data remain supported; new-mode pairs cannot be reinterpreted as legacy. DPO and GT losses are unchanged.

Static works with the scalar reward exporter and existing Tail-risk algorithm. Dynamic has no single candidate scalar, so scalar export, tail-weight generation and `tail_risk.enabled=true` are explicitly rejected. Ordinary dynamic DPO uses unit weights with Tail-risk disabled, as in the example config. Do not compare a Q tail threshold with an F-branch rejected score.

The existing IQA analyzer recognizes a static normalization sidecar and uses R rather than silently replacing it with the old Q. A sidecar explicitly declaring dynamic mode is rejected by its scalar reward-best analysis; use the pair builder's Q/F branch diagnostics instead. Static diagnostic scoring of a dynamically trained model is a separate evaluation choice, not a global dynamic ranking.

## Verification

```bash
python -m unittest discover -s tests -p 'test_reward_modes.py'
python -m unittest discover -s tests -p 'test_reward_normalization.py'
python -m unittest discover -s tests -p 'test_preference_provenance.py'
```

Fixtures verify metric direction, both orientations, threshold equality, F-branch Q-gap bypass, nontransitivity, 66 comparisons, deterministic selection, configurable limits, shared fidelity, missing values, frozen shared statistics, source preservation, config mismatch, scalar guards and resume metadata. They run neither a restoration model nor a training main.
