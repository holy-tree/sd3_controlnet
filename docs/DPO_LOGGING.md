# DPO Training And Validation Logs

DPO records training metrics at `logging_steps` and all six raw reward-related
metrics at every checkpoint validation: PSNR, SSIM, LPIPS, MUSIQ-SPAQ,
CLIP-IQA+, and NIMA-AVA. IQA scores are computed on current validation outputs,
not copied from offline candidate labels. They are not normalized on the small
validation batch and no new reward or training objective is introduced.

## Configuration

```yaml
training:
  report_to: tensorboard
  logging_dir: null             # <output_dir>/tensorboard
  logging_steps: 10
  checkpointing_steps: 500
  run_checkpoint_validation: true
  validation_compute_iqa: true
  validation_iqa_device: cpu    # cuda or cuda:0 is optional
```

The settings inherit into static/dynamic/Tail-risk configs. Six IQA metrics are
validation-time measurements, not extra losses or measurements at every training
microbatch. Increasing validation frequency increases inference/IQA cost.

CPU is the default for NR scorers to avoid extra training VRAM. GPU scorers are
offloaded to CPU after each metric. PyIQA and its model weights are required for
MUSIQ/CLIP-IQA/NIMA. Missing models or nonfinite results produce warnings, null
values, and explicit `n_<metric>` counts, never fake zeros. TensorBoard omits
unavailable metric values but records their valid counts. These scorers run
inside the existing validation RNG-preservation context.

## Output Layout

```text
<output_dir>/
  training.log
  training_metrics.jsonl
  validation_metrics.jsonl
  tensorboard/sd3_ra_dpo/events.out.tfevents.*
  checkpoint-N/
    validation_metrics.json
    validation_per_image.csv
    validation_weather_metrics.csv
```

`training.log` is an append-only readable training/validation metric log, not a
capture of every dependency's stdout. `training_metrics.jsonl` retains its
training-only schema. The independent `validation_metrics.jsonl` records step,
raw/EMA weights, timestamp, validation seed, scorer variants, Overall/weather
means, valid counts, and flattened TensorBoard scalars. Root-level histories and
TensorBoard events survive checkpoint pruning. Only the main process writes
logs in multi-GPU training.

TensorBoard preserves existing tags such as `validation/overall_psnr`, and adds
`validation/overall_musiq`, `validation/overall_clipiqa`,
`validation/overall_nima`, plus equivalent rain/snow/haze tags. Training tags
include `loss`, `dpo_loss_weighted`, `implicit_accuracy`, and `lr/controlnet`.
Validation events are flushed immediately; ordinary training events use the
writer's normal flushing policy.

## View In Jupyter

Run these notebook magics in the server's environment:

```python
%load_ext tensorboard
%tensorboard --logdir /root/autodl-tmp/sd3/experiment/SD3_ControlNet_RA_DPO_Static/tensorboard
```

If TensorBoard is absent, install the declared dependency with
`python -m pip install tensorboard`. The first validation curves appear only
after checkpoint validation finishes.

For a notebook table/plot after at least one validation, use pandas if installed:

```python
import json
from pathlib import Path
import pandas as pd

root = Path('/root/autodl-tmp/sd3/experiment/SD3_ControlNet_RA_DPO_Static')
rows = []
with (root / 'validation_metrics.jsonl').open() as handle:
    for line in handle:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # A final live-appended line may not be complete yet.

df = pd.json_normalize(rows).sort_values('step')
columns = ['step', 'weights'] + [f'overall.{m}' for m in
    ('psnr', 'ssim', 'lpips', 'musiq', 'clipiqa', 'nima')]
display(df[columns])
df.plot(x='step', y=columns[2:], subplots=True, figsize=(12, 15), grid=True)
```

PSNR, SSIM, MUSIQ, CLIP-IQA, and NIMA are higher-is-better; LPIPS is
lower-is-better. None becomes a missing plot value rather than zero.

## Existing Runs

Files append on resume; existing history is not truncated. Repeated validation
at the same step remains distinguishable by weights/timestamp. Removed old
checkpoints cannot have missing IQA history reconstructed automatically.

A running Python process does not reload changed code. To enable new logs for
an existing run, resume from a valid saved checkpoint with all original training
parameters unchanged; explicitly set `training.resume_from_checkpoint: latest`
in the launched config. Do not restart a completed/active experiment with
`resume_from_checkpoint: null`, which starts from the original SFT policy.

Logging/IQA settings do not change the DPO loss, beta, optimizer, or reward-mode
contract. Actual CUDA/IQA execution should be checked in the training environment;
unit tests use fake inference and mock IQA models without downloading weights.
