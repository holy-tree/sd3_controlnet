"""Tail-weighted DPO integration and sample-counted logging windows."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import torch

from .losses import diffusion_dpo_loss
from .tail_risk import TAIL_FIELDS, file_sha256, normalize_tail_risk_config


def tail_risk_resume_signature(train_config: dict) -> dict:
    tail = normalize_tail_risk_config(train_config.get("tail_risk"))
    if not tail["enabled"]:
        return {"enabled": False}
    signature = {"config": tail}
    candidate_file = tail["candidate_reward_file"]
    if candidate_file is not None:
        candidate_path = Path(candidate_file).expanduser().resolve()
        signature["candidate_reward_file"] = {
            "path": str(candidate_path), "sha256": file_sha256(candidate_path),
        }
    stats_path = Path(train_config["preference_manifest"]).expanduser().resolve().parent / "tail_risk_statistics.json"
    if stats_path.is_file():
        signature["statistics_sha256"] = file_sha256(stats_path)
        signature["preference_summary_sha256"] = file_sha256(stats_path.parent / "preference_summary.json")
    return signature


def validate_tail_risk_training(train_config: dict) -> dict:
    """Validate offline provenance before loading models; legacy pairs stay unit-weighted."""
    tail = normalize_tail_risk_config(train_config.get("tail_risk"))
    if not tail["enabled"]:
        return tail
    if bool(train_config.get("weight_by_psnr_gap", False)):
        raise ValueError(
            "tail_risk conflicts with training.weight_by_psnr_gap; explicitly set it to false"
        )
    path = Path(train_config["preference_manifest"]).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise ValueError("Preference manifest is empty")
    if not any(TAIL_FIELDS.intersection(row) for row in records):
        print("[tail-risk] Legacy pairs have no weights; using unit weights (no tail reweighting).")
        return tail
    if any(not TAIL_FIELDS.issubset(row) for row in records):
        raise ValueError("Weighted manifest has missing tail-risk fields or mixed weighted/legacy pairs")
    if tail["candidate_reward_file"] is None:
        raise ValueError("tail_risk.candidate_reward_file is required for weighted training")
    stats_path = path.parent / "tail_risk_statistics.json"
    with stats_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if normalize_tail_risk_config(metadata["config"]) != tail:
        raise ValueError("Tail-risk configuration differs from offline weight generation")
    if metadata["output_manifest"]["sha256"] != file_sha256(path):
        raise ValueError("Tail-risk weighted preference manifest hash mismatch")
    summary_path = path.parent / "preference_summary.json"
    if metadata["output_summary"]["sha256"] != file_sha256(summary_path):
        raise ValueError("Tail-risk output preference summary hash mismatch")
    candidate_path = Path(tail["candidate_reward_file"]).expanduser().resolve()
    candidate_info = metadata["input_files"]["candidate_reward_file"]
    if (
        Path(candidate_info["path"]).resolve() != candidate_path
        or candidate_info["sha256"] != file_sha256(candidate_path)
    ):
        raise ValueError("Tail-risk candidate reward file mismatch")
    for name, info in metadata["input_files"].items():
        if file_sha256(info["path"]) != info["sha256"]:
            raise ValueError(f"Tail-risk input file changed: {name}")
    by_weather = defaultdict(list)
    for row in records:
        weight = float(row["pair_weight"])
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("pair_weight must be positive and finite")
        if not isinstance(row["is_tail_pair"], bool):
            raise ValueError("is_tail_pair must be boolean")
        by_weather[row["weather"]].append(weight)
    for weather, weights in by_weather.items():
        if not math.isclose(statistics.fmean(weights), 1.0, rel_tol=0, abs_tol=1e-8):
            raise ValueError(f"{weather} pair weights must have offline mean 1")
        if metadata["per_weather"][weather]["pair_count"] != len(weights):
            raise ValueError(f"{weather} offline pair count mismatch")
    return tail


def training_dpo_loss(
    chosen_policy: torch.Tensor,
    rejected_policy: torch.Tensor,
    chosen_reference: torch.Tensor,
    rejected_reference: torch.Tensor,
    pair_weights: torch.Tensor,
    beta: float,
    sft_weight: float,
    psnr_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    per_pair_loss, stats = diffusion_dpo_loss(
        chosen_policy, rejected_policy, chosen_reference, rejected_reference,
        beta=beta, reduction="none",
    )
    weights = pair_weights.to(device=per_pair_loss.device, dtype=per_pair_loss.dtype)
    if weights.shape != per_pair_loss.shape:
        raise ValueError("pair_weight shape must match per-pair DPO loss")
    if psnr_weights is not None:
        if not torch.equal(weights, torch.ones_like(weights)):
            raise ValueError("Cannot combine tail pair weights with PSNR-gap weights")
        # Preserve the explicitly requested legacy PSNR-only weighting when tail risk is off.
        weights = psnr_weights.to(device=per_pair_loss.device, dtype=per_pair_loss.dtype)
    weighted_dpo = (weights * per_pair_loss).mean()
    loss = weighted_dpo + float(sft_weight) * chosen_policy.mean()
    stats.update(
        loss_dpo=weighted_dpo.detach(),
        dpo_loss_unweighted=per_pair_loss.mean().detach(),
        dpo_loss_weighted=weighted_dpo.detach(),
    )
    return loss, stats


class DPOLogWindow:
    """Aggregate all microbatches/processes in a logging window, without batch renormalization."""

    def __init__(self):
        self.count = 0
        self.sums = {}
        self.weight_min = None
        self.weight_max = None

    def update(self, stats: dict, loss: torch.Tensor, weights: torch.Tensor, tail_flags: torch.Tensor) -> None:
        count = weights.numel()
        values = {
            **stats, "loss": loss.detach(),
            "pair_weight_mean": weights.detach().double().mean(),
            "tail_pair_fraction": tail_flags.detach().double().mean(),
        }
        if self.count and self.sums.keys() != values.keys():
            raise ValueError("Logging keys changed within a window")
        for key, value in values.items():
            total = value.detach().double() * count
            self.sums[key] = self.sums.get(key, torch.zeros_like(total)) + total
        minimum, maximum = weights.detach().double().min(), weights.detach().double().max()
        self.weight_min = minimum if self.weight_min is None else torch.minimum(self.weight_min, minimum)
        self.weight_max = maximum if self.weight_max is None else torch.maximum(self.weight_max, maximum)
        self.count += count

    def flush(self, accelerator=None) -> dict[str, float]:
        if not self.count:
            raise ValueError("Cannot log an empty window")
        sample_count = self.weight_min.new_tensor(self.count)
        packed = torch.stack([*self.sums.values(), sample_count, self.weight_min, self.weight_max])[None]
        gathered = accelerator.gather(packed) if accelerator is not None else packed
        total_count = gathered[:, -3].sum()
        logs = {
            key: float(gathered[:, index].sum() / total_count)
            for index, key in enumerate(self.sums)
        }
        logs["pair_weight_min"] = float(gathered[:, -2].min())
        logs["pair_weight_max"] = float(gathered[:, -1].max())
        self.__init__()
        return logs
