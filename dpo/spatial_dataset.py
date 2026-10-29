"""Aligned local-preference stream plus independent all-source LQ/GT stream."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .dataset import collate_preference_pairs
from .spatial import aligned_rgb, gt_texture_mask, mask_checksum
from .spatial_config import SCHEMA
from .tail_risk import file_sha256


class SpatialPreferenceDataset(Dataset):
    """Every source, including sources without local preferences, trains GT MSE.

    GT source indices cycle independently of preference indices, with uniform
    epoch coverage. The common dataloader shuffle randomizes the combined rows.
    Geometry is deterministic resize/center-crop; optional horizontal flips are
    shared by candidate/GT/LQ/mask, not applied independently to each image.
    """
    def __init__(self, manifest_path, gt_source_manifest, settings):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        with self.manifest_path.open(encoding="utf-8") as f:
            self.records = [json.loads(line) for line in f if line.strip()]
        with Path(gt_source_manifest).open(encoding="utf-8") as f:
            self.gt_records = [json.loads(line) for line in f if line.strip()]
        if not self.gt_records:
            raise ValueError("Independent GT manifest is empty")
        if any(row.get("schema") != SCHEMA for row in [*self.records, *self.gt_records]):
            raise ValueError("Spatial dataset cannot consume old preference/source schemas")
        self.sources = {tuple(row["source_key"]): row for row in self.gt_records}
        if len(self.sources) != len(self.gt_records):
            raise ValueError("Duplicate GT sources")
        self.settings = settings
        self.num_preference_pairs = len(self.records)
        # Per-worker bounded decoded-source cache. No trainable mask parameters.
        self._source_cache = OrderedDict()

    def _cached_source(self, key):
        # Pickle-safe on Windows/spawn; bounded to two decoded sources per worker.
        value = self._source_cache.pop(key, None)
        if value is None:
            value = self._load_source(key)
        self._source_cache[key] = value
        while len(self._source_cache) > 2:
            self._source_cache.popitem(last=False)
        return value

    def _load_source(self, key):
        row = self.sources[key]
        for name in ("gt", "lq"):
            if file_sha256(row[name+"_path"]) != row[name+"_sha256"]:
                raise ValueError("GT/LQ source changed since texture preparation")
        gt, lq = (aligned_rgb(row[name+"_path"], self.settings["resolution"]) for name in ("gt", "lq"))
        mask = gt_texture_mask(gt[None], self.settings["mask"])
        if mask_checksum(mask) != row["mask_sha256"]:
            raise ValueError("GT mask/alignment differs from preference preparation")
        return gt*2-1, lq*2-1, mask[0]

    def __len__(self):
        return max(len(self.records), len(self.gt_records))

    def __getitem__(self, index):
        aux = self.gt_records[index % len(self.gt_records)]
        aux_key = tuple(aux["source_key"])
        aux_gt, aux_lq, _ = self._cached_source(aux_key)
        record = self.records[index % len(self.records)] if self.records else None
        if record is not None:
            key = tuple(record["source_key"])
            gt, lq, mask = self._cached_source(key)
            for name in ("chosen", "rejected"):
                if file_sha256(record[name+"_path"]) != record[name+"_checksum"]:
                    raise ValueError("Candidate changed since local preference scoring")
            chosen, rejected = (aligned_rgb(record[name+"_path"], self.settings["resolution"])*2-1
                                for name in ("chosen", "rejected"))
            prompt, weather = record["prompt"], record["weather"]
        else:
            # Collation-only placeholders, NOT exported aesthetic preferences.
            # Zero pair weight disables DPO; the independent GT objective remains.
            gt, lq, mask = self._cached_source(aux_key)
            chosen = rejected = gt
            prompt, weather = aux["prompt"], aux["weather"]
        probability = self.settings["horizontal_flip_probability"]
        if probability and torch.rand(()) < probability:
            chosen, rejected, gt, lq, mask = (x.flip(-1) for x in (chosen, rejected, gt, lq, mask))
        if probability and torch.rand(()) < probability:
            aux_gt, aux_lq = aux_gt.flip(-1), aux_lq.flip(-1)
        return {"chosen_pixel_values": chosen, "rejected_pixel_values": rejected,
                "gt_pixel_values": gt, "conditioning_pixel_values": lq,
                "texture_mask": mask, "aux_gt_pixel_values": aux_gt,
                "aux_conditioning_pixel_values": aux_lq, "aux_prompt": aux["prompt"],
                "prompt": prompt, "weather": weather,
                "psnr_gap": record["psnr_gap"] if record else 0.0,
                "reward_gap": record["reward_gap"] if record else 0.0,
                "pair_id": record["pair_id"] if record else "gt_only_placeholder",
                "pair_weight": 1.0 if record else 0.0, "is_tail_pair": False}


def collate_spatial_pairs(examples):
    result = collate_preference_pairs(examples)
    for name in ("texture_mask", "aux_gt_pixel_values", "aux_conditioning_pixel_values"):
        result[name] = torch.stack([row[name] for row in examples])
    result["aux_prompts"] = [row["aux_prompt"] for row in examples]
    return result
