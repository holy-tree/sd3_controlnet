"""JSONL preference-pair dataset for image restoration DPO."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PreferencePairDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        resolution: int = 512,
        tail_risk_enabled: bool = False,
    ):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]
        if not self.records:
            raise ValueError(f"Preference manifest is empty: {self.manifest_path}")
        self.tail_risk_enabled = tail_risk_enabled
        if tail_risk_enabled:
            for index, record in enumerate(self.records):
                try:
                    weight = float(record.get("pair_weight", 1.0))
                except (TypeError, ValueError, OverflowError) as error:
                    raise ValueError(
                        f"pair_weight must be positive and finite (record {index})"
                    ) from error
                if not math.isfinite(weight) or weight <= 0.0:
                    raise ValueError(
                        f"pair_weight must be positive and finite (record {index})"
                    )
        self.preprocess = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
        ])

    def _resolve(self, raw_path: str) -> Path:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = self.manifest_path.parent / path
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def _image(self, raw_path: str) -> torch.Tensor:
        with Image.open(self._resolve(raw_path)) as image:
            return self.preprocess(image.convert("RGB")) * 2.0 - 1.0

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        return {
            "chosen_pixel_values": self._image(record["chosen_path"]),
            "rejected_pixel_values": self._image(record["rejected_path"]),
            "gt_pixel_values": self._image(record["gt_path"]),
            "conditioning_pixel_values": self._image(record["lq_path"]),
            "prompt": str(record.get("prompt", "")),
            "weather": str(record["weather"]),
            "psnr_gap": float(record["psnr_gap"]),
            "reward_gap": float(record["reward_gap"]),
            "pair_id": str(record["pair_id"]),
            "pair_weight": (
                float(record.get("pair_weight", 1.0)) if self.tail_risk_enabled else 1.0
            ),
            "is_tail_pair": bool(record.get("is_tail_pair", False)),
        }

    def __len__(self) -> int:
        return len(self.records)


def collate_preference_pairs(examples: list[dict]) -> dict:
    return {
        "chosen_pixel_values": torch.stack([row["chosen_pixel_values"] for row in examples]),
        "rejected_pixel_values": torch.stack([row["rejected_pixel_values"] for row in examples]),
        "gt_pixel_values": torch.stack([row["gt_pixel_values"] for row in examples]),
        "conditioning_pixel_values": torch.stack([
            row["conditioning_pixel_values"] for row in examples
        ]),
        "prompts": [row["prompt"] for row in examples],
        "weather": [row["weather"] for row in examples],
        "psnr_gap": torch.tensor([row["psnr_gap"] for row in examples], dtype=torch.float32),
        "reward_gap": torch.tensor([row["reward_gap"] for row in examples], dtype=torch.float32),
        "pair_id": [row["pair_id"] for row in examples],
        "pair_weight": torch.tensor(
            [row.get("pair_weight", 1.0) for row in examples], dtype=torch.float32
        ),
        "is_tail_pair": torch.tensor(
            [row.get("is_tail_pair", False) for row in examples], dtype=torch.bool
        ),
    }
