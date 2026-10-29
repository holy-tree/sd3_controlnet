"""Save small synthetic GT/candidate/mask/crop diagnostics on CPU, without IQA."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dpo.spatial import gt_texture_mask, select_texture_patches, save_texture_visualization, texture_weighted_mse
from dpo.spatial_config import DEFAULTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="experiment/texture_dpo_check_v1")
    args = parser.parse_args()
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    size = 256
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    flat = torch.full((1,3,size,size), .5)
    edge = flat.clone()
    edge[...,size//2:] = 1
    texture = (((x//8+y//8)%2).float()*.8+.1)[None,None].repeat(1,3,1,1)
    summary = {"synthetic_only": True, "model_or_iqa_loaded": False, "cases": {}}
    for name, gt in (("flat",flat),("edge",edge),("texture",texture)):
        mask = gt_texture_mask(gt, DEFAULTS["mask"])
        patches = select_texture_patches(mask, DEFAULTS["patches"])
        save_texture_visualization(directory/name, gt[0], (gt[0]*.9+.03).clamp(0,1), mask, patches)
        summary["cases"][name] = {"mean": float(mask.mean()), "coverage": float((mask>=.5).float().mean()), "patches": patches}
    prediction, target = torch.randn(2,4,16,16), torch.randn(2,4,16,16)
    old = (prediction-target).square().flatten(1).mean(1)
    torch.testing.assert_close(texture_weighted_mse(prediction,target,torch.rand(2,1,256,256),0),old,rtol=0,atol=0)
    assert summary["cases"]["flat"]["mean"] == 0
    with (directory/"checks.json").open("w",encoding="utf-8") as f:
        json.dump(summary,f,indent=2)
    print(json.dumps(summary,indent=2))
    print("Saved diagnostics:",directory.resolve())


if __name__ == "__main__":
    main()
