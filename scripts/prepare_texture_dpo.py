"""Score full GT-selected texture crops or filter local-Q dynamic preferences."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dpo.tail_risk import load_config
from dpo.texture_preparation import score_texture_candidates, build_texture_preferences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("score", "filter"))
    parser.add_argument("--config", default="config/dpo_sd3_reward_dynamic_texture.yaml")
    parser.add_argument("--device", default="cpu", help="IQA scoring device; diffusion is never run")
    parser.add_argument("--restart-incomplete", action="store_true",
                        help="Score only: archive recognized incomplete output and restart; never overwrite completed results")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.restart_incomplete and args.stage != "score":
        parser.error("--restart-incomplete only applies to score")
    result = (score_texture_candidates(config, device=args.device, restart_incomplete=args.restart_incomplete)
              if args.stage == "score" else build_texture_preferences(config))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
