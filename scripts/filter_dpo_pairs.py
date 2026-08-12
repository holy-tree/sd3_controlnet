"""Filter offline SD3 candidates into preference pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dpo.filter_pairs import build_preference_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter SD3 restoration preference pairs")
    parser.add_argument("--config", default="./config/dpo_sd3.yaml")
    parser.add_argument("--candidate_metrics_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--skip_file_check", action="store_true")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    generation = config["candidate_generation"]
    filtering = config["preference_filter"]
    candidate_metrics = args.candidate_metrics_path or filtering.get(
        "candidate_metrics_path",
        f"{generation['output_dir']}/per_candidate_metrics.csv",
    )
    summary = build_preference_pairs(
        candidate_metrics_path=candidate_metrics,
        output_dir=args.output_dir or filtering["output_dir"],
        reward_config=config.get("reward"),
        selection=filtering,
        prompts=(
            config.get("weather_prompts")
            if bool(generation.get("use_prompt", False))
            else {}
        ),
        require_image_files=not args.skip_file_check,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
