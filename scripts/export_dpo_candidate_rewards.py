"""Export existing saved metric rewards, without IQA or metric normalization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dpo.tail_risk import export_candidate_rewards, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/dpo_sd3_tail_risk.yaml")
    parser.add_argument("--input_csv", help="Default: preference_filter.candidate_metrics_path")
    parser.add_argument("--output_csv", help="Default: tail_risk.candidate_reward_file (must be new)")
    args = parser.parse_args()
    metadata = export_candidate_rewards(load_config(args.config), args.input_csv, args.output_csv)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
