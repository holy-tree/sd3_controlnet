"""Weight original offline pairs using full TRAIN candidate reward quantiles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dpo.tail_risk import load_config, weight_tail_risk


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/dpo_sd3_tail_risk.yaml")
    parser.add_argument("--input_pairs", help="Default: preference_filter.output_dir/preference_pairs.jsonl")
    parser.add_argument("--output_dir", help="Default: training.preference_manifest parent; must be distinct/new")
    args = parser.parse_args()
    stats = weight_tail_risk(load_config(args.config), args.input_pairs, args.output_dir)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
