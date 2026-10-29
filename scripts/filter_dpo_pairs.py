"""Filter offline SD3 candidates into preference pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dpo.filter_pairs import build_preference_pairs
from dpo.tail_risk import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter SD3 restoration preference pairs")
    parser.add_argument("--config", default="./config/dpo_sd3.yaml")
    parser.add_argument("--candidate_metrics_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--skip_file_check", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if (config.get("reward") or {}).get("mode") == "dynamic_texture":
        if args.candidate_metrics_path or args.output_dir or args.skip_file_check:
            raise ValueError("Use explicit versioned texture config paths; CLI legacy overrides are unsupported")
        from dpo.texture_preparation import build_texture_preferences
        print(json.dumps(build_texture_preferences(config), indent=2, ensure_ascii=False))
        return
    generation = config["candidate_generation"]
    filtering = config["preference_filter"]
    candidate_metrics = args.candidate_metrics_path or filtering.get(
        "candidate_metrics_path",
        f"{generation['output_dir']}/per_candidate_metrics.csv",
    )
    target = Path(candidate_metrics).expanduser().resolve()
    if (not target.is_file() and not args.candidate_metrics_path
            and (config.get("reward") or {}).get("mode") in {"static", "dynamic"}):
        source = filtering.get("source_metrics_path")
        if not source or not Path(source).expanduser().is_file():
            raise FileNotFoundError(
                f"Shared FR/NR scores are missing: {target}. Cannot prepare from source_metrics_path={source!r}. "
                "Set preference_filter.source_metrics_path to the existing aesthetic candidate CSV "
                "containing MUSIQ/CLIP-IQA/NIMA/PSNR/SSIM/LPIPS scores. No model inference will be run."
            )
        from scripts.rescore_dpo_candidates import main as prepare_scores

        print(f"[pairs] auto-preparing shared FR/NR scores from saved metrics: {source} -> {target}")
        prepare_scores([
            "--config", args.config,
            "--input_csv", str(Path(source).expanduser().resolve()),
            "--output_csv", str(target),
            "--normalize_only",
        ])
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
    concise = {key: value for key, value in summary.items() if key not in {
        "evaluated_pairs_per_group", "normalization_meta", "candidate_policy",
    }}
    concise["summary_file"] = str(Path(summary["manifest_path"]).parent / "preference_summary.json")
    print(json.dumps(concise, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
