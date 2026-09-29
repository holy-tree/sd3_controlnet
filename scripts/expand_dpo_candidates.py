"""Incrementally expand saved DPO candidates without touching valid outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dpo.candidate_expansion import (  # noqa: E402
    CANDIDATE_FIELDS,
    FAILED_FIELDS,
    MANIFEST_FIELDS,
    PLAN_FIELDS,
    atomic_csv,
    atomic_json,
    backup_paths,
    build_expansion_plan,
    candidate_key,
    index_rows,
    manifest_rows_from_plan,
    read_csv,
    resolve_guidance_scales,
    valid_candidate_image,
    weather_action_summary,
)
from dpo.filter_pairs import build_preference_pairs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume-safe, non-destructive expansion of DPO candidate groups"
    )
    parser.add_argument("--config", default="./config/dpo_sd3.yaml")
    parser.add_argument("--target_candidates_per_group", type=int, default=None)
    parser.add_argument("--base_seed", type=int, default=None)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--verify_existing", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--overwrite_invalid", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--metrics_batch_size", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _configured(cli_value, config: Mapping, key: str, default):
    return cli_value if cli_value is not None else config.get(key, default)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _path_equal(left: object, right: object) -> bool:
    return Path(str(left)).expanduser().resolve() == Path(str(right)).expanduser().resolve()


def _validate_policy(
    summary: Mapping,
    generation: Mapping,
    model: Mapping,
    base_seed: int,
) -> None:
    settings = dict(summary.get("settings") or {})
    policy = dict(summary.get("candidate_policy") or {})
    if not policy:
        raise ValueError("summary.json lacks candidate_policy; expansion provenance is unsafe")
    recorded_seed = settings.get("candidate_seed")
    if recorded_seed is not None and int(recorded_seed) != base_seed:
        raise ValueError(
            f"Base-seed conflict: summary={recorded_seed}, configured={base_seed}"
        )

    for key in ("pretrained_model_name_or_path", "revision", "variant"):
        desired = model.get(key)
        recorded = policy.get(key)
        if desired is not None and recorded is not None and str(desired) != str(recorded):
            raise ValueError(f"Candidate-policy conflict for {key}: {recorded!r} != {desired!r}")
    for key in ("controlnet_model_path", "ra_fusion_path"):
        desired = model.get(key)
        recorded = policy.get(key)
        if desired and recorded and not _path_equal(desired, recorded):
            raise ValueError(f"Candidate-policy path conflict for {key}: {recorded!r} != {desired!r}")

    comparisons = {
        "controlnet_conditioning_scale": generation.get(
            "controlnet_conditioning_scale", model.get("controlnet_conditioning_scale", 1.0)
        ),
        "ra_fusion_scale": generation.get("ra_fusion_scale", model.get("ra_fusion_scale")),
        "ra_spatial_gate_scale": generation.get(
            "ra_spatial_gate_scale", model.get("ra_spatial_gate_scale")
        ),
        "ra_how_token_scale": generation.get(
            "ra_how_token_scale", model.get("ra_how_token_scale")
        ),
    }
    for key, desired in comparisons.items():
        recorded = policy.get(key)
        if desired is None or recorded is None:
            continue
        if not math.isclose(float(desired), float(recorded), abs_tol=1e-9):
            raise ValueError(f"Candidate-policy conflict for {key}: {recorded} != {desired}")


def _pipeline_config(config: Mapping, generation: Mapping, model: Mapping) -> dict:
    from utils.evaluate_sd3 import load_config

    eval_config = load_config(str(generation.get("eval_config", "./config/eval_sd3.yaml")))
    for key in (
        "pretrained_model_name_or_path",
        "controlnet_model_path",
        "ra_fusion_path",
        "revision",
        "variant",
    ):
        if model.get(key) is not None:
            eval_config[key] = model[key]
    for key in (
        "strength",
        "controlnet_conditioning_scale",
        "ra_fusion_scale",
        "ra_spatial_gate_scale",
        "ra_how_token_scale",
        "use_ra_fusion",
        "use_prompt",
    ):
        value = generation.get(key, model.get(key))
        if value is not None:
            eval_config[key] = value
    if generation.get("num_inference_steps") is not None:
        eval_config["num_inference_steps"] = generation["num_inference_steps"]
    eval_config["weather_prompts"] = dict(config.get("weather_prompts") or {})
    eval_config["load_transformer_lora"] = False
    return eval_config


def _atomic_image(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    image.save(temporary)
    with Image.open(temporary) as check:
        check.load()
        if check.mode != "RGB":
            raise ValueError(f"Generated candidate must be RGB: {temporary}")
    os.replace(temporary, path)


def _manifest_fields(rows: Sequence[Mapping]) -> list[str]:
    fields = list(MANIFEST_FIELDS)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def _write_manifest(path: Path, rows_by_key: Mapping[tuple[str, int], Mapping]) -> None:
    rows = [rows_by_key[key] for key in sorted(rows_by_key)]
    atomic_csv(path, rows, _manifest_fields(rows))


def _failure(item: Mapping, stage: str, error: Exception | str) -> dict:
    return {
        "group_id": item["group_id"],
        "weather": item["weather"],
        "subdataset": item["subdataset"],
        "global_index": item["global_index"],
        "candidate_index": item["candidate_index"],
        "stage": stage,
        "error": str(error),
    }


def _generate_candidates(
    tasks: Sequence[Mapping],
    representatives: Mapping[str, Mapping],
    manifest_path: Path,
    manifest_by_key: dict[tuple[str, int], dict],
    pipeline_config: dict,
    batch_size: int,
) -> tuple[dict[tuple[str, int], dict], list[dict]]:
    import torch

    from utils.randomness_check import (
        _get_lpips_model,
        build_preprocess,
        infer_latent_shape,
        load_image_batch,
        lpips_batch,
        output_checksum,
        psnr_batch,
        run_with_initial_noise,
        setup_pipeline,
        ssim_batch,
        tensor_to_pil,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    if pipeline_config.get("mixed_precision") == "fp16":
        dtype = torch.float16
    elif pipeline_config.get("mixed_precision") == "bf16":
        dtype = torch.bfloat16
    resolution = int(pipeline_config.get("resolution", 512))
    strength = float(pipeline_config.get("strength", 1.0))
    inference_steps = int(pipeline_config.get("num_inference_steps", 30))
    use_ra_fusion = bool(pipeline_config.get("use_ra_fusion", False))
    pipeline = setup_pipeline(
        pipeline_config,
        dtype,
        device,
        pipeline_config.get("ra_fusion_scale"),
        use_ra_fusion,
        pipeline_config.get("ra_spatial_gate_scale"),
        pipeline_config.get("ra_how_token_scale"),
    )
    if use_ra_fusion and type(pipeline.transformer).__name__ != "RAFusionSD3Transformer2DModel":
        raise RuntimeError("RA Fusion is enabled, but the transformer is not RA-aware")
    preprocess = build_preprocess(resolution)
    first = representatives[str(tasks[0]["group_id"])]
    first_lq = preprocess(Image.open(first["lq_path"]).convert("RGB"))
    latent_shape = infer_latent_shape(
        pipeline, Image.fromarray((first_lq.numpy().transpose(1, 2, 0) * 255).round().astype("uint8")),
        resolution, device,
    )
    try:
        lpips_model = _get_lpips_model(
            pipeline_config.get("lpips_net", "alex"), device=device
        )
    except Exception as error:
        raise RuntimeError("LPIPS is required for candidate expansion") from error

    grouped = defaultdict(list)
    for task in tasks:
        representative = representatives[str(task["group_id"])]
        prompt = str(representative.get("prompt", ""))
        grouped[(float(task["guidance_scale"]), prompt)].append(task)

    completed = {}
    failures = []
    for (guidance_scale, prompt), group_tasks in grouped.items():
        for start in range(0, len(group_tasks), batch_size):
            chunk = group_tasks[start : start + batch_size]
            records = [dict(representatives[str(item["group_id"])]) for item in chunk]
            try:
                lq_pils, _, gt_batch = load_image_batch(records, preprocess, device)
                noises = []
                for item in chunk:
                    generator = torch.Generator(device="cpu").manual_seed(
                        int(item["candidate_seed"])
                    )
                    noises.append(torch.randn(latent_shape, generator=generator))
                run_config = dict(pipeline_config)
                run_config["guidance_scale"] = guidance_scale
                predictions = run_with_initial_noise(
                    pipeline,
                    run_config,
                    device,
                    dtype,
                    lq_pils,
                    prompt,
                    torch.stack(noises),
                    strength,
                    inference_steps,
                    use_ra_fusion,
                )
                psnrs = psnr_batch(predictions, gt_batch)
                ssims = ssim_batch(predictions, gt_batch)
                lpips_values = lpips_batch(
                    lpips_model, predictions, gt_batch, device, dtype
                )
            except Exception as error:
                failures.extend(_failure(item, "generation", error) for item in chunk)
                continue

            for offset, item in enumerate(chunk):
                key = (str(item["group_id"]), int(item["candidate_index"]))
                representative = representatives[str(item["group_id"])]
                try:
                    candidate_path = Path(str(item["candidate_path"])).expanduser()
                    _atomic_image(candidate_path, tensor_to_pil(predictions[offset]))
                    row = {
                        field: representative.get(field, "")
                        for field in CANDIDATE_FIELDS
                    }
                    row.update({
                        "gt_path": representative["gt_path"],
                        "lq_path": representative["lq_path"],
                        "weather": representative["weather"],
                        "subdataset": representative.get("subdataset", representative["weather"]),
                        "pair_id": representative.get("pair_id", Path(representative["lq_path"]).stem),
                        "global_index": item["global_index"],
                        "candidate_index": item["candidate_index"],
                        "candidate_seed": item["candidate_seed"],
                        "noise_index": item["candidate_index"],
                        "guidance_scale": guidance_scale,
                        "psnr": psnrs[offset],
                        "ssim": ssims[offset],
                        "lpips": lpips_values[offset],
                        "prompt": prompt,
                        "candidate_path": str(candidate_path),
                        "output_checksum_sha256": output_checksum(predictions[offset]),
                    })
                    manifest_row = {"group_id": item["group_id"], "status": "complete", **row}
                    manifest_by_key[key] = manifest_row
                    _write_manifest(manifest_path, manifest_by_key)
                    completed[key] = row
                except Exception as error:
                    failures.append(_failure(item, "save", error))
    return completed, failures


def _stage_rescore_and_pairs(
    config_path: Path,
    config: Mapping,
    base_csv: Path,
    aesthetic_csv: Path,
    preference_dir: Path,
    metrics_batch_size: int,
) -> None:
    temporary_csv = aesthetic_csv.with_name(
        f".{aesthetic_csv.stem}.{os.getpid()}.tmp.csv"
    )
    temporary_stats = temporary_csv.with_name(
        f"{temporary_csv.stem}_normalization.json"
    )
    final_stats = aesthetic_csv.with_name(f"{aesthetic_csv.stem}_normalization.json")
    temporary_pairs = preference_dir.with_name(f".{preference_dir.name}.{os.getpid()}.tmp")
    if temporary_pairs.exists():
        shutil.rmtree(temporary_pairs)
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.rescore_dpo_candidates",
                "--input_csv",
                str(base_csv),
                "--output_csv",
                str(temporary_csv),
                "--batch_size",
                str(metrics_batch_size),
                "--overwrite",
            ],
            cwd=ROOT,
            check=True,
        )
        generation = config["candidate_generation"]
        filtering = config["preference_filter"]
        build_preference_pairs(
            candidate_metrics_path=temporary_csv,
            output_dir=temporary_pairs,
            reward_config=config.get("reward"),
            selection=filtering,
            prompts=(config.get("weather_prompts") if generation.get("use_prompt") else {}),
            require_image_files=True,
        )

        normalization = _load_json(temporary_stats)
        normalization["source_csv"] = str(base_csv)
        normalization["output_csv"] = str(aesthetic_csv)
        atomic_json(temporary_stats, normalization)
        preference_summary_path = temporary_pairs / "preference_summary.json"
        preference_summary = _load_json(preference_summary_path)
        preference_summary["candidate_metrics_path"] = str(aesthetic_csv)
        preference_summary["manifest_path"] = str(preference_dir / "preference_pairs.jsonl")
        atomic_json(preference_summary_path, preference_summary)

        aesthetic_csv.parent.mkdir(parents=True, exist_ok=True)
        preference_dir.mkdir(parents=True, exist_ok=True)
        os.replace(temporary_csv, aesthetic_csv)
        os.replace(temporary_stats, final_stats)
        for name in ("preference_pairs.jsonl", "preference_pairs.csv", "preference_summary.json"):
            os.replace(temporary_pairs / name, preference_dir / name)
    finally:
        temporary_csv.unlink(missing_ok=True)
        temporary_stats.unlink(missing_ok=True)
        if temporary_pairs.exists():
            shutil.rmtree(temporary_pairs)


def _update_summary(
    summary_path: Path,
    summary: dict,
    target_count: int,
    guidance_scales: Sequence[float],
    base_seed: int,
    plan_summary: Mapping,
) -> None:
    settings = summary.setdefault("settings", {})
    policy = summary.setdefault("candidate_policy", {})
    settings["candidate_seed"] = base_seed
    settings["num_candidates_per_image"] = target_count
    settings["candidate_guidance_scales"] = list(guidance_scales)
    policy["candidate_guidance_scales"] = list(guidance_scales)
    summary["candidate_expansion"] = {
        "target_candidates_per_group": target_count,
        "base_seed": base_seed,
        "guidance_scales": list(guidance_scales),
        "weather_summary": dict(plan_summary),
    }
    atomic_json(summary_path, summary)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    generation = config["candidate_generation"]
    expansion = dict(generation.get("expansion") or {})
    target_count = int(_configured(
        args.target_candidates_per_group,
        expansion,
        "target_candidates_per_group",
        12,
    ))
    base_seed = int(_configured(args.base_seed, expansion, "base_seed", generation.get("seed", 20240805)))
    resume = bool(_configured(args.resume, expansion, "resume", True))
    verify_existing = bool(_configured(args.verify_existing, expansion, "verify_existing", True))
    overwrite_invalid = bool(_configured(args.overwrite_invalid, expansion, "overwrite_invalid", False))
    batch_size = int(_configured(args.batch_size, expansion, "batch_size", 1))
    metrics_batch_size = int(_configured(args.metrics_batch_size, expansion, "metrics_batch_size", 8))
    if batch_size <= 0 or metrics_batch_size <= 0:
        raise ValueError("batch_size and metrics_batch_size must be positive")
    guidance_scales = resolve_guidance_scales(
        generation.get("candidate_guidance_scales") or [],
        target_count,
        expansion.get("guidance_scales"),
    )

    output_dir = Path(generation["output_dir"]).expanduser().resolve()
    base_csv = output_dir / "per_candidate_metrics.csv"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "candidate_seed_manifest.csv"
    plan_path = output_dir / "expansion_plan.csv"
    failed_path = output_dir / "failed_groups.csv"
    if not base_csv.is_file() or not summary_path.is_file():
        raise FileNotFoundError(
            f"Expansion requires existing {base_csv.name} and {summary_path.name} in {output_dir}"
        )
    csv_rows, csv_fields = read_csv(base_csv)
    manifest_rows = []
    if manifest_path.is_file() and resume:
        manifest_rows, _ = read_csv(manifest_path)
    summary = _load_json(summary_path)
    _validate_policy(summary, generation, config["model"], base_seed)
    plan, complete_rows, representatives = build_expansion_plan(
        csv_rows,
        manifest_rows,
        target_count,
        base_seed,
        guidance_scales,
        verify_existing,
        overwrite_invalid,
    )
    plan_summary = weather_action_summary(plan)
    print(json.dumps(plan_summary, indent=2, ensure_ascii=False))
    if args.dry_run:
        print(f"[expand] dry run: {sum(row['action'] != 'keep' for row in plan)} slots need work")
        return

    atomic_csv(plan_path, plan, PLAN_FIELDS)
    blocked = [row for row in plan if row["action"] == "blocked"]
    if blocked:
        failures = [_failure(row, "validation", row["reason"]) for row in blocked]
        atomic_csv(failed_path, failures, FAILED_FIELDS)
        raise RuntimeError(
            f"{len(blocked)} invalid candidates are blocked; rerun with --overwrite_invalid"
        )

    initial_manifest = manifest_rows_from_plan(plan, complete_rows)
    manifest_by_key = index_rows(initial_manifest, "seed manifest")
    _write_manifest(manifest_path, manifest_by_key)
    tasks = [row for row in plan if row["action"] in {"generate", "repair"}]
    generated = {}
    failures = []
    if tasks:
        generated, failures = _generate_candidates(
            tasks,
            representatives,
            manifest_path,
            manifest_by_key,
            _pipeline_config(config, generation, config["model"]),
            batch_size,
        )
    if failures:
        atomic_csv(failed_path, failures, FAILED_FIELDS)
    else:
        failed_path.unlink(missing_ok=True)

    merged = dict(complete_rows)
    merged.update(generated)
    for key, row in manifest_by_key.items():
        if row.get("status") in {"existing", "complete"}:
            valid, _ = valid_candidate_image(row, verify_checksum=True)
            if valid:
                merged[key] = {field: row.get(field, "") for field in CANDIDATE_FIELDS}
    missing = [
        (row["group_id"], int(row["candidate_index"]))
        for row in plan
        if (row["group_id"], int(row["candidate_index"])) not in merged
    ]

    backup_targets = [base_csv, summary_path]
    aesthetic_csv = Path(
        config["preference_filter"].get(
            "candidate_metrics_path", output_dir / "per_candidate_metrics_aesthetic.csv"
        )
    ).expanduser().resolve()
    preference_dir = Path(config["preference_filter"]["output_dir"]).expanduser().resolve()
    backup_targets.extend([
        aesthetic_csv,
        aesthetic_csv.with_name(f"{aesthetic_csv.stem}_normalization.json"),
        preference_dir,
    ])
    backup_root = backup_paths(backup_targets)
    if backup_root:
        print(f"[expand] backup -> {backup_root}")

    merged_fields = list(csv_fields)
    for field in CANDIDATE_FIELDS:
        if field not in merged_fields:
            merged_fields.append(field)
    merged_rows = [merged[key] for key in sorted(merged)]
    atomic_csv(base_csv, merged_rows, merged_fields)
    if failures or missing:
        raise RuntimeError(
            f"Expansion incomplete: failures={len(failures)}, missing={len(missing)}; "
            "successful candidates were checkpointed for --resume"
        )

    _update_summary(
        summary_path, summary, target_count, guidance_scales, base_seed, plan_summary
    )
    _stage_rescore_and_pairs(
        config_path,
        config,
        base_csv,
        aesthetic_csv,
        preference_dir,
        metrics_batch_size,
    )
    print(
        f"[expand] complete: groups={len(representatives)}, candidates={len(merged_rows)}, "
        f"generated_or_repaired={len(tasks)}"
    )


if __name__ == "__main__":
    main()
