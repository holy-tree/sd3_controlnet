"""Fixed-lambda experiment planning and source-paired reporting; no model loading."""
from __future__ import annotations

import copy
import csv
import math
import os
import shutil
import subprocess
import sys
import tempfile
from decimal import Decimal
from pathlib import Path

import yaml

from .spatial_config import training_signature
from .texture_release_validation import (
    METRICS, SCHEMA, WEATHERS, digest, finite, paired_summary, read_json, saved_probe, write_json,
)

SWEEP_SCHEMA = "texture_lambda_sweep_v1"
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_FILES = (
    "scripts/train_dpo_sd3.py", "utils/randomness_check.py", "utils/pipeline_inference.py",
    "dpo/texture_release_runtime.py", "dpo/texture_diagnostics_runtime.py", "dpo/spatial.py",
    "dpo/spatial_config.py", "dpo/spatial_dataset.py", "dpo/texture_lambda_sweep.py",
    "dpo/texture_release_validation.py", "utils/metrics.py", "utils/evaluate_sd3.py",
)


def lambda_name(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("lambda_fidelity must be finite and positive")
    return "lambda_" + format(Decimal(str(value)).normalize(), "f").replace(".", "p")


def verify_files(files):
    for path, expected in files.items():
        if digest(path) != expected:
            raise ValueError("Pinned input changed: " + str(path))


def check_snapshot_files(snapshot):
    verify_files({snapshot["config_path"]: snapshot["config_sha256"],
                  str(Path(snapshot["checkpoint"]) / "dpo_resume.json"): snapshot["metadata_sha256"],
                  **snapshot["weight_files_sha256"]})


def materialized_config(snapshot):
    cfg = copy.deepcopy(snapshot["config"])
    train = cfg["training"]
    for name in ("reward", "preference_filter", "tail_risk"):
        cfg[name] = copy.deepcopy(train.get(name, cfg.get(name, {"enabled": False} if name == "tail_risk" else {})))
    cfg["spatial_dynamic"] = copy.deepcopy(snapshot["metadata"]["spatial_dynamic"]["settings"])
    resolution = cfg["spatial_dynamic"].pop("resolution")
    if resolution != train["resolution"]:
        raise ValueError("Checkpoint/config resolution mismatch")
    for name in ("reward", "preference_filter", "spatial_dynamic", "tail_risk"):
        train.pop(name, None)
    effective = {**train, **{name: cfg[name] for name in ("reward", "preference_filter", "spatial_dynamic", "tail_risk")}}
    if training_signature(effective) != snapshot["metadata"]["spatial_dynamic"]:
        raise ValueError("Historical effective objective cannot be reproduced by current code")
    return cfg


def comparable_config(cfg):
    """Optimization conditions, excluding disabled callback/report/output settings."""
    cfg = copy.deepcopy(cfg)
    train = cfg["training"]
    for key in list(train):
        if key.startswith("validation_") or key in (
            "run_checkpoint_validation", "output_dir", "logging_dir", "report_to",
            "reward", "preference_filter", "spatial_dynamic", "tail_risk",
        ):
            train.pop(key)
    return {key: cfg.get(key) for key in ("model", "weather_prompts", "reward", "preference_filter", "spatial_dynamic", "tail_risk", "training")}


def validate_arm(snapshot, config, source, steps):
    """Reject different seed/budget/reference/optimizer/objective/data on reuse."""
    actual = materialized_config(snapshot)
    if training_signature(snapshot["config"]["training"]) != snapshot["metadata"]["spatial_dynamic"]:
        raise ValueError("Saved runtime defaults no longer match this checkpoint")
    if actual["training"].get("run_checkpoint_validation", True) or actual["training"].get("resume_from_checkpoint"):
        raise ValueError("Sweep arms must start at SFT with automatic validation disabled")
    if comparable_config(actual) != comparable_config(config):
        raise ValueError("Saved arm differs from the planned optimization conditions")
    meta, original = snapshot["metadata"], source["metadata"]
    if meta.get("global_step") != steps or meta.get("num_processes") != 1:
        raise ValueError("Require the exact optimizer-step budget and single-rank checkpoint")
    for key in original:
        if key in ("global_step", "epoch", "next_batch_index", "spatial_dynamic"):
            continue
        if meta.get(key) != original[key]:
            raise ValueError("Checkpoint training/data provenance differs: " + key)


def safe_separate(path, protected):
    for old in protected:
        old = Path(old).expanduser().resolve()
        if path == old or old in path.parents or path in old.parents:
            raise ValueError("Use separate experiment directories: " + str(path) + " vs " + str(old))


def prepare_sweep(validation_plan, output, run_root, lambdas=(1000, 2000, 5000, 10000), steps=200, reuse=None, storage_mode="full"):
    if storage_mode not in ("full", "weights-only"):
        raise ValueError("Unknown sweep storage mode")
    validation_plan = Path(validation_plan).expanduser().resolve()
    cohort = read_json(validation_plan)
    if cohort.get("schema") != SCHEMA or cohort.get("role") not in ("dpo_holdout", "external_validation"):
        raise ValueError("First prepare a fixed DPO-disjoint validation_plan.json")
    verify_files({cohort["validation_config"]: cohort["validation_config_sha256"]})
    source = cohort["snapshots"]["fidelity"]
    check_snapshot_files(source)
    if source["metadata"].get("num_processes") != 1:
        raise ValueError("Sweep supports single-card historical training conditions")
    template = materialized_config(source)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= int(template["training"].get("lr_warmup_steps", 50)):
        raise ValueError("Optimizer-step budget must exceed the original warmup")
    values = [float(value) for value in lambdas]
    names = [lambda_name(value) for value in values]
    if not names or len(set(names)) != len(names):
        raise ValueError("Supply unique positive lambda values")
    reuse = {lambda_name(key): Path(value).expanduser().resolve() for key, value in (reuse or {}).items()}
    if set(reuse) - set(names):
        raise ValueError("Reused lambda must appear in --lambdas")
    output, run_root = Path(output).expanduser().resolve(), Path(run_root).expanduser().resolve()
    protected = [validation_plan.parent, *(s["config"]["training"]["output_dir"] for s in cohort["snapshots"].values()),
                 *(Path(s["config_path"]).parent for s in cohort["snapshots"].values()), *reuse.values()]
    safe_separate(output, [*protected, run_root])
    safe_separate(run_root, protected)
    if output.exists():
        raise FileExistsError(output)
    inputs = {}
    for key, expected in (("preference_manifest", source["metadata"]["preference_manifest_sha256"]),
                          ("gt_source_manifest", source["metadata"]["reward_provenance"]["gt_manifest_sha256"])):
        path = str(Path(template["training"][key]).expanduser().resolve())
        inputs[path] = expected
    verify_files(inputs)
    arms = []
    for name, value in zip(names, values):
        cfg = copy.deepcopy(template)
        cfg["spatial_dynamic"]["lambda_fidelity"] = value
        target = run_root / name
        cfg["training"].update(output_dir=str(target), max_train_steps=steps, checkpointing_steps=steps,
            resume_from_checkpoint=None, logging_dir=None, run_checkpoint_validation=False,
            validation_before_training=False, validation_eval_config=None)
        cfg.setdefault("evaluation", {})["output_dir"] = str(target / "evaluation")
        snapshot = None
        if name in reuse:
            directory = reuse[name]
            snapshot = saved_probe(directory.parent, directory.name, directory / f"checkpoint-{steps}", steps)
            validate_arm(snapshot, cfg, source, steps)
        elif target.exists():
            raise FileExistsError("Training output already exists: " + str(target))
        arms.append({"method": name, "lambda": value, "config": str(output / (name + ".yaml")),
                     "output_dir": str(target), "reused_snapshot": snapshot, "settings": cfg})
    reference = comparable_config(arms[0]["settings"])
    reference["spatial_dynamic"].pop("lambda_fidelity")
    for arm in arms:
        comparable = comparable_config(arm["settings"])
        comparable["spatial_dynamic"].pop("lambda_fidelity")
        if comparable != reference:
            raise ValueError("Sweep arms differ in more than lambda/output paths")
    output.mkdir(parents=True, exist_ok=False)
    for arm in arms:
        with Path(arm["config"]).open("x", encoding="utf-8") as handle:
            yaml.safe_dump(arm.pop("settings"), handle, sort_keys=False, allow_unicode=True)
        arm["config_sha256"] = digest(arm["config"])
    plan = {"schema": SWEEP_SCHEMA, "validation_plan": str(validation_plan), "validation_plan_sha256": digest(validation_plan),
            "source_snapshot": source, "input_files_sha256": inputs, "run_root": str(run_root), "steps": steps,
            "storage_mode": storage_mode,
            "arms": arms, "protocol_sha256": {relative: digest(ROOT / relative) for relative in PROTOCOL_FILES},
            "comparison": "same SFT initialization, fixed lambda, equal optimizer-step budget; raw checkpoint vs SFT"}
    write_json(output / "sweep_plan.json", plan)
    write_json(output / "storage_estimate.json", storage_estimate(plan))
    return plan


def load_sweep(path):
    path = Path(path).expanduser().resolve()
    plan = read_json(path)
    if plan.get("schema") != SWEEP_SCHEMA:
        raise ValueError("Unknown sweep schema")
    verify_files({plan["validation_plan"]: plan["validation_plan_sha256"], **plan["input_files_sha256"],
                  **{arm["config"]: arm["config_sha256"] for arm in plan["arms"]},
                  **{str(ROOT / relative): expected for relative, expected in plan["protocol_sha256"].items()}})
    cohort = read_json(plan["validation_plan"])
    verify_files({cohort["validation_config"]: cohort["validation_config_sha256"]})
    return path, plan


def arm_snapshot(arm, plan):
    if arm["reused_snapshot"]:
        snapshot = arm["reused_snapshot"]
        check_snapshot_files(snapshot)
    else:
        directory = Path(arm["output_dir"])
        snapshot = saved_probe(directory.parent, directory.name, directory / f"checkpoint-{plan['steps']}", plan["steps"])
    with Path(arm["config"]).open(encoding="utf-8") as handle:
        validate_arm(snapshot, yaml.safe_load(handle), plan["source_snapshot"], plan["steps"])
    return snapshot


def storage_estimate(plan, pending=None):
    checkpoint = Path(plan["source_snapshot"]["checkpoint"])
    if not (checkpoint / "dpo_resume.json").is_file():
        raise FileNotFoundError("Source checkpoint needed for the storage estimate: " + str(checkpoint))
    size = sum(p.stat().st_size for p in checkpoint.rglob("*") if p.is_file())
    raw_bytes = sum(p.stat().st_size for sub in ("controlnet", "ra_fusion") for p in (checkpoint / sub).rglob("*") if p.is_file())
    exported = raw_bytes
    if (checkpoint / "ema_state.pt").is_file():
        exported += (checkpoint / "ema_state.pt").stat().st_size
    if pending is None:
        pending = [arm for arm in plan["arms"] if not arm["reused_snapshot"]]
    parent = Path(plan["run_root"])
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    cohort = read_json(plan["validation_plan"])
    # Three lossless RGB outputs (LQ/GT/prediction) per observation, plus overhead.
    validation = len(cohort["sources"]) * len(cohort["seeds"]) * (len(plan["arms"]) + 1) * (9 * cohort["resolution"] ** 2 + 4096)
    mode = plan.get("storage_mode", "full")
    retained = raw_bytes + (checkpoint / "dpo_resume.json").stat().st_size + Path(plan["source_snapshot"]["config_path"]).stat().st_size + 1024 ** 2
    # The trainer still writes a FULL run first. Compact only after successful
    # completion, retaining all previous raw policies during the next run.
    training_peak = (max(0, len(pending) - 1) * retained + size + exported if pending else 0) if mode == "weights-only" else len(pending) * (size + exported)
    estimate = training_peak + validation + 1024 ** 3
    return {"pending_training_runs": len(pending), "checkpoint_bytes": size, "per_run_checkpoint_plus_export_bytes": size + exported,
            "storage_mode": mode, "raw_policy_bytes": raw_bytes, "retained_per_new_run_bytes": retained if mode == "weights-only" else size + exported,
            "estimated_training_peak_bytes": training_peak,
            "estimated_validation_bytes": validation, "estimated_additional_peak_bytes": estimate, "free_bytes": free,
            "filesystem_check_path": str(parent), "enough_space_estimate": free >= estimate,
            "note": ("Explicit weights-only mode: each NEW completed run retains raw policy/config/metadata/logs; optimizer/RNG/EMA and redundant exports are removed. Reused runs are never compacted. Peak includes one full run before cleanup. " if mode == "weights-only" else "Retains every arm with full training states and final exports. ") +
                    "Validation reserved conservatively on training disk; check evaluation disk separately if different."}


def atomic_json(path, value):
    """Replace a small workflow record without exposing a truncated JSON."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        write_json(temporary, value)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def ownership_record(plan_path, arm):
    return {"schema": "texture_sweep_owned_run_v1", "sweep_plan_sha256": digest(plan_path),
            "config_sha256": arm["config_sha256"], "output_dir": arm["output_dir"], "method": arm["method"]}


def claim_training_run(plan_path, arm):
    """Record ownership BEFORE starting a new output, never adopt an old run."""
    if arm["reused_snapshot"] or Path(arm["output_dir"]).exists():
        raise FileExistsError("Cannot claim an existing/reused training run")
    folder = Path(plan_path).parent / "training_ownership"
    folder.mkdir(exist_ok=True)
    path = folder / (arm["method"] + ".json")
    expected = ownership_record(plan_path, arm)
    if path.exists() and read_json(path) != expected:
        raise ValueError("Training ownership record differs")
    atomic_json(path, expected)


def compact_new_arm(plan_path, arm, plan):
    """Opt-in cleanup of ONLY this sweep's owned, completed NEW run.

    Raw checkpoint/config/metadata stay byte-identical. An atomic journal allows
    retry after interruption; reused probes and unknown files are never removed.
    """
    if plan.get("storage_mode", "full") != "weights-only" or arm["reused_snapshot"]:
        raise ValueError("Compaction requires explicit weights-only mode and a new arm")
    root = Path(arm["output_dir"])
    expected_root = Path(plan["run_root"]) / lambda_name(arm["lambda"])
    if arm["method"] != lambda_name(arm["lambda"]) or root != expected_root or root.resolve() != root or root.is_symlink():
        raise ValueError("Compaction output is outside the owned run")
    owner = Path(plan_path).parent / "training_ownership" / (arm["method"] + ".json")
    if not owner.is_file() or read_json(owner) != ownership_record(plan_path, arm):
        raise ValueError("Refuse to compact a run without this sweep's pre-training ownership record")
    summary = read_json(root / "training_summary.json")
    snapshot = arm_snapshot(arm, plan)
    if summary.get("global_step") != plan["steps"] or summary.get("spatial_dynamic") != snapshot["metadata"]["spatial_dynamic"]:
        raise ValueError("Training did not complete the planned budget/objective; no cleanup performed")
    checkpoint = Path(snapshot["checkpoint"])
    raw_files = {str(Path(path).relative_to(checkpoint)) for path in snapshot["weight_files_sha256"]}
    allowed = {"ema_state.pt"}
    allowed.update(raw_files)  # Redundant final policy exports at the run root.
    allowed.update(str(Path(checkpoint.name) / "ema" / relative) for relative in raw_files)
    allowed.update(str(Path(checkpoint.name) / name) for name in
                   ("optimizer.bin", "scheduler.bin", "random_states_0.pkl", "scaler.pt", "ema_state.pt"))

    def safe_file(relative):
        candidate = root / relative
        if candidate.resolve() == root or root not in candidate.resolve().parents:
            raise ValueError("Cleanup path escapes the owned run")
        current = candidate
        while current != root:
            if current.is_symlink():
                raise ValueError("Refuse cleanup through a symlink: " + str(current))
            current = current.parent
        if candidate.exists() and not candidate.is_file():
            raise ValueError("Expected a generated file: " + str(candidate))
        return candidate

    for path in snapshot["weight_files_sha256"]:
        safe_file(str(Path(path).relative_to(root)))
    marker = root / "weights_only.json"
    if marker.exists():
        journal = read_json(marker)
        if journal.get("ownership") != ownership_record(plan_path, arm):
            raise ValueError("Compaction journal belongs to another sweep")
        check_snapshot_files(journal["retained_snapshot"])
        if journal["retained_snapshot"] != snapshot:
            raise ValueError("Retained raw checkpoint changed")
        if journal.get("status") == "completed_weights_only":
            return journal
    else:
        removals = []
        for relative in sorted(allowed):
            candidate = safe_file(relative)
            if candidate.is_file():
                removals.append({"relative_path": relative, "bytes": candidate.stat().st_size, "sha256": digest(candidate)})
        journal = {"schema": "texture_sweep_weights_only_v1", "status": "compacting", "ownership": ownership_record(plan_path, arm),
                   "retained_snapshot": snapshot, "removals": removals, "optimizer_resume_supported": False}
        atomic_json(marker, journal)
    # Preflight the entire remaining deletion set before deleting any file.
    for item in journal["removals"]:
        if item["relative_path"] not in allowed:
            raise ValueError("Unrecognized cleanup journal path")
        candidate = safe_file(item["relative_path"])
        if candidate.exists() and (candidate.stat().st_size != item["bytes"] or digest(candidate) != item["sha256"]):
            raise ValueError("Generated training-state file changed; no further cleanup: " + str(candidate))
    for item in journal["removals"]:
        candidate = safe_file(item["relative_path"])
        if candidate.exists():
            candidate.unlink()
    check_snapshot_files(snapshot)
    journal.update(status="completed_weights_only", removed_file_bytes=sum(item["bytes"] for item in journal["removals"]))
    atomic_json(marker, journal)
    return journal


def pending_training_arms(plan):
    pending = []
    for arm in plan["arms"]:
        if arm["reused_snapshot"] or Path(arm["output_dir"]).exists():
            if not arm["reused_snapshot"] and not (Path(arm["output_dir"]) / f"checkpoint-{plan['steps']}" / "dpo_resume.json").is_file():
                raise FileExistsError("Incomplete training output: " + arm["output_dir"] + "; preserve/rename it before retrying. The sweep never silently resumes or overwrites it")
            arm_snapshot(arm, plan)  # A complete, matching arm can be reused on retry.
        else:
            pending.append(arm)
    return pending


def train_sweep(plan_path, gpu="0"):
    path, plan = load_sweep(plan_path)
    if not gpu.isdigit() or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Use one physical GPU index from a single-process shell")
    check_snapshot_files(plan["source_snapshot"])
    pending = pending_training_arms(plan)
    if plan.get("storage_mode") == "weights-only":
        for arm in plan["arms"]:
            if not arm["reused_snapshot"] and arm not in pending:
                compact_new_arm(path, arm, plan)
    space = storage_estimate(plan, pending)
    write_json(path.parent / "storage_estimate.json", space)
    if pending and not space["enough_space_estimate"]:
        raise RuntimeError(f"Need approximately {space['estimated_additional_peak_bytes']/1024**3:.2f} GiB additional space; free {space['free_bytes']/1024**3:.2f} GiB. Choose a larger --run-root when preparing; no results were deleted")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    for index, arm in enumerate(pending):
        space = storage_estimate(plan, pending[index:])
        if not space["enough_space_estimate"]:
            raise RuntimeError("Available disk fell below the remaining-run estimate; completed arms are preserved")
        print(f"Training {arm['method']}: {plan['steps']} optimizer steps from SFT", flush=True)
        claim_training_run(path, arm)
        subprocess.run([sys.executable, "-m", "scripts.train_dpo_sd3", "--config", arm["config"]], cwd=str(ROOT), env=env, check=True)
        arm_snapshot(arm, plan)
        if plan.get("storage_mode") == "weights-only":
            journal = compact_new_arm(path, arm, plan)
            print(f"{arm['method']}: retained raw evaluation weights; removed {journal['removed_file_bytes']/1024**3:.2f} GiB of generated state/exports", flush=True)
    print("All planned checkpoints available; training finished. Evaluation is a separate command.")


def evaluation_plan(plan_path):
    path, sweep = load_sweep(plan_path)
    destination = path.parent / "evaluation" / "validation_plan.json"
    snapshots = {arm["method"]: arm_snapshot(arm, sweep) for arm in sweep["arms"]}
    plan = copy.deepcopy(read_json(sweep["validation_plan"]))
    for key in ("frozen_inference_rows", "frozen_baseline_identity", "cached_baseline_evidence", "expected_pyiqa_version"):
        plan.pop(key, None)
    plan.update(snapshots=snapshots, methods=["sft", *snapshots], comparisons=[[method, "sft"] for method in snapshots],
                inference_arms=[{"snapshot": method, "methods": (["sft", method] if i == 0 else [method])} for i, method in enumerate(snapshots)],
                protocol_sha256=sweep["protocol_sha256"], sweep_plan_sha256=digest(path),
                comparison_note=sweep["comparison"])
    if destination.exists():
        if read_json(destination) != plan:
            raise ValueError("Evaluation plan/models changed; preserve the old evaluation directory")
    else:
        destination.parent.mkdir(parents=True, exist_ok=False)
        write_json(destination, plan)
    return destination


def table_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def pareto_methods(points):
    """Observed PSNR/MUSIQ means only; excludes missing/partial observations."""
    valid = {key: pair for key, pair in points.items() if all(finite(v) is not None for v in pair)}
    return [key for key, (p, m) in valid.items() if not any(q >= p and n >= m and (q > p or n > m) for other, (q, n) in valid.items() if other != key)]


def tradeoff_plot(path, points, labels):
    """Dependency-free SVG data plot, not an architecture diagram."""
    from html import escape
    points = {key: values for key, values in points.items() if all(finite(v) is not None for v in values)}
    all_points = [(0., 0.), *points.values()]
    xs, ys = zip(*all_points)
    def bounds(values):
        low, high = min(values), max(values)
        margin = max((high - low) * .15, .01)
        return low - margin, high + margin
    xl, xh = bounds(xs); yl, yh = bounds(ys)
    def x(value): return 90 + (value - xl) / (xh - xl) * 580
    def y(value): return 410 - (value - yl) / (yh - yl) * 340
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="800" height="500" viewBox="0 0 800 500">',
             '<rect width="800" height="500" fill="white"/>',
             '<g font-family="sans-serif" font-size="13" fill="#222">',
             '<text x="90" y="28">Fixed validation: mean deltas vs SFT (raw checkpoints)</text>']
    for i in range(6):
        px, py = xl + (xh - xl) * i / 5, yl + (yh - yl) * i / 5
        lines += [f'<path d="M{x(px):.2f} 70V410 M90 {y(py):.2f}H670" stroke="#eee"/>',
                  f'<text x="{x(px):.2f}" y="435" text-anchor="middle">{px:.3g}</text>',
                  f'<text x="80" y="{y(py):.2f}" text-anchor="end">{py:.3g}</text>']
    lines += [f'<path d="M{x(0):.2f} 70V410 M90 {y(0):.2f}H670" stroke="#999" stroke-dasharray="4 4"/>',
              '<text x="260" y="475">PSNR delta (dB; higher is better)</text>',
              '<text transform="translate(22 330) rotate(-90)">Whole-image MUSIQ delta (higher is better)</text>']
    for key, (p, m) in {"sft": (0., 0.), **points}.items():
        color = "#666" if key == "sft" else "#146ac2"
        lines += [f'<circle cx="{x(p):.2f}" cy="{y(m):.2f}" r="5" fill="{color}"/>',
                  f'<text x="{x(p)+8:.2f}" y="{y(m)-8:.2f}">{escape(labels.get(key,key))}</text>']
    lines += ['</g></svg>']
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def report_sweep(plan_path):
    path, sweep = load_sweep(plan_path)
    root = path.parent / "evaluation"
    eval_path = root / "validation_plan.json"
    plan, data = read_json(eval_path), read_json(root / "validation_results.json")
    if plan["sweep_plan_sha256"] != digest(path) or data["plan_sha256"] != digest(eval_path):
        raise ValueError("Sweep/evaluation evidence changed")
    if set(data["model_evidence"]) != {arm["method"] for arm in sweep["arms"]}:
        raise ValueError("Missing real-model evidence for one or more lambda arms")
    for evidence in data["model_evidence"].values():
        verify_files({evidence["path"]: evidence["sha256"]})
        state = read_json(evidence["path"])
        if not all(state.get(key) is True for key in ("policy_parameters_unchanged", "reference_unchanged", "buffers_unchanged")):
            raise ValueError("Evaluation changed model/reference/buffers")
    summaries, changes = paired_summary(data["rows"], plan)
    table_csv(root / "per_image_metrics.csv", data["rows"])
    table_csv(root / "paired_summary.csv", summaries)
    table_csv(root / "paired_source_deltas.csv", changes)
    seed_rows = []
    for seed in plan["seeds"]:
        subset, _ = paired_summary([r for r in data["rows"] if int(r["seed"]) == seed], {**plan, "seeds": [seed]})
        seed_rows.extend({"seed": seed, **row} for row in subset)
    table_csv(root / "per_seed_summary.csv", seed_rows)
    indexed = {(r["method"], r["scope"], r["metric"]): r for r in summaries}
    points, compact = {}, []
    for arm in sweep["arms"]:
        method = arm["method"]
        for scope in ("all", *WEATHERS):
            row = {"lambda": arm["lambda"], "method": method, "scope": scope, "steps": sweep["steps"]}
            for metric in METRICS:
                item = indexed[(method, scope, metric)]
                for field in ("method_mean", "reference_mean", "delta_mean", "ci_low", "ci_high", "n_sources"):
                    row[metric + "_" + field] = item[field]
            compact.append(row)
        p, m = (indexed[(method, "all", key)] for key in ("psnr", "musiq"))
        if p["n_sources"] == m["n_sources"] == len(plan["sources"]):
            points[method] = (p["delta_mean"], m["delta_mean"])
    front = pareto_methods({"sft": (0., 0.), **points})
    table_csv(root / "lambda_comparison.csv", compact)
    tradeoff_plot(root / "psnr_musiq_tradeoff.svg", points, {arm["method"]: "lambda=" + f"{arm['lambda']:g}" for arm in sweep["arms"]})
    def fmt(value): return "N/A" if value is None else f"{value:+.6g}"
    lines = ["# 固定 lambda 等预算比较", "", f"- 每组 {sweep['steps']} 个 optimizer steps；固定 SFT 初始化、训练 seed、偏好、GT、beta、学习率和有效 batch。",
             f"- 存储模式：`{sweep.get('storage_mode','full')}`；weights-only 新组只保留 raw 评估权重，不能完整恢复 optimizer 续训。",
             f"- 数据角色：`{plan['role']}`；{len(plan['sources'])} 张源图 × {len(plan['seeds'])} 个固定推理 seed；SFT 仅生成一套共同输出。",
             "- 下面差值均为候选减 SFT；PSNR/SSIM/美学越大越好，LPIPS/DISTS 越小越好。",
             "- DPO 留出不代表 SFT 未见；此处是选参验证，最终 test 应在模型冻结后单独进行。", "",
             "|lambda|范围|PSNR差(dB)|95%配对CI|整图MUSIQ差|SSIM差|LPIPS差|DISTS差|CLIP-IQA差|NIMA差|TOPIQ-NR差|",
             "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in compact:
        metric_values = [fmt(row[key + "_delta_mean"]) for key in ("musiq", "ssim", "lpips", "dists", "clipiqa", "nima", "topiq_nr")]
        lines.append(f"|{row['lambda']:g}|{row['scope']}|{fmt(row['psnr_delta_mean'])}|[{fmt(row['psnr_ci_low'])}, {fmt(row['psnr_ci_high'])}]|" + "|".join(metric_values) + "|")
    lines += ["", "![PSNR/MUSIQ折中](psnr_musiq_tradeoff.svg)", "", "## 如何判断", "",
              "- 均值 PSNR/MUSIQ 的非支配点（包含 SFT）：`" + ", ".join(front) + "`。这只是当前预算下观测到的 Pareto 前沿，不自动决定最佳 lambda。",
              "- CSV 同时保存各方法/SFT绝对均值、逐图指标、源图内平均 seed 的配对差、95% CI、中位数、p05/p95、退化比例及每 seed 复查。缺失指标不会填零，`n_sources` 应等于该范围源图数。",
              "- CI 以源图为单位，先平均同图多个 seed，总体按天气分层 bootstrap；属于逐比较区间，未作多重比较校正，不能作为选择后泛化保证。",
              "- 先看各天气保真退化和最差源图，再看美学收益；较大 lambda 是否更好由实际结果决定，不从损失权重大小直接推断。",
              "- 本工具不根据验证分数改变训练权重、不自动续训；200步筛选也不证明长期收敛或每图保真不下降。"]
    (root / "lambda_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(root / "lambda_summary.json", {"sweep_plan_sha256": digest(path), "results_sha256": digest(root / "validation_results.json"),
               "observed_psnr_musiq_pareto": front, "comparison": compact})
    return root / "lambda_report.md"
