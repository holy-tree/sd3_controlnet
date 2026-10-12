"""Generate offline candidates by forwarding DPO YAML values to randomness_check."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dpo.candidate_sharding import (
    SCHEMA_VERSION, _fingerprint, load_generation_plan, load_group_cache,
    merge_candidate_shards, shard_output_dir, shard_records,
    validate_shard_completion,
)
from dpo.tail_risk import load_config


class CandidateProgress:
    """Count atomic group journals, without repeatedly reading candidate PNGs."""

    def __init__(self, output_root, plan, gpu_ids=None, verify_existing=False):
        self.root, self.plan = Path(output_root), plan
        if gpu_ids is not None and len(gpu_ids) != plan["num_shards"]:
            raise ValueError("Progress GPU labels must match the frozen shard count")
        self.records = [{int(row["global_index"]): row for row in shard_records(plan, i)}
                        for i in range(plan["num_shards"])]
        self.labels = [f"GPU{gpu_ids[i]}" if gpu_ids else f"shard{i}"
                       for i in range(plan["num_shards"])]
        self.completed = [set() for _ in self.records]
        self.signatures = {}
        if verify_existing:
            existing = [(i, index, row) for i, rows in enumerate(self.records)
                        for index, row in rows.items()
                        if (shard_output_dir(self.root, i) / "group_cache" / f"{index}.json").is_file()]
            for i, index, row in tqdm(existing, desc="Verify cached source groups", unit="group", leave=False):
                path = shard_output_dir(self.root, i) / "group_cache" / f"{index}.json"
                signature = self.file_signature(path)
                if signature is not None:
                    self.signatures[(i, index)] = signature
                    if load_group_cache(shard_output_dir(self.root, i), row, plan) is not None:
                        self.completed[i].add(index)
        initial = sum(map(len, self.completed))
        self.bar = tqdm(total=len(plan["sample_records"]), initial=initial,
                        desc="Generate source groups", unit="group", dynamic_ncols=True, mininterval=2)
        self.refresh()

    @staticmethod
    def file_signature(path):
        try:
            stat = path.stat()
            return stat.st_mtime_ns, stat.st_size, stat.st_ino
        except OSError:
            return None

    def refresh(self):
        for i, records in enumerate(self.records):
            directory = shard_output_dir(self.root, i) / "group_cache"
            present = set()
            if directory.is_dir():
                for path in directory.glob("*.json"):
                    if not path.stem.isdecimal():
                        continue
                    index = int(path.stem)
                    if index not in records or path.name != f"{index}.json":
                        continue
                    present.add(index)
                    signature = self.file_signature(path)
                    if signature is None or self.signatures.get((i, index)) == signature:
                        continue
                    self.signatures[(i, index)] = signature
                    self.completed[i].discard(index)
                    try:
                        payload = json.loads(path.read_text(encoding="utf-8"))
                        result = payload.get("result", {})
                        if (payload.get("schema_version") == SCHEMA_VERSION
                                and payload.get("plan_fingerprint") == self.plan["fingerprint"]
                                and payload.get("global_index") == index
                                and result.get("status") in {"retained", "rejected"}
                                and payload.get("fingerprint") == _fingerprint({
                                    key: value for key, value in payload.items() if key != "fingerprint"
                                })):
                            self.completed[i].add(index)
                    except (OSError, ValueError, TypeError, AttributeError):
                        pass
            self.completed[i].intersection_update(present)
            for index in records.keys() - present:
                self.signatures.pop((i, index), None)
        total = sum(map(len, self.completed))
        if total != self.bar.n:
            self.bar.update(total - self.bar.n)
        status = "loading/first group" if not total else (
            "checking/merging" if total == len(self.plan["sample_records"]) else "generating"
        )
        self.bar.set_postfix({**{label: f"{len(done)}/{len(rows)}" for label, done, rows in
                               zip(self.labels, self.completed, self.records)}, "stage": status})

    def close(self):
        self.bar.close()


def monitor_candidates(output_root, gpu_ids=None):
    """Observe an existing coordinator without launching or stopping any worker."""
    root = Path(output_root)
    plan = load_generation_plan(root / "generation_plan.json")
    progress = CandidateProgress(root, plan, gpu_ids, verify_existing=True)
    lock = root.parent / f".{root.name}.generation.lock"
    try:
        while True:
            progress.refresh()
            try:
                complete = json.loads((root / "COMPLETE.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                complete = {}
            if complete.get("plan_fingerprint") == plan["fingerprint"]:
                print("[candidate] merged generation complete", flush=True)
                return
            if not lock.is_file():
                print("[candidate] coordinator is not active and merged COMPLETE is absent; check shard logs", flush=True)
                return
            time.sleep(2)
    except KeyboardInterrupt:
        print("[candidate] monitor stopped; generation workers are not affected", flush=True)
    finally:
        progress.close()


@contextmanager
def generation_lock(output_root: Path):
    # Keep the lock outside the root so a fresh plan still requires an empty root.
    if output_root.resolve() != output_root or output_root.is_symlink():
        raise ValueError(f"Use a canonical, non-symlink candidate output directory: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    path = output_root.parent / f".{output_root.name}.generation.lock"
    try:
        handle = path.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise RuntimeError(
            f"Candidate generation lock exists: {path}. Check for an active job; "
            "remove the lock only after confirming its recorded process has stopped."
        ) from error
    try:
        with handle:
            json.dump({"pid": os.getpid(), "output_dir": str(output_root)}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        yield
    finally:
        path.unlink(missing_ok=True)


def generate_shards(command: list[str], output_root: Path, gpu_ids: list[str], resume: bool) -> dict:
    count = len(gpu_ids)
    extra = ["--resume"] if resume else []
    with generation_lock(output_root):
        plan_command = command + ["--plan_only", "--num_shards", str(count), *extra]
        print("[candidate] preparing global source plan", flush=True)
        subprocess.run(plan_command, cwd=ROOT, check=True)
        plan_path = output_root / "generation_plan.json"
        plan = load_generation_plan(plan_path)
        if plan["num_shards"] != count:
            raise ValueError("Generation plan shard count differs from requested GPUs")
        print(f"[candidate] source groups={len(plan['sample_records'])}; "
              f"groups per GPU={[len(shard_records(plan, i)) for i in range(count)]}", flush=True)
        running, handles = [], []
        progress = CandidateProgress(output_root, plan, gpu_ids, verify_existing=resume)
        try:
            for index, gpu in enumerate(gpu_ids):
                if resume and validate_shard_completion(output_root, plan, index):
                    print(f"[candidate] shard={index} complete and verified; reusing", flush=True)
                    continue
                # A stale final marker must not claim completion during repair.
                (output_root / "COMPLETE.json").unlink(missing_ok=True)
                environment = dict(os.environ)
                for name in ("LOCAL_RANK", "RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
                    environment.pop(name, None)
                environment.update(CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1")
                log_path = output_root / f"shard-{index:04d}.log"
                handle = log_path.open("a", encoding="utf-8")
                handles.append(handle)
                worker = command + [
                    "--generation_plan", str(plan_path), "--num_shards", str(count),
                    "--shard_index", str(index), *extra,
                ]
                print(f"[candidate] GPU={gpu} shard={index} log={log_path}", flush=True)
                process = subprocess.Popen(
                    worker, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT,
                )
                running.append((index, process))
            pending = list(running)
            while pending:
                progress.refresh()
                remaining = []
                for index, process in pending:
                    code = process.poll()
                    if code is None:
                        remaining.append((index, process))
                    elif code != 0:
                        raise RuntimeError(
                            f"Candidate shard {index} failed with exit code {code}; "
                            f"see {output_root / f'shard-{index:04d}.log'}. "
                            "No merged completion was published; retry with --resume."
                        )
                    else:
                        print(f"[candidate] shard={index} finished", flush=True)
                pending = remaining
                if pending:
                    time.sleep(1)
        finally:
            # Stop only our children; completed group journals remain available.
            for _, process in running:
                if process.poll() is None:
                    process.terminate()
            for _, process in running:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            for handle in handles:
                handle.close()
            progress.refresh()
            progress.close()
        print("[candidate] verifying and merging all shards", flush=True)
        summary = merge_candidate_shards(output_root, plan)
        print(json.dumps(summary["parallel_generation"], indent=2), flush=True)
        print(f"[candidate] complete: {output_root / 'per_candidate_metrics.csv'}", flush=True)
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SD3 restoration DPO candidates")
    parser.add_argument("--config", default="./config/dpo_sd3.yaml")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--gpu_ids", nargs="+", help="Independent workers, e.g. --gpu_ids 0 1 2; not DDP")
    parser.add_argument("--output_dir", help="Override candidate_generation.output_dir; use a fresh directory")
    parser.add_argument("--resume", action="store_true", help="Reuse a matching frozen plan and verified groups")
    parser.add_argument("--monitor", action="store_true", help="Read-only progress for an already running generation job")
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("Candidate generation is not DDP; use ordinary python with --gpu_ids 0 1 2")
    if args.gpu_ids and (len(set(args.gpu_ids)) != len(args.gpu_ids) or any(
        not gpu or "," in gpu or gpu.startswith("-") for gpu in args.gpu_ids
    )):
        raise ValueError("gpu_ids must be unique CUDA device IDs or UUIDs, one per worker")
    if args.resume and not args.gpu_ids:
        raise ValueError("--resume requires the coordinated --gpu_ids generation mode")
    if args.monitor and (args.resume or args.dry_run):
        raise ValueError("--monitor is read-only; do not combine it with --resume or --dry_run")
    config = load_config(args.config)
    generation = dict(config["candidate_generation"])
    if args.output_dir:
        generation["output_dir"] = args.output_dir
    if args.monitor:
        monitor_candidates(Path(generation["output_dir"]).expanduser().absolute(), args.gpu_ids)
        return
    model = config["model"]
    command = [
        sys.executable,
        "-m",
        "utils.randomness_check",
        "--config",
        str(generation.get("eval_config", "./config/eval_sd3.yaml")),
        "--output_dir",
        str(generation["output_dir"]),
        "--num_candidates_per_image",
        str(generation.get("num_candidates_per_image", 8)),
        "--seed",
        str(generation.get("seed", 20240805)),
        "--rain_psnr_gap",
        str(generation.get("weather_psnr_gap_thresholds", {}).get("rain", 0.2)),
        "--snow_psnr_gap",
        str(generation.get("weather_psnr_gap_thresholds", {}).get("snow", 0.62)),
        "--haze_psnr_gap",
        str(generation.get("weather_psnr_gap_thresholds", {}).get("haze", 2.5)),
        "--max_saved_groups_per_weather",
        str(generation.get("max_saved_groups_per_weather", 7000)),
        "--pretrained_model_name_or_path",
        str(model["pretrained_model_name_or_path"]),
        "--controlnet_model_path",
        str(model["controlnet_model_path"]),
        "--ra_fusion_path",
        str(model["ra_fusion_path"]),
        "--controlnet_conditioning_scale",
        str(generation.get(
            "controlnet_conditioning_scale",
            model.get("controlnet_conditioning_scale", 1.0),
        )),
        "--ra_fusion_scale",
        str(generation.get("ra_fusion_scale", model.get("ra_fusion_scale", 1.0))),
        "--no-load_transformer_lora",
    ]
    for key, flag in (
        ("ra_spatial_gate_scale", "--ra_spatial_gate_scale"),
        ("ra_how_token_scale", "--ra_how_token_scale"),
    ):
        value = generation.get(key, model.get(key))
        if value is not None:
            command.extend([flag, str(value)])
    optional = {
        "max_samples_per_weather": "--max_samples_per_weather",
        "strength": "--strength",
        "num_inference_steps": "--max_inference_steps",
        "dataset_rain": "--dataset_rain",
        "dataset_snow": "--dataset_snow",
        "dataset_haze": "--dataset_haze",
        "selection_manifest": "--selection_manifest",
        "rain_max_samples": "--rain_max_samples",
        "snow_max_samples": "--snow_max_samples",
        "haze_max_samples": "--haze_max_samples",
        "sample_mode": "--sample_mode",
    }
    for key, flag in optional.items():
        value = generation.get(key)
        if value is not None:
            command.extend([flag, str(value)])
    splits = generation.get("splits")
    if splits:
        command.append("--splits")
        command.extend(str(split) for split in splits)
    guidance_scales = generation.get("candidate_guidance_scales")
    if guidance_scales:
        command.append("--candidate_guidance_scales")
        command.extend(str(value) for value in guidance_scales)
    for key in ("revision", "variant"):
        if model.get(key) is not None:
            command.extend([f"--{key}", str(model[key])])
    for key, flag in (("use_ra_fusion", "--use_ra_fusion"), ("use_prompt", "--use_prompt")):
        value = generation.get(key)
        if value is not None:
            command.append(flag if value else f"--no-{flag.removeprefix('--')}")
    print("[candidate] " + " ".join(command))
    if args.dry_run:
        if args.gpu_ids:
            print(f"[candidate] plan-only preflight; {len(args.gpu_ids)} disjoint source shards")
            for index, gpu in enumerate(args.gpu_ids):
                print(f"[candidate] GPU={gpu} global_index % {len(args.gpu_ids)} == {index}; "
                      "independent output, then one global merge")
        return
    if args.gpu_ids:
        generate_shards(command, Path(generation["output_dir"]).expanduser().absolute(), args.gpu_ids, args.resume)
    else:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
