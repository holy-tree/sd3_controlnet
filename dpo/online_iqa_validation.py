"""Generate independent, paired SFT/DPO inputs for offline candidate-pool IQA.

No historical candidate is written here. Diffusion dependencies are imported only
by the real backend (or the prompt fallback when an enabled prompt is missing).
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import uuid
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

from dpo.provenance import checkpoint_checksum


NOISE_STRATEGY = "explicit_cpu_float32_randn_manual_seed_reset_per_source_and_seed"
SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(payload: object) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _publish(path: Path, payload, image: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        if image:
            payload.save(temporary, format="PNG")
        else:
            temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _checkpoint_components(raw_path: str, weights: str) -> dict:
    """Preflight exact paths; never pass a training-output root to the resolver."""
    if weights not in {"raw", "ema"}:
        raise ValueError("Checkpoint weights must be 'raw' or 'ema'")
    supplied = Path(raw_path).expanduser().resolve()
    if (supplied / "config.json").is_file():
        root = supplied.parent
        controlnet = supplied
    elif (supplied / "controlnet" / "config.json").is_file():
        root = supplied
        controlnet = root / "controlnet"
    else:
        raise FileNotFoundError(
            f"Specify an exact checkpoint root or controlnet component, not a training-output "
            f"root (no latest checkpoint fallback): {supplied}"
        )
    # A directly supplied EMA root/component is already selected. Raw never means EMA.
    if root.name == "ema":
        if weights != "ema":
            raise ValueError(f"raw weights requested for an EMA component: {supplied}")
    elif weights == "ema":
        root = root / "ema"
        controlnet = root / "controlnet"
    ra = root / "ra_fusion"
    weight_files = [controlnet / name for name in (
        "diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin",
    )]
    indexes = [controlnet / name for name in (
        "diffusion_pytorch_model.safetensors.index.json", "diffusion_pytorch_model.bin.index.json",
    ) if (controlnet / name).is_file()]
    if not (controlnet / "config.json").is_file() or (not any(
        path.is_file() and path.stat().st_size > 0 for path in weight_files
    ) and not indexes):
        raise FileNotFoundError(f"Missing complete {weights} ControlNet weights/config: {controlnet}")
    for index in indexes:
        shards = _json(index).get("weight_map", {})
        if not isinstance(shards, dict) or not shards or any(
            not isinstance(shard, str) or not (controlnet / shard).resolve().is_relative_to(controlnet)
            or not (controlnet / shard).is_file() or (controlnet / shard).stat().st_size == 0
            for shard in shards.values()
        ):
            raise FileNotFoundError(f"Incomplete {weights} ControlNet shards: {index}")
    if (not (ra / "config.json").is_file() or not (ra / "ra_fusion.safetensors").is_file()
            or (ra / "ra_fusion.safetensors").stat().st_size == 0):
        raise FileNotFoundError(f"Missing complete {weights} RA Fusion weights/config: {ra}; no fallback")
    _json(controlnet / "config.json")
    _json(ra / "config.json")
    return {
        "requested_checkpoint": str(supplied), "weights": weights,
        "controlnet_model_path": str(controlnet), "ra_fusion_path": str(ra),
        "controlnet_checksum_sha256": checkpoint_checksum(controlnet),
        "ra_fusion_checksum_sha256": checkpoint_checksum(ra),
        "controlnet_index_sha256": {index.name: _sha256(index) for index in controlnet.glob("*.index.json")},
    }


class _DiffusionBackend:
    def __init__(self, config: dict):
        import torch
        from utils.evaluate_sd3 import resolve_controlnet_path
        from utils.randomness_check import build_preprocess, setup_pipeline

        self.config = dict(config)
        resolved = resolve_controlnet_path(config["controlnet_model_path"])
        if Path(resolved).resolve() != Path(config["controlnet_model_path"]).resolve():
            raise ValueError("ControlNet resolver changed the explicitly selected component")
        self.device = torch.device(config["device"])
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
            config["inference_dtype"]
        ]
        self.preprocess = build_preprocess(config["resolution"])
        self.pipeline = None
        try:
            self.pipeline = setup_pipeline(
                config, self.dtype, self.device, config.get("ra_fusion_scale"),
                config["use_ra_fusion"], config.get("ra_spatial_gate_scale"),
                config.get("ra_how_token_scale"),
            )
        except Exception:
            self.close()
            raise

    def generate(self, source: dict, seeds: list[int]) -> list:
        import torch
        from utils.randomness_check import (
            infer_latent_shape, load_image_batch, run_with_initial_noise, tensor_to_pil,
        )

        with torch.inference_mode():
            # GT is validated and recorded, but does not need to occupy CUDA memory.
            lq_pils, lq_tensor, gt_tensor = load_image_batch([source], self.preprocess, "cpu")
            del lq_tensor, gt_tensor
            shape = infer_latent_shape(
                self.pipeline, lq_pils[0], self.config["resolution"], self.device,
            )
            images = []
            # Batch one intentionally keeps outputs invariant to resume and batch policy.
            for seed in seeds:
                noise = torch.randn(
                    (1, *shape), generator=torch.Generator(device="cpu").manual_seed(seed),
                    dtype=torch.float32, device="cpu",
                )
                prediction = run_with_initial_noise(
                    self.pipeline, self.config, self.device, self.dtype, lq_pils,
                    source["prompt"], noise, self.config["strength"],
                    self.config["num_inference_steps"], self.config["use_ra_fusion"],
                )
                images.append(tensor_to_pil(prediction[0]))
            return images

    def close(self) -> None:
        import torch

        self.pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _fallback_prompt(source: dict, settings: dict, sample_seed: int) -> str:
    if not settings.get("use_prompt", False):
        return ""
    from utils.evaluate_sd3 import maybe_make_prompt

    return maybe_make_prompt(
        source["identity"][0], {**settings, "seed": sample_seed},
        sample_key="|".join(source["identity"]),
    )


def _settings(config: dict, candidate_csv: Path) -> tuple[dict, dict, Path | None]:
    evaluation = config.get("eval_config")
    if isinstance(evaluation, dict):
        defaults = dict(evaluation)
    elif evaluation:
        path = Path(evaluation).expanduser().resolve()
        if path.suffix.lower() == ".json":
            defaults = _json(path)
        else:
            import yaml

            defaults = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(defaults, dict):
            raise ValueError("eval_config must contain a mapping")
    else:
        defaults = {}
    summary_path = next((path for path in (
        candidate_csv.with_name(f"{candidate_csv.stem}_summary.json"),
        candidate_csv.parent / "summary.json",
    ) if path.is_file()), None)
    summary = _json(summary_path) if summary_path else {}
    historical = summary.get("settings", {})
    if not isinstance(historical, dict):
        raise ValueError("Candidate summary.settings must be a mapping")
    keys = (
        "pretrained_model_name_or_path", "revision", "variant", "resolution",
        "num_inference_steps", "guidance_scale", "strength", "controlnet_conditioning_scale",
        "use_ra_fusion", "ra_fusion_scale", "ra_spatial_gate_scale", "ra_how_token_scale",
        "ra_disable_global", "ra_disable_spatial", "ra_disable_deformable",
        "use_prompt", "weather_prompts", "prompt_ratio", "negative_prompt", "load_transformer_lora",
    )
    settings = {
        "resolution": 512, "num_inference_steps": 30, "guidance_scale": 1.5,
        "strength": 1.0, "controlnet_conditioning_scale": 1.0,
        "use_ra_fusion": True, "load_transformer_lora": False,
    }
    nullable = {"revision", "variant", "negative_prompt", "ra_fusion_scale",
                "ra_spatial_gate_scale", "ra_how_token_scale"}
    for payload in (defaults, historical, summary.get("candidate_policy", {})):
        if not isinstance(payload, dict):
            raise ValueError("Candidate generation settings/policy must be mappings")
        for key in keys:
            if key in payload and (payload[key] is not None or key in nullable):
                settings[key] = payload[key]
        if payload.get("base_model"):
            settings["pretrained_model_name_or_path"] = payload["base_model"]
    for external, internal in (
        ("resolution", "resolution"), ("inference_steps", "num_inference_steps"),
        ("guidance_scale", "guidance_scale"), ("strength", "strength"),
        ("ra_fusion_scale", "ra_fusion_scale"), ("controlnet_scale", "controlnet_conditioning_scale"),
    ):
        if config.get(external) is not None:
            settings[internal] = config[external]
    settings["resolution"] = int(settings["resolution"])
    if historical.get("resolution") is not None and settings["resolution"] != int(historical["resolution"]):
        raise ValueError("Online resolution must match historical candidate summary.settings.resolution")
    settings["num_inference_steps"] = int(settings["num_inference_steps"])
    if settings["resolution"] <= 0 or settings["resolution"] % 8 or settings["num_inference_steps"] <= 0:
        raise ValueError("resolution must be positive/divisible by 8 and inference_steps positive")
    for key in ("strength", "guidance_scale", "controlnet_conditioning_scale", "ra_fusion_scale",
                "ra_spatial_gate_scale", "ra_how_token_scale"):
        if settings.get(key) is not None:
            settings[key] = float(settings[key])
            if not math.isfinite(settings[key]) or settings[key] < 0:
                raise ValueError(f"Invalid inference setting: {key}")
    if not 0 < settings["strength"] <= 1:
        raise ValueError("strength must be in (0, 1]")
    if not settings.get("pretrained_model_name_or_path"):
        raise ValueError("A base model is required in candidate summary or eval_config")
    settings.update({
        "device": config.get("device", "cuda"),
        "inference_dtype": config.get("inference_dtype", "bf16"),
        "inference_batch_size": int(config.get("inference_batch_size", 1)),
        "effective_inference_batch_size": 1,
        "deterministic_controlnet_vae": True,
        "controlnet_vae_conditioning": "posterior_mode", "vae_decode_dtype": "fp32",
        "noise_strategy": NOISE_STRATEGY,
    })
    if settings["inference_dtype"] not in {"bf16", "fp16", "fp32"}:
        raise ValueError("inference_dtype must be bf16, fp16, or fp32")
    if settings["inference_batch_size"] <= 0:
        raise ValueError("inference_batch_size must be positive")
    return settings, summary, summary_path


def prepare_online_inputs(config: dict, backend_factory=None) -> dict:
    """Return a new analysis config after paired inference, with strict cache reuse.

    ``backend_factory(model_config)`` returns an object implementing
    ``generate(source, seeds) -> list[PIL.Image]`` and ``close()``. Sources carry
    canonical ``identity``, original display IDs, original LQ/GT, hashes, and one
    exact shared prompt. A failure publishes diagnostics and raises, never COMPLETE.
    """
    from PIL import Image
    from dpo.offline_iqa_analysis import (
        PATH_ALIASES, _first, _meaningful_group_id, _read_rows, _resolve_artifact_path, canonical_path,
        find_normalization, identity_key, load_candidates, load_normalization, write_csv,
    )

    seeds = list(config.get("seeds", [42, 43, 44]))
    reference_seed = config.get("reference_seed", 42)
    if not seeds or any(type(seed) is not int or not 0 <= seed < 2 ** 63 for seed in seeds):
        raise ValueError("seeds must be nonnegative integers smaller than 2**63")
    if type(reference_seed) is not int or len(set(seeds)) != len(seeds) or reference_seed not in seeds:
        raise ValueError("seeds must be unique and include reference_seed")
    limit = int(config.get("num_samples_per_weather", 20))
    maximum = config.get("max_images")
    if limit < 0 or (maximum is not None and int(maximum) <= 0):
        raise ValueError("num_samples_per_weather must be >= 0; max_images must be positive")
    sample_seed = int(config.get("sample_seed", 2026))
    candidate_csv = Path(config["candidate_csv"]).expanduser().resolve()
    output_dir = Path(config["output_dir"]).expanduser().resolve()
    output = (output_dir / "online_validation").resolve()
    # Refuse layouts that could overwrite any historical inputs.
    if not output.is_relative_to(output_dir):
        raise ValueError("online_validation output must not escape output_dir through a symlink")
    if candidate_csv.is_relative_to(output_dir):
        raise ValueError("Historical candidate manifest must be outside output_dir")
    settings, historical, summary_path = _settings(config, candidate_csv)
    normalization_path = find_normalization(candidate_csv, config.get("normalization_json"))
    evaluation = config.get("eval_config")
    evaluation_path = Path(evaluation).expanduser().resolve() if isinstance(evaluation, (str, Path)) else None
    for path in (summary_path, normalization_path, evaluation_path):
        if path and path.is_relative_to(output_dir):
            raise ValueError(f"Historical metadata must be outside output_dir: {path}")
    candidates, skips, candidate_counts = load_candidates(
        candidate_csv, load_normalization(normalization_path),
    )
    raw_by_path = {}
    raw_groups = defaultdict(list)
    for row_number, raw in enumerate(_read_rows(candidate_csv), 2):
        lq = _resolve_artifact_path(raw.get("lq_path"), candidate_csv)
        gt = _resolve_artifact_path(raw.get("gt_path"), candidate_csv)
        for field in (*PATH_ALIASES, "lq_path", "gt_path"):
            path = _resolve_artifact_path(raw.get(field), candidate_csv)
            if path and path.is_relative_to(output_dir):
                raise ValueError(f"Historical candidate/source paths must be outside output_dir: {path}")
        group_id, _ = _meaningful_group_id(raw, lq, row_number)
        raw_identity = identity_key(
            raw.get("weather"), raw.get("subdataset") or raw.get("source"), group_id,
        )
        raw_groups[raw_identity].append((raw, str(lq or ""), str(gt or "")))
        value = _first(raw, PATH_ALIASES)
        if value:
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = candidate_csv.parent / path
            raw_by_path.setdefault(canonical_path(path), raw)
    groups = defaultdict(list)
    for candidate in candidates:
        groups[candidate["identity"]].append(candidate)
    incomplete = {row.get("identity") for row in skips if row["reason"] == "missing_candidate_file"}
    expected_pool_size = int(historical.get("settings", {}).get("num_candidates_per_image") or 0)
    if expected_pool_size < 0:
        raise ValueError("Historical num_candidates_per_image must be positive when specified")
    sources_by_weather = defaultdict(list)
    from tqdm import tqdm

    ranked_groups = sorted(groups.items(), key=lambda item: (
        item[0][0], _fingerprint([sample_seed, item[0][0], list(item[0])]),
    ))
    validation_limit = min(limit, int(maximum)) if limit and maximum is not None else limit
    if not validation_limit and maximum is not None:
        validation_limit = int(maximum)
    for identity, pool in tqdm(ranked_groups, desc="Validate candidate sources", unit="group", leave=False):
        if validation_limit and len(sources_by_weather[identity[0]]) >= validation_limit:
            continue
        first = pool[0]
        source = {
            "identity": list(identity), "weather": first["weather_display"],
            "subdataset": first["subdataset_display"], "source_id": first["source_id_display"],
            "lq_path": first["lq_path"], "gt_path": first["gt_path"],
            "candidate_count": len(pool),
        }
        try:
            if first["identity_text"] in incomplete:
                raise ValueError("incomplete historical candidate pool")
            if expected_pool_size and len(pool) < expected_pool_size:
                raise ValueError(f"incomplete historical candidate pool: {len(pool)} rows, expected at least {expected_pool_size}")
            if any(lq != source["lq_path"] or gt != source["gt_path"]
                   for _, lq, gt in raw_groups[identity]):
                raise ValueError("conflicting source paths for canonical identity")
            sizes = []
            for kind in ("lq", "gt"):
                path = Path(source[f"{kind}_path"])
                if not source[f"{kind}_path"] or not path.is_file():
                    raise ValueError(f"missing {kind.upper()} source: {path}")
                with Image.open(path) as image:
                    image.load()
                    sizes.append(image.size)
                source[f"{kind}_sha256"] = _sha256(path)
            if sizes[0] != sizes[1]:
                raise ValueError("GT/LQ correspondence: source dimensions differ")
            if canonical_path(source["lq_path"]) == canonical_path(source["gt_path"]):
                raise ValueError("GT/LQ correspondence: identical source paths")
            for candidate in pool:
                try:
                    with Image.open(candidate["candidate_path"]) as image:
                        image.load()
                        if image.size != (settings["resolution"], settings["resolution"]):
                            raise ValueError("historical candidate dimensions differ from inference resolution")
                except OSError as error:
                    raise ValueError(f"incomplete historical candidate pool: unreadable {candidate['candidate_path']}") from error
            source["source_dimensions"] = list(sizes[0])
            raw_pool = [raw_by_path[canonical_path(row["candidate_path"])] for row in pool]
            prompts = {row["prompt"] for row, _, _ in raw_groups[identity]
                       if "prompt" in row and row["prompt"] is not None}
            if len(prompts) > 1:
                raise ValueError("inconsistent historical prompts within selected source group")
            source["prompt"] = next(iter(prompts)) if prompts else _fallback_prompt(source, settings, sample_seed)
            source["prompt_origin"] = "candidate_manifest" if prompts else "maybe_make_prompt_fallback"
            source["prompt_fingerprint"] = _fingerprint(source["prompt"])
            source["candidate_cfg_rows"] = [
                {"candidate_index": row.get("candidate_index", ""),
                 "guidance_scale": row.get("guidance_scale", "")}
                for row in raw_pool
            ]
            source["candidate_cfg_rows"].sort(key=_fingerprint)
            sources_by_weather[identity[0]].append(source)
        except (OSError, ValueError) as error:
            skips.append({"stage": "source_validation", "identity": "|".join(identity), "reason": str(error)})
    # Hash ranking supplies stable prefixes even when sampling limits expand.
    for weather, sources in sources_by_weather.items():
        sources.sort(key=lambda source: _fingerprint([sample_seed, weather, source["identity"]]))
        if limit:
            sources_by_weather[weather] = sources[:limit]
    sources = []
    weather_order = sorted(sources_by_weather)
    for index in range(max((len(rows) for rows in sources_by_weather.values()), default=0)):
        for weather in weather_order:
            if index < len(sources_by_weather[weather]):
                sources.append(sources_by_weather[weather][index])
    if maximum is not None:
        sources = sources[:int(maximum)]
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "generation_skips.csv", skips, ("stage", "identity", "reason"))
    if not sources:
        raise ValueError("No valid candidate-manifest LQ/GT sources remain; see generation_skips.csv")
    checkpoints = {
        model: _checkpoint_components(config[f"{model}_checkpoint"], config.get(f"{model}_weights", default))
        for model, default in (("sft", "raw"), ("dpo", "ema"))
    }
    for model, checkpoint in checkpoints.items():
        if Path(checkpoint["requested_checkpoint"]).is_relative_to(output_dir):
            raise ValueError(f"Input {model} checkpoint must be outside output_dir")
    versions = {}
    for package in ("torch", "torchvision", "diffusers", "transformers", "numpy", "Pillow", "safetensors"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    root = Path(__file__).resolve().parents[1]
    implementations = {name: _sha256(root / name) for name in (
        "dpo/online_iqa_validation.py", "dpo/provenance.py", "utils/randomness_check.py",
        "utils/evaluate_sd3.py", "utils/pipeline_inference.py", "utils/restoration_condition.py",
        "models/ra_fusion_sd3.py",
    )}
    factory = backend_factory or _DiffusionBackend
    backend_identity = f"{getattr(factory, '__module__', type(factory).__module__)}.{getattr(factory, '__qualname__', type(factory).__qualname__)}"
    try:
        backend_hash = _fingerprint(inspect.getsource(factory if inspect.isfunction(factory) or inspect.isclass(factory) else type(factory)))
    except (OSError, TypeError):
        backend_hash = None
    base = Path(settings["pretrained_model_name_or_path"]).expanduser()
    if base.is_dir() and base.resolve().is_relative_to(output_dir):
        raise ValueError("Input base model must be outside output_dir")
    base_checksum = checkpoint_checksum(base) if base.is_dir() else None
    base_metadata_hashes = {
        path.relative_to(base).as_posix(): _sha256(path)
        for path in sorted(base.rglob("*"))
        if path.is_file() and path.suffix in {".json", ".txt", ".model"}
    } if base.is_dir() else {}
    policies = {}
    for model, checkpoint in checkpoints.items():
        model_config = {
            **settings, **{key: value for key, value in checkpoint.items() if key != "requested_checkpoint"},
            "model": model, "seeds": seeds, "reference_seed": reference_seed,
        }
        if settings["load_transformer_lora"]:
            lora = Path(checkpoint["controlnet_model_path"]).parent / "transformer_lora"
            if not (lora / "pytorch_lora_weights.safetensors").is_file():
                raise FileNotFoundError(f"Missing exact checkpoint Transformer LoRA: {lora}")
            model_config.update(transformer_lora_path=str(lora), transformer_lora_checksum=checkpoint_checksum(lora))
        policies[model] = {
            "schema_version": SCHEMA_VERSION, "inference_config": model_config,
            "versions": versions, "implementation_hashes": implementations,
            "backend": backend_identity, "base_model_checksum": base_checksum,
            "backend_implementation_hash": backend_hash,
            "base_model_metadata_hashes": base_metadata_hashes,
        }
    policy = {"schema_version": SCHEMA_VERSION, "models": policies}
    policy_fingerprint = _fingerprint(policy)
    policy_path = output / "generation_policy.json"
    resume = bool(config.get("resume", True))
    source_policies = {_fingerprint(source["identity"]): _fingerprint(source) for source in sources}
    if resume and not policy_path.exists() and any((output / model).exists() for model in ("sft", "dpo")):
        raise ValueError("Missing shared generation policy for existing outputs; use a new output_dir or no-resume")
    if resume and policy_path.exists():
        try:
            previous = _json(policy_path)
        except (OSError, ValueError) as error:
            raise ValueError("Unreadable shared generation policy; use a new output_dir or no-resume") from error
        if (previous.get("fingerprint") != policy_fingerprint
                or _fingerprint({"schema_version": previous.get("schema_version"), "models": previous.get("models")}) != policy_fingerprint):
            raise ValueError("Stale online generation fingerprint: checkpoint/inference policy changed; use a new output_dir or no-resume")
        prior_sources = previous.get("source_fingerprints")
        if (not isinstance(prior_sources, dict)
                or previous.get("source_manifest_fingerprint") != _fingerprint(prior_sources)):
            raise ValueError("Unreadable shared source fingerprints; use a new output_dir or no-resume")
        if any(key in prior_sources and prior_sources[key] != value for key, value in source_policies.items()):
            raise ValueError("Stale source/prompt fingerprint at shared manifest level; use a new output_dir or no-resume")
        source_policies = {**prior_sources, **source_policies}
    _publish(policy_path, {
        "fingerprint": policy_fingerprint, "source_fingerprints": source_policies,
        "source_manifest_fingerprint": _fingerprint(source_policies), **policy,
    })
    write_csv(output / "source_manifest.csv", [{
        **source, "identity": json.dumps(source["identity"]),
        "candidate_cfg_rows": json.dumps(source["candidate_cfg_rows"]),
    } for source in sources])
    warnings = [
        "Population is training-candidate diagnostics, not heldout generalization.",
        f"Historical candidate best-of-K has a selection advantage over the compared {len(seeds)}-seed outputs (default 3 seeds); the pool is not regenerated or rewritten.",
    ]
    candidate_checkpoint = (historical.get("candidate_policy", {}).get("controlnet_model_path")
                            or historical.get("settings", {}).get("checkpoint_controlnet") or "unspecified")
    warnings.append(
        f"Recorded candidate SFT component: {candidate_checkpoint}; compared SFT component: "
        f"{checkpoints['sft']['controlnet_model_path']}. Checkpoints may differ; do not attribute "
        "candidate-versus-model gaps solely to DPO or selection."
    )
    if not summary_path:
        warnings.append("No adjacent candidate summary.settings found; compatibility defaults come from eval_config.")
    fallback_count = sum(source["prompt_origin"] == "maybe_make_prompt_fallback" for source in sources)
    if fallback_count:
        warnings.append(f"Original per-candidate prompts are absent for {fallback_count} selected sources; sample_seed-fixed fallback prompts cannot establish exact historical prompt compatibility.")
    if base_checksum is None and not settings.get("revision"):
        warnings.append("Remote base model has no pinned revision; its remote contents cannot be verified by local cache provenance.")
    if settings["inference_batch_size"] != 1:
        warnings.append("Requested inference_batch_size is recorded; effective inference batch size remains 1 for robust deterministic resume.")
    metadata = {
        "status": "RUNNING", "sample": {
            "sample_seed": sample_seed, "num_samples_per_weather": limit, "max_images": maximum,
            "count": len(sources), "counts_by_weather": dict(Counter(source["identity"][0] for source in sources)),
            "identities": [source["identity"] for source in sources], "seeds": seeds, "reference_seed": reference_seed,
            "sampling": "sha256_rank_per_weather_round_robin_global_limit",
        },
        "checkpoints": checkpoints, "settings": settings,
        "provenance": {
            "candidate_csv": str(candidate_csv), "candidate_csv_sha256": _sha256(candidate_csv),
            "candidate_summary_path": str(summary_path) if summary_path else None,
            "candidate_generation_config": historical, "candidate_population": candidate_counts,
            "normalization_json": str(normalization_path) if normalization_path else None,
            "normalization_sha256": _sha256(normalization_path) if normalization_path else None,
            "policy_fingerprint": policy_fingerprint, "policies": policies,
            "source_manifest": str(output / "source_manifest.csv"),
            "candidate_cfg_distribution": dict(Counter(
                str(row["guidance_scale"]) for source in sources for row in source["candidate_cfg_rows"]
            )),
        }, "results": {}, "warnings": warnings,
    }
    run_path = output / "run_metadata.json"
    _publish(run_path, metadata)
    errors = []
    manifests = {}
    for model in ("sft", "dpo"):
        model_config = policies[model]["inference_config"]
        model_fingerprint = _fingerprint(policies[model])
        rows = []
        generated = reused = 0
        backend = None
        model_run = {"status": "RUNNING", "policy": policies[model], "sample": metadata["sample"]}
        model_run_path = output / model / "run_metadata.json"
        if not model_run_path.parent.resolve().is_relative_to(output):
            raise ValueError("Generated model directory must not escape output through a symlink")
        _publish(model_run_path, model_run)
        progress = tqdm(sources, desc=f"Online {model.upper()}", unit="source", leave=False)
        try:
            for source in progress:
                source_key = _fingerprint(source["identity"])
                missing = []
                entries = []
                for seed in seeds:
                    prediction = output / model / source_key / f"seed_{seed}.png"
                    if not prediction.parent.resolve().is_relative_to(output):
                        raise ValueError("Generated source directory must not escape output through a symlink")
                    sidecar = prediction.with_suffix(".json")
                    record_policy = {
                        "model_fingerprint": model_fingerprint, "model_policy": policies[model],
                        "source": source, "seed": seed,
                    }
                    record_fingerprint = _fingerprint(record_policy)
                    valid = False
                    if resume and prediction.is_file() and sidecar.is_file():
                        try:
                            cached = _json(sidecar)
                            # Shared policy/source fingerprints already reject stale runs.
                            # A mismatched sidecar here is an individual corrupt cache entry.
                            valid = (
                                cached.get("schema_version") == SCHEMA_VERSION
                                and cached.get("fingerprint") == record_fingerprint
                                and _fingerprint(cached.get("provenance")) == record_fingerprint
                                and cached.get("output_sha256") == _sha256(prediction)
                            )
                            if valid:
                                with Image.open(prediction) as image:
                                    image.load()
                                    valid = image.size == (settings["resolution"], settings["resolution"])
                        except (OSError, ValueError):
                            valid = False
                    entry = (seed, prediction, sidecar, record_policy, record_fingerprint)
                    entries.append(entry)
                    if valid:
                        reused += 1
                    else:
                        missing.append(entry)
                if missing:
                    if backend is None:
                        backend = factory(deepcopy(model_config))
                    images = list(backend.generate(deepcopy(source), [entry[0] for entry in missing]))
                    if len(images) != len(missing):
                        raise ValueError("Backend returned wrong number of generated images")
                    for image, (_, prediction, sidecar, record_policy, record_fingerprint) in zip(images, missing):
                        if image.size != (settings["resolution"], settings["resolution"]):
                            raise ValueError("Backend output resolution does not match inference policy")
                        _publish(prediction, image, image=True)
                        _publish(sidecar, {
                            "schema_version": SCHEMA_VERSION, "fingerprint": record_fingerprint,
                            "provenance": record_policy, "output_sha256": _sha256(prediction),
                            "prediction_path": str(prediction),
                        })
                        generated += 1
                for seed, prediction, sidecar, _, record_fingerprint in entries:
                    rows.append({
                        "weather": source["weather"], "subdataset": source["subdataset"],
                        "source_id": source["source_id"], "seed": seed,
                        "prediction_path": str(prediction), "lq_path": source["lq_path"],
                        "gt_path": source["gt_path"], "prompt": source["prompt"],
                        "identity": json.dumps(source["identity"]),
                        "provenance_path": str(sidecar), "provenance_fingerprint": record_fingerprint,
                    })
                progress.set_postfix(source="|".join(source["identity"]), generated=generated,
                                     reused=reused, refresh=False)
            model_run.update(status="COMPLETE", generated=generated, reused=reused, output_count=len(rows))
        except Exception as error:
            errors.append({"model": model, "stage": "generation", "reason": str(error),
                           "identity": "|".join(source["identity"])})
            model_run.update(status="FAILED", generated=generated, reused=reused, error=str(error))
            metadata["status"] = "FAILED"
            _publish(model_run_path, model_run)
            _publish(run_path, metadata)
            write_csv(output / "generation_errors.csv", errors, ("model", "stage", "identity", "reason"))
            raise
        finally:
            progress.close()
            if backend is not None:
                try:
                    backend.close()
                except Exception as error:
                    errors.append({"model": model, "stage": "cleanup", "reason": str(error)})
                    model_run.update(status="FAILED", error=str(error))
                    metadata["status"] = "FAILED"
                    _publish(model_run_path, model_run)
                    _publish(run_path, metadata)
                    write_csv(output / "generation_errors.csv", errors, ("model", "stage", "identity", "reason"))
                    raise
                finally:
                    backend = None
        manifest = output / model / "manifest.csv"
        write_csv(manifest, rows)
        manifests[model] = str(manifest)
        metadata["results"][model] = {"generated": generated, "reused": reused, "output_count": len(rows), "manifest": str(manifest)}
        _publish(model_run_path, model_run)
    write_csv(output / "generation_errors.csv", errors, ("model", "stage", "identity", "reason"))
    metadata["status"] = "COMPLETE"
    _publish(run_path, metadata)
    return {
        **config, "sft_input": manifests["sft"], "dpo_input": manifests["dpo"],
        "resolution": settings["resolution"],
        "online_selected_identities": [source["identity"] for source in sources],
        "online_validation": metadata,
    }
