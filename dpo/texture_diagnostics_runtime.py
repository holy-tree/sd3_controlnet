"""REAL-model-only diagnostic runtime. Not imported by the trainer."""
from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.func import functional_call
from tqdm.auto import tqdm

from .losses import diffusion_dpo_loss
from .reference import snapshot_reference_parameters
from .spatial import aligned_rgb, decoded_gt_mse, gt_texture_mask, gt_timestep_indices, texture_weighted_mse
from .spatial_config import spatial_settings, training_signature
from .spatial_dataset import SpatialPreferenceDataset, collate_spatial_pairs
from .tail_risk_training import training_dpo_loss
from .texture_diagnostics import (GradientLedger, SCOPES, correlation, distribution,
                                  individual_gradient_norms, parameter_registry, tensor_digest)


def sampling_plan(dataset, windows, batch_size, accumulation, seed):
    buckets = defaultdict(list)
    for index in range(len(dataset)):
        gt = dataset.gt_records[index % len(dataset.gt_records)]
        pair = dataset.records[index % len(dataset.records)] if dataset.records else gt
        buckets[(pair["weather"], gt["weather"])].append(index)
    weather = [w for w in ("rain", "snow", "haze") if any(key[0] == w for key in buckets)]
    if not weather:
        raise ValueError("No diagnostic training sources")
    rng = random.Random(seed)
    plan = []
    for window in range(windows):
        w = weather[window % len(weather)]
        key = (w, w) if (w, w) in buckets else sorted(k for k in buckets if k[0] == w)[0]
        plan.append({"weather": key[0], "gt_weather": key[1],
                     "indices": [[rng.choice(buckets[key]) for _ in range(batch_size)] for _ in range(accumulation)]})
    return plan, {"/".join(key): len(value) for key, value in buckets.items()}


def source_is_training(row):
    if row.get("split") in ("test", "validation", "val"):
        return False
    return not any("test" in Path(row[key]).parts for key in ("gt_path", "lq_path"))


def save_image(path, tensor):
    array = tensor.detach().float().cpu().clamp(0, 1)
    if array.shape[0] == 1:
        pixels = (array[0].numpy()*255).round().astype("uint8")
    else:
        pixels = (array.permute(1, 2, 0).numpy()*255).round().astype("uint8")
    Image.fromarray(pixels).save(path)


class PipelineRunner(torch.nn.Module):
    """Functional substitution reuses real modules without copying into policy weights."""
    def __init__(self, engine, pipeline, options):
        super().__init__()
        self.controlnet, self.transformer = engine.controlnet, engine.transformer
        self.pipeline, self.options = pipeline, options
        self.device, self.dtype = engine.device, engine.dtype

    def forward(self, pil, noise, prompt, pooled, negative, negative_pooled):
        from utils.randomness_check import run_with_initial_noise
        return run_with_initial_noise(self.pipeline, self.options, self.device, self.dtype,
                                      [pil], None, noise, self.options["strength"], self.options["num_inference_steps"], True,
                                      prompt_embeds=prompt, pooled_prompt_embeds=pooled,
                                      negative_prompt_embeds=negative, negative_pooled_prompt_embeds=negative_pooled)


class RealEngine:
    def __init__(self, args, snapshot, runtime):
        # Deliberately import the actual loader, not a reimplementation or mock.
        from scripts import train_dpo_sd3 as trainer
        self.trainer, self.args = trainer, args
        self.config = copy.deepcopy(snapshot["config"])
        self.train = self.config["training"]
        self.model_config = copy.deepcopy(self.config["model"])
        self.checkpoint = Path(args.checkpoint).expanduser().resolve()
        self.meta = snapshot["metadata"]
        if self.meta is None or self.meta.get("version") != 6:
            raise ValueError("Require an actual dynamic_texture v6 checkpoint")
        self.settings = spatial_settings(self.train)
        if training_signature(self.train) != self.meta.get("spatial_dynamic"):
            raise ValueError("Saved config and checkpoint spatial training signatures differ")
        if not self.train.get("upcast_vae", True):
            raise ValueError("Saved training disabled required FP32 VAE")
        self.dataset = SpatialPreferenceDataset(self.train["preference_manifest"], self.train["gt_source_manifest"], self.settings)
        trainer.validate_resume_metadata(self.checkpoint, self.train, self.train["train_controlnet"],
                                         self.train["train_ra_fusion"], self.meta["num_processes"], len(self.dataset))
        trainer.validate_candidate_policy(self.model_config, self.train)
        print("[diagnose] loading real SD3/VAE and initial SFT ControlNet/RA reference...", flush=True)
        self.device = torch.device(args.device)
        mixed = self.train.get("mixed_precision", "no")
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32, "float32": torch.float32}[mixed]
        self.loss_scale = 1.0
        if mixed == "fp16":
            scaler_file = self.checkpoint/"scaler.pt"
            if args.loss_scale is not None:
                self.loss_scale = args.loss_scale
            elif scaler_file.is_file():
                self.loss_scale = float(torch.load(scaler_file, map_location="cpu", weights_only=True)["scale"])
            else:
                raise ValueError("FP16 checkpoint scaler state unavailable; supply an explicit recorded --loss-scale")
        elif args.loss_scale is not None:
            raise ValueError("--loss-scale only applies to an actual FP16 run")
        if not math.isfinite(self.loss_scale) or self.loss_scale <= 0:
            raise ValueError("Invalid loss scale")
        self.accumulation = int(self.train["gradient_accumulation_steps"])
        self.batch_size = int(self.train["train_batch_size"])
        self.backward_factor = self.loss_scale/self.accumulation
        self.plan, bucket_counts = sampling_plan(self.dataset, args.batches, self.batch_size, self.accumulation, args.seed)
        runtime.update(sampling_buckets=bucket_counts, sampling_plan=self.plan,
                       missing_dpo_weathers=[w for w in ("rain", "snow", "haze") if not any(p["weather"] == w for p in self.plan)],
                       weights="raw checkpoint training parameters; not EMA", loss_scale=self.loss_scale,
                       backward_factor=self.backward_factor, original_world_size=self.meta["num_processes"],
                       accumulation_scope="single_rank_configured_microbatch_mean_average")
        trainer.set_seed(int(self.train.get("seed", 42)))
        model_path = self.model_config["pretrained_model_name_or_path"]
        revision = self.model_config.get("revision")
        self.scheduler = trainer.FlowMatchEulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler", revision=revision)
        self.vae = trainer.AutoencoderKL.from_pretrained(model_path, subfolder="vae", revision=revision,
                                                       variant=self.model_config.get("variant"), torch_dtype=torch.float32).to(self.device)
        self.vae.requires_grad_(False).eval()
        self.controlnet = trainer._load_controlnet_smart(trainer.resolve_controlnet_path(self.model_config["controlnet_model_path"])).to(self.device)
        self.controlnet.requires_grad_(True).train()
        for p in self.controlnet.parameters():
            if p.is_floating_point() and p.dtype != torch.float32:
                p.data = p.data.float()  # identical initialization precision to trainer
        for buffer in self.controlnet.buffers():
            if buffer.is_floating_point() and buffer.dtype != self.dtype:
                buffer.data = buffer.data.to(self.dtype)
        self.reference_controlnet = snapshot_reference_parameters(self.controlnet)
        self.sft_control_buffers = {k:v.detach().clone() for k,v in self.controlnet.named_buffers()}
        if self.train.get("gradient_checkpointing", True):
            self.controlnet.enable_gradient_checkpointing()
        self.model_config["device"] = self.device
        self.transformer = trainer.load_ra_transformer(self.model_config, self.dtype, self.train["train_ra_fusion"])
        if self.train.get("gradient_checkpointing", True):
            self.transformer.enable_gradient_checkpointing()
        self.reference_ra = snapshot_reference_parameters(self.transformer, prefix="ra_")
        self.sft_ra_buffers = {k:v.detach().clone() for k,v in self.transformer.named_buffers()}
        print("[diagnose] loading raw checkpoint policy (no optimizer state or updates)...", flush=True)
        loaded = trainer._load_controlnet_smart(str(self.checkpoint/"controlnet"))
        self.controlnet.load_state_dict(loaded.state_dict(), strict=True)
        del loaded
        self.transformer.load_ra_fusion(self.checkpoint/"ra_fusion")
        self.registry, info = parameter_registry(self.controlnet, self.transformer)
        self.parameters = [entry["parameter"] for entry in self.registry]
        runtime["parameter_registry"] = info
        runtime["parameter_registry"]["entries"] = [{"name": e["name"], "numel": e["numel"], "groups": sorted(e["groups"])} for e in self.registry]
        runtime["vae_trainable_parameters"] = sum(p.numel() for p in self.vae.parameters() if p.requires_grad)
        runtime["real_model_types"] = {name:type(module).__module__+"."+type(module).__name__ for name,module in
                                       (("controlnet",self.controlnet),("transformer",self.transformer),("vae",self.vae),("scheduler",self.scheduler))}
        runtime["vae_parameterization"] = {"scaling_factor":float(self.vae.config.scaling_factor),
                                            "shift_factor":float(self.vae.config.shift_factor), "dtype":str(self.vae.dtype)}
        runtime["reference_requires_grad"] = any(t.requires_grad for _, t in self.reference_tensors())
        runtime["stochastic_or_mutable_modules"] = [root+"."+name+":"+type(module).__name__ for root, model in
            (("controlnet", self.controlnet), ("ra", self.transformer)) for name, module in model.named_modules()
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            or (isinstance(module, torch.nn.modules.dropout._DropoutNd) and module.p > 0)]
        print("[diagnose] hashing all policy/VAE and reference parameters for no-update verification...", flush=True)
        self.initial_policy = tensor_digest(self.named_policy_parameters())
        self.initial_reference = tensor_digest(self.reference_tensors())
        self.initial_buffers = tensor_digest(self.named_buffers())
        runtime["parameter_checksums_before"] = {"all_policy_and_vae":self.initial_policy,
                                                 "reference":self.initial_reference,"buffers":self.initial_buffers}
        self.timesteps, self.sigmas = self.scheduler.timesteps.to(self.device), self.scheduler.sigmas.to(self.device)
        self.gt_indices = gt_timestep_indices(self.sigmas, self.scheduler.config.num_train_timesteps, self.settings["gt"])
        runtime["actual_gt_sigma_grid"] = {"count":len(self.gt_indices),
            "minimum":float(self.sigmas[self.gt_indices].to(self.dtype).min()),
            "maximum":float(self.sigmas[self.gt_indices].to(self.dtype).max())}
        self.force_zero = bool(getattr(self.controlnet.config, "force_zeros_for_pooled_projection", False))
        prompts = {""}
        for item in self.plan:
            for indices in item["indices"]:
                for index in indices:
                    pair = self.dataset.records[index % len(self.dataset.records)] if self.dataset.records else None
                    if pair:
                        prompts.add(str(pair.get("prompt", "")))
                    prompts.add(str(self.dataset.gt_records[index % len(self.dataset.gt_records)].get("prompt", "")))
        tokenizers, encoders = trainer.load_text_components(model_path, revision, self.model_config.get("variant"), self.device, self.dtype)
        self.prompt_cache = trainer.cache_validation_prompt_embeddings(encoders, tokenizers, sorted(prompts),
                                                                      int(self.train.get("max_sequence_length", 77)), self.device)
        del tokenizers, encoders
        gc.collect()
        torch.cuda.empty_cache()
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    def autocast(self):
        return torch.autocast("cuda", dtype=self.dtype) if self.dtype != torch.float32 else nullcontext()

    def reference_tensors(self):
        return [("controlnet."+k, v) for k, v in self.reference_controlnet.items()] + [("ra."+k, v) for k, v in self.reference_ra.items()]

    def named_buffers(self):
        return [(prefix+"."+name, value) for prefix, model in (("controlnet", self.controlnet), ("ra", self.transformer), ("vae",self.vae)) for name, value in model.named_buffers()]

    def named_policy_parameters(self):
        seen = set()
        for prefix, model in (("controlnet", self.controlnet), ("transformer", self.transformer), ("vae",self.vae)):
            for name, value in model.named_parameters():
                if id(value) not in seen:
                    seen.add(id(value))
                    yield prefix+"."+name, value

    def embeddings(self, prompts):
        return tuple(torch.stack([self.prompt_cache[p][i] for p in prompts]).to(self.device, dtype=self.dtype) for i in (0, 1))

    @torch.no_grad()
    def conditions(self, pixels):
        posterior = self.vae.encode(pixels.to(self.device, dtype=self.vae.dtype)).latent_dist.mode()
        restoration = (posterior-self.vae.config.shift_factor)*self.vae.config.scaling_factor
        control = (posterior-(0.0 if self.force_zero else self.vae.config.shift_factor))*self.vae.config.scaling_factor
        return control.to(self.dtype), restoration.to(self.dtype)

    def forward(self, noisy, timestep, prompt, pooled, control, restoration, reference=False, full_sft=False):
        kwargs = dict(hidden_states=noisy, timestep=timestep, encoder_hidden_states=prompt,
                      pooled_projections=torch.zeros_like(pooled) if self.force_zero else pooled,
                      controlnet_cond=control, conditioning_scale=float(self.model_config.get("controlnet_conditioning_scale", 1)), return_dict=False)
        with self.autocast():
            reference_control = {**self.reference_controlnet, **self.sft_control_buffers} if full_sft else self.reference_controlnet
            reference_ra = {**self.reference_ra, **self.sft_ra_buffers} if full_sft else self.reference_ra
            samples = (functional_call(self.controlnet, reference_control, (), kwargs, strict=False)
                       if reference else self.controlnet(**kwargs))[0]
        transformer_kwargs = dict(hidden_states=noisy, timestep=timestep, encoder_hidden_states=prompt,
                                  pooled_projections=pooled, block_controlnet_hidden_states=[x.to(self.dtype) for x in samples],
                                  restoration_cond=restoration, return_dict=False)
        with self.autocast():
            return (functional_call(self.transformer, reference_ra, (), transformer_kwargs, strict=False)
                    if reference else self.transformer(**transformer_kwargs))[0]

    def dpo(self, batch):
        b = len(batch["prompts"])
        with torch.no_grad():
            pixels = torch.cat([batch["chosen_pixel_values"], batch["rejected_pixel_values"]]).to(self.device, dtype=self.vae.dtype)
            latents = self.vae.encode(pixels).latent_dist.mode()
            latents = ((latents-self.vae.config.shift_factor)*self.vae.config.scaling_factor).to(self.dtype)
            control, restoration = self.conditions(batch["conditioning_pixel_values"])
            prompt, pooled = self.embeddings(batch["prompts"])
            indices = torch.randint(0, self.scheduler.config.num_train_timesteps, (b,), device=self.device)
            sigma = self.sigmas[indices].to(self.dtype).view(-1,1,1,1)
            noise = torch.randn_like(latents[:b], dtype=self.dtype)
            sigma_pair, noise_pair = torch.cat([sigma, sigma]), torch.cat([noise, noise])
            noisy = (1-sigma_pair)*latents+sigma_pair*noise_pair
            target = noise_pair-latents
            mask = batch["texture_mask"].to(self.device, dtype=torch.float32)
            mask = torch.cat([mask, mask])
        inputs = (noisy, torch.cat([self.timesteps[indices]]*2), torch.cat([prompt]*2), torch.cat([pooled]*2), torch.cat([control]*2), torch.cat([restoration]*2))
        policy = self.forward(*inputs)
        policy_error = texture_weighted_mse(policy, target, mask, self.settings["texture_strength"])
        with torch.no_grad():
            reference = self.forward(*inputs, reference=True)
            reference_error = texture_weighted_mse(reference, target, mask, self.settings["texture_strength"])
        pc, pr = policy_error.split(b)
        rc, rr = reference_error.split(b)
        weights = batch["pair_weight"].to(self.device)
        loss, _ = training_dpo_loss(pc, pr, rc, rr, pair_weights=weights, beta=float(self.train["beta"]), sft_weight=float(self.train.get("sft_weight", 0)))
        per_pair, _ = diffusion_dpo_loss(pc, pr, rc, rr, beta=float(self.train["beta"]), reduction="none")
        logit = float(self.train["beta"])*((pr-pc)-(rr-rc))
        data = []
        for i in range(b):
            dc, dr = float((pc[i]-rc[i]).detach()), float((pr[i]-rr[i]).detach())
            positive = max(0.0, dr)+max(0.0, -dc)
            p = float(logit[i].detach().sigmoid())
            data.append(dict(E_policy_chosen=float(pc[i].detach()), E_policy_rejected=float(pr[i].detach()),
                E_reference_chosen=float(rc[i]), E_reference_rejected=float(rr[i]),
                A_policy=float((pr[i]-pc[i]).detach()), A_reference=float(rr[i]-rc[i]), logit=float(logit[i].detach()), sigmoid=p,
                dpo_pair_loss=float(per_pair[i].detach()), pair_weight=float(weights[i]),
                dpo_logit_coefficient=float((float(self.train["beta"])*weights[i]/b*torch.sigmoid(-logit[i])).detach()),
                sigmoid_rounded_zero=p == 0, sigmoid_rounded_one=p == 1,
                chosen_error_delta_vs_reference=dc, rejected_error_delta_vs_reference=dr,
                both_fm_errors_worse_but_margin_better=dc > 0 and dr > 0 and dr-dc > 0,
                rejected_push_fraction= max(0.0, dr)/positive if positive else None,
                dpo_sigma=float(sigma[i].flatten()[0]), dpo_timestep=float(self.timesteps[indices[i]]),
                mask_mean=float(mask[i].mean()), mask_coverage=float((mask[i]>=.5).float().mean())))
            data[-1]["dpo_metrics_finite"] = all(math.isfinite(data[-1][k]) for k in (
                "E_policy_chosen", "E_policy_rejected", "E_reference_chosen", "E_reference_rejected", "logit", "dpo_pair_loss"))
        return loss, per_pair*weights/b, data

    def gt(self, batch):
        with torch.no_grad():
            pixels = batch["aux_gt_pixel_values"].to(self.device, dtype=self.vae.dtype)
            latents = self.vae.encode(pixels).latent_dist.mode()
            latents = ((latents-self.vae.config.shift_factor)*self.vae.config.scaling_factor).to(self.dtype)
            control, restoration = self.conditions(batch["aux_conditioning_pixel_values"])
            prompt, pooled = self.embeddings(batch["aux_prompts"])
            chosen = self.gt_indices[torch.randint(len(self.gt_indices), (len(pixels),), device=self.device)]
            sigma = self.sigmas[chosen].to(self.dtype).view(-1,1,1,1)
            noise = torch.randn_like(latents)
            noisy = (1-sigma)*latents+sigma*noise
        prediction = self.forward(noisy, self.timesteps[chosen], prompt, pooled, control, restoration)
        graph = {"gt_prediction_requires_grad": prediction.requires_grad}
        def observe(module, inputs):
            graph["vae_decoder_input_requires_grad"] = inputs[0].requires_grad
        hook = self.vae.decoder.register_forward_pre_hook(observe)
        try:
            loss = decoded_gt_mse(self.vae, prediction, noisy, sigma, pixels)
        finally:
            hook.remove()
        graph.update(gt_loss_requires_grad=loss.requires_grad, gt_batch_mean_loss=float(loss.detach()),
                     gt_sigmas=[float(x) for x in sigma.flatten()], gt_timesteps=[float(x) for x in self.timesteps[chosen]])
        return loss, graph

    def finish_integrity(self):
        policy = tensor_digest(self.named_policy_parameters())
        reference = tensor_digest(self.reference_tensors())
        buffers = tensor_digest(self.named_buffers())
        return {"policy_parameters_unchanged": self.initial_policy == policy,
                "reference_unchanged": self.initial_reference == reference,
                "buffers_unchanged": self.initial_buffers == buffers,
                "parameter_checksums_after":{"all_policy_and_vae":policy,"reference":reference,"buffers":buffers},
                "reference_has_gradient": any(t.grad is not None for _, t in self.reference_tensors()),
                "vae_has_gradient": any(p.grad is not None for p in self.vae.parameters())}

    @torch.no_grad()
    def inference_check(self, source_records, rows, directory):
        from diffusers import StableDiffusion3ControlNetPipeline
        from torchvision.transforms.functional import to_pil_image
        from utils.evaluate_sd3 import _get_lpips_model, lpips_batch, psnr_batch, ssim_batch
        self.controlnet.eval()
        self.transformer.eval()
        options = {"resolution": self.settings["resolution"],
                   "num_inference_steps": int(self.train["validation_num_inference_steps"]),
                   "guidance_scale": float(self.train["validation_guidance_scale"]),
                   "strength": float(self.train["validation_strength"]),
                   "controlnet_conditioning_scale": float(self.model_config.get("controlnet_conditioning_scale", 1))}
        pipeline = StableDiffusion3ControlNetPipeline(scheduler=copy.deepcopy(self.scheduler), vae=self.vae,
            text_encoder=None, tokenizer=None, text_encoder_2=None, tokenizer_2=None, text_encoder_3=None, tokenizer_3=None,
            transformer=self.transformer, controlnet=self.controlnet)
        pipeline.set_progress_bar_config(disable=True)
        runner = PipelineRunner(self, pipeline, options)
        substitutions = {"controlnet."+name:value for name,value in {**self.reference_controlnet, **self.sft_control_buffers}.items()}
        substitutions.update({"transformer."+name:value for name,value in {**self.reference_ra, **self.sft_ra_buffers}.items()})
        lpips_model = _get_lpips_model(self.train.get("validation_lpips_net", "alex"), device=self.device)
        sorted_indices = sorted(self.gt_indices.tolist(), key=lambda i: float(self.sigmas[i]))
        probes = [("low", sorted_indices[0]), ("middle", sorted_indices[len(sorted_indices)//2]), ("high", sorted_indices[-1])]
        saved_outputs = []
        for source in tqdm(source_records, desc="Paired real inference", unit="source"):
            if not source_is_training(source):
                raise ValueError("Refusing held-out/test paths in training diagnostic")
            key = tuple(source["source_key"])
            sid = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:12]
            folder = directory/"visualizations"/(key[0]+"_"+sid)
            folder.mkdir(parents=True, exist_ok=False)
            gt01 = aligned_rgb(source["gt_path"], self.settings["resolution"])[None].to(self.device)
            lq01 = aligned_rgb(source["lq_path"], self.settings["resolution"])[None].to(self.device)
            pixels = gt01*2-1
            latents = self.vae.encode(pixels.to(self.vae.dtype)).latent_dist.mode()
            latents = ((latents-self.vae.config.shift_factor)*self.vae.config.scaling_factor).to(self.dtype)
            control, restoration = self.conditions(lq01*2-1)
            prompt, pooled = self.embeddings([source["prompt"]])
            negative, negative_pooled = self.embeddings([""])
            seed = int.from_bytes(hashlib.sha256(f"{self.args.seed}|{key}".encode()).digest()[:8], "little") % (2**63-1)
            generator = torch.Generator(device="cpu").manual_seed(seed)
            noise = torch.randn(latents.shape, generator=generator, dtype=torch.float32).to(self.device, dtype=self.dtype)
            mask = gt_texture_mask(gt01, self.settings["mask"])
            save_image(folder/"gt.png", gt01[0])
            save_image(folder/"lq.png", lq01[0])
            save_image(folder/"mask.png", mask[0])
            base = dict(weather=key[0], gt_weather=key[0], source_key=key, source_id=key[-1], gt_path=source["gt_path"],
                        lq_path=source["lq_path"], seed=seed, prompt=source["prompt"], weights="raw", inference_parameters=options,
                        shared_noise_sha256=tensor_digest([("noise",noise)]), shared_gt_latent_sha256=tensor_digest([("gt_latent",latents)]))
            proxy_changes = []
            for label, index in probes:
                sigma = self.sigmas[index].to(self.dtype).view(1,1,1,1)
                noisy = (1-sigma)*latents+sigma*noise
                images, losses = {}, {}
                for name, reference in (("sft", True), ("current", False)):
                    prediction = self.forward(noisy, self.timesteps[index:index+1], prompt, pooled, control, restoration,
                                              reference=reference, full_sft=reference)
                    x0 = noisy.float()-sigma.float()*prediction.float()
                    with torch.autocast("cuda", enabled=False):
                        decoded = self.vae.decode((x0/self.vae.config.scaling_factor+self.vae.config.shift_factor).to(self.vae.dtype), return_dict=False)[0].float()
                    images[name] = (decoded+1)/2
                    losses[name] = float((images[name]-gt01).square().mean())
                    if not math.isfinite(losses[name]):
                        raise FloatingPointError("Nonfinite matched one-step GT reconstruction: " + name)
                    save_image(folder/f"one_step_{label}_{name}.png", images[name][0])
                errors = {name:(image-gt01).square().mean(1,keepdim=True) for name,image in images.items()}
                scale = max(1e-12, *(float(e.max()) for e in errors.values()))
                for name, error in errors.items():
                    save_image(folder/f"one_step_{label}_{name}_error.png", error[0]/scale)
                change = losses["current"]-losses["sft"]
                proxy_changes.append(change)
                rows.append({**base, "row_type":"one_step_compare", "probe":label, "gt_sigma":float(sigma.flatten()[0]),
                             "scheduler_sigma":float(self.sigmas[index]), "gt_timestep":float(self.timesteps[index]),
                             "sft_gt_mse":losses["sft"], "current_gt_mse":losses["current"], "gt_mse_change":change,
                             "error_visual_scale":scale})
            images, metrics = {}, {}
            pil = to_pil_image(lq01[0].cpu())
            for name, reference in (("sft",True),("current",False)):
                with self.autocast():
                    image = (functional_call(runner, substitutions, (pil,noise,prompt,pooled,negative,negative_pooled), strict=False)
                             if reference else runner(pil,noise,prompt,pooled,negative,negative_pooled))
                images[name] = image
                metrics[name] = {"psnr":float(psnr_batch(image,gt01)[0]), "ssim":float(ssim_batch(image,gt01)[0]),
                                 "lpips":float(lpips_batch(lpips_model,image,gt01,self.device,self.dtype)[0])}
                if not all(math.isfinite(v) for v in metrics[name].values()):
                    raise FloatingPointError("Nonfinite full-inference metrics: " + name)
                save_image(folder/f"full_{name}.png",image[0])
                saved_outputs.append({"source":source, "method":name, "path":str(folder/f"full_{name}.png")})
            errors = {name:(image-gt01).square().mean(1,keepdim=True) for name,image in images.items()}
            scale = max(1e-12, *(float(e.max()) for e in errors.values()))
            for name,error in errors.items():
                save_image(folder/f"full_{name}_error.png",error[0]/scale)
            delta = metrics["current"]["psnr"]-metrics["sft"]["psnr"]
            mean_proxy = sum(proxy_changes)/len(proxy_changes)
            case = "C_proxy_better_full_psnr_worse" if mean_proxy < 0 and delta < 0 else "D_proxy_and_full_psnr_worse" if mean_proxy > 0 and delta < 0 else "other_or_tie"
            rows.append({**base, "row_type":"inference_compare", **{m+"_"+k:v for m, values in metrics.items() for k,v in values.items()},
                         "full_psnr_change":delta,"mean_gt_mse_change":mean_proxy,
                         "sigma_probe_signs":[int(x>0)-int(x<0) for x in proxy_changes], "case":case, "error_visual_scale":scale})
        del runner, pipeline
        return saved_outputs


def summarize_gradients(rows):
    summary = {}
    for kind in ("micro_gradient", "accum_gradient"):
        records = [row for row in rows if row.get("row_type") == kind]
        combinations = sorted({(r["weather"],r["gt_weather"]) for r in records})
        for group in SCOPES:
            for weather, gt_weather in [("all","all"), *combinations]:
                selected = [r for r in records if r["scope"] == group and (weather == "all" or (r["weather"],r["gt_weather"]) == (weather,gt_weather))]
                cos = [r["cosine"] for r in selected if r.get("cosine") is not None]
                dot = [r["scheduled_gt_dot_total"] for r in selected if r.get("scheduled_gt_dot_total") is not None and r.get("gt_due")]
                summary[f"{kind}/{group}/DPO={weather},GT={gt_weather}"] = {
                    **{name:distribution([r.get(name) for r in selected]) for name in ("dpo_norm","gt_norm","weighted_gt_norm","gt_to_dpo_ratio","cosine","gt_dot_total","scheduled_gt_dot_total")},
                    "negative_cosine_count":sum(x<0 for x in cos),"cosine_valid_count":len(cos),
                    "negative_cosine_fraction":sum(x<0 for x in cos)/len(cos) if cos else None,
                    "negative_scheduled_gt_dot_total_count":sum(x<0 for x in dot),"scheduled_dot_valid_count":len(dot),
                    "status_counts":dict(Counter(r["status"] for r in selected))}
    return summary


def summarize_runtime(runtime, rows):
    runtime["gradient_summary"] = summarize_gradients(rows)
    all_pairs = [r for r in rows if r.get("row_type") == "pair_sample" and r.get("pair_weight",0)>0]
    pairs = [r for r in all_pairs if r.get("dpo_metrics_finite")]
    findings = []
    findings.append(f"有效偏好样本中 DPO 数值非有限的对={len(all_pairs)-len(pairs)}；这些样本不作为正常的拉差/饱和统计。")
    runtime["dpo_summary"] = {}
    for weather in ["all", *sorted({r["weather"] for r in pairs})]:
        chosen = [r for r in pairs if weather == "all" or r["weather"] == weather]
        if chosen:
            n = len(chosen)
            runtime["dpo_summary"][weather] = {
                **{key:distribution([r.get(key) for r in chosen]) for key in (
                    "E_policy_chosen", "E_policy_rejected", "E_reference_chosen", "E_reference_rejected",
                    "A_policy", "A_reference", "logit", "sigmoid", "dpo_pair_loss", "dpo_logit_coefficient", "rejected_push_fraction")},
                "both_fm_errors_worse_but_margin_better":sum(r["both_fm_errors_worse_but_margin_better"] for r in chosen),
                "sigmoid_rounded_zero_count":sum(r["sigmoid_rounded_zero"] for r in chosen),
                "sigmoid_rounded_one_count":sum(r["sigmoid_rounded_one"] for r in chosen)}
            findings.append(f"DPO {weather}: {sum(r['both_fm_errors_worse_but_margin_better'] for r in chosen)}/{n} 对在 chosen/rejected FM 误差都高于参考时仍得到更优偏好差；sigmoid 数值舍入到1的对={sum(r['sigmoid_rounded_one'] for r in chosen)}，logit导数为0的对={sum(r['dpo_logit_coefficient']==0 for r in chosen)}。")
    runtime["per_sample_gradient_summary"] = {}
    for scope in SCOPES:
        field = "individual_gradient_norm_"+scope
        available = [r for r in pairs if field in r]
        norms = [r.get(field) for r in available]
        per_batch = defaultdict(list)
        per_window = defaultdict(list)
        for r in available:
            per_batch[(r["window"],r["microbatch"])].append(r.get(field))
            per_window[r["window"]].append(r.get(field))
        effective = []
        for values in per_batch.values():
            if values and all(v is not None and math.isfinite(v) for v in values) and sum(v*v for v in values)>0:
                effective.append(sum(values)**2/sum(v*v for v in values))
        runtime["per_sample_gradient_summary"][scope] = {
            "measured":bool(available), "individual_norm":distribution(norms),
            "within_microbatch_norm_share":distribution([r.get("individual_norm_share_"+scope) for r in available]),
            "effective_norm_contributors_per_microbatch":distribution(effective),
            "effective_norm_contributors_per_accumulation_window":distribution([
                sum(v)**2/sum(x*x for x in v) for v in per_window.values()
                if v and all(x is not None and math.isfinite(x) for x in v) and sum(x*x for x in v)>0]),
            "largest_norm_share_per_accumulation_window":distribution([
                max(v)/sum(v) for v in per_window.values()
                if v and all(x is not None and math.isfinite(x) for x in v) and sum(v)>0]),
            "largest_norm_samples":[{k:r.get(k) for k in ("pair_id","weather","gt_weather","source_key","dpo_sigma",field)}
                                    for r in sorted((r for r in available if r.get(field) is not None and math.isfinite(r[field])), key=lambda r:r[field],reverse=True)[:3]]}
    combined = [r for r in rows if r.get("row_type") == "accum_gradient" and r.get("scope") == "all"]
    ratios = distribution([r.get("gt_to_dpo_ratio") for r in combined])
    finite_cos = [r["cosine"] for r in combined if r.get("cosine") is not None]
    dot_bad = [r for r in combined if r.get("gt_due") and r.get("scheduled_gt_dot_total") is not None and r["scheduled_gt_dot_total"]<0]
    findings.append(f"累计梯度：加权GT/DPO范数比中位数={ratios['median']}；有效余弦中负值={sum(x<0 for x in finite_cos)}/{len(finite_cos)}；实际启用GT的窗口中 g_F·g_total<0 的窗口={len(dot_bad)}。这些数值不设经验合理性阈值。")
    samples_with_alignment = [r for r in pairs if "mask_alignment_close_at_1e_5" in r]
    findings.append(f"返回张量的GT/mask对齐数值复查：{sum(not r['mask_alignment_close_at_1e_5'] for r in samples_with_alignment)}/{len(samples_with_alignment)} 超出1e-5数值容差；该检查包含RGB归一化往返误差，详情见CSV，不能单独当作裁剪错位的证据。")
    full = [r for r in rows if r.get("row_type") == "inference_compare"]
    cases = Counter(r["case"] for r in full)
    if full:
        runtime["inference_summary"] = {"source_count":len(full),"case_counts":dict(cases),
            "correlation_of_proxy_improvement_and_full_psnr_change":correlation([-r["mean_gt_mse_change"] for r in full],[r["full_psnr_change"] for r in full]),
            "full_psnr_change":distribution([r["full_psnr_change"] for r in full]), "mean_gt_mse_change":distribution([r["mean_gt_mse_change"] for r in full])}
        runtime["inference_summary"]["per_weather"] = {w:{
            "n":len([r for r in full if r["weather"] == w]),
            "full_psnr_change":distribution([r["full_psnr_change"] for r in full if r["weather"] == w]),
            "mean_gt_mse_change":distribution([r["mean_gt_mse_change"] for r in full if r["weather"] == w])}
            for w in sorted({r["weather"] for r in full})}
        findings.append(f"匹配的完整推理共{len(full)}个训练源图，按三个sigma的平均一步MSE分类：C={cases['C_proxy_better_full_psnr_worse']}，D={cases['D_proxy_and_full_psnr_worse']}；每个sigma明细见CSV。")
    else:
        findings.append("未执行配对完整推理，情况C/D尚未验证。")
    priorities = []
    if (not runtime.get("policy_parameters_unchanged",True) or not runtime.get("reference_unchanged",True)
        or len(all_pairs) != len(pairs) or any(r.get("status") == "nonfinite_gradient" for r in combined)):
        priorities.append("优先核查参数完整性/非有限梯度证据，不能先靠提高λ解释或掩盖问题。")
    if any(r.get("gt_norm") == 0 or r.get("gt_gradient_state") == "all_missing" for r in combined):
        priorities.append("存在GT梯度缺失或全零窗口：先核查图连接、样本和数值状态，不能直接用更大的λ修复零梯度。")
    if cases["C_proxy_better_full_psnr_worse"]:
        priorities.append("存在情况C样本：优先复查一步GT代理与完整采样的一致性；先复现该小组样本，不直接增加GT权重。")
    if dot_bad:
        priorities.append("存在累计梯度一阶冲突：优先复查对应来源/时间步的冲突；加权GT范数比需同时考虑，不能只根据负余弦选λ。")
    if not priorities:
        priorities.append("先结合累计范数比及完整推理复现结果判断：只有GT相对弱、冲突证据少且代理方向一致时，再做一次加强固定GT权重的受控验证；本次不自动修改。")
    if not full:
        priorities.append("补做 --full-inference，以区分代理失配(C)与两者同步退化(D)，避免仅凭梯度比例决定改训练代理。")
    runtime.update(findings=findings, priorities=priorities)
    runtime["measured_microbatches"] = sum(r.get("row_type") == "micro_gradient" and r.get("scope") == "all" for r in rows)
    runtime["measured_accumulation_windows"] = len(combined)


def run_diagnostics(args, state, rows, directory):
    runtime = state["runtime"]
    engine = None
    ledger = None
    outputs = []
    try:
        engine = RealEngine(args, state["snapshot"], runtime)
        disk_bytes = 3*4*sum(e["numel"] for e in engine.registry)
        runtime["gradient_scratch_bytes"] = disk_bytes
        print(f"[diagnose] temporary gradient scratch: {disk_bytes/1024**3:.2f} GiB; CPU vectors are streamed in chunks", flush=True)
        if shutil.disk_usage(directory).free < disk_bytes:
            raise RuntimeError(f"Need {disk_bytes} bytes of free diagnostic scratch disk for bounded-RAM gradient vectors")
        ledger = GradientLedger(directory/"gradient_buffers", engine.registry, engine.accumulation)
        coefficient = engine.settings["lambda_fidelity"]
        runtime["executed"] = True
        for window, plan in enumerate(engine.plan):
            ledger.reset()
            step = int(engine.meta["global_step"])+window
            due = step % engine.settings["gt"]["interval"] == 0
            for micro, indices in enumerate(plan["indices"]):
                batch = collate_spatial_pairs([engine.dataset[index] for index in indices])
                loss, per_pair, detail = engine.dpo(batch)
                records = []
                for i, (index, item) in enumerate(zip(indices, detail)):
                    pair = engine.dataset.records[index % len(engine.dataset.records)] if engine.dataset.records else None
                    gt = engine.dataset.gt_records[index % len(engine.dataset.gt_records)]
                    if not source_is_training(gt):
                        raise ValueError("Diagnostic GT source appears to be a held-out/test sample")
                    record = {"row_type":"pair_sample", "window":window,"microbatch":micro,"sample":i,
                              "weather":plan["weather"],"gt_weather":gt["weather"], "dataset_index":index,
                              "pair_id":pair["pair_id"] if pair else "GT_only",
                              "source_key":pair["source_key"] if pair else gt["source_key"], "gt_source_key":gt["source_key"],
                              "chosen_path":pair["chosen_path"] if pair else None, "rejected_path":pair["rejected_path"] if pair else None,
                              "pair_gt_path":pair["gt_path"] if pair else gt["gt_path"],
                              "pair_lq_path":pair["lq_path"] if pair else gt["lq_path"],
                              "gt_path":gt["gt_path"],"lq_path":gt["lq_path"], **item}
                    record["dpo_batch_mean_loss"] = float(loss.detach())
                    if args.per_sample_gradients:
                        if per_pair[i].requires_grad:
                            (per_pair[i]*engine.backward_factor).backward(retain_graph=True)
                        gradients = [parameter.grad for parameter in engine.parameters]
                        norms = individual_gradient_norms(gradients, engine.registry, engine.backward_factor)
                        for parameter in engine.parameters:
                            parameter.grad = None
                        del gradients
                        for scope, values in norms.items():
                            record["individual_gradient_norm_"+scope] = values["norm"]
                            record["individual_missing_parameters_"+scope] = values["missing"]
                            record["individual_nonfinite_chunks_"+scope] = values["nonfinite_chunks"]
                    records.append(record)
                if loss.requires_grad:
                    (loss*engine.backward_factor).backward()
                ledger.collect_q(engine.backward_factor)
                del loss, per_pair
                gt_loss, graph = engine.gt(batch)
                # Geometry check on the ACTUAL returned tensors, including any
                # shared random flip. Numerical tolerance is not a loss-ratio cutoff.
                aligned_mask = gt_texture_mask((batch["gt_pixel_values"]+1)/2, engine.settings["mask"])
                alignment_error = float((aligned_mask-batch["texture_mask"]).abs().max())
                graph["mask_alignment_max_abs"] = alignment_error
                graph["mask_alignment_close_at_1e_5"] = bool(torch.allclose(aligned_mask, batch["texture_mask"], atol=1e-5, rtol=1e-5))
                for record, sigma, timestep in zip(records, graph["gt_sigmas"], graph["gt_timesteps"]):
                    record.update(gt_sigma=sigma, gt_timestep=timestep, gt_due=due, **{k:v for k,v in graph.items() if k not in ("gt_sigmas","gt_timesteps")})
                if args.per_sample_gradients:
                    for scope in SCOPES:
                        norms = [r["individual_gradient_norm_"+scope] for r in records]
                        valid = all(n is not None for n in norms)
                        mass = sum(norms) if valid else None
                        for r, n in zip(records, norms):
                            r["individual_norm_share_"+scope] = n/mass if mass else None
                rows.extend(records)
                if gt_loss.requires_grad:
                    (gt_loss*engine.backward_factor).backward()
                measured = ledger.collect_f(engine.backward_factor, coefficient, due)
                del gt_loss
                for scope, values in measured.items():
                    rows.append({"row_type":"micro_gradient","window":window,"microbatch":micro,
                                 "weather":plan["weather"],"gt_weather":plan["gt_weather"],"scope":scope,
                                 "diagnostic_step":step,"GT_probe_only":not due, **values})
                print(f"[diagnose] window={window+1}/{args.batches} micro={micro+1}/{engine.accumulation} DPO={plan['weather']} GT={plan['gt_weather']}", flush=True)
            for scope, values in ledger.accumulated(coefficient, due).items():
                rows.append({"row_type":"accum_gradient","window":window,"weather":plan["weather"],
                             "gt_weather":plan["gt_weather"],"scope":scope,"microbatch_count":engine.accumulation,
                             "local_effective_batch":engine.batch_size*engine.accumulation, **values})
        ledger.close(args.keep_gradient_buffers)
        ledger = None
        if args.full_inference:
            selected, counts, seen = [], Counter(), set()
            for plan in engine.plan:
                for indices in plan["indices"]:
                    for index in indices:
                        source = engine.dataset.gt_records[index % len(engine.dataset.gt_records)]
                        key = tuple(source["source_key"])
                        if key not in seen and counts[source["weather"]] < args.inference_samples_per_weather:
                            seen.add(key)
                            selected.append(source)
                            counts[source["weather"]] += 1
            outputs = engine.inference_check(selected, rows, directory)
    finally:
        if ledger is not None:
            ledger.close(args.keep_gradient_buffers)
        if engine is not None:
            try:
                runtime.update(engine.finish_integrity())
            except Exception as error:
                runtime.update(policy_parameters_unchanged=None, reference_unchanged=None,
                               buffers_unchanged=None, integrity_check_error=str(error))
        summarize_runtime(runtime, rows)
    # Optional IQA is done only after SD3 has been released, with ONE saved
    # normalization artifact for both methods; never fit separate distributions.
    frozen_train = copy.deepcopy(engine.train) if engine is not None else None
    frozen_config = copy.deepcopy(engine.config) if engine is not None else None
    if engine is not None:
        del engine
    gc.collect()
    torch.cuda.empty_cache()
    if args.local_iqa:
        from .texture_preparation import PyIQAPatchScorer
        from .reward_normalization import apply_robust_statistics
        from .spatial_config import LOCAL_METRICS, LOCAL_COEFFICIENTS
        with Path(frozen_train["preference_manifest"]).with_name("preference_summary.json").open(encoding="utf-8") as f:
            summary = json.load(f)
        with Path(summary["normalization_path"]).open(encoding="utf-8") as f:
            norm = json.load(f)
        from .tail_risk import file_sha256
        if file_sha256(summary["normalization_path"]) != summary["normalization_sha256"]:
            raise ValueError("Frozen local IQA normalization changed")
        runtime["comparison_local_normalization"] = {"path":summary["normalization_path"],"sha256":summary["normalization_sha256"]}
        scorer = PyIQAPatchScorer(frozen_config, args.device)
        same_backend = scorer.metadata == norm["iqa_backend"]
        runtime["local_iqa_normalization_status"] = "same_frozen_statistics" if same_backend else "backend_changed_raw_scores_only_no_refit"
        for output in tqdm(outputs, desc="Matched frozen local IQA"):
            source = output["source"]
            if not source["patches"]:
                rows.append({"row_type":"local_iqa","method":output["method"],"source_key":source["source_key"],"status":"no_reliable_texture_patches"})
                continue
            image = aligned_rgb(output["path"], norm["preparation"]["resolution"])
            scores = scorer(torch.stack([image[:,y:y+h,x:x+w] for x,y,w,h in source["patches"]]))
            scores["weather"] = source["weather"]
            q = None
            if same_backend:
                apply_robust_statistics([scores], norm["statistics"], metrics=LOCAL_METRICS, clip_z=norm["clip_z"])
                q = sum(w*float(scores[m+"_z"]) for m,w in LOCAL_COEFFICIENTS.items())
            rows.append({"row_type":"local_iqa","method":output["method"],"source_key":source["source_key"],"weather":source["weather"],"Q_texture":q,**scores})
