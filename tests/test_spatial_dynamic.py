"""CPU numerical/gradient/alignment checks; fake IQA only, no model downloads."""
from __future__ import annotations

import ast
import copy
import csv
import hashlib
import json
import pickle
import tempfile
import types
import unittest
from contextlib import nullcontext
from pathlib import Path

import torch
from PIL import Image
from torch.func import functional_call
from torchvision.transforms.functional import to_pil_image

from dpo.losses import diffusion_dpo_loss, flow_matching_gt_losses
from dpo.reference import snapshot_reference_parameters
from dpo.spatial import (decoded_gt_mse, gt_texture_mask, gt_timestep_indices,
                         texture_weighted_mse, select_texture_patches)
from dpo.spatial_config import (MODE, SCHEMA, spatial_settings, training_signature,
                                preparation_contract, texture_filter_contract)
from dpo.spatial_dataset import SpatialPreferenceDataset, collate_spatial_pairs
from dpo.tail_risk_training import DPOLogWindow, training_dpo_loss
from dpo.texture_preparation import (score_texture_candidates, build_texture_preferences,
                                     _verify_candidate_pixels, _candidate_file_checksum, FIELDS)
from dpo.preference_provenance import validate_preference_reward

ROOT = Path(__file__).resolve().parents[1]


def image_cases(size=64):
    flat = torch.full((1, 3, size, size), 0.5)
    edge = flat.clone()
    edge[..., size//2:] = 1
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    texture = (((x//4 + y//4) % 2).float()*0.8+0.1)[None, None].repeat(1, 3, 1, 1)
    return flat, edge, texture


def config_at(root, output_name="scores"):
    return {"reward": {"mode": MODE, "min_texture_q_gap": 0.01},
            "spatial_dynamic": {"mask": {"window": 3, "smooth_window": 3},
                                "patches": {"size": 32, "stride": 16, "max_patches": 2}},
            "training": {"resolution": 64, "train_controlnet": True, "train_ra_fusion": True,
                         "gt_flow_weight": 0, "gt_x0_l1_weight": 0, "sft_weight": 0,
                         "preference_manifest": str(root/(output_name+"_prefs")/"preference_pairs.jsonl"),
                         "gt_source_manifest": str(root/output_name/"gt_sources.jsonl")},
            "texture_scoring": {"input_csv": str(root/"input.csv"), "output_dir": str(root/output_name), "visualize_groups": 2},
            "preference_filter": {"candidate_metrics_path": str(root/output_name/"per_candidate_texture.csv"),
                                  "output_dir": str(root/(output_name+"_prefs")), "pairs_per_group": 12,
                                  "min_psnr_gap": -0.15, "random_seed": 42, "shuffle": True,
                                  "fidelity_constraints": {"baseline_mode": "group_median", "baseline_psnr_tolerance": .5,
                                                           "baseline_dists_tolerance": .02, "max_dists_pair_degradation": .01}},
            "weather_prompts": {"rain": "clear"}}


class FakeScorer:
    metadata = {"backend": "synthetic_test_only"}
    def __init__(self):
        self.inputs = []
    def __call__(self, patches):
        self.inputs.append(patches.clone())
        value = float(patches.mean())
        return {name+"_texture": value*scale for name, scale in (("musiq", 100), ("clipiqa", 1), ("nima", 10))}


def prepare_fixture(root, all_flat=False, bad_fidelity=False):
    cfg = config_at(root)
    rows = []
    flat, _, texture = image_cases()
    for group, gt in enumerate((flat, flat if all_flat else texture)):
        folder = root/f"source_{group}"
        folder.mkdir()
        to_pil_image(gt[0]).save(folder/"gt.png")
        to_pil_image(gt[0]*0.8).save(folder/"lq.png")
        for index in range(4):
            path = folder/f"candidate_{index}.png"
            to_pil_image((gt[0]*.6 + .05*index).clamp(0, 1)).save(path)
            rows.append({"weather": "rain", "subdataset": "test", "global_index": group,
                         "noise_index": index, "candidate_index": index, "guidance_scale": 1,
                         "candidate_path": str(path), "gt_path": str(folder/"gt.png"), "lq_path": str(folder/"lq.png"),
                         "psnr": 30+.01*index, "dists": .10-.001*index, "lpips": .2-.001*index})
            # Real generator CSVs contain PIXEL checksums, not PNG file hashes.
            with Image.open(path) as saved:
                rgb = saved.convert("RGB")
                pixels = (b"".join(c.tobytes() for c in rgb.split()) if index % 2 == 0 else rgb.tobytes())
            rows[-1]["output_checksum_sha256"] = hashlib.sha256(pixels).hexdigest()
            if bad_fidelity and group == 1:
                if index == 3:
                    rows[-1]["psnr"] = 29
                if index == 2:
                    rows[-1]["dists"] = .5
    with (root/"input.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    scorer = FakeScorer()
    score = score_texture_candidates(cfg, scorer=scorer)
    summary = build_texture_preferences(cfg)
    train = {**cfg["training"], "reward": cfg["reward"], "preference_filter": cfg["preference_filter"],
             "spatial_dynamic": cfg["spatial_dynamic"]}
    return cfg, scorer, score, summary, train


class SpatialNumericsTest(unittest.TestCase):
    def setUp(self):
        self.settings = spatial_settings(config_at(Path("unused")))

    def test_flat_noise_edge_and_texture(self):
        flat, edge, texture = image_cases()
        masks = [gt_texture_mask(x.requires_grad_(), self.settings["mask"]) for x in (flat, edge, texture)]
        self.assertEqual(float(masks[0].max()), 0)
        self.assertGreater(float(masks[1].mean()), 0.01)
        self.assertGreater(float(masks[2].mean()), float(masks[1].mean()))
        self.assertEqual(float(gt_texture_mask(flat.detach()+torch.rand_like(flat)*1e-6, self.settings["mask"]).max()), 0)
        for mask in masks:
            self.assertFalse(mask.requires_grad)
            self.assertEqual(mask.dtype, torch.float32)
            self.assertTrue(bool(torch.isfinite(mask).all()))
            self.assertGreaterEqual(float(mask.min()), 0)
            self.assertLessEqual(float(mask.max()), 1)
        self.assertEqual(select_texture_patches(masks[0], self.settings["patches"]), [])
        a = select_texture_patches(masks[2], self.settings["patches"])
        self.assertTrue(a)
        self.assertEqual(a, select_texture_patches(masks[2].clone(), self.settings["patches"]))

    def test_spatial_reduction_uniform_and_zero_strength(self):
        torch.manual_seed(41)
        prediction, target = torch.randn(2, 4, 8, 8), torch.randn(2, 4, 8, 8)
        old = (prediction.float()-target.float()).square().flatten(1).mean(1)
        arbitrary = torch.rand(2, 1, 64, 64)
        torch.testing.assert_close(texture_weighted_mse(prediction, target, arbitrary, 0), old, rtol=0, atol=0)
        for constant in (0, .3, 1):
            torch.testing.assert_close(texture_weighted_mse(prediction, target, torch.full_like(arbitrary, constant), 2), old)
        mask = torch.zeros(2, 1, 8, 8)
        mask[..., :4] = 1
        error = (prediction-target).square().mean(1, keepdim=True)
        expected = ((1+2*mask)*error).sum((1, 2, 3))/(1+2*mask).sum((1, 2, 3))
        torch.testing.assert_close(texture_weighted_mse(prediction, target, mask, 2), expected)

    def test_scheduler_bounds_and_inverse_vae_parameterization(self):
        sigmas = torch.tensor([1, .9, .7, .5, .1, .05, 0])
        self.assertEqual(gt_timestep_indices(sigmas, 7, self.settings["gt"]).tolist(), [2, 3, 4, 5])
        with self.assertRaises(ValueError):
            gt_timestep_indices(torch.ones(5), 5, self.settings["gt"])
        vae = ToyVAE()
        clean = torch.randn(2, 3, 8, 8)
        sigma = torch.tensor([.05, .9999]).view(2, 1, 1, 1)
        noise = torch.randn_like(clean)
        noisy = (1-sigma)*clean+sigma*noise
        gt = clean/vae.config.scaling_factor+vae.config.shift_factor
        perfect = noise-clean
        self.assertLess(float(decoded_gt_mse(vae, perfect, noisy, sigma, gt)), 1e-11)
        pred = (perfect+.1).requires_grad_()
        loss = decoded_gt_mse(vae, pred, noisy, sigma, gt)
        loss.backward()
        self.assertGreater(float(pred.grad.abs().sum()), 0)
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertTrue(all(p.grad is None for p in vae.parameters()))

    def test_contract_ablation_and_static_guard(self):
        cfg = config_at(Path("unused"))
        ablation = copy.deepcopy(cfg)
        ablation["spatial_dynamic"]["texture_strength"] = 0
        self.assertEqual(preparation_contract(cfg), preparation_contract(ablation))
        self.assertEqual(texture_filter_contract(cfg), texture_filter_contract(ablation))
        train = {**cfg["training"], "reward": cfg["reward"], "spatial_dynamic": cfg["spatial_dynamic"]}
        other = {**train, "spatial_dynamic": ablation["spatial_dynamic"]}
        self.assertNotEqual(training_signature(train), training_signature(other))
        with self.assertRaises(ValueError):
            spatial_settings({**cfg, "reward": {"mode": "static"}})
        with self.assertRaises(ValueError):
            training_signature({**train, "gt_flow_weight": .1})


class TexturePipelineTest(unittest.TestCase):
    def test_generator_chw_and_hwc_pixel_hashes_are_not_file_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tree = ast.parse((ROOT/"utils/randomness_check.py").read_text(encoding="utf-8"))
            definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"tensor_to_pil", "output_checksum"}]
            namespace = {"torch": torch, "hashlib": hashlib, "Image": Image}
            exec(compile(ast.Module(body=definitions, type_ignores=[]), "actual_generation_checksums", "exec"), namespace)
            image = torch.arange(3*8*8).reshape(3,8,8).float()/(3*8*8)
            path = root/"candidate.png"
            namespace["tensor_to_pil"](image).save(path)
            chw = namespace["output_checksum"](image)
            with Image.open(path) as saved:
                hwc = hashlib.sha256(saved.convert("RGB").tobytes()).hexdigest()
            container = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(len({chw, hwc, container}), 3)
            _verify_candidate_pixels(path, chw)
            _verify_candidate_pixels(path, hwc)
            with self.assertRaisesRegex(ValueError, "PIXEL checksum mismatch"):
                _verify_candidate_pixels(path, container)
            self.assertEqual(_candidate_file_checksum({"candidate_path": str(path), "candidate_file_sha256": container,
                                                       "output_checksum_sha256": chw}), container)
            # Completed artifacts from the initial release remain readable.
            self.assertEqual(_candidate_file_checksum({"candidate_path": str(path), "output_checksum_sha256": container}), container)
            Image.new("RGB", (8,8), "red").save(path)
            with self.assertRaisesRegex(ValueError, "PIXEL checksum mismatch"):
                _verify_candidate_pixels(path, chw)

    def test_archive_incomplete_and_restart_preserves_old_files_and_rejects_completed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg, _, _, _, _ = prepare_fixture(root)
            target = root/"failed_legacy"
            target.mkdir()
            old_raw = target/"per_candidate_texture_raw.csv"
            with old_raw.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([name for name in FIELDS if name != "candidate_file_sha256"])
            before = old_raw.read_bytes()
            cfg["texture_scoring"]["output_dir"] = str(target)
            with self.assertRaisesRegex(FileExistsError, "restart-incomplete"):
                score_texture_candidates(cfg, scorer=FakeScorer())
            fresh = score_texture_candidates(cfg, scorer=FakeScorer(), restart_incomplete=True)
            backups = list(root.glob("failed_legacy.incomplete-*"))
            self.assertEqual(len(backups), 1)
            self.assertEqual((backups[0]/old_raw.name).read_bytes(), before)
            self.assertEqual(fresh["num_candidate_rows"], 8)
            self.assertTrue((target/"texture_scoring_attempt.json").is_file())
            with self.assertRaisesRegex(FileExistsError, "Completed"):
                score_texture_candidates(cfg, scorer=FakeScorer(), restart_incomplete=True)
            unknown = root/"unrelated"
            unknown.mkdir()
            (unknown/"notes.txt").write_text("user work", encoding="utf-8")
            cfg["texture_scoring"]["output_dir"] = str(unknown)
            with self.assertRaisesRegex(ValueError, "unrelated files"):
                score_texture_candidates(cfg, scorer=FakeScorer(), restart_incomplete=True)
            self.assertEqual((unknown/"notes.txt").read_text(), "user work")

    def test_same_crops_mask_bindings_independent_gt_and_versioning(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg, scorer, score, summary, train = prepare_fixture(root)
            self.assertEqual(score["rejected"]["no_reliable_texture_group"], 1)
            self.assertEqual(summary["num_gt_sources"], 2)
            self.assertEqual(summary["num_groups_with_pairs"], 1)
            self.assertGreater(summary["num_preference_pairs"], 0)
            with (root/"input.csv").open(newline="", encoding="utf-8") as f:
                original = list(csv.DictReader(f))
            with Path(score["candidate_metrics_path"]).open(newline="", encoding="utf-8") as f:
                scored = list(csv.DictReader(f))
            for old, new in zip(original, scored):
                self.assertEqual(new["output_checksum_sha256"], old["output_checksum_sha256"])
                self.assertEqual(new["candidate_file_sha256"], hashlib.sha256(Path(new["candidate_path"]).read_bytes()).hexdigest())
            # Exactly the texture source was scored, with identical crop layouts.
            self.assertEqual(len(scorer.inputs), 4)
            self.assertTrue(all(x.shape == scorer.inputs[0].shape for x in scorer.inputs))
            with Path(score["gt_manifest_path"]).open() as f:
                sources = [json.loads(line) for line in f]
            gt = image_cases()[2][0]
            for batch, index in zip(scorer.inputs, range(4)):
                candidate = ((gt*.6+.05*index)*255).byte().float()/255
                expected = torch.stack([candidate[:, y:y+h, x:x+w] for x,y,w,h in sources[1]["patches"]])
                torch.testing.assert_close(batch, expected, rtol=0, atol=0)
            proof = validate_preference_reward(train)
            self.assertEqual(proof["schema"], SCHEMA)
            ds = SpatialPreferenceDataset(train["preference_manifest"], train["gt_source_manifest"], spatial_settings(cfg))
            self.assertIn("source_0", ds.gt_records[0]["gt_path"])
            self.assertEqual(ds[0]["aux_prompt"], "clear")
            torch.testing.assert_close(ds[0]["aux_gt_pixel_values"], torch.full((3,64,64), 127/255*2-1))
            batch = collate_spatial_pairs([ds[0], ds[1]])
            self.assertEqual(batch["texture_mask"].shape, (2,1,64,64))
            self.assertGreater(float(batch["texture_mask"].mean()), 0)
            self.assertEqual(len(pickle.loads(pickle.dumps(ds))), len(ds))
            flipped_cfg = copy.deepcopy(cfg)
            flipped_cfg["spatial_dynamic"]["horizontal_flip_probability"] = 1
            flipped = SpatialPreferenceDataset(train["preference_manifest"], train["gt_source_manifest"], spatial_settings(flipped_cfg))[0]
            for key in ("chosen_pixel_values", "rejected_pixel_values", "gt_pixel_values", "conditioning_pixel_values", "texture_mask"):
                torch.testing.assert_close(flipped[key], ds[0][key].flip(-1), rtol=0, atol=0)
            self.assertTrue((root/"scores"/"visualizations"/"group_0001"/"patch_locations.png").is_file())
            with self.assertRaises(FileExistsError):
                score_texture_candidates(cfg, scorer=FakeScorer())
            ablation = copy.deepcopy(train)
            ablation["spatial_dynamic"]["texture_strength"] = 0
            self.assertEqual(validate_preference_reward(ablation), proof)
            with self.assertRaises(ValueError):
                validate_preference_reward({**train, "reward": {"mode": "dynamic"}})
            sources[1]["mask_sha256"] = "changed"
            with Path(train["gt_source_manifest"]).open("w") as f:
                f.writelines(json.dumps(x)+"\n" for x in sources)
            with self.assertRaises(ValueError):
                validate_preference_reward(train)

    def test_version6_resume_rejects_spatial_ablation_and_legacy_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, _, _, train = prepare_fixture(root)
            tree = ast.parse((ROOT/"scripts/train_dpo_sd3.py").read_text(encoding="utf-8"))
            names = {"file_checksum", "effective_ra_learning_rate", "write_resume_metadata", "validate_resume_metadata"}
            definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
            namespace = {"Path": Path, "hashlib": hashlib, "json": json,
                         "tail_risk_resume_signature": lambda c: {"enabled": False}}
            exec(compile(ast.Module(body=definitions, type_ignores=[]), "spatial_resume", "exec"), namespace)
            checkpoint = root/"checkpoint"
            checkpoint.mkdir()
            namespace["write_resume_metadata"](checkpoint, 0, 0, 0, train, True, True, 1, 6)
            meta = namespace["validate_resume_metadata"](checkpoint, train, True, True, 1, 6)
            self.assertEqual(meta["version"], 6)
            modified = copy.deepcopy(train)
            modified["spatial_dynamic"]["texture_strength"] = 0
            with self.assertRaisesRegex(ValueError, "spatial_dynamic"):
                namespace["validate_resume_metadata"](checkpoint, modified, True, True, 1, 6)
            del meta["spatial_dynamic"]
            meta["version"] = 5
            (checkpoint/"dpo_resume.json").write_text(json.dumps(meta), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "spatial_dynamic"):
                namespace["validate_resume_metadata"](checkpoint, train, True, True, 1, 6)

    def test_no_texture_still_has_gt_only_training_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg, scorer, _, summary, train = prepare_fixture(Path(temporary), all_flat=True)
            self.assertEqual(summary["num_preference_pairs"], 0)
            self.assertEqual(len(scorer.inputs), 0)
            validate_preference_reward(train)
            ds = SpatialPreferenceDataset(train["preference_manifest"], train["gt_source_manifest"], spatial_settings(cfg))
            self.assertEqual(len(ds), 2)
            self.assertEqual(ds[0]["pair_weight"], 0)
            self.assertEqual(ds.num_preference_pairs, 0)

    def test_full_pipeline_preserves_hard_gates_and_does_not_pad_pairs(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, _, summary, _ = prepare_fixture(Path(temporary), bad_fidelity=True)
            self.assertEqual(summary["num_preference_pairs"], 1)
            self.assertGreater(summary["rejected"]["psnr_gap"], 0)
            self.assertGreater(summary["rejected"]["dists_gap"], 0)
            self.assertEqual(summary["num_gt_sources"], 2)

    def test_alignment_and_changed_candidate_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg, _, _, _, train = prepare_fixture(root)
            ds = SpatialPreferenceDataset(train["preference_manifest"], train["gt_source_manifest"], spatial_settings(cfg))
            Image.new("RGB", (64,64), "red").save(ds.records[0]["chosen_path"])
            with self.assertRaisesRegex(ValueError, "Candidate changed"):
                ds[0]
            Image.new("RGB", (32,64), "gray").save(root/"source_0"/"lq.png")
            cfg["texture_scoring"]["output_dir"] = str(root/"misaligned_scores")
            with self.assertRaisesRegex(ValueError, "aspect ratios"):
                score_texture_candidates(cfg, scorer=FakeScorer())


class ToyVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.config = types.SimpleNamespace(scaling_factor=.5, shift_factor=.2)
    @property
    def dtype(self):
        return torch.float32
    def encode(self, pixels):
        return types.SimpleNamespace(latent_dist=types.SimpleNamespace(mode=lambda: pixels))
    def decode(self, latent, return_dict=False):
        return (latent*self.anchor,)


class ToyControl(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.3))
    def forward(self, hidden_states, controlnet_cond, **kwargs):
        return ([self.weight*(hidden_states+controlnet_cond*.1)],)


class ToyRA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ra_gain = torch.nn.Parameter(torch.tensor(.4))
    def forward(self, hidden_states, block_controlnet_hidden_states, restoration_cond, **kwargs):
        return (hidden_states*.2+self.ra_gain*block_controlnet_hidden_states[0]+restoration_cond*.1,)


def execute_trainer_microbatch(tree=None, spatial_mode=True, pair_active=True, global_step=0, gt_interval=1):
    """Execute the ACTUAL trainer's microbatch AST with tiny CPU policy/VAE stubs."""
    torch.manual_seed(923)
    tree = tree or ast.parse((ROOT/"scripts/train_dpo_sd3.py").read_text(encoding="utf-8"))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    block = next(n for n in ast.walk(main) if isinstance(n, ast.With) and any(
        isinstance(i.context_expr, ast.Call) and isinstance(i.context_expr.func, ast.Attribute)
        and i.context_expr.func.attr == "accumulate" for i in n.items))
    control, transformer, vae = ToyControl(), ToyRA(), ToyVAE()
    references = (snapshot_reference_parameters(control), snapshot_reference_parameters(transformer))
    grads = []
    class Accelerator:
        sync_gradients = True
        def accumulate(self, *args): return nullcontext()
        def autocast(self): return nullcontext()
        def backward(self, loss):
            before = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in (*control.parameters(), *transformer.parameters())]
            loss.backward()
            grads.append([p.grad.clone()-old for p, old in zip((*control.parameters(), *transformer.parameters()), before)])
        def clip_grad_norm_(self, *args): pass
    batch = {name:torch.rand(2,3,8,8)*1.5-.75 for name in (
        "chosen_pixel_values", "rejected_pixel_values", "gt_pixel_values", "conditioning_pixel_values",
        "aux_gt_pixel_values", "aux_conditioning_pixel_values")}
    batch.update(prompts=["clear"]*2, aux_prompts=["clear"]*2,
                 texture_mask=torch.rand(2,1,8,8), pair_weight=torch.full((2,), float(pair_active)),
                 is_tail_pair=torch.zeros(2,dtype=torch.bool), psnr_gap=torch.ones(2))
    optimizer = torch.optim.SGD([*control.parameters(),*transformer.parameters()], lr=.01)
    settings = spatial_settings(config_at(Path("unused"))) if spatial_mode else None
    if settings is not None:
        settings["gt"]["interval"] = gt_interval
    namespace = dict(torch=torch, accelerator=Accelerator(), accumulation_models=[control, transformer], train_controlnet=True, train_ra_fusion=True,
        controlnet=control, transformer=transformer, raw_controlnet=control, raw_transformer=transformer,
        reference_controlnet=references[0], reference_ra=references[1], vae=vae,
        device=torch.device("cpu"), weight_dtype=torch.float32, batch=batch,
        force_zero_pooled=False, model_config={}, prompt_embeddings=lambda prompts:(torch.zeros(len(prompts),1,1),torch.zeros(len(prompts),1)),
        scheduler=types.SimpleNamespace(config=types.SimpleNamespace(num_train_timesteps=4)),
        scheduler_timesteps=torch.tensor([1000.,700.,400.,100.]),scheduler_sigmas=torch.tensor([1.,.7,.4,.1]),
        spatial_gt_indices=torch.tensor([1,2,3]), spatial=settings, functional_call=functional_call,
        per_sample_mse=lambda p,t:(p.float()-t.float()).square().flatten(1).mean(1),
        texture_weighted_mse=texture_weighted_mse, training_dpo_loss=training_dpo_loss,
        flow_matching_gt_losses=flow_matching_gt_losses, decoded_gt_mse=decoded_gt_mse,
        beta=.1, train_config={}, mean_psnr_gap=1, gt_supervision_enabled=True,
        gt_flow_weight=.1, gt_x0_l1_weight=.01, global_step=global_step,
        log_window=DPOLogWindow(), trainable=[*control.parameters(),*transformer.parameters()],
        optimizer=optimizer, ema=None, lr_scheduler=types.SimpleNamespace(step=lambda:None))
    exec(compile(ast.Module(body=[block], type_ignores=[]), "actual_trainer_microbatch", "exec"), namespace)
    return namespace, grads, references


class TrainerIntegrationTest(unittest.TestCase):
    def test_actual_trainer_both_objectives_update_controlnet_ra_reference_frozen(self):
        namespace, grads, references = execute_trainer_microbatch()
        self.assertEqual(len(grads), 2)
        for objective_grads in grads:
            for grad in objective_grads:
                self.assertTrue(torch.isfinite(grad).all())
                self.assertGreater(float(grad.abs().sum()), 0)
        for snapshot in references:
            self.assertTrue(all(value.grad is None and not value.requires_grad for value in snapshot.values()))
        self.assertGreater(float(namespace["gt_rgb_mse"].detach()), 0)
        torch.testing.assert_close(namespace["weighted_gt_loss"], namespace["gt_rgb_mse"]*.1)
        self.assertEqual(float(namespace["gt_flow_loss"]), 0)
        self.assertEqual(float(namespace["gt_x0_l1_loss"]), 0)

    def test_gt_only_and_configured_interval_use_actual_trainer_path(self):
        namespace, grads, _ = execute_trainer_microbatch(pair_active=False)
        self.assertEqual(float(namespace["dpo_loss"]), 0)
        self.assertTrue(all(float(g.abs().sum()) == 0 for g in grads[0]))
        self.assertTrue(all(float(g.abs().sum()) > 0 for g in grads[1]))
        namespace, grads, _ = execute_trainer_microbatch(global_step=1, gt_interval=2)
        self.assertEqual(len(grads), 1)
        self.assertEqual(float(namespace["weighted_gt_loss"]), 0)
        self.assertEqual(float(namespace["stats"]["gt_aux_active"]), 0)


if __name__ == "__main__":
    unittest.main()
