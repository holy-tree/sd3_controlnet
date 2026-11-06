"""Validation sampling/statistics/config tests, not real SD3 model validation."""
import copy
import json
import tempfile
import unittest
from unittest.mock import patch
import types
from pathlib import Path

from PIL import Image
import torch
import yaml

from dpo.spatial import aligned_rgb
from dpo.texture_release_validation import (canonical_pixels,digest,prepare_validation,paired_summary,
    validation_decision,continuation_config,WEATHERS,METRICS)


def make_fixture(root):
    roots={};sources=[]
    for wi,w in enumerate(WEATHERS):
        folder=root/"train"/w/"sub";folder.mkdir(parents=True)
        (folder/"GT").mkdir();(folder/"LQ").mkdir()
        roots[w]=str(root/"train"/w)
        for i in range(13):
            color=(i+wi*50,(i*7+wi*29)%256,(i*13+wi*17)%256)
            lqcolor=tuple((x+23)%256 for x in color)
            Image.new("RGB",(48,32),color).save(folder/"GT"/f"{i}.png")
            Image.new("RGB",(48,32),lqcolor).save(folder/"LQ"/f"{i}.png")
        cache=root/"cache"/w;cache.mkdir(parents=True)
        Image.new("RGB",(32,32),(wi*50,wi*29,wi*17)).save(cache/"gt.png")
        Image.new("RGB",(32,32),tuple((x+23)%256 for x in (wi*50,wi*29,wi*17))).save(cache/"lq.png")
        sources.append({"weather":w,"gt_path":str(cache/"gt.png"),"lq_path":str(cache/"lq.png"),
                        "gt_sha256":digest(cache/"gt.png"),"lq_sha256":digest(cache/"lq.png"),"prompt":"clear "+w})
    manifest=root/"training_sources.jsonl"
    manifest.write_text("".join(json.dumps(s)+"\n" for s in sources),encoding="utf-8")
    import csv
    original=root/"original_candidates.csv"
    with original.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=["gt_path","lq_path"]);writer.writeheader()
        for w in WEATHERS:
            writer.writerow({"gt_path":str(root/"train"/w/"sub"/"GT"/"0.png"),"lq_path":str(root/"train"/w/"sub"/"LQ"/"0.png")})
    (root/"texture_scores.json").write_text(json.dumps({"input_csv":str(original),"input_sha256":digest(original)}),encoding="utf-8")
    probe=root/"probe"
    for name,coefficient in (("baseline",.1),("fidelity",1000)):
        directory=probe/name;directory.mkdir(parents=True)
        train={"reward":{"mode":"dynamic_texture"},"preference_filter":{},
               "spatial_dynamic":{"lambda_fidelity":coefficient},"beta":5000,"resolution":32,
               "train_batch_size":2,"gradient_accumulation_steps":4,"seed":42,"controlnet_learning_rate":1e-6,
               "ra_fusion_learning_rate":2e-6,"gt_source_manifest":str(manifest),"output_dir":str(directory),
               "validation_guidance_scale":1.,"validation_strength":1.,"validation_num_inference_steps":30,
               "lr_scheduler":"constant_with_warmup"}
        cfg={"training":train,"model":{"controlnet_model_path":"SFT/controlnet","ra_fusion_path":"SFT/ra_fusion"}}
        (directory/"dpo_config.yaml").write_text(yaml.safe_dump(cfg),encoding="utf-8")
        checkpoint=directory/"checkpoint-200";checkpoint.mkdir()
        for part in ("controlnet","ra_fusion"):
            (checkpoint/part).mkdir()
            # Opaque metadata fixture only; no model loader is called by tests.
            (checkpoint/part/"fixture.bin").write_bytes(b"metadata fixture, not model weights")
        meta={"version":6,"global_step":200,"preference_manifest_sha256":"unchanged",**{k:train[k] for k in
              ("beta","resolution","train_batch_size","gradient_accumulation_steps","seed","controlnet_learning_rate","ra_fusion_learning_rate")},
              "spatial_dynamic":{"settings":{"lambda_fidelity":coefficient,"resolution":32,"texture_strength":2}},
              "reward_provenance":{"gt_manifest_sha256":digest(manifest)}}
        (checkpoint/"dpo_resume.json").write_text(json.dumps(meta),encoding="utf-8")
        for name in ("optimizer.bin","scheduler.bin","random_states_0.pkl"):
            (checkpoint/name).write_bytes(b"opaque resume-file fixture; not executed")
    return probe,roots


def analytical_rows(plan,psnr_candidate=.1):
    rows=[]
    for source in plan["sources"]:
        for seed in plan["seeds"]:
            for method in ("sft","baseline","fidelity"):
                row={"source_id":source["id"],"weather":source["weather"],"seed":seed,"method":method}
                for metric,direction in METRICS.items():
                    base=30 if metric=="psnr" else 60 if metric=="musiq" else .5
                    if method=="sft":delta=0
                    elif method=="baseline":delta=-.2 if metric=="psnr" else (-.01*direction if metric not in ("musiq","clipiqa","nima","topiq_nr") else .6)
                    else:delta=psnr_candidate if metric=="psnr" else (.5 if metric=="musiq" else .01*direction)
                    row[metric]=base+delta
                rows.append(row)
    return rows


class TextureReleaseValidationTest(unittest.TestCase):
    def test_canonical_hash_matches_project_geometry(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"image.png"
            tensor=torch.arange(3*43*65).reshape(3,43,65).remainder(256).byte()
            Image.fromarray(tensor.permute(1,2,0).numpy()).save(p)
            pixels=(aligned_rgb(p,32).permute(1,2,0).numpy()*255).round().astype("uint8")
            self.assertEqual(canonical_pixels(p,32),hashlib.sha256(pixels.tobytes()).hexdigest())

    def test_fixed_dpo_holdout_excludes_canonical_training_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);probe,roots=make_fixture(root)
            plan=prepare_validation(probe,roots,root/"validation",10,(42,2024),materialization="copy")
            self.assertEqual(len(plan["sources"]),30)
            for w in WEATHERS:self.assertGreaterEqual(plan["exclusions"][w+":seen_in_DPO"],1)
            self.assertTrue(all(Path(r["gt_path"]).stem!="0" for r in plan["sources"]))
            self.assertEqual(len({r["canonical_gt_sha256"] for r in plan["sources"]}),30)
            for r in plan["sources"]:
                self.assertEqual(digest(r["validation_gt_path"]),r["gt_sha256"])
            cfg=yaml.safe_load(Path(plan["validation_config"]).read_text(encoding="utf-8"))
            self.assertEqual(cfg["splits"],["val"])
            self.assertEqual(plan["role"],"dpo_holdout")
            self.assertIn("SFT",plan["sft_exposure"])

    def test_source_cluster_statistics_and_continuation_are_not_final_goal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);probe,roots=make_fixture(root)
            plan=prepare_validation(probe,roots,root/"validation",10,(42,2024),materialization="copy")
            rows=analytical_rows(plan,psnr_candidate=-.02)
            summary,_=paired_summary(rows,plan)
            all_psnr=next(r for r in summary if r["scope"]=="all" and r["metric"]=="psnr" and r["method"]=="fidelity" and r["reference"]=="sft")
            self.assertEqual(all_psnr["n_sources"],30) # not 60 seed-observations
            self.assertAlmostEqual(all_psnr["delta_mean"],-.02)
            decision=validation_decision(summary,plan)
            self.assertEqual(decision["status"],"NOT_MET")
            self.assertTrue(decision["supports_budgeted_continuation"])
            cfg=continuation_config(plan,decision,root/"continue",2000)
            self.assertEqual(Path(cfg["training"]["resume_from_checkpoint"]),(probe/"fidelity"/"checkpoint-200").resolve())
            self.assertEqual(cfg["training"]["max_train_steps"],2000)
            self.assertFalse(cfg["training"]["validation_compute_iqa"])
            self.assertFalse(cfg["training"]["validation_use_ema"])
            self.assertEqual(cfg["model"],plan["snapshots"]["fidelity"]["config"]["model"])
            with self.assertRaises(ValueError):continuation_config(plan,decision,root/"long_run",20000)

    def test_missing_results_cannot_pass_and_positive_budget_can_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);probe,roots=make_fixture(root)
            plan=prepare_validation(probe,roots,root/"validation",10,(42,2024),materialization="copy")
            rows=analytical_rows(plan)
            summary,_=paired_summary(rows,plan)
            self.assertEqual(validation_decision(summary,plan)["status"],"PASS")
            rows[0]["dists"]=None
            summary,_=paired_summary(rows,plan)
            self.assertEqual(validation_decision(summary,plan)["status"],"INCOMPLETE")
            with self.assertRaises(ValueError):paired_summary(rows[:-1],plan)

    def test_test_roots_are_not_validation_fallback(self):
        from dpo.texture_release_validation import discover_pairs
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"test";root.mkdir()
            with self.assertRaises(ValueError):list(discover_pairs(root))

    def test_checkpoint_storage_shortage_is_detected_before_creating_training_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);probe,roots=make_fixture(root)
            plan=prepare_validation(probe,roots,root/"validation",10,(42,2024),materialization="copy")
            summary,_=paired_summary(analytical_rows(plan),plan)
            decision=validation_decision(summary,plan)
            with patch("dpo.texture_release_validation.shutil.disk_usage",return_value=types.SimpleNamespace(free=1)):
                with self.assertRaisesRegex(RuntimeError,"larger disk"):
                    continuation_config(plan,decision,root/"training_new",2000)
            self.assertFalse((root/"training_new").exists())


if __name__=="__main__":unittest.main()
