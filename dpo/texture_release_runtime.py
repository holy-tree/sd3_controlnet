"""Real paired validation through the existing checkpoint-validation path."""
from __future__ import annotations

import copy
import csv
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.func import functional_call
from tqdm.auto import tqdm
import yaml

from .spatial import aligned_rgb
from .texture_diagnostics import tensor_digest
from .texture_diagnostics_runtime import RealEngine
from .texture_release_validation import digest, read_json, write_json, SCHEMA


def inference_groups(plan):
    groups=plan.get("inference_arms",[{"snapshot":"baseline","methods":["sft","baseline"]},
                                       {"snapshot":"fidelity","methods":["fidelity"]}])
    methods=[method for group in groups for method in group["methods"]]
    if len(set(methods))!=len(methods) or set(methods)!=set(plan.get("methods",("sft","baseline","fidelity"))):
        raise ValueError("Invalid inference arm/method assignment")
    if any(group["snapshot"] not in plan["snapshots"] for group in groups):raise ValueError("Unknown inference snapshot")
    return groups


class ValidationCall(torch.nn.Module):
    def __init__(self,engine,kwargs):
        super().__init__()
        self.controlnet,self.transformer=engine.controlnet,engine.transformer
        self.kwargs,self.callback=kwargs,engine.trainer.run_checkpoint_validation
    def forward(self):return self.callback(**self.kwargs)


def evaluate_case(engine,plan,method,seed,folder,lookup):
    with Path(plan["validation_config"]).open(encoding="utf-8") as f:cfg=yaml.safe_load(f)
    cfg.update(validation_seed=seed,validation_compute_iqa=False,validation_save_images=True)
    records=engine.trainer.build_validation_records(cfg)
    if len(records)!=len(plan["sources"]):raise ValueError("Linked validation population changed")
    for record in records:
        if (record["gt_path"],record["lq_path"]) not in lookup:raise ValueError("Unexpected linked validation source")
    kwargs=dict(checkpoint_dir=folder,step=0 if method=="sft" else engine.meta["global_step"],validation_config=cfg,
                validation_records=records,prompt_cache=engine.prompt_cache,controlnet=engine.controlnet,
                transformer=engine.transformer,vae=engine.vae,scheduler=engine.scheduler,device=engine.device,
                weight_dtype=engine.dtype,weights_name="raw")
    call=ValidationCall(engine,kwargs)
    if method=="sft":
        substitutions={"controlnet."+k:v for k,v in {**engine.reference_controlnet,**engine.sft_control_buffers}.items()}
        substitutions.update({"transformer."+k:v for k,v in {**engine.reference_ra,**engine.sft_ra_buffers}.items()})
        functional_call(call,substitutions,(),strict=False)
    else:call()
    with (folder/"validation_per_image.csv").open(newline="",encoding="utf-8") as f:measured=list(csv.DictReader(f))
    if len(measured)!=len(records):raise ValueError("Incomplete real validation output")
    preprocess=engine.trainer.build_preprocess(engine.settings["resolution"])
    lq_pils,_,_=engine.trainer.load_image_batch([records[0]],preprocess,engine.device)
    with torch.no_grad():shape=engine.trainer.infer_latent_shape(_shape_pipeline(engine),lq_pils[0],engine.settings["resolution"],engine.device)
    output=[]
    for record,row in zip(records,measured):
        source=lookup[(record["gt_path"],record["lq_path"])]
        if row["weather"]!=source["weather"] or row["name"]!=Path(record["gt_path"]).stem:
            raise ValueError("Validation output order/identity mismatch")
        noise,seeds=engine.trainer.make_candidate_noise([record],0,shape,seed)
        output.append({**row,"source_id":source["id"],"original_subdataset":source["subdataset"],"seed":seed,"method":method,
                       "global_index":record["global_index"],"noise_seed":seeds[0],"noise_sha256":tensor_digest([("noise",noise)]),
                       "prediction_sha256":digest(row["prediction_path"]),"source_gt_sha256":source["gt_sha256"],
                       "source_lq_sha256":source["lq_sha256"],"aligned_gt_sha256":digest(row["gt_path"])})
    return output


def _shape_pipeline(engine):
    from diffusers import StableDiffusion3ControlNetPipeline
    return StableDiffusion3ControlNetPipeline(scheduler=copy.deepcopy(engine.scheduler),vae=engine.vae,
        text_encoder=None,tokenizer=None,text_encoder_2=None,tokenizer_2=None,text_encoder_3=None,tokenizer_3=None,
        transformer=engine.transformer,controlnet=engine.controlnet)


def run_inference(plan_path,device):
    plan_path=Path(plan_path).expanduser().resolve();root=plan_path.parent
    plan=read_json(plan_path);plan_hash=digest(plan_path)
    if plan.get("schema")!=SCHEMA or digest(plan["validation_config"])!=plan["validation_config_sha256"]:
        raise ValueError("Validation plan/config mismatch")
    for relative,expected in plan.get("protocol_sha256",{}).items():
        if digest(Path(__file__).resolve().parents[1]/relative)!=expected:raise ValueError("Validation protocol changed: "+relative)
    for source in plan["sources"]:
        for field in ("gt","lq"):
            if digest(source[field+"_path"])!=source[field+"_sha256"]:raise ValueError("Validation source changed")
            if digest(source["validation_"+field+"_path"])!=source[field+"_sha256"]:raise ValueError("Materialized validation source changed")
    lookup={(str(Path(s["validation_gt_path"]).resolve()),str(Path(s["validation_lq_path"]).resolve())):s for s in plan["sources"]}
    rows=list(plan.get("frozen_inference_rows",[]));model_evidence={}
    if plan.get("cached_baseline_evidence"):
        item=plan["cached_baseline_evidence"]
        if digest(item["path"])!=item["sha256"]:raise ValueError("Cached real-model evidence changed")
        model_evidence["baseline_cached"]=item
    frozen=plan.get("frozen_baseline_identity")
    reference_hash=frozen["reference"] if frozen else None
    frozen_hash=frozen["frozen_parameters"] if frozen else None
    sft_buffers_hash=frozen["sft_buffers"] if frozen else None
    frozen_methods={r["method"] for r in rows}
    for row in rows:
        if digest(row["prediction_path"])!=row["prediction_sha256"] or digest(row["gt_path"])!=row["aligned_gt_sha256"]:
            raise ValueError("Cached baseline image changed")
    for group in inference_groups(plan):
        arm,methods=group["snapshot"],group["methods"]
        if set(methods)<=frozen_methods:continue
        snapshot=plan["snapshots"][arm]
        if digest(snapshot["config_path"])!=snapshot["config_sha256"]:raise ValueError("Saved training config changed")
        if digest(Path(snapshot["checkpoint"])/"dpo_resume.json")!=snapshot["metadata_sha256"]:
            raise ValueError("Probe checkpoint metadata changed")
        for path,expected in snapshot["weight_files_sha256"].items():
            if digest(path)!=expected:raise ValueError("Probe model file changed: "+path)
        args=SimpleNamespace(checkpoint=snapshot["checkpoint"],device=device,seed=42,batches=3,inference_only=True,loss_scale=None)
        evidence={}
        engine=RealEngine(args,snapshot,evidence,extra_prompts={s["prompt"] for s in plan["sources"]})
        if reference_hash is not None and reference_hash!=engine.initial_reference:raise ValueError("Different SFT reference initialization")
        reference_hash=engine.initial_reference
        shared=tensor_digest((name,p) for name,p in engine.named_policy_parameters() if not p.requires_grad)
        buffers=tensor_digest([*(('controlnet.'+k,v) for k,v in engine.sft_control_buffers.items()),
                               *(('ra.'+k,v) for k,v in engine.sft_ra_buffers.items())])
        if frozen_hash is not None and frozen_hash!=shared:raise ValueError("Different shared SD3/VAE frozen weights")
        if sft_buffers_hash is not None and sft_buffers_hash!=buffers:raise ValueError("Different SFT buffers")
        frozen_hash,sft_buffers_hash=shared,buffers
        evidence.update(shared_frozen_parameters_sha256=shared,sft_buffers_sha256=buffers)
        try:
            for method in methods:
                for seed in plan["seeds"]:
                    folder=root/"inference"/method/f"seed_{seed}"
                    completed=folder/"validation_complete.json"
                    if completed.is_file():
                        previous=read_json(completed)
                        if previous["plan_sha256"]!=plan_hash or previous["reference_sha256"]!=reference_hash or previous["policy_sha256"]!=engine.initial_policy:
                            raise ValueError("Cached validation model/plan changed")
                        case=previous["rows"]
                        for row in case:
                            if digest(row["prediction_path"])!=row["prediction_sha256"]:raise ValueError("Saved prediction changed")
                    else:
                        if folder.exists():raise FileExistsError(f"Incomplete validation case {folder}; preserve/rename it before retry")
                        folder.mkdir(parents=True,exist_ok=False)
                        case=evaluate_case(engine,plan,method,seed,folder,lookup)
                        write_json(completed,{"plan_sha256":plan_hash,"reference_sha256":reference_hash,"policy_sha256":engine.initial_policy,"rows":case})
                    rows.extend(case)
            integrity=engine.finish_integrity()
            if not all(integrity[k] for k in ("policy_parameters_unchanged","reference_unchanged","buffers_unchanged")):
                raise RuntimeError("Validation changed parameters/reference/buffers; investigate instead of training")
            write_json(root/(arm+"_model_evidence.json"),{**evidence,**integrity})
            model_evidence[arm]={"path":str(root/(arm+"_model_evidence.json")),"sha256":digest(root/(arm+"_model_evidence.json"))}
        finally:
            del engine
            gc.collect();torch.cuda.empty_cache()
    # Per-source, per-seed initial noise must be identical for all methods.
    identities={}
    for row in rows:
        key=(row["source_id"],int(row["seed"]))
        condition=(row["noise_sha256"],row["source_gt_sha256"],row["source_lq_sha256"],row["prompt"])
        if key in identities and identities[key]!=condition:raise ValueError("Unpaired inference conditions")
        identities[key]=condition
    write_json(root/"inference_results.json",{"plan_sha256":plan_hash,"model_evidence":model_evidence,
                                            "protocol":"real trainer.run_checkpoint_validation; raw matched precision", "rows":rows})


def score_outputs(plan_path,device):
    """Whole-image metrics use the SAME existing evaluator wrappers for all models."""
    from utils import metrics
    from utils.evaluate_sd3 import IQA_PANEL_SPEC
    plan_path=Path(plan_path).expanduser().resolve();root=plan_path.parent
    plan=read_json(plan_path);data=read_json(root/"inference_results.json")
    import importlib.metadata
    version=importlib.metadata.version("pyiqa")
    if plan.get("expected_pyiqa_version",version)!=version:raise ValueError("IQA backend changed since baseline validation")
    if data["plan_sha256"]!=digest(plan_path):raise ValueError("Plan changed after inference")
    for evidence in data["model_evidence"].values():
        if digest(evidence["path"])!=evidence["sha256"]:raise ValueError("Real model validation evidence changed")
    rows=copy.deepcopy(data["rows"])
    for metric in ("dists","musiq","clipiqa","nima","topiq_nr"):
        function,options,_=IQA_PANEL_SPEC[metric]
        for row in tqdm(rows,desc="Whole-image "+metric,unit="image"):
            if digest(row["prediction_path"])!=row["prediction_sha256"]:raise ValueError("Prediction changed before scoring")
            if digest(row["gt_path"])!=row["aligned_gt_sha256"]:raise ValueError("Aligned validation GT changed")
            prediction=aligned_rgb(row["prediction_path"],plan["resolution"])[None]
            with torch.no_grad():
                result=function(prediction,aligned_rgb(row["gt_path"],plan["resolution"])[None],device=device,**options) if metric=="dists" else function(prediction,device=device,**options)
            value=float(result[0]);row[metric]=value if __import__("math").isfinite(value) else None
        # This is a separate validation process. Release each metric completely,
        # rather than keeping five CPU copies under a small container RAM limit.
        metrics._IQA_CACHE.clear();gc.collect();torch.cuda.empty_cache()
    write_json(root/"validation_results.json",{"plan_sha256":digest(plan_path),"inference_sha256":digest(root/"inference_results.json"),
                                              "pyiqa_version":version,"model_evidence":data["model_evidence"],
                                              "scoring":"raw whole-image scores; no per-method normalization", "rows":rows})
