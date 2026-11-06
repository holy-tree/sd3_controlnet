"""Fixed DPO-held-out validation planning, paired statistics, and continuation.

No model is loaded here. SFT exposure of ordinary train-directory holdouts is
unknown: these are held out from DPO, not automatically from the whole pipeline.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

WEATHERS = ("rain", "snow", "haze")
METRICS = {"psnr":1,"ssim":1,"lpips":-1,"dists":-1,"musiq":1,"clipiqa":1,"nima":1,"topiq_nr":1}
SCHEMA = "texture_release_validation_v1"
EXTENSIONS = {".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp"}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as f:return json.load(f)


def write_json(path,value):
    with Path(path).open("w",encoding="utf-8") as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)


def canonical_image(path,resolution):
    """Match PIL Resize(shorter) + torchvision CenterCrop, before tensor scaling."""
    with Image.open(path) as opened:
        image=opened.convert("RGB")
        w,h=image.size
        size=(resolution,int(resolution*h/w)) if w<=h else (int(resolution*w/h),resolution)
        if image.size!=size:image=image.resize(size,Image.Resampling.BILINEAR)
        x,y=int(round((image.width-resolution)/2)),int(round((image.height-resolution)/2))
        image=image.crop((x,y,x+resolution,y+resolution))
        return image


def canonical_pixels(path,resolution):
    return hashlib.sha256(canonical_image(path,resolution).tobytes()).hexdigest()


def saved_probe(root,name,checkpoint_path=None,required_step=200):
    directory=Path(root)/name
    cfg_path=directory/"dpo_config.yaml"
    with cfg_path.open(encoding="utf-8") as f:cfg=yaml.safe_load(f)
    train=cfg["training"]
    if not all(k in train for k in ("reward","preference_filter","spatial_dynamic")):
        raise ValueError("Require actual saved dpo_config.yaml, not the editable default")
    if train["reward"].get("mode")!="dynamic_texture":raise ValueError("Not dynamic_texture")
    checkpoint=Path(checkpoint_path).expanduser().resolve() if checkpoint_path else directory/"checkpoint-200"
    meta=read_json(checkpoint/"dpo_resume.json")
    if meta.get("version")!=6 or (required_step is not None and meta.get("global_step")!=required_step):raise ValueError("Require the expected raw v6 checkpoint")
    cfg.update(reward=train["reward"],preference_filter=train["preference_filter"],spatial_dynamic=train["spatial_dynamic"])
    weights={str(p.resolve()):digest(p) for sub in ("controlnet","ra_fusion") for p in sorted((checkpoint/sub).rglob("*"))
             if p.is_file() and p.suffix in (".json",".bin",".safetensors")}
    if not weights:raise ValueError("Probe model files are missing")
    return {"config_path":str(cfg_path.resolve()),"config_sha256":digest(cfg_path),"config":cfg,
            "checkpoint":str(checkpoint.resolve()),"metadata":meta,"metadata_sha256":digest(checkpoint/"dpo_resume.json"),"weight_files_sha256":weights}


def discover_pairs(root):
    """Paired GT/LQ folders; no test-directory fallback."""
    root=Path(root).expanduser().resolve()
    if not root.is_dir():raise FileNotFoundError(root)
    if "test" in [part.lower() for part in root.parts]:raise ValueError("Do not use test roots for tuning")
    folders=[root,*sorted(p for p in root.rglob("*") if p.is_dir() and p.name.lower() in ("gt","lq"))]
    seen=set()
    for folder in folders:
        parent=folder.parent if folder.name.lower() in ("gt","lq") else folder
        if parent in seen:continue
        seen.add(parent)
        if any(part.lower()=="test" for part in parent.relative_to(root).parts):continue
        children={p.name.lower():p for p in parent.iterdir() if p.is_dir()}
        if not {"gt","lq"}<=set(children):continue
        def index_images(directory):
            result={}
            for p in sorted(directory.iterdir()):
                if p.is_file() and p.suffix.lower() in EXTENSIONS:
                    if p.stem in result:raise ValueError("Duplicate paired-image stem: "+str(p))
                    result[p.stem]=p
            return result
        gt=index_images(children["gt"]);lq=index_images(children["lq"])
        sub=str(parent.relative_to(root));sub="root" if sub=="." else sub
        for stem in sorted(set(gt)&set(lq)):yield str(gt[stem].resolve()),str(lq[stem].resolve()),sub


def prepare_validation(probe_root,roots,output,count=30,seeds=(42,2024),role="dpo_holdout",psnr_tolerance=0.0,materialization="symlink"):
    from tqdm.auto import tqdm
    output=Path(output).expanduser().resolve()
    if count<10 or len(set(seeds))!=len(seeds) or len(seeds)<2:raise ValueError("Use at least 10 sources/weather and two unique seeds")
    if not math.isfinite(psnr_tolerance) or psnr_tolerance<0:raise ValueError("Invalid predeclared PSNR budget")
    snapshots={name:saved_probe(Path(probe_root).expanduser().resolve(),name) for name in ("baseline","fidelity")}
    a,b=(snapshots[name] for name in ("baseline","fidelity"))
    for key in ("model",):
        if a["config"][key]!=b["config"][key]:raise ValueError("The two runs have different initial SFT model configurations")
    for key in ("beta","resolution","train_batch_size","gradient_accumulation_steps","seed","controlnet_learning_rate","ra_fusion_learning_rate"):
        if a["metadata"].get(key)!=b["metadata"].get(key):raise ValueError("Probe conditions differ: "+key)
    settings=copy.deepcopy(b["metadata"]["spatial_dynamic"]["settings"])
    original_settings=copy.deepcopy(a["metadata"]["spatial_dynamic"]["settings"])
    settings.pop("lambda_fidelity");original_settings.pop("lambda_fidelity")
    if settings!=original_settings:raise ValueError("Probe objectives differ in more than lambda")
    for key in ("preference_manifest_sha256",):
        if a["metadata"][key]!=b["metadata"][key]:raise ValueError("Different preference data")
    train=b["config"]["training"]
    if float(train["validation_guidance_scale"])!=1.0:raise ValueError("Use the existing trainer's CFG=1 validation protocol")
    resolution=int(train["resolution"])
    for snapshot in snapshots.values():
        protected=Path(snapshot["config"]["training"]["output_dir"]).resolve()
        if output==protected or protected in output.parents:raise ValueError("Validation output must be outside training runs")
    source_manifest=Path(train["gt_source_manifest"])
    if digest(source_manifest)!=b["metadata"]["reward_provenance"]["gt_manifest_sha256"]:raise ValueError("Training GT manifest changed")
    sources=[json.loads(line) for line in source_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    used_gt,used_lq=set(),set()
    used_gt_paths={str(Path(row["gt_path"]).resolve()) for row in sources}
    used_lq_paths={str(Path(row["lq_path"]).resolve()) for row in sources}
    score_metadata=read_json(source_manifest.parent/"texture_scores.json")
    original_csv=Path(score_metadata["input_csv"]).expanduser().resolve()
    if digest(original_csv)!=score_metadata["input_sha256"]:
        raise ValueError("Original candidate population changed; cannot reliably exclude original train paths")
    with original_csv.open(newline="",encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            for field,collection in (("gt_path",used_gt_paths),("lq_path",used_lq_paths)):
                p=Path(row[field]).expanduser()
                collection.add(str((p if p.is_absolute() else original_csv.parent/p).resolve()))
    prompts=defaultdict(set)
    for row in tqdm(sources,desc="Hash DPO exclusions",unit="source"):
        for field,collection in (("gt",used_gt),("lq",used_lq)):
            if digest(row[field+"_path"])!=row[field+"_sha256"]:raise ValueError("DPO source changed")
            collection.add(canonical_pixels(row[field+"_path"],resolution))
        prompts[row["weather"]].add(row.get("prompt",""))
    if any(len(prompts[w])!=1 for w in WEATHERS):raise ValueError("Need one attested weather prompt per weather")
    selected=[];selected_gt=set();selected_lq=set();exclusions=Counter()
    for weather in WEATHERS:
        root=Path(roots[weather]).expanduser().resolve()
        if role=="external_validation" and any(p.lower()=="train" for p in root.parts):
            raise ValueError("A train root cannot be labeled external validation")
        pools=defaultdict(list)
        for gt,lq,sub in discover_pairs(root):pools[sub].append((gt,lq))
        rng=random.Random(42)
        for pool in pools.values():rng.shuffle(pool)
        chosen=[]
        while len(chosen)<count and any(pools.values()):
            for sub in sorted(pools):
                if not pools[sub] or len(chosen)>=count:continue
                gt,lq=pools[sub].pop()
                if gt in used_gt_paths or lq in used_lq_paths:
                    exclusions[weather+":seen_in_DPO"]+=1;continue
                with Image.open(gt) as gi,Image.open(lq) as li:
                    if not math.isclose(gi.width/gi.height,li.width/li.height,rel_tol=1e-6,abs_tol=1e-6):
                        exclusions[weather+":geometry_mismatch"]+=1;continue
                gh,lh=canonical_pixels(gt,resolution),canonical_pixels(lq,resolution)
                if gh in used_gt or lh in used_lq:
                    exclusions[weather+":seen_in_DPO"]+=1;continue
                if gh in selected_gt or lh in selected_lq:
                    exclusions[weather+":duplicate_validation_content"]+=1;continue
                selected_gt.add(gh);selected_lq.add(lh)
                identity=hashlib.sha256(json.dumps([weather,gt,lq]).encode()).hexdigest()[:20]
                chosen.append({"id":identity,"weather":weather,"subdataset":sub,"gt_path":gt,"lq_path":lq,
                               "gt_sha256":digest(gt),"lq_sha256":digest(lq),"canonical_gt_sha256":gh,"canonical_lq_sha256":lh,
                               "prompt":next(iter(prompts[weather]))})
        if len(chosen)!=count:raise ValueError(f"Only {len(chosen)} eligible {weather} sources; need {count}. Do not silently use test data")
        selected.extend(chosen)
    output.mkdir(parents=True,exist_ok=False)
    for row in selected:
        folder=output/"dataset"/row["weather"]/"val"
        for field,part in (("gt","GT"),("lq","LQ")):
            target=folder/part;target.mkdir(parents=True,exist_ok=True)
            destination=target/(row["id"]+Path(row[field+"_path"]).suffix.lower())
            if materialization=="copy":shutil.copy2(row[field+"_path"],destination)
            elif materialization=="symlink":destination.symlink_to(row[field+"_path"])
            else:raise ValueError("Unknown validation materialization")
            row["validation_"+field+"_path"]=str(destination)
    eval_config={"dataset_root":str(output/"dataset"),"weather_types":list(WEATHERS),"splits":["val"],
                 "resolution":resolution,"use_prompt":True,"weather_prompts":{w:next(iter(prompts[w])) for w in WEATHERS},
                 "validation_num_samples_per_weather":count,"max_samples_per_weather":0,"sample_mode":"head",
                 "guidance_scale":1.0,"strength":float(train["validation_strength"]),
                 "validation_guidance_scale":1.0,"validation_strength":float(train["validation_strength"]),
                 "validation_num_inference_steps":int(train["validation_num_inference_steps"]),
                 "controlnet_conditioning_scale":float(b["config"]["model"].get("controlnet_conditioning_scale",1.0)),
                 "validation_seed":seeds[0],"validation_save_images":True,"validation_compute_iqa":False,
                 "lpips_net":train.get("validation_lpips_net","alex")}
    with (output/"validation.yaml").open("w",encoding="utf-8") as f:yaml.safe_dump(eval_config,f,sort_keys=False,allow_unicode=True)
    plan={"schema":SCHEMA,"role":role,"materialization":materialization,"sft_exposure":"unknown; DPO holdout does not imply SFT-unseen",
          "snapshots":snapshots,"sources":selected,"seeds":list(seeds),"resolution":resolution,
          "count_per_weather":count,"selection_seed":42,"exclusions":dict(exclusions),"gt_training_manifest_sha256":digest(source_manifest),
          "original_candidate_csv_sha256":score_metadata["input_sha256"],
          "validation_config":str(output/"validation.yaml"),"validation_config_sha256":digest(output/"validation.yaml"),
          "criteria":{"psnr_tolerance_db":psnr_tolerance,"bootstrap_repetitions":2000,"confidence":0.95,
                     "primary_aesthetic":"whole_image_musiq","auxiliary_mean_nonregression":["ssim","lpips","dists","clipiqa","nima","topiq_nr"]}}
    plan["protocol_sha256"]={relative:digest(Path(__file__).resolve().parents[1]/relative) for relative in
                             ("scripts/train_dpo_sd3.py","utils/randomness_check.py","utils/pipeline_inference.py",
                              "dpo/texture_release_runtime.py","dpo/texture_diagnostics_runtime.py","dpo/spatial.py")}
    write_json(output/"validation_plan.json",plan)
    return plan


def finite(value):
    try:
        x=float(value);return x if math.isfinite(x) else None
    except (TypeError,ValueError):return None


def paired_summary(rows,plan):
    """Average seeds WITHIN source, then source-cluster bootstrap paired deltas."""
    methods=plan.get("methods",("sft","baseline","fidelity"))
    comparisons=plan.get("comparisons",(("baseline","sft"),("fidelity","sft"),("fidelity","baseline")))
    if len(set(methods))!=len(methods) or any(a not in methods or b not in methods or a==b for a,b in comparisons):
        raise ValueError("Invalid paired comparison methods")
    expected={(s["id"],seed,method) for s in plan["sources"] for seed in plan["seeds"] for method in methods}
    indexed={}
    for row in rows:
        key=(row["source_id"],int(row["seed"]),row["method"])
        if key in indexed:raise ValueError("Duplicate validation observation")
        indexed[key]=row
    if set(indexed)!=expected:raise ValueError("Missing/extra validation source/seed/method observations")
    rng=np.random.default_rng(42)
    summaries=[];source_deltas=[]
    for scope in ("all",*WEATHERS):
        chosen=[s for s in plan["sources"] if scope=="all" or s["weather"]==scope]
        for method,reference in comparisons:
            for metric,direction in METRICS.items():
                triples=[];labels=[]
                for source in chosen:
                    x=[finite(indexed[(source["id"],seed,method)].get(metric)) for seed in plan["seeds"]]
                    y=[finite(indexed[(source["id"],seed,reference)].get(metric)) for seed in plan["seeds"]]
                    if any(v is None for v in (*x,*y)):continue
                    a,b=float(np.mean(x)),float(np.mean(y));triples.append((a,b,a-b));labels.append(source["weather"])
                    source_deltas.append({"scope":scope,"source_id":source["id"],"weather":source["weather"],"subdataset":source["subdataset"],"metric":metric,
                                          "method":method,"reference":reference,"delta":a-b})
                values=np.asarray(triples,dtype=float)
                if len(values):
                    deltas=values[:,2]
                    # Preserve fixed weather quotas in the overall bootstrap.
                    parts=[]
                    for w in sorted(set(labels)):
                        subset=deltas[np.asarray(labels)==w]
                        parts.append(np.mean(subset[rng.integers(0,len(subset),(plan["criteria"]["bootstrap_repetitions"],len(subset)))],axis=1)*len(subset)/len(deltas))
                    boot=sum(parts)
                    result={"n_sources":len(deltas),"method_mean":float(values[:,0].mean()),"reference_mean":float(values[:,1].mean()),
                            "delta_mean":float(deltas.mean()),"delta_median":float(np.median(deltas)),
                            "delta_p05":float(np.quantile(deltas,.05)),"delta_p95":float(np.quantile(deltas,.95)),
                            "ci_low":float(np.quantile(boot,.025)),"ci_high":float(np.quantile(boot,.975)),
                            "worse_fraction":float(np.mean(deltas*direction<0))}
                else:result={"n_sources":0,"method_mean":None,"reference_mean":None,"delta_mean":None,"ci_low":None,"ci_high":None}
                summaries.append({"scope":scope,"metric":metric,"method":method,"reference":reference,**result})
    return summaries,source_deltas


def validation_decision(summaries,plan):
    selected={(r["scope"],r["metric"]):r for r in summaries if r["method"]=="fidelity" and r["reference"]=="sft"}
    violations=[];uncertain=[];missing=[]
    tolerance=plan["criteria"]["psnr_tolerance_db"]
    for scope in ("all",*WEATHERS):
        row=selected[(scope,"psnr")]
        required=len([s for s in plan["sources"] if scope=="all" or s["weather"]==scope])
        if row["n_sources"]!=required:missing.append(scope+":psnr");continue
        if row["delta_mean"] < -tolerance:violations.append(scope+":PSNR observed budget violation")
        elif row["ci_low"] < -tolerance:uncertain.append(scope+":PSNR CI crosses budget")
    aesthetic=selected[("all","musiq")]
    if aesthetic["n_sources"]!=len(plan["sources"]):missing.append("all:musiq")
    elif aesthetic["delta_mean"]<=0:violations.append("whole-image MUSIQ did not improve")
    elif aesthetic["ci_low"]<=0:uncertain.append("whole-image MUSIQ CI includes non-improvement")
    for metric in plan["criteria"]["auxiliary_mean_nonregression"]:
        for scope in ("all",*WEATHERS):
            row=selected[(scope,metric)]
            required=len([s for s in plan["sources"] if scope=="all" or s["weather"]==scope])
            if row["n_sources"]!=required:missing.append(scope+":"+metric)
            elif row["delta_mean"]*METRICS[metric]<0:uncertain.append(scope+":"+metric+" mean regresses; review required")
    status="INCOMPLETE" if missing else "NOT_MET" if violations else "INCONCLUSIVE" if uncertain else "PASS"
    between={(r["scope"],r["metric"]):r for r in summaries if r["method"]=="fidelity" and r["reference"]=="baseline"}
    direction_reasons=[]
    if missing:direction_reasons.append("Required metrics incomplete")
    if between[("all","psnr")]["ci_low"] is None or between[("all","psnr")]["ci_low"]<=0:
        direction_reasons.append("Paired candidate-vs-baseline PSNR improvement is not established")
    if aesthetic["ci_low"] is None or aesthetic["ci_low"]<=0:
        direction_reasons.append("Candidate aesthetic gain above SFT is not established")
    for scope in ("all",*WEATHERS):
        for metric in ("psnr","ssim","lpips","dists"):
            row=between[(scope,metric)]
            if row["delta_mean"] is None or row["delta_mean"]*METRICS[metric]<0:
                direction_reasons.append(scope+":"+metric+" candidate regresses vs baseline")
    for metric in ("clipiqa","nima","topiq_nr"):
        row=selected[("all",metric)]
        if row["delta_mean"] is None or row["delta_mean"]<0:
            direction_reasons.append("Candidate "+metric+" mean below SFT")
    return {"status":status,"missing":missing,"observed_violations":violations,"uncertainty_or_auxiliary_review":uncertain,
            "supports_budgeted_continuation":not direction_reasons,"continuation_blockers":direction_reasons,
            "meaning":"predeclared validation evidence only; not a guarantee about future training or unseen tests"}


def continuation_storage(checkpoint,train,output_dir,total_steps,step=200):
    checkpoint,output_dir=Path(checkpoint),Path(output_dir).expanduser().resolve()
    checkpoint_bytes=sum(p.stat().st_size for p in checkpoint.rglob("*") if p.is_file())
    saves=max(1,(total_steps-step+499)//500)
    keep=int(train.get("checkpoints_total_limit",2));keep=keep if keep>0 else saves
    exported=sum(p.stat().st_size for sub in ("controlnet","ra_fusion") for p in (checkpoint/sub).rglob("*") if p.is_file())
    if (checkpoint/"ema_state.pt").is_file():exported+=(checkpoint/"ema_state.pt").stat().st_size
    estimated=max(min(saves,keep+1)*checkpoint_bytes,min(saves,keep)*checkpoint_bytes+exported)+1024**3
    parent=output_dir.parent
    while not parent.exists():parent=parent.parent
    free=shutil.disk_usage(parent).free
    return {"checkpoint_bytes":checkpoint_bytes,"estimated_peak_bytes":estimated,"free_bytes":free,
            "filesystem_check_path":str(parent),"training_output":str(output_dir),"enough_space_estimate":free>=estimated,
            "note":"Estimate includes write-before-prune checkpoint peak, final exports/EMA and 1GiB margin; actual quotas/size may vary"}


def continuation_config(plan,decision,output_dir,total_steps=2000):
    if decision["status"]!="PASS" and not decision["supports_budgeted_continuation"]:
        raise ValueError("Validation does not support continuation; inspect the paired results rather than claiming it passed")
    if decision["status"]!="PASS" and total_steps>2000:
        raise ValueError("Final preservation target is not met; only a monitored continuation up to total step 2000 is supported")
    snapshot=plan["snapshots"]["fidelity"]
    checkpoint=Path(snapshot["checkpoint"])
    required=["dpo_resume.json","optimizer.bin","scheduler.bin","random_states_0.pkl"]
    if snapshot["config"]["training"].get("use_ema",False):required.append("ema_state.pt")
    for name in required:
        if not (checkpoint/name).is_file():raise FileNotFoundError("Cannot resume: missing "+str(checkpoint/name))
    if digest(checkpoint/"dpo_resume.json")!=snapshot["metadata_sha256"]:raise ValueError("Resume metadata changed after validation")
    for path,expected in snapshot["weight_files_sha256"].items():
        if digest(path)!=expected:raise ValueError("Validated checkpoint weights changed")
    if digest(plan["validation_config"])!=plan["validation_config_sha256"]:raise ValueError("Fixed validation config changed")
    cfg=copy.deepcopy(snapshot["config"]);effective=cfg["training"]
    for name in ("reward","preference_filter","spatial_dynamic","tail_risk"):
        if name in effective:cfg[name]=copy.deepcopy(effective[name])
    for name in ("reward","preference_filter","spatial_dynamic","tail_risk"):effective.pop(name,None)
    if total_steps<=snapshot["metadata"]["global_step"]:raise ValueError("Total steps must exceed the resumed step")
    if total_steps%500:raise ValueError("Use a total-step multiple of 500 so the final raw checkpoint is explicitly saved")
    if cfg["training"].get("lr_scheduler","constant_with_warmup") not in ("constant","constant_with_warmup"):
        raise ValueError("Changing total steps would reinterpret a nonconstant LR schedule")
    output_dir=Path(output_dir).expanduser().resolve()
    original=Path(snapshot["config"]["training"]["output_dir"]).resolve()
    if output_dir.exists() or output_dir==original or original in output_dir.parents:raise FileExistsError("Use a separate, new continuation output")
    if snapshot["metadata"].get("num_processes",1)!=1:raise ValueError("Single-card continuation cannot silently reinterpret multi-rank resume state")
    storage=continuation_storage(checkpoint,effective,output_dir,total_steps,snapshot["metadata"]["global_step"])
    if not storage["enough_space_estimate"]:
        raise RuntimeError(f"Continuation output disk {storage['filesystem_check_path']} has {storage['free_bytes']/1024**3:.2f} GiB; estimated checkpoint/EMA peak needs about {storage['estimated_peak_bytes']/1024**3:.2f} GiB. Pick --training-output on a larger disk; do not retrain the 200-step probes")
    effective.update(output_dir=str(output_dir),resume_from_checkpoint=snapshot["checkpoint"],max_train_steps=total_steps,
                     checkpointing_steps=500,logging_dir=None,run_checkpoint_validation=True,
                     validation_eval_config=plan["validation_config"],validation_num_samples_per_weather=plan["count_per_weather"],
                     validation_seed=plan["seeds"][0],validation_use_ema=False,
                     validation_compute_iqa=False,validation_save_images=True)
    cfg.setdefault("evaluation",{})["output_dir"]=str(output_dir/"evaluation")
    cfg["validation_release"]={"final_target_status":decision["status"],"data_role":plan["role"],
                               "budgeted_continuation":decision["status"]!="PASS",
                               "estimated_output_disk_peak_bytes":storage["estimated_peak_bytes"],"available_output_disk_bytes":storage["free_bytes"]}
    return cfg


def recheck_plan(old_plan_path,candidate_run,checkpoint_path,output):
    """Reuse the exact held-out cohort/criteria and cached SFT/baseline images."""
    old_plan_path=Path(old_plan_path).resolve();old_root=old_plan_path.parent
    plan=read_json(old_plan_path)
    results=read_json(old_root/"validation_results.json")
    if results["plan_sha256"]!=digest(old_plan_path):raise ValueError("Old validation evidence changed")
    for relative,expected in plan["protocol_sha256"].items():
        if digest(Path(__file__).resolve().parents[1]/relative)!=expected:raise ValueError("Validation protocol changed; cannot silently reuse old baselines")
    candidate_run=Path(candidate_run).resolve()
    updated=saved_probe(candidate_run.parent,candidate_run.name,checkpoint_path,required_step=None)
    old=plan["snapshots"]["fidelity"]
    if updated["metadata"]["spatial_dynamic"]!=old["metadata"]["spatial_dynamic"] or updated["config"]["model"]!=old["config"]["model"]:
        raise ValueError("Recheck cannot change objective or initial SFT reference")
    if updated["metadata"]["preference_manifest_sha256"]!=old["metadata"]["preference_manifest_sha256"]:
        raise ValueError("Recheck training data changed")
    plan["snapshots"]["fidelity"]=updated
    evidence=read_json(old_root/"baseline_model_evidence.json")
    plan["frozen_baseline_identity"]={"reference":evidence["parameter_checksums_before"]["reference"],
                                      "frozen_parameters":evidence["shared_frozen_parameters_sha256"],
                                      "sft_buffers":evidence["sft_buffers_sha256"]}
    plan["cached_baseline_evidence"]={"path":str(old_root/"baseline_model_evidence.json"),"sha256":digest(old_root/"baseline_model_evidence.json")}
    plan["frozen_inference_rows"]=[r for r in results["rows"] if r["method"] in ("sft","baseline")]
    plan["comparison_note"]="Candidate continued beyond 200 steps; candidate-vs-baseline is a trajectory check, not a pure equal-budget lambda ablation"
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    content=Path(plan["validation_config"]).read_bytes()
    (output/"validation.yaml").write_bytes(content)
    plan["validation_config"]=str(output/"validation.yaml")
    plan["validation_config_sha256"]=digest(output/"validation.yaml")
    plan["expected_pyiqa_version"]=results["pyiqa_version"]
    write_json(output/"validation_plan.json",plan)
    return plan
