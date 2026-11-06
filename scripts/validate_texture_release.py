"""Fixed-sample real SFT/baseline/fidelity validation and bounded continuation."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from dpo.texture_release_validation import (WEATHERS,METRICS,prepare_validation,paired_summary,
    validation_decision,continuation_config,digest,read_json,write_json,canonical_image,recheck_plan,continuation_storage)


def table_csv(path,rows):
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)


def review_panels(root,plan,rows,changes):
    import numpy as np
    from PIL import Image,ImageDraw
    directory=root/"review_panels";directory.mkdir(exist_ok=True)
    indexed={(r["source_id"],int(r["seed"]),r["method"]):r for r in rows}
    sources={s["id"]:s for s in plan["sources"]};panels={}
    seed=plan["seeds"][0]
    for weather in WEATHERS:
        worst=sorted((r for r in changes if r['scope']==weather and r['method']=='fidelity' and r['reference']=='sft' and r['metric']=='psnr'),key=lambda r:r['delta'])[:3]
        for item in worst:
            source=sources[item["source_id"]]
            gt=canonical_image(source["gt_path"],plan["resolution"])
            lq=canonical_image(source["lq_path"],plan["resolution"])
            images=[lq,gt]
            for method in ("sft","baseline","fidelity"):
                with Image.open(indexed[(source["id"],seed,method)]["prediction_path"]) as image:images.append(image.convert("RGB"))
            errors=[np.abs(np.asarray(image,dtype=np.float32)-np.asarray(gt,dtype=np.float32)).mean(2)/255 for image in images[2:]]
            scale=max(1e-8,*(float(error.max()) for error in errors))
            panel=Image.new("RGB",(1280,560),"white");draw=ImageDraw.Draw(panel)
            labels=["LQ","GT","SFT","lambda="+str(plan["snapshots"]["baseline"]["metadata"]["spatial_dynamic"]["settings"]["lambda_fidelity"]),
                    "lambda="+str(plan["snapshots"]["fidelity"]["metadata"]["spatial_dynamic"]["settings"]["lambda_fidelity"])]
            for i,(image,label) in enumerate(zip(images,labels)):
                draw.text((i*256+4,4),label,fill="black");panel.paste(image.resize((256,256)),(i*256,24))
            draw.text((4,300),"Absolute RGB error; common scale="+f"{scale:.5g}"+"; seed="+str(seed),fill="black")
            for i,error in enumerate(errors,2):
                view=Image.fromarray((error/scale*255).round().astype("uint8")).convert("RGB").resize((256,256))
                panel.paste(view,(i*256,304))
            target=directory/(weather+"_"+source["id"]+".png");panel.save(target)
            panels[source["id"]]=str(target.relative_to(root))
    return panels


def summarize(plan_path):
    plan_path=Path(plan_path).expanduser().resolve();root=plan_path.parent
    plan=read_json(plan_path);data=read_json(root/"validation_results.json")
    if data["plan_sha256"]!=digest(plan_path):raise ValueError("Criteria or selection changed after evaluation")
    for evidence in data["model_evidence"].values():
        if digest(evidence["path"])!=evidence["sha256"]:raise ValueError("Real-model evidence changed")
    summaries,changes=paired_summary(data["rows"],plan)
    decision=validation_decision(summaries,plan)
    table_csv(root/"per_image_metrics.csv",data["rows"])
    table_csv(root/"paired_source_deltas.csv",changes)
    table_csv(root/"paired_summary.csv",summaries)
    # Per-seed results, without treating repeated seeds as independent images.
    seed_rows=[]
    for seed in plan["seeds"]:
        subset=[r for r in data["rows"] if int(r["seed"])==seed]
        one_plan={**plan,"seeds":[seed]}
        seed_summary,_=paired_summary(subset,one_plan)
        seed_rows.extend({"seed":seed,**r} for r in seed_summary)
        for row in seed_summary:
            if row["method"]=="fidelity" and row["reference"]=="sft" and row["metric"]=="psnr" and row["delta_mean"] is not None and row["delta_mean"] < -plan["criteria"]["psnr_tolerance_db"]:
                decision["uncertainty_or_auxiliary_review"].append(f"seed={seed}, {row['scope']}: PSNR budget violated")
            if row["method"]=="fidelity" and row["reference"]=="baseline" and row["metric"]=="psnr" and (row["delta_mean"] is None or row["delta_mean"]<0 or (row["scope"]=="all" and row["delta_mean"]==0)):
                decision["supports_budgeted_continuation"]=False
                decision["continuation_blockers"].append(f"seed={seed}, {row['scope']}: candidate PSNR improvement vs baseline not reproduced")
            if row["scope"]=="all" and row["method"]=="fidelity" and row["reference"]=="sft" and row["metric"]=="musiq" and (row["delta_mean"] is None or row["delta_mean"]<=0):
                decision["uncertainty_or_auxiliary_review"].append(f"seed={seed}: whole-image MUSIQ gain not reproduced")
                decision["supports_budgeted_continuation"]=False
                decision["continuation_blockers"].append(f"seed={seed}: aesthetic direction not stable")
    if decision["status"]=="PASS" and decision["uncertainty_or_auxiliary_review"]:decision["status"]="INCONCLUSIVE"
    table_csv(root/"per_seed_summary.csv",seed_rows)
    decision.update(plan_sha256=digest(plan_path),results_sha256=digest(root/"validation_results.json"),role=plan["role"])
    write_json(root/"validation_decision.json",decision)
    panels=review_panels(root,plan,data["rows"],changes)
    def fmt(value):return "N/A" if value is None else f"{value:.6g}"
    baseline_lambda=plan['snapshots']['baseline']['metadata']['spatial_dynamic']['settings']['lambda_fidelity']
    candidate_lambda=plan['snapshots']['fidelity']['metadata']['spatial_dynamic']['settings']['lambda_fidelity']
    lines=["# 固定验证集三模型验证", "", "## 主要结论",
           f"- 最终保真/美学目标判据：**{decision['status']}**。",
           f"- 是否支持固定方案、监控下继续到总步数2000：**{decision['supports_budgeted_continuation']}**。这不等于最终目标已达标。",
           f"- 数据角色：`{plan['role']}`；{len(plan['sources'])} 个源图，每图 {len(plan['seeds'])} 个固定种子。",
           "- DPO-heldout 不等于 SFT 未见：若来源是原 train 目录，仅能宣称未进入本次 DPO 偏好及全 GT 流；不能冒充全流程独立验证或最终测试。",
           f"- PSNR 预算预先固定为 {plan['criteria']['psnr_tolerance_db']} dB；没有根据结果重新拟合阈值。",
           "- 模型、选样、噪声协议和所有条件固定；本工具没有训练模型。",
           "", "## 目标预算 / 不确定性", "", "```json",json.dumps(decision,indent=2,ensure_ascii=False),"```",
           "", "## 候选相对 SFT", "", f"|范围|指标|有效源图|SFT|lambda{candidate_lambda}候选|差值|95%配对CI|变差源图比例|",
           "|---|---|---:|---:|---:|---:|---|---:|"]
    if plan.get("comparison_note"):lines.insert(9,"- "+plan["comparison_note"])
    for r in summaries:
        if r["method"]=="fidelity" and r["reference"]=="sft":
            lines.append(f"|{r['scope']}|{r['metric']}|{r['n_sources']}|{fmt(r['reference_mean'])}|{fmt(r['method_mean'])}|{fmt(r['delta_mean'])}|[{fmt(r['ci_low'])},{fmt(r['ci_high'])}]|{fmt(r.get('worse_fraction'))}|")
    lines += ["", f"## 候选相对 lambda{baseline_lambda} 基线", "", "|范围|PSNR差|95%CI|全图MUSIQ差|", "|---|---:|---|---:|"]
    by={(r['scope'],r['metric']):r for r in summaries if r['method']=='fidelity' and r['reference']=='baseline'}
    for scope in ("all",*WEATHERS):
        p,m=by[(scope,'psnr')],by[(scope,'musiq')]
        lines.append(f"|{scope}|{fmt(p['delta_mean'])}|[{fmt(p['ci_low'])},{fmt(p['ci_high'])}]|{fmt(m['delta_mean'])}|")
    lines += ["", "## 必须查看的逐图情况", "", "下面为候选相对 SFT 最差的 PSNR 源图，按种子先求每图均值。所有对应输出可在 inference 目录检查。",
              "", "|天气|源图ID|PSNR差|拼图：上排原图，下排共同尺度误差|", "|---|---|---:|---|"]
    for weather in WEATHERS:
        worst=sorted((r for r in changes if r['scope']==weather and r['method']=='fidelity' and r['reference']=='sft' and r['metric']=='psnr'),key=lambda r:r['delta'])[:3]
        for r in worst:lines.append(f"|{weather}|{r['source_id']}|{r['delta']:+.6g}|[查看]({panels[r['source_id']]})|")
    subgroup=defaultdict(lambda:defaultdict(list))
    for r in changes:
        if r['scope']=='all' and r['method']=='fidelity' and r['reference']=='sft' and r['metric'] in ('psnr','musiq'):
            subgroup[(r['weather'],r['subdataset'])][r['metric']].append(r['delta'])
    lines += ["", "## 子数据集方向（小样本，仅辅助）", "", "|天气/子集|PSNR有效源图|PSNR平均差|全图MUSIQ平均差|", "|---|---:|---:|---:|"]
    for (weather,sub),values in sorted(subgroup.items()):
        p,m=values.get('psnr',[]),values.get('musiq',[])
        lines.append(f"|{weather}/{sub}|{len(p)}|{fmt(sum(p)/len(p) if p else None)}|{fmt(sum(m)/len(m) if m else None)}|")
    lines += ["", "## 验证边界与执行建议",
              "- CI 以源图为单位 bootstrap，先在源图内平均 seed；总体按天气分层，不把两个seed当成两张独立图片。全量均值/中位数/p05/p95/下降比例见CSV。",
              "- 本轮评估整图 MUSIQ/CLIP-IQA/NIMA 及非奖励 TOPIQ-NR；这些都使用同一次原始评分，不给每个方法另拟合归一化。",
              "- `PASS` 才表示在预设验证预算内得到证据；`supports_budgeted_continuation` 只支持固定lambda的受控中程训练，不得写成已满足最终不下降目标。",
              "- 少量独立样本和两种seed不能保证未来全部训练步或所有图像都不下降。正式 test 留待最终冻结模型后评测。",
              "- PSNR预算保护总体与各天气的平均指标，不保证每张图不下降；p05/下降比例和最差图用于揭示均值掩盖的退化。",
              "- 继续训练前人工查看最差样本，注意色偏、结构错位、过锐和虚构纹理。没有按美学评分筛选验证图片。",
              "- 不重跑 GT 梯度或一步sigma诊断；此前已定位训练信号，本轮优先检查DPO未见输入上的完整推理质量。"]
    (root/"validation_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("Final target:",decision['status'],"; monitored continuation:",decision['supports_budgeted_continuation'])
    print("Report:",root/"validation_report.md")
    return decision


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="stage",required=True)
    prepare=sub.add_parser("prepare",help="Freeze DPO-disjoint validation sources before inference")
    prepare.add_argument("--probe-root",required=True)
    for w in WEATHERS:prepare.add_argument("--"+w+"-root",required=True)
    prepare.add_argument("--output",required=True)
    prepare.add_argument("--samples-per-weather",type=int,default=30)
    prepare.add_argument("--seeds",default="42,2024")
    prepare.add_argument("--data-role",choices=("dpo_holdout","external_validation"),default="dpo_holdout")
    prepare.add_argument("--psnr-tolerance",type=float,default=0.0,help="Predeclare allowed dB drop; 0 means strict nonregression")
    prepare.add_argument("--materialization",choices=("symlink","copy"),default="symlink")
    run=sub.add_parser("run",help="Real inference + whole-image IQA + paired report; no training")
    score=sub.add_parser("score",help="Rescore already saved predictions, without loading SD3")
    report=sub.add_parser("report",help="CPU paired summary")
    cont=sub.add_parser("continue-config",help="Prepare a separate resumed run; does not start it")
    recheck=sub.add_parser("recheck",help="Same cohort/criteria for a later candidate; reuse fixed SFT/baseline outputs")
    for parser in (run,score,report,cont,recheck):parser.add_argument("--plan",required=True)
    for parser in (run,score):parser.add_argument("--device",default="cuda:0")
    cont.add_argument("--training-output",required=True)
    cont.add_argument("--total-steps",type=int,default=2000)
    recheck.add_argument("--candidate-run",required=True)
    recheck.add_argument("--checkpoint",required=True)
    recheck.add_argument("--output",required=True)
    space=sub.add_parser("space",help="CPU estimate of future checkpoint/EMA disk peak, before doing expensive validation")
    space.add_argument("--probe-root",required=True)
    space.add_argument("--training-output",required=True)
    space.add_argument("--total-steps",type=int,default=2000)
    args=p.parse_args()
    if args.stage=="space":
        import yaml
        run=Path(args.probe_root).resolve()/"fidelity"
        with (run/"dpo_config.yaml").open(encoding="utf-8") as f:cfg=yaml.safe_load(f)
        storage=continuation_storage(run/"checkpoint-200",cfg["training"],args.training_output,args.total_steps)
        print(json.dumps(storage,indent=2))
        print(f"Need approximately {storage['estimated_peak_bytes']/1024**3:.2f} GiB; free {storage['free_bytes']/1024**3:.2f} GiB")
        if not storage["enough_space_estimate"]:raise SystemExit("Choose a larger output disk before continuation; the trained 200-step probes are reusable")
        return
    if args.stage=="prepare":
        plan=prepare_validation(args.probe_root,{w:getattr(args,w+"_root") for w in WEATHERS},args.output,
                                args.samples_per_weather,tuple(int(x) for x in args.seeds.split(",")),args.data_role,
                                args.psnr_tolerance,args.materialization)
        print("Prepared:",Path(args.output)/"validation_plan.json","; sources:",len(plan['sources']))
        return
    if args.stage=="recheck":
        recheck_plan(args.plan,args.candidate_run,args.checkpoint,args.output)
        print("Prepared fixed-cohort recheck:",Path(args.output)/"validation_plan.json")
        return
    if args.stage in ("run","score"):
        import torch
        if not torch.cuda.is_available() or not args.device.startswith("cuda"):
            raise RuntimeError("Real three-model validation requires CUDA; no model-stub fallback")
        from dpo.texture_release_runtime import run_inference,score_outputs
        if args.stage=="run":run_inference(args.plan,args.device)
        score_outputs(args.plan,args.device)
        summarize(args.plan)
    elif args.stage=="report":summarize(args.plan)
    else:
        import yaml
        plan_path=Path(args.plan).resolve();root=plan_path.parent
        plan,decision=read_json(plan_path),read_json(root/"validation_decision.json")
        if decision['plan_sha256']!=digest(plan_path) or decision['results_sha256']!=digest(root/"validation_results.json"):
            raise ValueError("Validation evidence changed")
        cfg=continuation_config(plan,decision,args.training_output,args.total_steps)
        output=root/f"continue_{args.total_steps}.yaml"
        with output.open("x",encoding="utf-8") as f:yaml.safe_dump(cfg,f,sort_keys=False,allow_unicode=True)
        print("Prepared, NOT started:",output)


if __name__=="__main__":main()
