"""Read-only-parameter diagnostics for REAL dynamic_texture checkpoints.

No optimizer is constructed. --inspect-only works without diffusion libraries.
--batches counts complete, single-rank accumulation windows, not microbatches.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def environment():
    deps = {name: importlib.util.find_spec(name) is not None
            for name in ("torch", "torchvision", "numpy", "yaml", "accelerate", "diffusers", "transformers", "pyiqa", "lpips")}
    result = {"python": sys.version, "platform": platform.platform(), "dependencies": deps,
              "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if deps["torch"]:
        try:
            import torch
            result.update(torch=torch.__version__, cuda_available=torch.cuda.is_available(),
                          cuda_devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
        except Exception as error:
            result.update(torch_import_error=str(error), cuda_available=False)
    return result


def code_audit():
    files = ("scripts/train_dpo_sd3.py", "dpo/spatial.py", "dpo/spatial_dataset.py",
             "dpo/reference.py", "dpo/losses.py", "dpo/tail_risk_training.py",
             "scripts/diagnose_dynamic_texture.py", "dpo/texture_diagnostics.py", "dpo/texture_diagnostics_runtime.py")
    audit = {"files": {}, "findings": []}
    for relative in files:
        path = ROOT/relative
        text = path.read_text(encoding="utf-8")
        audit["files"][relative] = {"sha256": sha(path), "lines": len(text.splitlines())}
    spatial_tree = ast.parse((ROOT/"dpo/spatial.py").read_text(encoding="utf-8"))
    decode = next(n for n in spatial_tree.body if isinstance(n, ast.FunctionDef) and n.name == "decoded_gt_mse")
    calls = [n for n in ast.walk(decode) if isinstance(n, ast.Call)]
    blocked = [n.func.attr for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr in ("detach", "no_grad", "clamp", "clamp_")]
    audit["gt_decode_blocking_calls"] = blocked
    audit["findings"] += [
        {"evidence": f"dpo/spatial.py:{decode.lineno}", "finding": "GT 一步估计采用 x0=z_sigma-sigma*v，按 VAE scale/shift 逆变换后计算 RGB [0,1] 全图 MSE。"},
        {"evidence": f"dpo/spatial.py:{decode.lineno}", "finding": "decoded_gt_mse 内未发现 detach/no_grad/硬 clamp。" if not blocked else "发现可能阻断操作："+str(blocked)},
        {"evidence": "scripts/train_dpo_sd3.py GT branch", "finding": "GT latent 与独立辅助 LQ 编码位于 no_grad 中；ControlNet/RA 的 GT 预测和 VAE decode 在其外。日志 detach 位于两次 backward 之后。"},
        {"evidence": "dpo/spatial_dataset.py", "finding": "偏好数据与全源图 GT 流相互独立；同一偏好的 GT/LQ/候选/mask 共用几何变换和翻转，辅助 GT/LQ 自身保持对齐。"},
        {"evidence": "dpo/reference.py:snapshot_reference_parameters", "finding": "原训练在载入 DPO checkpoint 前，从初始 SFT 参数 detach+clone 固定参考；未单独快照 buffers，需在真实运行检查其变化。"},
        {"evidence": "dpo/losses.py; dpo/tail_risk_training.py", "finding": "DPO 优化候选 FM 误差的相对偏好差距，不直接优化任一候选的最终 RGB 保真。"},
    ]
    return audit


def read_snapshot(args):
    if not args.config:
        return None
    import yaml
    path = Path(args.config).expanduser().resolve()
    with path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    train = config.get("training") or {}
    # These embedded blocks are actually written by the trainer. Do not replace
    # them with current repository YAML defaults or infer an old experiment.
    if not all(key in train for key in ("reward", "preference_filter", "spatial_dynamic")):
        raise ValueError("--config must be the experiment's saved dpo_config.yaml with embedded effective training blocks")
    if train["reward"].get("mode") != "dynamic_texture":
        raise ValueError("Saved training mode is not dynamic_texture")
    config["reward"], config["preference_filter"], config["spatial_dynamic"] = (
        train["reward"], train["preference_filter"], train["spatial_dynamic"])
    metadata = None
    if args.checkpoint:
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        with (checkpoint/"dpo_resume.json").open(encoding="utf-8") as f:
            metadata = json.load(f)
        for name in ("beta", "resolution", "train_batch_size", "gradient_accumulation_steps", "seed"):
            if name not in train or train[name] != metadata.get(name):
                raise ValueError("Saved config/checkpoint disagree or do not attest actual " + name)
    return {"config_path": str(path), "config_sha256": sha(path), "config": config, "metadata": metadata}


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def write_csv(path, rows):
    columns = list(dict.fromkeys(["row_type", "window", "microbatch", "weather", "gt_weather", "scope",
                                 *[key for row in rows for key in row]]))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list, tuple)) else v
                             for k, v in clean(row).items()})


def report(directory, state, rows):
    state, rows = clean(state), clean(rows)
    (directory/"diagnostic_state.json").write_text(json.dumps(state, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    write_csv(directory/"diagnostics.csv", rows)
    lines = ["# dynamic_texture 诊断报告", "", "## 主要发现"]
    runtime = state.get("runtime", {})
    if not runtime.get("executed"):
        lines += ["", "**尚无真实训练运行发现。未用合成数据或小模型替代。**",
                  "", "原因："+str(state.get("error") or "仅执行 inspect-only，未加载实际 checkpoint。"),
                  "", "当前证据不足以判定 GT 梯度弱、梯度冲突或 GT 代理失配，也不能据此调整 λ。"]
    else:
        lines += ["", "### 真实运行发现", "", "运行状态："+str(state.get("status")),
                  "", f"参数值未变化：{runtime.get('policy_parameters_unchanged')}；参考快照未变化：{runtime.get('reference_unchanged')}；共享 buffers 未变化：{runtime.get('buffers_unchanged')}"]
        for finding in runtime.get("findings", []):
            lines.append("- "+finding)
    lines += ["", "### 代码检查发现"]
    for item in state["code_audit"]["findings"]:
        lines.append(f"- `{item['evidence']}`：{item['finding']}")
    lines += ["", "### 尚未验证的推测", "",
              "- GT 权重较弱、目标冲突、一步 GT 代理和完整采样的分布差异都可能相关；默认 YAML 的 λ 或 β 数值本身不是证据。",
              "- 没有 paired full-inference 数据时，不能区分情况 C（一步代理改善、完整 PSNR 下降）和 D（两者都变差）。"]
    snapshot = state.get("snapshot")
    lines += ["", "## 实际配置与输入"]
    if snapshot:
        cfg, meta = snapshot["config"], snapshot.get("metadata") or {}
        train = cfg["training"]
        settings = ((meta.get("spatial_dynamic") or {}).get("settings") or {})
        values = {"saved_config": snapshot["config_path"], "saved_config_sha256": snapshot["config_sha256"],
                  "checkpoint": state["arguments"].get("checkpoint"), "checkpoint_step": meta.get("global_step"),
                  "beta": train.get("beta"), "lambda_fidelity": settings.get("lambda_fidelity"),
                  "texture_strength": settings.get("texture_strength"), "GT": settings.get("gt"),
                  "microbatch": train.get("train_batch_size"), "accumulation": train.get("gradient_accumulation_steps"),
                  "original_world_size": meta.get("num_processes"), "preference_manifest": train.get("preference_manifest"),
                  "gt_manifest": train.get("gt_source_manifest")}
        if all(values.get(k) is not None for k in ("microbatch", "accumulation", "original_world_size")):
            values["original_effective_global_batch"] = values["microbatch"]*values["accumulation"]*values["original_world_size"]
        lines += ["", "```json", json.dumps(values, indent=2, ensure_ascii=False), "```"]
    else:
        lines += ["", "未取得实际训练快照，β、λ、纹理强度、GT sigma/间隔和有效 batch 均未核实；没有使用默认 YAML 冒充。"]
    lines += ["", "## 梯度统计", "", "完整均值、中位数、p05/p25/p75/p95 与缺失状态见 `diagnostic_state.json`，逐样本值见 `diagnostics.csv`。", "",
              "|层级/参数/天气条件|有效窗口/微批|DPO范数中位数|加权GT范数中位数|GT/DPO 中位数 [p05,p95]|余弦均值|负余弦|负 scheduled GT·total|状态计数|",
              "|---|---:|---:|---:|---|---:|---:|---:|---|"]
    def fmt(value):
        return f"{value:.5g}" if isinstance(value, (float,int)) else "N/A"
    for key, summary in runtime.get("gradient_summary", {}).items():
        q, f, r, c = (summary[n] for n in ("dpo_norm", "weighted_gt_norm", "gt_to_dpo_ratio", "cosine"))
        lines.append(f"|{key}|{q['n']}|{fmt(q['median'])}|{fmt(f['median'])}|{fmt(r['median'])} [{fmt(r['p05'])},{fmt(r['p95'])}]|{fmt(c['mean'])}|{summary['negative_cosine_count']}/{summary['cosine_valid_count']}|{summary['negative_scheduled_gt_dot_total_count']}/{summary['scheduled_dot_valid_count']}|{summary['status_counts']}|")
    lines += ["", "## DPO 拉差与样本贡献"]
    for weather, values in runtime.get("dpo_summary", {}).items():
        lines.append(f"- {weather}：logit 中位数={fmt(values['logit']['median'])}，p05={fmt(values['logit']['p05'])}，p95={fmt(values['logit']['p95'])}；rejected 正贡献份额中位数={fmt(values['rejected_push_fraction']['median'])}。")
    for scope, values in runtime.get("per_sample_gradient_summary", {}).items():
        lines.append(f"- {scope}：真实逐对梯度已测={values['measured']}；微批内范数贡献有效数量中位数={fmt(values['effective_norm_contributors_per_microbatch']['median'])}。贡献范数分布和最大样本身份见 JSON。")
    if runtime.get("inference_summary"):
        lines += ["", "## 同条件的一步 GT 与完整推理", "", "```json",
                  json.dumps(runtime["inference_summary"], indent=2, ensure_ascii=False), "```"]
    if runtime.get("comparison_local_normalization"):
        lines += ["", "局部IQA比较状态："+str(runtime.get("local_iqa_normalization_status")),
                  "；唯一冻结统计：`"+runtime["comparison_local_normalization"]["path"]+"`；哈希：`"+
                  runtime["comparison_local_normalization"]["sha256"]+"`。原始分数与可用的统一z/Q见CSV。"]
    if state.get("error") and runtime.get("executed"):
        lines += ["", "本次真实运行未完整完成："+state["error"]]
    lines += ["", "## 解释边界与下一步优先级", "",
              "- g_F·g_total<0 仅指普通小步梯度下降下该 GT loss 的一阶上升趋势；不代表 Adam 的实际更新或最终推理 PSNR 一定下降。",
              "- 本轮参数/参考哈希比较只验证诊断前后；没有历史参考快照日志时，不能追溯证明此前整个训练过程从未改变参考。",
              "- micro 梯度按单个 batch mean 计；accumulation 梯度是配置累计步数的平均。此工具单进程运行，不声称测得多卡 all-reduce 的全局梯度。",
              "- GT 间隔>1 时仍额外探测原始 GT 梯度；fixed-lambda 列回答题设公式，scheduled 列才表示该步真实启用/跳过的联合梯度。",
              "- 按 `(DPO天气, GT天气)` 条件汇总；不把跨天气辅助 GT 的梯度误称为纯单天气梯度。抽样是分层诊断抽样，不是精确重放下一批训练数据。",
              "- per-sample 范数份额是各贡献向量范数的份额，不是向量和中的可加比例。sigmoid 舍入成 1 不等于梯度严格为 0，另记录稳定计算的 logit 导数。",
              "- E_policy/E_reference 是候选分支的空间加权 FM 误差，不是最终 RGB 保真。DPO 低于参考中性值 log(2) 不是训练时间序列上的 loss 降幅。",
              "- 一步 RGB MSE 在 clamp 之前计算；可视化才 clamp。完整推理沿用项目的 PIL/量化输出与 PSNR/SSIM/LPIPS 评测路径。",
              "- 全部完整推理样本来自保存的训练 GT manifest，不载入测试集配置；少量训练样本只能给出诊断线索，不能用于宣称泛化质量或唯一原因。"]
    priorities = runtime.get("priorities") or ["先在具备真实快照、权重及依赖的环境运行本工具；本地静态检查不能替代三个问题的实测。"]
    lines += ["", "### 基于当前证据的优先级"]+[f"{i}. {text}" for i, text in enumerate(priorities, 1)]
    lines += ["", "## 环境", "", "```json", json.dumps(state["environment"], indent=2, ensure_ascii=False), "```"]
    (directory/"diagnostic_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", help="ACTUAL experiment output/dpo_config.yaml; no default YAML fallback")
    p.add_argument("--checkpoint", help="Complete raw checkpoint directory containing controlnet, ra_fusion, dpo_resume.json")
    p.add_argument("--batches", type=int, default=3, help="Number of complete accumulation windows; microbatches=batches*saved_accumulation")
    p.add_argument("--output", required=True, help="New diagnostic directory, never training/checkpoint output")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--inspect-only", action="store_true")
    p.add_argument("--full-inference", action="store_true", help="Paired real SFT/current inference on sampled TRAINING GT sources")
    p.add_argument("--inference-samples-per-weather", type=int, default=1)
    p.add_argument("--local-iqa", action="store_true", help="After releasing SD3, score both full outputs with SAME frozen local stats")
    p.add_argument("--per-sample-gradients", action=argparse.BooleanOptionalAction, default=True,
                   help="Measure actual per-pair gradient contribution norms (extra backward passes)")
    p.add_argument("--keep-gradient-buffers", action="store_true", help="Retain diagnostic-only disk vectors; otherwise delete them after measuring")
    p.add_argument("--loss-scale", type=float, help="Explicit FP16 scaler override, recorded; otherwise read checkpoint scaler.pt")
    args = p.parse_args()
    if args.batches <= 0 or args.inference_samples_per_weather <= 0:
        p.error("counts must be positive")
    if not args.inspect_only and (not args.config or not args.checkpoint):
        p.error("real diagnostics require --config and --checkpoint")
    if args.local_iqa and not args.full_inference:
        p.error("--local-iqa requires --full-inference")
    output = Path(args.output).expanduser().resolve()
    for value in (args.checkpoint, Path(args.config).parent if args.config else None):
        if value:
            protected = Path(value).expanduser().resolve()
            if output == protected or protected in output.parents:
                p.error("diagnostic output must be separate from the training/checkpoint directory")
    return args


def main():
    args = parse_args()
    directory = Path(args.output).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)
    state = {"arguments": vars(args), "environment": environment(), "code_audit": code_audit(),
             "snapshot": None, "runtime": {"executed": False}, "status": "inspect_only"}
    rows = []
    failed = False
    try:
        state["snapshot"] = read_snapshot(args)
        if not args.inspect_only:
            required = ("torch", "torchvision", "numpy", "yaml", "accelerate", "diffusers", "transformers")
            if args.full_inference:
                required += ("lpips",)
            if args.local_iqa:
                required += ("pyiqa",)
            missing = [name for name in required if not state["environment"]["dependencies"][name]]
            if missing:
                raise RuntimeError("Missing real-model dependencies: "+", ".join(missing))
            if not state["environment"].get("cuda_available") or not args.device.startswith("cuda"):
                raise RuntimeError("Real SD3 gradient diagnostic requires CUDA; no CPU/tiny-model fallback")
            from dpo.texture_diagnostics_runtime import run_diagnostics
            state["status"] = "running_real_models"
            report(directory, state, rows)  # pending evidence survives a process-level kill
            run_diagnostics(args, state, rows, directory)
            state["status"] = "completed_real_models"
    except Exception as error:
        failed = True
        state.update(status="failed_or_partial", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    finally:
        report(directory, state, rows)
    print("Report:", directory/"diagnostic_report.md")
    print("CSV:", directory/"diagnostics.csv")
    if failed:
        raise SystemExit(state["error"])


if __name__ == "__main__":
    main()
