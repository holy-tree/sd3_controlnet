"""CPU-only postprocessing of REAL diagnostics and a fixed-lambda A/B probe.

Never loads a model, edits an old config, or starts training. Uses the effective
training snapshot embedded in diagnostic_state.json, not current default YAML.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import shlex
from collections import defaultdict
from pathlib import Path

import yaml


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def number(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Missing/non-numeric " + label) from error
    if not math.isfinite(result):
        raise ValueError("Nonfinite " + label)
    return result


def projected_metrics(q, f, dot, coefficient):
    """Algebra at the measured parameter point, NOT an optimizer simulation."""
    if q <= 0 or f <= 0:
        raise ValueError("Cannot calibrate from a zero/missing objective gradient")
    result = {"lambda": coefficient, "weighted_gt_norm": coefficient*f,
              "gt_to_dpo_ratio": coefficient*f/q,
              "gt_dot_total": dot+coefficient*f*f,
              "total_norm": math.sqrt(max(0.0, q*q+2*coefficient*dot+coefficient*coefficient*f*f))}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("Nonfinite projected weighting metrics; no experiment configs will be generated")
    return result


def parse_preview(value):
    if value in (None, ""):
        return None
    result = json.loads(value)
    if not isinstance(result, list) or not all(isinstance(n, str) for n in result):
        raise ValueError("Invalid missing-gradient name preview")
    return result


def read_evidence(directory, candidate):
    directory = Path(directory).expanduser().resolve()
    state_path, csv_path = directory/"diagnostic_state.json", directory/"diagnostics.csv"
    with state_path.open(encoding="utf-8") as f:
        state = json.load(f)
    if state.get("status") != "completed_real_models" or not state.get("runtime", {}).get("executed"):
        raise ValueError("Requires completed REAL-model diagnostics, not inspect-only or synthetic data")
    runtime = state["runtime"]
    if any(runtime.get(k) is not True for k in ("policy_parameters_unchanged", "reference_unchanged", "buffers_unchanged")):
        raise ValueError("Parameter/reference/buffer integrity must be verified before preparing a weighting probe")
    snapshot = state.get("snapshot") or {}
    config, metadata = snapshot.get("config"), snapshot.get("metadata")
    if not config or not metadata or metadata.get("version") != 6:
        raise ValueError("Missing actual saved training snapshot/v6 checkpoint metadata")
    if metadata.get("num_processes") != 1:
        raise ValueError("This single-GPU probe does not reinterpret a multi-rank diagnostic as a global gradient")
    train = config["training"]
    if train.get("reward", {}).get("mode") != "dynamic_texture":
        raise ValueError("Effective historical mode is not dynamic_texture")
    settings = metadata["spatial_dynamic"]["settings"]
    coefficient = number(settings["lambda_fidelity"], "checkpoint lambda")
    if coefficient <= 0:
        raise ValueError("Historical lambda must be positive")
    if train.get("spatial_dynamic", {}).get("lambda_fidelity", coefficient) != coefficient:
        raise ValueError("Snapshot/checkpoint disagree about lambda")
    if candidate is not None and (not math.isfinite(candidate) or candidate <= 0):
        raise ValueError("Candidate lambda must be finite and positive")

    measured, missing = [], defaultdict(lambda: {"q_names":set(), "f_names":set(), "q_missing_events":0,
                                               "f_missing_events":0, "preview_available":True})
    windows = set()
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("row_type") != "accum_gradient":
                continue
            if row.get("status") == "nonfinite_gradient":
                raise ValueError("Nonfinite accumulation gradients: investigate before changing lambda")
            q = number(row.get("dpo_norm"), "DPO norm")
            gt = number(row.get("gt_norm"), "GT norm")
            cosine = number(row.get("cosine"), "gradient cosine")
            dot = number(row.get("gt_dot_dpo"), "GT dot DPO")
            weighted = number(row.get("weighted_gt_norm"), "weighted GT norm")
            if not math.isclose(weighted, coefficient*gt, rel_tol=1e-6, abs_tol=1e-12):
                raise ValueError("CSV and snapshot lambda disagree")
            if q <= 0 or gt <= 0 or not math.isclose(dot/(q*gt), cosine, rel_tol=1e-5, abs_tol=1e-6):
                raise ValueError("Missing/zero/inconsistent accumulation gradient evidence")
            due = str(row.get("gt_due", "")).lower() in ("true", "1")
            threshold = max(0.0, -dot/(gt*gt))
            if not math.isfinite(threshold):
                raise ValueError("Nonfinite SGD boundary")
            item = {"window":row["window"], "weather":row["weather"], "gt_weather":row["gt_weather"],
                    "scope":row["scope"], "gt_due":due, "dpo_norm":q, "gt_norm":gt, "cosine":cosine,
                    "gt_dot_dpo":dot, "lambda_sgd_nonincrease_boundary":threshold,
                    "original":projected_metrics(q, gt, dot, coefficient),
                    "candidate":projected_metrics(q, gt, dot, candidate) if candidate is not None else None}
            measured.append(item)
            if row["scope"] == "all":
                key = row["window"]
                if key in windows:
                    raise ValueError("Duplicate all-parameter accumulation window")
                windows.add(key)
            group = missing[row["scope"]]
            for kind in ("q", "f"):
                group[kind+"_missing_events"] += int(row.get(kind+"_missing") or 0)
                names = parse_preview(row.get(kind+"_missing_names_preview"))
                group["preview_available"] &= names is not None
                if names is not None:
                    group[kind+"_names"].update(names)
    if not measured or not windows:
        raise ValueError("No usable real accumulation windows")
    active = [r for r in measured if r["scope"] == "all" and r["gt_due"]]
    if not active:
        raise ValueError("No actual GT-active accumulation window")
    if len(windows) != runtime.get("measured_accumulation_windows"):
        raise ValueError("CSV/state accumulation-window count mismatch")
    boundary = max(r["lambda_sgd_nonincrease_boundary"] for r in active)
    rounded = (math.ceil(boundary/10**math.floor(math.log10(boundary)))*10**math.floor(math.log10(boundary))) if boundary else coefficient
    missing_out = {}
    for scope, values in missing.items():
        missing_out[scope] = {"q_missing_events":values["q_missing_events"], "f_missing_events":values["f_missing_events"],
                              "preview_available":values["preview_available"],
                              "common_missing_name_preview":sorted(values["q_names"] & values["f_names"]),
                              "gt_only_missing_name_preview":sorted(values["f_names"] - values["q_names"]),
                              "dpo_only_missing_name_preview":sorted(values["q_names"] - values["f_names"])}
    return {"schema":"texture_fixed_lambda_probe_v1", "diagnostic_directory":str(directory),
            "state_sha256":sha(state_path), "csv_sha256":sha(csv_path), "snapshot":snapshot,
            "original_lambda":coefficient, "candidate_lambda":candidate,
            "measured":measured, "missing_gradient_audit":missing_out,
            "maximum_observed_all_parameter_sgd_boundary":boundary,
            "rounded_boundary_candidate_not_optimum":rounded,
            "source_counts":{"accumulation_windows":len(windows),"gt_active_windows":len(active)}}


def verify_validation_config(path):
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    splits = config.get("splits")
    if not isinstance(splits, list) or not splits or not set(splits) <= {"val", "validation"}:
        raise ValueError("Independent validation config must explicitly use splits: [val] or [validation], not test/train")
    for name in ("dataset_root", "dataset_rain", "dataset_snow", "dataset_haze"):
        if config.get(name) and "test" in Path(config[name]).parts:
            raise ValueError("Test roots cannot be used for weighting feedback")
    return str(path)


def generate_configs(evidence, directory, run_root, steps, validation_config=None):
    candidate = evidence["candidate_lambda"]
    if candidate is None:
        raise ValueError("--write-configs requires an explicitly chosen --candidate-lambda")
    if candidate == evidence["original_lambda"]:
        raise ValueError("Candidate equals original lambda; this would not be a weighting comparison")
    source = copy.deepcopy(evidence["snapshot"]["config"])
    original_train = source["training"]
    warmup = int(original_train.get("lr_warmup_steps", 0))
    if steps <= warmup or steps <= 0:
        raise ValueError(f"Probe steps must exceed historical warmup ({warmup})")
    validation = verify_validation_config(validation_config) if validation_config else None
    run_root = Path(run_root).expanduser().resolve()
    old_output = Path(original_train["output_dir"]).expanduser().resolve()
    if run_root == old_output or old_output in run_root.parents:
        raise ValueError("Probe runs must be separate from the original training directory")
    generated = {}
    for name, coefficient in (("baseline",evidence["original_lambda"]),("fidelity",candidate)):
        config = copy.deepcopy(source)
        config["reward"] = copy.deepcopy(original_train["reward"])
        config["preference_filter"] = copy.deepcopy(original_train["preference_filter"])
        # Materialize the checkpoint's EFFECTIVE defaults, rather than letting
        # today's default values reinterpret a historical partial config.
        config["spatial_dynamic"] = copy.deepcopy(evidence["snapshot"]["metadata"]["spatial_dynamic"]["settings"])
        resolution = config["spatial_dynamic"].pop("resolution")
        if resolution != original_train["resolution"]:
            raise ValueError("Checkpoint/config resolution mismatch")
        config["spatial_dynamic"]["lambda_fidelity"] = coefficient
        config["tail_risk"] = copy.deepcopy(original_train.get("tail_risk", source.get("tail_risk", {"enabled":False})))
        train = config["training"]
        # These are historical runtime copies, not another independent source of
        # settings. The actual trainer will save new effective copies itself.
        for key in ("reward", "preference_filter", "spatial_dynamic", "tail_risk"):
            train.pop(key, None)
        target = run_root/name
        if target.exists():
            raise FileExistsError("Probe training output already exists: " + str(target))
        train.update(output_dir=str(target), max_train_steps=steps, checkpointing_steps=steps,
                     resume_from_checkpoint=None, logging_dir=None,
                     run_checkpoint_validation=validation is not None,
                     validation_before_training=validation is not None,
                     validation_eval_config=validation)
        config.setdefault("evaluation", {})["output_dir"] = str(target/"evaluation")
        path = directory/(name+".yaml")
        if path.exists():
            raise FileExistsError(path)
        generated[name] = {"path":str(path), "config":config}
    # Enforce a single non-path difference BETWEEN the two probe arms.
    def comparable(config):
        config = copy.deepcopy(config)
        config["spatial_dynamic"].pop("lambda_fidelity")
        config["training"].pop("output_dir")
        config["evaluation"].pop("output_dir")
        return config
    if comparable(generated["baseline"]["config"]) != comparable(generated["fidelity"]["config"]):
        raise ValueError("Probe arms differ in more than lambda/output paths")
    for item in generated.values():
        with Path(item["path"]).open("x", encoding="utf-8") as f:
            yaml.safe_dump(item["config"], f, sort_keys=False, allow_unicode=True)
    return {name:{"config":item["path"],"output_dir":item["config"]["training"]["output_dir"],
                  "lambda":item["config"]["spatial_dynamic"]["lambda_fidelity"],
                  "steps":steps,"held_out_validation":validation} for name,item in generated.items()}


def write_report(directory, evidence, configs):
    report = ["# 固定 GT 权重校准与短程对照", "", "## 主要结论",
        "- 输入为已完成的真实模型诊断；本工具仅做 CPU 代数和配置准备，不加载模型、不启动训练。",
        "- lambda_sgd_boundary 仅表示原参数点上，普通一致步长 SGD 令当前 GT loss 一阶不增的边界；不是最佳 lambda，更不是 Adam/PSNR 保证。",
        "- 该边界来自 37000 等指定检查点，不可直接当作 SFT 初始化处或以后每一步的边界。",
        f"- 原 lambda={evidence['original_lambda']}；观测到的全参数 GT-active 最大边界={evidence['maximum_observed_all_parameter_sgd_boundary']:.8g}。",
        f"- 向上取一位有效数字的诊断候选={evidence['rounded_boundary_candidate_not_optimum']:.8g}；实际指定候选={evidence['candidate_lambda']}。",
        "", "```text", "GT_dot_total(lambda) = dot(g_GT,g_DPO) + lambda * ||g_GT||^2",
        "lambda_sgd_boundary = max(0, -dot(g_GT,g_DPO) / ||g_GT||^2)", "```",
        "", "## 累计窗口代数复查", "",
        "|天气/GT天气|参数集合|原GT/DPO|SGD边界|候选GT/DPO|候选GT·total|",
        "|---|---|---:|---:|---:|---:|"]
    for row in evidence["measured"]:
        other = row["candidate"]
        report.append(f"|{row['weather']}/{row['gt_weather']}|{row['scope']}|{row['original']['gt_to_dpo_ratio']:.7g}|{row['lambda_sgd_nonincrease_boundary']:.7g}|"+
                      (f"{other['gt_to_dpo_ratio']:.7g}|{other['gt_dot_total']:.7g}|" if other else "N/A|N/A|"))
    report += ["", "## 缺失梯度核对", "", "预览最多列出诊断器保存的部分名字，不是完整参数集合；不要仅凭共同缺失就认定正常。",
               "GT 独有缺失应先结合前向结构核对；旧诊断若没有预览列，需要更新诊断器后才能定位名字。",
               "", "```json", json.dumps(evidence["missing_gradient_audit"],indent=2,ensure_ascii=False), "```"]
    if configs:
        report += ["", "## 已生成、尚未执行的对照", "", "两组从原始 SFT 路径初始化，不 resume 37000；相同偏好/GT、beta、学习率、纹理强度、sigma、有效batch、seed及训练预算。",
                   "仅 lambda 和输出路径不同。未提供独立 val 配置时，关闭自动验证并清空 validation_eval_config，避免回落到项目 test 配置。",
                   "此时只是训练信号/稳定性探针，不能据训练诊断图选择最终模型或宣称泛化 PSNR 改善。", "", "```bash"]
        for item in configs.values():
            report.append("CUDA_VISIBLE_DEVICES=0 python -m scripts.train_dpo_sd3 --config "+shlex.quote(item["config"]))
        report += ["```", "", "各运行只到约定探针步数并保存 checkpoint。之后分别用各自保存的 dpo_config.yaml 和 checkpoint 运行相同真实诊断；不要将 λ 不同的配置直接 resume 原 checkpoint。"]
    else:
        report += ["", "未生成训练配置。若决定做单一候选的对照，显式使用 --write-configs、--candidate-lambda、--run-root。"]
    report += ["", "## 判断顺序", "", "1. 先核对缺失梯度与非有限状态，再看真实相对梯度和联合方向。",
               "2. 短程对照只检查两个目标是否实际参与、一步代理是否改善及同条件完整推理的线索。",
               "3. 最终 PSNR/美学是否兼顾，必须在独立验证集上判断；不得复用 test 集调 lambda。没有 val 数据时暂不作此结论。",
               "4. 若标定后仍有可复现反向冲突，再研究冲突处理；若反复出现代理改善而推理退化，再优先修改 GT 代理。"]
    (directory/"calibration_report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    with (directory/"calibration.json").open("x",encoding="utf-8") as f:
        json.dump({**evidence,"generated_configs":configs},f,indent=2,ensure_ascii=False,allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-dir",required=True)
    parser.add_argument("--output",required=True,help="New directory for the algebra report and optional YAMLs")
    parser.add_argument("--candidate-lambda",type=float)
    parser.add_argument("--write-configs",action="store_true")
    parser.add_argument("--run-root",help="New sibling experiment root, outside original training output")
    parser.add_argument("--steps",type=int,default=200,help="Shared short-probe optimizer-step budget")
    parser.add_argument("--validation-config",help="Optional independent val/validation config; never test")
    args = parser.parse_args()
    if args.write_configs and (args.candidate_lambda is None or not args.run_root):
        parser.error("--write-configs requires --candidate-lambda and --run-root")
    evidence = read_evidence(args.diagnostic_dir,args.candidate_lambda)
    directory = Path(args.output).expanduser().resolve()
    for protected in (Path(evidence["diagnostic_directory"]),
                      Path(evidence["snapshot"]["config"]["training"]["output_dir"]).expanduser().resolve()):
        if directory == protected or protected in directory.parents:
            raise ValueError("Planning output must be outside the old training/diagnostic directories")
    directory.mkdir(parents=True,exist_ok=False)
    configs = generate_configs(evidence,directory,args.run_root,args.steps,args.validation_config) if args.write_configs else {}
    write_report(directory,evidence,configs)
    print("Observed SGD boundary (not optimal lambda):",evidence["maximum_observed_all_parameter_sgd_boundary"])
    print("Missing-gradient audit and report:",directory/"calibration_report.md")
    if configs:
        print("Configs created; training NOT started:")
        for item in configs.values():
            print(item["config"])


if __name__ == "__main__":
    main()
