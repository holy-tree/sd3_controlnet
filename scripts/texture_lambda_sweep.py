"""Prepare, explicitly train, and compare fixed lambda_fidelity candidates to SFT."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dpo.texture_lambda_sweep import (
    evaluation_plan, load_sweep, pending_training_arms, prepare_sweep, report_sweep, storage_estimate, train_sweep,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    prepare = sub.add_parser("prepare", help="Write independent configs and a fixed plan; never start training")
    prepare.add_argument("--validation-plan", required=True, help="Prepared by scripts.validate_texture_release prepare; does not require prior inference")
    prepare.add_argument("--output", required=True, help="New config/report directory")
    prepare.add_argument("--run-root", required=True, help="Separate training output root with enough disk")
    prepare.add_argument("--lambdas", default="1000,2000,5000,10000")
    prepare.add_argument("--steps", type=int, default=200)
    prepare.add_argument("--storage-mode", choices=("full", "weights-only"), default="full",
                         help="weights-only: after each NEW run completes, retain raw evaluation weights/config/logs and delete its generated optimizer/RNG/EMA/redundant exports; reused runs are preserved")
    prepare.add_argument("--reuse", action="append", default=[], metavar="LAMBDA=RUN_DIRECTORY", help="Explicitly reuse a matching SFT-started run; repeatable")
    train = sub.add_parser("train", help="Explicitly start single-GPU serial training; stop on first failure")
    train.add_argument("--gpu", default="0", help="One physical CUDA GPU index")
    run = sub.add_parser("run", help="Real paired inference, whole-image scoring and report; no training")
    score = sub.add_parser("score", help="Retry scoring saved predictions without loading SD3")
    report = sub.add_parser("report", help="CPU paired statistics from saved metrics")
    space = sub.add_parser("space", help="Estimate retained checkpoints/EMA/exports and evaluation storage")
    for command in (train, run, score, report, space):
        command.add_argument("--plan", required=True)
    for command in (run, score):
        command.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.stage == "prepare":
        reuse = {}
        for item in args.reuse:
            key, separator, value = item.partition("=")
            if not separator or not value or float(key) in reuse:
                parser.error("--reuse expects unique LAMBDA=RUN_DIRECTORY entries")
            reuse[float(key)] = value
        plan = prepare_sweep(args.validation_plan, args.output, args.run_root,
                             [float(v) for v in args.lambdas.split(",")], args.steps, reuse, args.storage_mode)
        print("Prepared; training NOT started:", Path(args.output).resolve() / "sweep_plan.json")
        print("Arms:", ", ".join(f"{a['lambda']:g}" + (" (reuse)" if a["reused_snapshot"] else " (new training)") for a in plan["arms"]))
        print(json.dumps(storage_estimate(plan, pending_training_arms(plan)), indent=2))
    elif args.stage == "train":
        train_sweep(args.plan, args.gpu)
    elif args.stage == "space":
        _, plan = load_sweep(args.plan)
        print(json.dumps(storage_estimate(plan, pending_training_arms(plan)), indent=2))
    elif args.stage == "report":
        print("Report:", report_sweep(args.plan))
    else:
        import torch
        if not args.device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("Real SD3 validation requires CUDA")
        from dpo.texture_release_runtime import run_inference, score_outputs
        eval_plan = evaluation_plan(args.plan)
        if args.stage == "run":
            run_inference(eval_plan, args.device)
        score_outputs(eval_plan, args.device)
        print("Report:", report_sweep(args.plan))


if __name__ == "__main__":
    main()
