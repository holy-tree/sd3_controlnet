"""Algebra/config preparation only; no models or real-training claims."""
import copy
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.plan_texture_fidelity_check import read_evidence, generate_configs, verify_validation_config, projected_metrics


def fixture(root):
    settings = {"version":1,"resolution":512,"lambda_fidelity":.1,"texture_strength":2.0,
                "mask":{"window":7},"gt":{"interval":1,"sigma_min":.05,"sigma_max":.7}}
    train = {"reward":{"mode":"dynamic_texture","min_texture_q_gap":.05},
             "preference_filter":{"candidate_metrics_path":"unchanged.csv","random_seed":42},
             "spatial_dynamic":{"version":1,"lambda_fidelity":.1,"texture_strength":2},
             "output_dir":str(root/"original"),"resolution":512,"max_train_steps":40000,
             "preference_manifest":"original_pairs.jsonl","gt_source_manifest":"all_gt.jsonl",
             "controlnet_learning_rate":1e-6,"ra_fusion_learning_rate":2e-6,
             "beta":5000,"lr_warmup_steps":50,"train_batch_size":2,"gradient_accumulation_steps":4,
             "resume_from_checkpoint":"latest","run_checkpoint_validation":True,
             "validation_eval_config":"config/eval_sd3.yaml","seed":42}
    config = {"training":train,"model":{"controlnet_model_path":"SFT/controlnet","ra_fusion_path":"SFT/ra_fusion"},
              "spatial_dynamic":train["spatial_dynamic"],"reward":train["reward"],"preference_filter":train["preference_filter"]}
    state = {"status":"completed_real_models","runtime":{"executed":True,"policy_parameters_unchanged":True,
             "reference_unchanged":True,"buffers_unchanged":True,"measured_accumulation_windows":3},
             "snapshot":{"config":config,"metadata":{"version":6,"global_step":37000,"num_processes":1,"spatial_dynamic":{"settings":settings}}}}
    (root/"diagnostic_state.json").write_text(json.dumps(state),encoding="utf-8")
    rows=[]
    # Rounded figures from the example report, used solely to test algebra.
    for i,(weather,q,f,c) in enumerate((("haze",31.602,.043721,-.13446),("rain",1293.9,.049995,.063236),("snow",443.67,.093822,-.19434))):
        rows.append({"row_type":"accum_gradient","window":i,"weather":weather,"gt_weather":weather,
                     "scope":"all","status":"partial_missing_gradients","dpo_norm":q,"gt_norm":f,
                     "cosine":c,"gt_dot_dpo":q*f*c,"weighted_gt_norm":.1*f,"gt_due":True,
                     "q_missing":4,"f_missing":4,"q_missing_names_preview":json.dumps(["example.unused"]),
                     "f_missing_names_preview":json.dumps(["example.unused"])})
    with (root/"diagnostics.csv").open("w",newline="",encoding="utf-8") as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    return state


class FixedFidelityPlanTest(unittest.TestCase):
    def test_generated_full_configs_pass_real_project_configuration_contracts(self):
        from dpo.spatial_config import training_signature, preparation_contract
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            original=yaml.safe_load((Path(__file__).resolve().parents[1]/"config/dpo_sd3_reward_dynamic_texture.yaml").read_text(encoding="utf-8"))
            actual={**original["training"],"reward":original["reward"],"preference_filter":original["preference_filter"],
                    "spatial_dynamic":original["spatial_dynamic"],"tail_risk":original["tail_risk"]}
            original["training"]=actual
            evidence={"snapshot":{"config":original,"metadata":{"spatial_dynamic":training_signature(actual)}},
                      "candidate_lambda":1000,"original_lambda":.1}
            output=root/"plan";output.mkdir()
            prepared=generate_configs(evidence,output,root/"runs",200)
            for name,item in prepared.items():
                cfg=yaml.safe_load(Path(item["config"]).read_text(encoding="utf-8"))
                effective={**cfg["training"],"reward":cfg["reward"],"preference_filter":cfg["preference_filter"],
                           "spatial_dynamic":cfg["spatial_dynamic"],"tail_risk":cfg["tail_risk"]}
                signature=training_signature(effective)
                self.assertEqual(signature["settings"]["lambda_fidelity"], .1 if name=="baseline" else 1000)
                self.assertEqual(preparation_contract(cfg),preparation_contract(original))

    def test_boundary_and_candidate_projection_are_measured_point_algebra(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            fixture(root)
            evidence=read_evidence(root,1000)
            self.assertGreater(evidence["maximum_observed_all_parameter_sgd_boundary"],900)
            self.assertLess(evidence["maximum_observed_all_parameter_sgd_boundary"],950)
            self.assertEqual(evidence["rounded_boundary_candidate_not_optimum"],1000)
            self.assertTrue(all(row["candidate"]["gt_dot_total"]>=0 for row in evidence["measured"]))
            self.assertEqual(evidence["missing_gradient_audit"]["all"]["common_missing_name_preview"],["example.unused"])
            q,f,dot=2.,3.,-3.
            self.assertEqual(projected_metrics(q,f,dot,1/3)["gt_dot_total"],0)

    def test_only_lambda_differs_and_test_evaluation_is_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            state=fixture(root)
            evidence=read_evidence(root,1000)
            before=(root/"diagnostic_state.json").read_bytes()
            output=root/"plan";output.mkdir()
            result=generate_configs(evidence,output,root/"runs",200)
            configs=[yaml.safe_load(Path(result[name]["config"]).read_text(encoding="utf-8")) for name in ("baseline","fidelity")]
            for cfg in configs:
                self.assertFalse(cfg["training"]["run_checkpoint_validation"])
                self.assertIsNone(cfg["training"]["validation_eval_config"])
                self.assertIsNone(cfg["training"]["resume_from_checkpoint"])
                self.assertEqual(cfg["training"]["max_train_steps"],200)
                self.assertEqual(cfg["training"]["checkpointing_steps"],200)
                self.assertEqual(cfg["training"]["beta"],5000)
                self.assertEqual(cfg["training"]["controlnet_learning_rate"],1e-6)
                self.assertEqual(cfg["model"],state["snapshot"]["config"]["model"])
                self.assertNotIn("reward",cfg["training"])
                # Historical effective settings were materialized.
                self.assertEqual(cfg["spatial_dynamic"]["gt"]["sigma_max"],.7)
            self.assertEqual(configs[0]["spatial_dynamic"]["lambda_fidelity"],.1)
            self.assertEqual(configs[1]["spatial_dynamic"]["lambda_fidelity"],1000)
            self.assertEqual((root/"diagnostic_state.json").read_bytes(),before)
            self.assertFalse((root/"runs").exists())  # no training output or process created
            with self.assertRaises(FileExistsError):
                generate_configs(evidence,output,root/"runs",200)

    def test_inspect_only_or_nonfinite_or_missing_gradients_cannot_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);state=fixture(root)
            state["runtime"]["executed"]=False
            (root/"diagnostic_state.json").write_text(json.dumps(state),encoding="utf-8")
            with self.assertRaises(ValueError):read_evidence(root,1000)
            fixture(root)
            with self.assertRaises(ValueError):read_evidence(root,float("nan"))
            data=(root/"diagnostics.csv").read_text(encoding="utf-8").replace("partial_missing_gradients","nonfinite_gradient")
            (root/"diagnostics.csv").write_text(data,encoding="utf-8")
            with self.assertRaises(ValueError):read_evidence(root,1000)

    def test_validation_must_be_explicitly_held_out_and_probe_exceeds_warmup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root)
            validation=root/"validation.yaml"
            validation.write_text("splits: [test]\ndataset_rain: /data/test/rain\n",encoding="utf-8")
            with self.assertRaises(ValueError):verify_validation_config(validation)
            validation.write_text("splits: [val]\ndataset_rain: /data/validation/rain\n",encoding="utf-8")
            self.assertEqual(verify_validation_config(validation),str(validation.resolve()))
            evidence=read_evidence(root,1000)
            output=root/"plan";output.mkdir()
            with self.assertRaises(ValueError):generate_configs(evidence,output,root/"runs",50)
            result=generate_configs(evidence,output,root/"runs",200,validation)
            for item in result.values():
                cfg=yaml.safe_load(Path(item["config"]).read_text(encoding="utf-8"))
                self.assertTrue(cfg["training"]["run_checkpoint_validation"])
                self.assertTrue(cfg["training"]["validation_before_training"])


if __name__ == "__main__":
    unittest.main()
